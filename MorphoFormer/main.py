# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import warnings
import logging
from pathlib import Path
from collections import defaultdict
import shutil

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
import utils
from cfg import get_args_parser
from models.new import DualDilateformer

# ============= 这里导入 Grad-CAM =============
from gradcam import GradCAM, save_cam_on_image

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 日志 ----------
def setup_logger(logfile="train_log.txt"):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger()


# ---------- 单个 fold 的训练流程 ----------
def run_single_fold(args, fold_idx=None, num_folds=1):
    utils.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + utils.get_rank() + (fold_idx * 100 if fold_idx is not None else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args, fold_idx=fold_idx, num_folds=num_folds)
    dataset_val, _ = build_dataset(is_train=False, args=args, fold_idx=fold_idx, num_folds=num_folds)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=val_sampler,
        batch_size=int(1.5 * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = (args.mixup > 0) or (args.cutmix > 0.) or (args.cutmix_minmax is not None)
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating model: {args.model}")
    model = DualDilateformer(num_classes=args.nb_classes)
    model.to(device)

    model_without_ddp = model
    optimizer = create_optimizer(args, model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    best_epoch = -1
    best_test_stats = None

    output_dir = Path(args.output_dir if args.output_dir else "./output")
    if fold_idx is not None:
        fold_output_dir = output_dir / f"fold_{fold_idx + 1}"
    else:
        fold_output_dir = output_dir / "single_training"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn, 1, args.distributed,
            amp=args.amp,
            finetune=args.finetune
        )

        lr_scheduler.step(epoch)
        test_stats = evaluate(data_loader_val, model, device, 1, logger, distributed=args.distributed, amp=args.amp)

        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            best_epoch = epoch
            best_test_stats = test_stats.copy()

        if utils.is_main_process():
            checkpoint_path = fold_output_dir / 'model_best.pth'
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'scaler': loss_scaler.state_dict(),
                'max_accuracy': max_accuracy,
                'test_acc1': float(test_stats['acc1']),
                'test_loss': float(test_stats['loss'])
            }
            utils.save_on_master(state_dict, checkpoint_path)

    return {"best_acc": max_accuracy, "best_epoch": best_epoch}


# ---------- Grad-CAM 流程 ----------
def run_gradcam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['im_Dyskeratotic', 'im_Koilocytotic', 'im_Metaplastic',
               'im_Parabasal', 'im_Superficial-Intermediate']

    # 模型
    model = DualDilateformer(num_classes=len(classes))
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()

    target_layer = list(model.modules())[-2]  # 选最后一层卷积
    cam_extractor = GradCAM(model, target_layer)

    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    for idx, (images, labels, paths) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # 针对真实标签
        cams_true, _ = cam_extractor(images, class_idx=labels)
        # 针对预测标签
        cams_pred, preds = cam_extractor(images, class_idx=None)

        for i in range(images.size(0)):
            img_path = paths[i]
            gt_class = classes[labels[i].item()]
            pred_class = classes[preds[i].item()]

            save_dir_true = Path(args.output_dir) / "true_label" / gt_class
            save_path_true = save_dir_true / f"{Path(img_path).stem}_true.png"
            save_cam_on_image(img_path, cams_true[i].unsqueeze(0), save_path_true)

            save_dir_pred = Path(args.output_dir) / "pred_label" / pred_class
            save_path_pred = save_dir_pred / f"{Path(img_path).stem}_pred.png"
            save_cam_on_image(img_path, cams_pred[i].unsqueeze(0), save_path_pred)

        if idx % 20 == 0:
            print(f"👉 已处理 {idx}/{len(data_loader)} 张图像")

    print(f"🎉 Grad-CAM 热力图保存到 {args.output_dir}")


# ---------- main ----------
def main(args):
    if args.gradcam:
        run_gradcam(args)
    else:
        run_single_fold(args)


if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument("--gradcam", action="store_true", help="运行 Grad-CAM 可视化而不是训练")
    parser.add_argument("--resume", default="./output/model_best.pth", type=str, help="模型权重路径")
    parser.add_argument("--output-dir", default="./GradCAM", type=str, help="保存热力图路径")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
