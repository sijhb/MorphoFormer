# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset building utilities.
Mostly adapted from https://github.com/facebookresearch/deit/blob/main/datasets.py
"""

import os
import torch
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args, fold_idx=None, num_folds=5):
    """
    根据传入参数构建数据集及其类别数
    支持的数据集：SIPakMeD（5类）、Herlev（7类）、SIPakMeD2（2类）、Herlev2（2类）、SIPakMeD3（3类）、BCCD4(4类)
    """
    transform = build_transform(is_train, args)

    dataset_root = os.path.join(args.data_path, args.data_set)
    
    if args.data_set == 'SIPakMeD':
        nb_classes = 5
    elif args.data_set == 'Herlev':
        nb_classes = 7
    elif args.data_set == 'SIPakMeD2':
        nb_classes = 2
    elif args.data_set == 'Herlev2':
        nb_classes = 2
    elif args.data_set == 'SIPakMeD3':
        nb_classes = 3
    elif args.data_set == 'BCCD4':
        nb_classes = 4
    else:
        raise ValueError(f"Unknown dataset: {args.data_set}")

   # 如果是交叉验证模式，从train文件夹加载完整数据
    if fold_idx is not None and num_folds > 1:
        # 使用train文件夹作为完整数据集进行交叉验证
        dataset_root = os.path.join(args.data_path, args.data_set, 'train')
        full_dataset = datasets.ImageFolder(dataset_root, transform=transform)
        dataset = create_cross_validation_split(full_dataset, fold_idx, num_folds, is_train)
    else:
        # 原始的单次训练验证模式
        dataset_root = os.path.join(args.data_path, args.data_set, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(dataset_root, transform=transform)

    return dataset, nb_classes


def create_cross_validation_split(full_dataset, fold_idx, num_folds, is_train):
    """
    创建K折交叉验证的数据分割
    """
    # 获取所有样本的路径和标签
    samples = full_dataset.samples
    targets = [s[1] for s in samples]
    
    # 使用分层K折确保每折的类别分布一致
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # 生成所有fold的分割
    all_folds = list(skf.split(range(len(targets)), targets))
    
    # 获取当前fold的分割
    train_indices, val_indices = all_folds[fold_idx]
    
    if is_train:
        subset = torch.utils.data.Subset(full_dataset, train_indices)
    else:
        subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # 保留类别信息
    subset.classes = full_dataset.classes
    subset.class_to_idx = full_dataset.class_to_idx
    
    return subset


def build_transform(is_train, args):
    """
    构造数据预处理流程，训练和验证分别使用不同的增强和变换方式
    """
    resize_im = args.input_size > 32
    if is_train:
        # 训练时用timm自带的增强方法，兼容AutoAugment、color jitter、随机擦除等
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # 对小尺寸图片替换Resize为RandomCrop带padding
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    # 验证集使用resize + center crop + normalize
    t = []
    if resize_im:
        size = int(args.crop_ratio * args.input_size)
        # Resize保持长宽比例，再中心裁剪
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)