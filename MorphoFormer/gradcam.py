# gradcam.py
import os
from pathlib import Path
import argparse
import importlib
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np

from datasets import build_dataset

# -------------------------
# 自动寻找目标层（最后一个 Conv2d）
# -------------------------
def find_target_layer(model: nn.Module):
    target = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target = m
    if target is None:
        raise RuntimeError("❌ 没有找到 Conv2d 层，请手动在 gradcam.py 里指定 target_layer（或在模型里选一个合适的卷积层）")
    return target


# -------------------------
# Grad-CAM 实现
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)  # [N, num_classes]
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        scores = output.gather(1, class_idx.view(-1, 1)).squeeze(1)
        self.model.zero_grad()
        scores.backward(torch.ones_like(scores), retain_graph=True)

        grads = self.gradients
        activs = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activs, dim=1)
        cam = F.relu(cam)

        n, h, w = cam.shape
        cam = cam.view(n, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(n, 1, h, w)
        return cam, output.argmax(dim=1)


# -------------------------
# 工具函数：保存热力图叠加图
# -------------------------
def save_cam_on_image(img_tensor, cam, save_path):
    # 添加调试信息
    print(f"DEBUG: cam shape: {cam.shape}, dim: {cam.dim()}")
    print(f"DEBUG: img_tensor shape: {img_tensor.shape}")
    
    # 修复维度问题
    if cam.dim() == 1:
        # 如果cam是1D，尝试reshape为2D
        side_len = int(cam.shape[0] ** 0.5)
        if side_len * side_len == cam.shape[0]:
            cam = cam.reshape(1, 1, side_len, side_len)
        else:
            # 如果无法reshape，创建默认的cam
            h, w = 14, 14  # 默认大小
            cam = torch.ones(1, 1, h, w) * 0.5
    elif cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif cam.dim() == 3:
        if cam.shape[0] == 1:
            cam = cam.unsqueeze(0)  # [1, 1, H, W]
        else:
            cam = cam.unsqueeze(1)  # [B, 1, H, W]
    
    print(f"DEBUG: after fix cam shape: {cam.shape}")
    
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.uint8(255 * img)

    h, w, _ = img.shape
    
    # 插值
    cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
    cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) * 0.4 + np.float32(img) * 0.6
    overlay = np.uint8(overlay)

    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
    
    # 创建对比图：原图 | 热力图叠加
    margin = 5  # 图像间距
    comparison = np.ones((h, w * 2 + margin, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 左侧：原图（转换为BGR格式）
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    comparison[:, :w] = img_bgr
    
    # 右侧：热力图叠加（转换为BGR格式）
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    comparison[:, w + margin:] = overlay_bgr

    # 保存对比图
    cv2.imwrite(str(save_path), comparison)


# -------------------------
# 主流程：生成真实标签与预测标签两套 CAM
# -------------------------
def run_gradcam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = [
        "im_Dyskeratotic",
        "im_Koilocytotic",
        "im_Metaplastic",
        "im_Parabasal",
        "im_Superficial-Intermediate",
    ]

    # ========== 简化的模型导入逻辑 ==========
    try:
        # 直接导入 new.py 中的 DualDilateformer
        from models.new import DualDilateformer
        print("✅ 成功导入 DualDilateformer")
        
    except ImportError as e:
        # 如果失败，尝试其他导入方式
        try:
            # 尝试从 df.py 导入
            from models.df import DualDilateformer
            print("✅ 从 df.py 导入 DualDilateformer 成功")
        except ImportError:
            try:
                # 尝试导入 Dilateformer
                from models.df import Dilateformer as DualDilateformer
                print("✅ 使用 Dilateformer 作为 DualDilateformer")
            except ImportError as e:
                msg = (
                    f"❌ 导入模型类失败：{e}\n\n"
                    "排查建议：\n"
                    "1) 确认 models/new.py 中有 `class DualDilateformer(nn.Module):`\n"
                    "2) 或者 models/df.py 中有 `class DualDilateformer(nn.Module):` 或 `class Dilateformer(nn.Module):`\n"
                    "3) 运行以下命令测试导入：\n"
                    "   python -c \"from models.new import DualDilateformer; print('导入成功')\"\n"
                    "4) 检查模型文件是否存在：ls models/*.py\n"
                )
                raise RuntimeError(msg) from e

    print("🔧 正在加载模型权重...")
    model = DualDilateformer(num_classes=len(classes))
    ckpt = torch.load(args.resume, map_location="cpu")
    
    # 适配不同的checkpoint格式
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
        
    model.to(device).eval()
    print("✅ 模型加载完成！")

    # 选择目标层
    target_layer = find_target_layer(model)
    print(f"🎯 使用的目标层：{target_layer.__class__.__name__}")
    cam_extractor = GradCAM(model, target_layer)

    # 注意：很多仓库的 build_dataset 返回 (train, val, nb_classes)
    ds_tuple = build_dataset(is_train=False, args=args)
    if isinstance(ds_tuple, (list, tuple)) and len(ds_tuple) == 3:
        _, dataset_val, _ = ds_tuple
    else:
        # 如果你的 build_dataset 返回的是 (val, nb_classes)
        dataset_val = ds_tuple[0] if isinstance(ds_tuple, (list, tuple)) else ds_tuple

    data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    print("📊 开始生成 Grad-CAM 热力图...")
    for idx, batch in enumerate(data_loader):
        # 兼容两种 DataLoader 输出：(images, labels) 或 (images, labels, paths)
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = [None] * images.size(0)

        images, labels = images.to(device), labels.to(device)

        # 真实标签 CAM
        cams_true, _ = cam_extractor(images, class_idx=labels)
        # 预测标签 CAM
        cams_pred, preds = cam_extractor(images, class_idx=None)

        for i in range(images.size(0)):
            gt_name = classes[labels[i].item()]
            pred_name = classes[preds[i].item()]

            # 文件名友好化
            stem = f"img{idx:05d}"
            if paths[i] is not None:
                p = Path(paths[i])
                stem = p.stem

            # 真实标签
            out_dir_true = Path(args.output_dir) / "true_label" / gt_name
            out_path_true = out_dir_true / f"{stem}_true.png"
            save_cam_on_image(images[i], cams_true[i], out_path_true)

            # 预测标签
            out_dir_pred = Path(args.output_dir) / "pred_label" / pred_name
            out_path_pred = out_dir_pred / f"{stem}_pred.png"
            save_cam_on_image(images[i], cams_pred[i], out_path_pred)

        if idx % 20 == 0:
            print(f"👉 已处理 {idx}/{len(data_loader)} 张图像")

    print(f"🎉 Grad-CAM 全部生成完成！保存路径：{args.output_dir}")


# -------------------------
# 脚本入口
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grad-CAM 可视化")
    parser.add_argument("--resume", default="./output/single_training/model_best.pth", type=str, help="模型权重路径")
    parser.add_argument("--data-path", default="./dataset", type=str, help="数据集路径")
    parser.add_argument("--data-set", default="SIPakMeD", type=str, help="数据集名称")
    parser.add_argument("--output-dir", default="./GradCAM", type=str, help="保存热力图路径")
    parser.add_argument("--input-size", default=224, type=int, help="输入图像尺寸")
    parser.add_argument("--crop-ratio", default=0.875, type=float, help="随机裁剪比例（默认 0.875）")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_gradcam(args)