import math
import os
import time
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from einops import rearrange
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
import utils

logger = logging.getLogger('train')

# 全局变量用于跟踪最佳性能
best_acc = 0.0
best_epoch = 0


def get_class_names(data_loader):
    """获取数据集的类别名称，兼容 Subset 和普通 Dataset"""
    dataset = data_loader.dataset

    # 处理 Subset 对象
    if isinstance(dataset, torch.utils.data.Subset):
        # 尝试从 Subset 或其原始 dataset 中获取 classes
        if hasattr(dataset, 'classes'):
            return dataset.classes
        elif hasattr(dataset.dataset, 'classes'):
            return dataset.dataset.classes
    # 处理普通 Dataset 对象
    elif hasattr(dataset, 'classes'):
        return dataset.classes

    # fallback：如果有 nb_classes 属性，用索引列表表示类别名
    nb = getattr(dataset, 'nb_classes', None)
    if nb is not None:
        return [str(i) for i in range(nb)]

    # 最后退回到 dataset 的长度估计（不太理想，但可用）
    try:
        # 如果 dataset 有 targets 或 samples，从中推断类别数
        if hasattr(dataset, 'targets'):
            labels = getattr(dataset, 'targets')
            return sorted(list(set([str(x) for x in labels])))
    except Exception:
        pass

    # 默认返回占位类名
    return [str(i) for i in range(1000)]


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    finetune: bool = False):
    """训练一个 epoch"""
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            raise ValueError(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if amp:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            if max_norm and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        try:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        except Exception:
            pass

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)

    logger.info(f"Epoch {epoch} - Loss: {metric_logger.loss.global_avg:.4f}, "
                f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, world_size=1, logger=None, distributed=True, amp=False, epoch=0, fold=0):
    """评估模型性能"""
    global best_acc, best_epoch

    if logger is None:
        _logger = logging.getLogger('train')
    else:
        _logger = logger

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('precision', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('recall', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('f1', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    class_names = get_class_names(data_loader)
    num_classes = len(class_names)

    outputs_batches = []
    targets_batches = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            output = model(images)

        if distributed and torch.distributed.is_initialized():
            gathered_out = concat_all_gather(output)
            gathered_tgt = concat_all_gather(target)
            outputs_batches.append(gathered_out)
            targets_batches.append(gathered_tgt)
        else:
            outputs_batches.append(output)
            targets_batches.append(target)

    if len(outputs_batches) == 0:
        _logger.warning("No outputs collected during evaluation (empty dataloader?)")
        return {'acc1': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': 0.0}

    outputs = torch.cat(outputs_batches, dim=0)
    targets = torch.cat(targets_batches, dim=0)

    try:
        num_data = len(data_loader.dataset)
        outputs = outputs[:num_data]
        targets = targets[:num_data]
    except Exception:
        pass

    acc1 = accuracy(outputs, targets, topk=(1,))[0]
    preds = outputs.argmax(dim=1).cpu().numpy()
    labels = targets.cpu().numpy()

    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    real_loss = criterion(outputs, targets)

    metric_logger.update(loss=real_loss.item())
    metric_logger.meters['acc1'].update(acc1.item())
    metric_logger.meters['precision'].update(precision)
    metric_logger.meters['recall'].update(recall)
    metric_logger.meters['f1'].update(f1)

    os.makedirs('plots', exist_ok=True)
    
    current_acc = metric_logger.acc1.global_avg
    is_best = current_acc > best_acc
    
    if is_best:
        best_acc = current_acc
        best_epoch = epoch
        
        # 保持混淆矩阵和ROC曲线的更新逻辑不变
        for file in os.listdir('plots'):
            if file.startswith('best_'):
                os.remove(os.path.join('plots', file))
        
        roc_fname = f'plots/best_roc_curve_epoch{epoch}_acc{current_acc:.4f}.png'
        cm_fname = f'plots/best_confusion_matrix_epoch{epoch}_acc{current_acc:.4f}.png'
        
        _logger.info(f'🎯 新的最佳模型！准确率: {current_acc:.4f}, 正在保存评估结果...')
    else:
        roc_fname = 'plots/roc_curve5(S).png'
        cm_fname = 'plots/confusion_matrix5(S).png'

    # ROC曲线绘制 - 修改的核心部分
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    y_true = labels

    plt.figure(figsize=(10, 8))
    fpr, tpr, roc_auc = {}, {}, {}

    # 根据分类数量选择合适的颜色映射
    if num_classes <= 10:
        # 使用 tab10 色图，最多10种颜色
        colors = cycle(plt.cm.tab10(np.linspace(0, 1, num_classes)))
    elif num_classes <= 20:
        # 使用 tab20 色图，最多20种颜色
        colors = cycle(plt.cm.tab20(np.linspace(0, 1, num_classes)))
    else:
        # 对于超过20类的情况，使用连续的色图
        colors = cycle(plt.cm.viridis(np.linspace(0, 1, num_classes)))

    # 为每个类别绘制ROC曲线
    for i, color in zip(range(num_classes), colors):
        try:
            fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i], tpr[i],
                color=color,
                linewidth=2,
                label=f'{class_names[i]} (AUC={roc_auc[i]:.4f})'
            )
        except ValueError:
            _logger.warning(f"Class {i} 无有效样本，跳过ROC绘制")
            continue

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    # 动态设置标题，显示分类数量
    plt.title(f'ROC Curve ({num_classes}-classes)', fontsize=14, fontweight='bold')
    
    # 调整坐标轴范围
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 调整图例位置，避免遮挡曲线
    if num_classes <= 5:
        plt.legend(loc="lower right", fontsize=10, frameon=True)
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_fname, dpi=300, bbox_inches='tight')
    plt.close()

    # 混淆矩阵绘制
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(max(8, num_classes), max(6, num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix ({num_classes}-classes)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_fname, dpi=300, bbox_inches='tight')
    plt.close()

    _logger.info('* Acc@1 {:.3f}  Precision {:.3f} Recall {:.3f} F1 {:.3f} Loss {:.3f}'.format(
        metric_logger.acc1.global_avg,
        metric_logger.precision.global_avg,
        metric_logger.recall.global_avg,
        metric_logger.f1.global_avg,
        metric_logger.loss.global_avg
    ))

    return {
        'acc1': metric_logger.acc1.global_avg,
        'precision': metric_logger.precision.global_avg,
        'recall': metric_logger.recall.global_avg,
        'f1': metric_logger.f1.global_avg,
        'loss': metric_logger.loss.global_avg,
        'roc_plot': roc_fname,
        'confusion_matrix_plot': cm_fname,
        'is_best': is_best
    }


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor):
    """分布式环境下收集 tensor"""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensor

    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def save_best_model(model, optimizer, epoch, acc, fold=0, model_dir='checkpoints'):
    """保存最佳模型：删除对应fold的旧pth文件，保存新的最佳模型"""
    global best_acc, best_epoch
    
    os.makedirs(model_dir, exist_ok=True)
    
    # 删除当前fold下所有旧的最佳模型文件
    old_model_files = [f for f in os.listdir(model_dir) if f.startswith(f'best_model_fold{fold}_')]
    for old_file in old_model_files:
        old_file_path = os.path.join(model_dir, old_file)
        try:
            os.remove(old_file_path)
            logger.info(f"已删除旧的最佳模型文件: {old_file_path}")
        except Exception as e:
            logger.warning(f"删除旧模型文件失败: {old_file_path}, 错误: {e}")
    
    # 保存新的最佳模型
    model_path = os.path.join(model_dir, f'best_model_fold{fold}_epoch{epoch}_acc{acc:.4f}.pth')
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': acc,
        'fold': fold
    }
    torch.save(save_dict, model_path)
    logger.info(f"已保存新的最佳模型至: {model_path}")