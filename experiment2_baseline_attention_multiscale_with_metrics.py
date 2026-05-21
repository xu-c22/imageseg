# -*- coding: utf-8 -*-
import os
import time
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 路径与训练参数
# =========================
TRAIN_CT_ROOT = r"U:\xucheng\image-seg\CHAOS_Train_Sets\Train_Sets\CT"
SAVE_DIR = r"U:\xucheng\image-seg\experiment2_attention_multiscale_runs"

IMG_SIZE = 256

SEED = 42
VAL_CASE_RATIO = 0.2
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 30

LR = 3e-4
WEIGHT_DECAY = 3e-5

BCE_WEIGHT = 0.3
DICE_WEIGHT = 0.7

SAVE_VAL_PRED = True
VAL_PRED_MAX_SAVE = 30

EARLY_STOP_PATIENCE = 14
GRAD_CLIP_NORM = 1.0

THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
POSTPROCESS_MAX_COMPONENT = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 数据读取
# =========================
def collect_case_samples(ct_root: str) -> Dict[str, List[Tuple[str, str]]]:
    root = Path(ct_root)
    if not root.exists():
        raise FileNotFoundError(f"训练集根目录不存在: {root}")

    case_dict = {}
    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    case_dirs = sorted(case_dirs, key=lambda x: x.name)

    for case_dir in case_dirs:
        image_dir = case_dir / "Image_png"
        mask_dir = case_dir / "Ground_bin"

        if not image_dir.exists() or not mask_dir.exists():
            print(f"[跳过] 病例 {case_dir.name} 缺少 Image_png 或 Ground_bin")
            continue

        img_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() == ".png"])
        mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"])

        if len(img_files) == 0 or len(mask_files) == 0:
            print(f"[跳过] 病例 {case_dir.name} 中图像或标签为空")
            continue

        img_name_set = {p.name for p in img_files}
        mask_name_set = {p.name for p in mask_files}
        common_names = sorted(list(img_name_set & mask_name_set))

        if len(common_names) == 0:
            print(f"[跳过] 病例 {case_dir.name} 中图像与标签无同名文件")
            continue

        pairs = []
        for name in common_names:
            img_path = str(image_dir / name)
            mask_path = str(mask_dir / name)
            pairs.append((img_path, mask_path))

        case_dict[case_dir.name] = pairs
        print(f"[病例 {case_dir.name}] 配对样本数: {len(pairs)}")

    if len(case_dict) == 0:
        raise RuntimeError("未找到可用病例，请检查 Image_png / Ground_bin 是否准备好")

    return case_dict


def split_cases(case_dict: Dict[str, List[Tuple[str, str]]], val_ratio=0.2, seed=42):
    case_ids = sorted(list(case_dict.keys()), key=lambda x: int(x) if x.isdigit() else x)
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    n_val = max(1, int(len(case_ids) * val_ratio))
    val_case_ids = case_ids[:n_val]
    train_case_ids = case_ids[n_val:]

    train_samples = []
    val_samples = []

    for cid in train_case_ids:
        train_samples.extend(case_dict[cid])
    for cid in val_case_ids:
        val_samples.extend(case_dict[cid])

    print("\n========== 数据划分 ==========")
    print("训练病例:", train_case_ids)
    print("验证病例:", val_case_ids)
    print(f"训练切片数: {len(train_samples)}")
    print(f"验证切片数: {len(val_samples)}")
    print("=============================\n")

    return train_case_ids, val_case_ids, train_samples, val_samples


def resize_image_mask(img: np.ndarray, mask: np.ndarray, size: int):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def random_flip(img: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    if random.random() < 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()
    return img, mask


def random_rotate_90(img: np.ndarray, mask: np.ndarray):
    k = random.randint(0, 3)
    img = np.rot90(img, k).copy()
    mask = np.rot90(mask, k).copy()
    return img, mask


def normalize_image(img: np.ndarray):
    img = img.astype(np.float32) / 255.0
    return img


class ChaosCT2DDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]], img_size=256, train=True):
        self.samples = samples
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"图像读取失败: {img_path}")
        if mask is None:
            raise RuntimeError(f"标签读取失败: {mask_path}")

        mask = (mask > 0).astype(np.uint8)

        img, mask = resize_image_mask(img, mask, self.img_size)

        if self.train:
            img, mask = random_flip(img, mask)
            img, mask = random_rotate_90(img, mask)

        img = normalize_image(img)

        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": img,
            "mask": mask,
            "img_path": img_path,
            "mask_path": mask_path
        }


# =========================
# 模型模块
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class MultiScaleBottleneck(nn.Module):
    """
    不做分支注意力，只做多尺度并行增强
    """
    def __init__(self, in_ch):
        super().__init__()
        branch_ch = in_ch // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        self.refine = DoubleConv(in_ch, in_ch)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fuse(out)
        out = out + x
        out = self.refine(out)
        return out


class UNetAttentionMS2D(nn.Module):
    """
    实验2：
    Baseline + Attention + Multi-Scale
    先 bottleneck attention，再 multi-scale bottleneck
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.attn = SEBlock(base_ch * 16, reduction=16)
        self.ms_bottleneck = MultiScaleBottleneck(base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch, base_ch)

        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.attn(x5)
        x5 = self.ms_bottleneck(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


# =========================
# 损失函数
# =========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# =========================
# 评估指标
# =========================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def reset_dir(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def postprocess_largest_component(mask_2d: np.ndarray) -> np.ndarray:
    mask_2d = (mask_2d > 0).astype(np.uint8)

    if mask_2d.sum() == 0:
        return mask_2d

    num_labels, labels = cv2.connectedComponents(mask_2d)

    if num_labels <= 1:
        return mask_2d

    max_area = 0
    max_label = 0
    for label_id in range(1, num_labels):
        area = (labels == label_id).sum()
        if area > max_area:
            max_area = area
            max_label = label_id

    out = (labels == max_label).astype(np.uint8)
    return out


def compute_binary_metrics_np(pred: np.ndarray, target: np.ndarray, eps=1e-6):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    intersection = np.logical_and(pred == 1, target == 1).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    return dice, iou, precision, recall


def get_boundary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded
    return (boundary > 0).astype(np.uint8)


def compute_boundary_f1(pred: np.ndarray, target: np.ndarray, dilation_ratio=0.003):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    pred_b = get_boundary(pred)
    gt_b = get_boundary(target)

    h, w = pred.shape
    diag = np.sqrt(h * h + w * w)
    dilation = max(1, int(round(dilation_ratio * diag)))
    kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)

    pred_dil = cv2.dilate(pred_b, kernel, iterations=1)
    gt_dil = cv2.dilate(gt_b, kernel, iterations=1)

    gt_match = gt_b * pred_dil
    pred_match = pred_b * gt_dil

    n_pred = pred_b.sum()
    n_gt = gt_b.sum()

    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    precision = pred_match.sum() / (n_pred + 1e-6)
    recall = gt_match.sum() / (n_gt + 1e-6)
    bf1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(bf1)


def _surface_distances(pred: np.ndarray, target: np.ndarray):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    pred_b = get_boundary(pred)
    gt_b = get_boundary(target)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        # 一方空，一方非空，给很大惩罚
        h, w = pred.shape
        diag = np.sqrt(h * h + w * w)
        return np.array([diag], dtype=np.float32), np.array([diag], dtype=np.float32)

    dt_gt = cv2.distanceTransform((1 - gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((1 - pred_b).astype(np.uint8), cv2.DIST_L2, 3)

    d_pred_to_gt = dt_gt[pred_b > 0]
    d_gt_to_pred = dt_pred[gt_b > 0]
    return d_pred_to_gt.astype(np.float32), d_gt_to_pred.astype(np.float32)


def compute_hd95_assd(pred: np.ndarray, target: np.ndarray):
    d1, d2 = _surface_distances(pred, target)
    all_d = np.concatenate([d1, d2], axis=0)

    hd95 = float(np.percentile(all_d, 95))
    assd = float((d1.mean() + d2.mean()) / 2.0)
    return hd95, assd


def save_mask_visual(image_tensor, gt_np, pred_np, save_path):
    image = image_tensor.squeeze().cpu().numpy() * 255.0
    image = image.astype(np.uint8)

    gt = (gt_np * 255).astype(np.uint8)
    pred = (pred_np * 255).astype(np.uint8)

    canvas = np.concatenate([image, gt, pred], axis=1)
    cv2.imwrite(str(save_path), canvas)


@torch.no_grad()
def compute_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device,
                       thresholds=None,
                       postprocess=False,
                       measure_time=False):
    model.eval()

    if thresholds is None:
        thresholds = [0.5]

    total_loss = 0.0
    n_batches = 0

    threshold_stats = {
        thr: {
            "dice_sum": 0.0,
            "iou_sum": 0.0,
            "precision_sum": 0.0,
            "recall_sum": 0.0,
            "hd95_sum": 0.0,
            "assd_sum": 0.0,
            "bf1_sum": 0.0,
            "count": 0,
        }
        for thr in thresholds
    }

    infer_time_sum = 0.0
    infer_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if measure_time and device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(images)

        if measure_time and device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if measure_time:
            infer_time_sum += (t1 - t0)
            infer_count += images.size(0)

        loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        targets = masks.detach().cpu().numpy()[:, 0]
        targets = (targets > 0.5).astype(np.uint8)

        bs = probs.shape[0]

        for thr in thresholds:
            preds = (probs > thr).astype(np.uint8)

            for i in range(bs):
                pred_i = preds[i]
                gt_i = targets[i]

                if postprocess:
                    pred_i = postprocess_largest_component(pred_i)

                dice, iou, precision, recall = compute_binary_metrics_np(pred_i, gt_i)
                hd95, assd = compute_hd95_assd(pred_i, gt_i)
                bf1 = compute_boundary_f1(pred_i, gt_i)

                threshold_stats[thr]["dice_sum"] += dice
                threshold_stats[thr]["iou_sum"] += iou
                threshold_stats[thr]["precision_sum"] += precision
                threshold_stats[thr]["recall_sum"] += recall
                threshold_stats[thr]["hd95_sum"] += hd95
                threshold_stats[thr]["assd_sum"] += assd
                threshold_stats[thr]["bf1_sum"] += bf1
                threshold_stats[thr]["count"] += 1

    best_thr = thresholds[0]
    best_dice = -1.0
    best_metrics = None

    for thr in thresholds:
        cnt = max(threshold_stats[thr]["count"], 1)
        dice = threshold_stats[thr]["dice_sum"] / cnt
        iou = threshold_stats[thr]["iou_sum"] / cnt
        precision = threshold_stats[thr]["precision_sum"] / cnt
        recall = threshold_stats[thr]["recall_sum"] / cnt
        hd95 = threshold_stats[thr]["hd95_sum"] / cnt
        assd = threshold_stats[thr]["assd_sum"] / cnt
        bf1 = threshold_stats[thr]["bf1_sum"] / cnt

        if dice > best_dice:
            best_dice = dice
            best_thr = thr
            best_metrics = {
                "loss": total_loss / max(n_batches, 1),
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "hd95": hd95,
                "assd": assd,
                "bf1": bf1,
                "best_threshold": best_thr,
            }

    if measure_time:
        ms_per_image = (infer_time_sum / max(infer_count, 1)) * 1000.0
        best_metrics["ms_per_image"] = ms_per_image

    return best_metrics


@torch.no_grad()
def export_validation_predictions(model, loader, device, save_dir,
                                  threshold=0.5,
                                  postprocess=False,
                                  max_save=30):
    model.eval()
    reset_dir(Path(save_dir))

    saved_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        targets = masks.detach().cpu().numpy()[:, 0]
        targets = (targets > 0.5).astype(np.uint8)

        bs = probs.shape[0]
        preds = (probs > threshold).astype(np.uint8)

        for i in range(bs):
            if saved_count >= max_save:
                return

            pred_i = preds[i]
            gt_i = targets[i]

            if postprocess:
                pred_i = postprocess_largest_component(pred_i)

            img_name = Path(batch["img_path"][i]).stem
            save_path = Path(save_dir) / f"{img_name}.png"
            save_mask_visual(images[i], gt_i, pred_i, save_path)
            saved_count += 1


# =========================
# 训练函数
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        metrics = compute_metrics_from_logits(logits.detach(), masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
        "precision": total_precision / max(n_batches, 1),
        "recall": total_recall / max(n_batches, 1),
    }


# =========================
# 主函数
# =========================
def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    val_vis_dir = Path(SAVE_DIR) / "val_pred_best"

    print(f"Using device: {DEVICE}")
    print(f"Train root: {TRAIN_CT_ROOT}")

    case_dict = collect_case_samples(TRAIN_CT_ROOT)
    _, _, train_samples, val_samples = split_cases(case_dict, VAL_CASE_RATIO, SEED)

    train_dataset = ChaosCT2DDataset(train_samples, img_size=IMG_SIZE, train=True)
    val_dataset = ChaosCT2DDataset(val_samples, img_size=IMG_SIZE, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    model = UNetAttentionMS2D(in_ch=1, out_ch=1, base_ch=32).to(DEVICE)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    best_dice = -1.0
    best_thr = 0.5
    no_improve_epochs = 0

    log_txt = Path(SAVE_DIR) / "train_log.txt"
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,train_dice,train_iou,"
            "val_loss,val_dice,val_iou,val_precision,val_recall,val_hd95,val_assd,val_bf1,val_best_thr,lr\n"
        )

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE,
            thresholds=THRESHOLDS,
            postprocess=POSTPROCESS_MAX_COMPONENT,
            measure_time=False
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["dice"])

        msg = (
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
            f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, "
            f"HD95: {val_metrics['hd95']:.4f}, ASSD: {val_metrics['assd']:.4f}, BF1: {val_metrics['bf1']:.4f}, "
            f"Thr: {val_metrics['best_threshold']:.2f} | LR: {current_lr:.6f}"
        )
        print(msg)

        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},"
                f"{train_metrics['loss']:.6f},{train_metrics['dice']:.6f},{train_metrics['iou']:.6f},"
                f"{val_metrics['loss']:.6f},{val_metrics['dice']:.6f},{val_metrics['iou']:.6f},"
                f"{val_metrics['precision']:.6f},{val_metrics['recall']:.6f},"
                f"{val_metrics['hd95']:.6f},{val_metrics['assd']:.6f},{val_metrics['bf1']:.6f},"
                f"{val_metrics['best_threshold']:.2f},{current_lr:.8f}\n"
            )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_thr = val_metrics["best_threshold"]
            no_improve_epochs = 0

            best_path = Path(SAVE_DIR) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_dice,
                    "best_threshold": best_thr,
                    "img_size": IMG_SIZE,
                },
                best_path,
            )

            print(f"[保存最佳模型] Epoch={epoch}, Val Dice={best_dice:.4f}, Best Thr={best_thr:.2f}")

            if SAVE_VAL_PRED:
                export_validation_predictions(
                    model,
                    val_loader,
                    DEVICE,
                    val_vis_dir,
                    threshold=best_thr,
                    postprocess=POSTPROCESS_MAX_COMPONENT,
                    max_save=VAL_PRED_MAX_SAVE
                )

        else:
            no_improve_epochs += 1
            print(f"[未提升] 连续 {no_improve_epochs} 个 epoch 未刷新 best")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"[Early Stop] 连续 {EARLY_STOP_PATIENCE} 个 epoch 未提升，停止训练")
            break

    # 最终再用最佳权重做一次完整评估（含推理时间）
    best_ckpt = torch.load(Path(SAVE_DIR) / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    final_metrics = validate_one_epoch(
        model,
        val_loader,
        criterion,
        DEVICE,
        thresholds=THRESHOLDS,
        postprocess=POSTPROCESS_MAX_COMPONENT,
        measure_time=True
    )

    last_path = Path(SAVE_DIR) / "last_model.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_dice,
            "best_threshold": best_thr,
            "img_size": IMG_SIZE,
        },
        last_path,
    )

    summary_path = Path(SAVE_DIR) / "final_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== Experiment 2: Baseline + Attention + Multi-Scale =====\n")
        f.write(f"Best Val Dice: {final_metrics['dice']:.6f}\n")
        f.write(f"Best Threshold: {final_metrics['best_threshold']:.2f}\n")
        f.write(f"IoU: {final_metrics['iou']:.6f}\n")
        f.write(f"Precision: {final_metrics['precision']:.6f}\n")
        f.write(f"Recall: {final_metrics['recall']:.6f}\n")
        f.write(f"HD95: {final_metrics['hd95']:.6f}\n")
        f.write(f"ASSD: {final_metrics['assd']:.6f}\n")
        f.write(f"Boundary F1: {final_metrics['bf1']:.6f}\n")
        f.write(f"Params: {total_params}\n")
        f.write(f"Trainable Params: {trainable_params}\n")
        f.write(f"Inference Time (ms/image): {final_metrics['ms_per_image']:.6f}\n")

    print("\n训练完成。")
    print(f"最佳验证 Dice: {final_metrics['dice']:.4f}")
    print(f"最佳阈值: {final_metrics['best_threshold']:.2f}")
    print(f"HD95: {final_metrics['hd95']:.4f}")
    print(f"ASSD: {final_metrics['assd']:.4f}")
    print(f"Boundary F1: {final_metrics['bf1']:.4f}")
    print(f"Params: {total_params:,}")
    print(f"Inference Time (ms/image): {final_metrics['ms_per_image']:.4f}")
    print(f"模型保存目录: {SAVE_DIR}")


if __name__ == "__main__":
    main()
