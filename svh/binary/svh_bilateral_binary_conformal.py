#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilateral (two-hand) SvH Binary Classification (0 vs >0) with a ViT backbone.

GitHub-safe version:
- ✅ Removes absolute/local machine paths (use CLI args or environment variables)
- ✅ Removes run commands that may reveal environment specifics
- ✅ Keeps the same logic: build bilateral dataset from joint CSV, patient-level split, AMP, split conformal

Task definition (binary):
- erosion: keep bilateral images with exactly 26 valid erosion joints; label = (sum > 0)
- jsn: keep bilateral images with exactly 18 valid jsn joints; label = (sum > 0)

Selection:
- Save best by VAL AUC (avoid test leakage)
- Print conformal stats on VAL(calib-self) and TEST
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import AutoModel, AutoImageProcessor, AutoConfig
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


# =========================
# Args / Config
# =========================
def parse_args():
    import argparse

    p = argparse.ArgumentParser()

    # task
    p.add_argument("--task", type=str, default=os.environ.get("TASK", "jsn"),
                   choices=["erosion", "jsn"], help="Binary task: 0 vs >0 based on bilateral sum.")

    # expected joints
    p.add_argument("--expected_joints_erosion", type=int, default=int(os.environ.get("EXPECTED_JOINTS_EROSION", "26")))
    p.add_argument("--expected_joints_jsn", type=int, default=int(os.environ.get("EXPECTED_JOINTS_JSN", "18")))

    # data
    p.add_argument("--csv_path", type=str, default=os.environ.get("CSV_PATH", ""),
                   help="Joint-level CSV path with patient_id/timepoint/side/joint_type/index/file_name/Erosion_score/JSN_score.")
    p.add_argument("--whole_img_root", type=str, default=os.environ.get("WHOLE_IMG_ROOT", ""),
                   help="Directory containing bilateral whole images referenced by inferred whole_image names.")

    # model init
    p.add_argument("--dinov3_model_name", type=str,
                   default=os.environ.get("DINOV3_MODEL_NAME", "facebook/dinov3-vitb16-pretrain-lvd1689m"))
    p.add_argument("--fm_ckpt_path", type=str, default=os.environ.get("FM_CKPT_PATH", ""),
                   help="Foundation checkpoint containing key 'ema_state' (optional if --random_init).")
    p.add_argument("--random_init", action="store_true",
                   help="Randomly initialize encoder (ignores --fm_ckpt_path).")

    # train
    p.add_argument("--img_size", type=int, default=int(os.environ.get("IMG_SIZE", "224")))
    p.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", "64")))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "200")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-5")))
    p.add_argument("--freeze_encoder", action="store_true", help="Train heads only.")
    p.add_argument("--pos_weight_main", type=float, default=float(os.environ.get("POS_WEIGHT_MAIN", "5.0")))

    # split
    p.add_argument("--seed", type=int, default=int(os.environ.get("SPLIT_SEED", "3407")))
    p.add_argument("--train_ratio", type=float, default=float(os.environ.get("TRAIN_RATIO", "0.8")))
    p.add_argument("--val_ratio", type=float, default=float(os.environ.get("VAL_RATIO", "0.1")))

    # conformal
    p.add_argument("--use_conformal", action="store_true", help="Enable split conformal prediction (calib=val).")
    p.add_argument("--conformal_alpha", type=float, default=float(os.environ.get("CONFORMAL_ALPHA", "0.10")))

    # io
    p.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "./outputs/svh_bilateral_binary"),
                   help="Directory to save checkpoints/logs (state_dict).")

    args = p.parse_args()

    if not args.csv_path:
        raise ValueError("Missing --csv_path (or set $CSV_PATH).")
    if not args.whole_img_root:
        raise ValueError("Missing --whole_img_root (or set $WHOLE_IMG_ROOT).")

    if (not args.random_init) and (not args.fm_ckpt_path):
        raise ValueError("Missing --fm_ckpt_path (or set $FM_CKPT_PATH), unless --random_init is used.")

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    return args


# =========================
# Device / seed / logging
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Utilities
# =========================
def clean_label(v, nan_val=-1):
    if pd.isna(v):
        return nan_val
    s = str(v).strip()
    if s == "":
        return nan_val
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return nan_val


def extract_whole_image_name(file_name: str) -> str:
    """
    Infer whole-image name from joint-level file_name:
        <base>_<side>_<joint_type>_<index>.tif -> <base>.tif
    """
    base = os.path.basename(file_name)
    base_no_ext = base[:-4] if base.lower().endswith(".tif") else Path(base).stem
    parts = base_no_ext.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected joint file name pattern: {file_name}")
    whole_base = "_".join(parts[:-3])
    return whole_base + ".tif"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def describe_label_dist(df: pd.DataFrame, col: str, name: str):
    vals = df[col].values
    unique, counts = np.unique(vals, return_counts=True)
    info = ", ".join([f"{int(k)}:{int(v)}" for k, v in zip(unique, counts)])
    log(f"[Label Dist] {name} {col}: {info}")


# =========================
# Dataset & Transforms
# =========================
class BilateralImageDataset(Dataset):
    """Returns: image, labels_tensor([Erosion_sum, JSN_sum])"""
    def __init__(self, df: pd.DataFrame, root: Path, tf):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row.whole_image
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        eros_sum = float(row.Erosion_sum)
        jsn_sum = float(row.JSN_sum)
        labels = torch.tensor([eros_sum, jsn_sum], dtype=torch.float32)
        return img, labels


# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
    def __init__(self, model_name: str, device: str, init_mode: str = "pretrained"):
        super().__init__()
        self.device = device

        if init_mode == "random":
            cfg = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(cfg)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        self.encoder.to(self.device)

        cfg = self.encoder.config
        self.hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "hidden_dim", None)
            or getattr(cfg, "embed_dim", None)
            or getattr(cfg, "width", None)
        )
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        out = self.encoder(pixel_values=x, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D]

        if self.num_register_tokens > 0:
            cls_tok = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_register_tokens:, :]
            tokens = torch.cat([cls_tok, patches], dim=1)

        return tokens


def load_student_from_ckpt(encoder: VisionEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ema_state" not in ckpt:
        raise KeyError("Expected key 'ema_state' in foundation checkpoint.")
    missing, unexpected = encoder.load_state_dict(ckpt["ema_state"], strict=False)
    log("[FM] Loaded encoder weights from checkpoint (ema_state).")
    if missing:
        log(f"[FM] Missing keys: {len(missing)} (first 5) {missing[:5]}")
    if unexpected:
        log(f"[FM] Unexpected keys: {len(unexpected)} (first 5) {unexpected[:5]}")


# =========================
# Model (LN + Linear head)
# =========================
class BilateralSVHModel(nn.Module):
    def __init__(self, encoder: VisionEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size
        self.head_e = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        self.head_j = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

    def forward(self, x):
        feat = self.encoder(x)         # [B, 1+P, D]
        pooled = feat.mean(dim=1)      # [B, D]
        logit_e = self.head_e(pooled).squeeze(1)
        logit_j = self.head_j(pooled).squeeze(1)
        return logit_e, logit_j


# =========================
# Loss (masked BCE)
# =========================
class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        valid_pred = pred[mask]
        valid_tgt = (target[mask] > 0.0).float()
        if valid_tgt.numel() == 0:
            return pred.new_tensor(0.0)

        base_loss = F.binary_cross_entropy_with_logits(valid_pred, valid_tgt, reduction="none")
        if self.pos_weight is not None and self.pos_weight != 1.0:
            w = torch.ones_like(base_loss)
            w[valid_tgt > 0.5] = self.pos_weight
            base_loss = base_loss * w

        if self.reduction == "mean":
            return base_loss.mean()
        if self.reduction == "sum":
            return base_loss.sum()
        return base_loss


# =========================
# Metrics (AUC/ACC/AUPR/F1)
# =========================
@torch.no_grad()
def compute_binary_metrics_from_logits(pred_logits: np.ndarray, target_scores: np.ndarray) -> Dict[str, float]:
    mask = target_scores > -0.5
    if mask.sum() == 0:
        return dict(auc=np.nan, acc=np.nan, aupr=np.nan, f1=np.nan)

    sums = target_scores[mask]
    y_true = (sums > 0).astype(int)

    logits = pred_logits[mask]
    probs = sigmoid_np(logits)
    y_pred = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    try:
        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan

    try:
        aupr = float(average_precision_score(y_true, probs)) if np.sum(y_true) > 0 else np.nan
    except Exception:
        aupr = np.nan

    return dict(auc=auc, acc=acc, aupr=aupr, f1=float(f1))


def log_binary_metrics(title: str, m: Dict[str, float]):
    log(f"📊 {title} | AUC={m['auc']:.3f} ACC={m['acc']:.3f} AUPR={m['aupr']:.3f} F1={m['f1']:.3f}")


# =========================
# Conformal prediction (split conformal)
# =========================
def fit_split_conformal_scores(cal_logits: np.ndarray, cal_targets_bin: np.ndarray) -> np.ndarray:
    probs_pos = sigmoid_np(cal_logits)
    p_true = np.where(cal_targets_bin == 1, probs_pos, 1.0 - probs_pos)
    scores = 1.0 - p_true
    return scores.astype(np.float64)


def conformal_predict_sets(test_logits: np.ndarray, cal_scores: np.ndarray, alpha: float) -> Tuple[np.ndarray, Dict[str, float]]:
    n_cal = len(cal_scores)
    probs_pos = sigmoid_np(test_logits)
    p0 = 1.0 - probs_pos
    p1 = probs_pos

    t0 = 1.0 - p0
    t1 = 1.0 - p1

    cal = cal_scores.reshape(1, -1)
    t0c = t0.reshape(-1, 1)
    t1c = t1.reshape(-1, 1)

    pval0 = (np.sum(cal >= t0c, axis=1) + 1.0) / (n_cal + 1.0)
    pval1 = (np.sum(cal >= t1c, axis=1) + 1.0) / (n_cal + 1.0)

    in0 = pval0 > alpha
    in1 = pval1 > alpha
    pred_sets = np.stack([in0, in1], axis=1)

    set_sizes = pred_sets.sum(axis=1)
    stats = {
        "avg_set_size": float(np.mean(set_sizes)),
        "empty_rate": float(np.mean(set_sizes == 0)),
        "both_rate": float(np.mean(set_sizes == 2)),
        "singleton_rate": float(np.mean(set_sizes == 1)),
    }
    return pred_sets, stats


def conformal_coverage(pred_sets: np.ndarray, true_bin: np.ndarray) -> float:
    ok = pred_sets[np.arange(len(true_bin)), true_bin]
    return float(np.mean(ok))


def log_conformal_stats(title: str, alpha: float, coverage: float, stats: Dict[str, float]):
    log(
        f"🧪 Conformal {title} | alpha={alpha:.2f} "
        f"coverage={coverage:.3f} avg|S|={stats['avg_set_size']:.3f} "
        f"empty={stats['empty_rate']:.3f} both={stats['both_rate']:.3f} singleton={stats['singleton_rate']:.3f}"
    )


# =========================
# Train / Eval (with AMP)
# =========================
def train_epoch(model, loader, optimizer, loss_main, target_idx: int, scaler: Optional[GradScaler]) -> float:
    model.train()
    total, n = 0.0, 0

    for img, labels in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        eros_t = labels[:, 0]
        jsn_t = labels[:, 1]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and DEVICE == "cuda":
            with autocast():
                logit_e, logit_j = model(img)
                tgt, pred = (eros_t, logit_e) if target_idx == 0 else (jsn_t, logit_j)
                mask = tgt >= 0.0
                loss = loss_main(pred, tgt, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logit_e, logit_j = model(img)
            tgt, pred = (eros_t, logit_e) if target_idx == 0 else (jsn_t, logit_j)
            mask = tgt >= 0.0
            loss = loss_main(pred, tgt, mask)
            loss.backward()
            optimizer.step()

        bs = img.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(1, n)


@torch.no_grad()
def eval_loader_collect(model, loader, target_idx: int):
    model.eval()
    pred_list, tgt_list = [], []

    for img, labels in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        eros_t = labels[:, 0]
        jsn_t = labels[:, 1]

        if DEVICE == "cuda":
            with autocast():
                logit_e, logit_j = model(img)
        else:
            logit_e, logit_j = model(img)

        pred, tgt = (logit_e, eros_t) if target_idx == 0 else (logit_j, jsn_t)
        pred_list.append(pred.detach().cpu().numpy())
        tgt_list.append(tgt.detach().cpu().numpy())

    pred_logits = np.concatenate(pred_list, axis=0)
    tgt_scores = np.concatenate(tgt_list, axis=0)
    return pred_logits, tgt_scores


def eval_metrics(model, loader, target_idx: int) -> Dict[str, float]:
    pred_logits, tgt_scores = eval_loader_collect(model, loader, target_idx)
    return compute_binary_metrics_from_logits(pred_logits, tgt_scores)


# =========================
# Patient-level split helper
# =========================
def patient_level_split_3way(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    patients = df["patient_id"].astype(str).unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    df_train = df[df["patient_id"].astype(str).isin(train_p)].reset_index(drop=True)
    df_val = df[df["patient_id"].astype(str).isin(val_p)].reset_index(drop=True)
    df_test = df[df["patient_id"].astype(str).isin(test_p)].reset_index(drop=True)
    return df_train, df_val, df_test


# =========================
# Build bilateral dataframe (per task)
# =========================
def build_bilateral_df(csv_path: str, whole_img_root: Path, task: str,
                       expected_joints_erosion: int, expected_joints_jsn: int) -> pd.DataFrame:
    df_all = pd.read_csv(csv_path)

    required_cols = ["patient_id", "timepoint", "side", "joint_type",
                     "index", "Erosion_score", "JSN_score", "file_name"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df_all["Erosion_clean"] = df_all["Erosion_score"].apply(clean_label)
    df_all["JSN_clean"] = df_all["JSN_score"].apply(clean_label)
    df_all["whole_image"] = df_all["file_name"].apply(extract_whole_image_name)

    if task == "erosion":
        df_valid = df_all[(df_all["Erosion_clean"] >= 0) & (df_all["Erosion_clean"] <= 5)].copy()
        expected_joints = expected_joints_erosion
        log(f"[erosion] joint rows with valid erosion scores: {len(df_valid)}")
    else:
        df_valid = df_all[(df_all["JSN_clean"] >= 0) & (df_all["JSN_clean"] <= 4)].copy()
        expected_joints = expected_joints_jsn
        log(f"[jsn] joint rows with valid JSN scores: {len(df_valid)}")

    log(f"Unique whole_image candidates in CSV (task={task}): {df_valid['whole_image'].nunique()}")

    group_cols = ["patient_id", "timepoint", "whole_image"]
    group_list = []

    for (pid, tp, wimg), g in df_valid.groupby(group_cols):
        img_path = whole_img_root / wimg
        if not img_path.is_file():
            continue

        if len(g) != expected_joints:
            continue

        if task == "erosion":
            eros_sum = int(g["Erosion_clean"].sum())
            jsn_sum = -1
        else:
            eros_sum = -1
            jsn_sum = int(g["JSN_clean"].sum())

        group_list.append({
            "patient_id": pid,
            "timepoint": tp,
            "whole_image": wimg,
            "Erosion_sum": eros_sum,
            "JSN_sum": jsn_sum,
            "num_joints": len(g),
        })

    df_bi = pd.DataFrame(group_list)
    log(f"Built bilateral df (task={task}): N={len(df_bi)} with expected_joints={expected_joints} and existing whole images.")
    return df_bi


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    task = args.task
    whole_root = Path(args.whole_img_root)

    label_col = "Erosion_sum" if task == "erosion" else "JSN_sum"
    target_idx = 0 if task == "erosion" else 1

    log(f"Running bilateral task={task} (Binary 0 vs >0)")
    log(f"Split=train/val/test={args.train_ratio:.2f}/{args.val_ratio:.2f}/{1.0-args.train_ratio-args.val_ratio:.2f} (patient-level), seed={args.seed}")
    log(f"Conformal={args.use_conformal} alpha={args.conformal_alpha:.2f}")

    df_bi = build_bilateral_df(
        csv_path=args.csv_path,
        whole_img_root=whole_root,
        task=task,
        expected_joints_erosion=args.expected_joints_erosion,
        expected_joints_jsn=args.expected_joints_jsn,
    )
    if df_bi.empty:
        raise RuntimeError(f"No valid bilateral samples found for task={task}.")

    # ---- split ----
    df_tr, df_val, df_te = patient_level_split_3way(
        df_bi, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    log(
        f"Patient split: train={df_tr['patient_id'].nunique()} pts, "
        f"val={df_val['patient_id'].nunique()} pts, "
        f"test={df_te['patient_id'].nunique()} pts"
    )
    log(f"Samples: train={len(df_tr)} val={len(df_val)} test={len(df_te)}")

    describe_label_dist(df_tr, label_col, "Train")
    describe_label_dist(df_val, label_col, "Val")
    describe_label_dist(df_te, label_col, "Test")

    # ---- transforms ----
    proc = AutoImageProcessor.from_pretrained(args.dinov3_model_name)
    image_mean, image_std = proc.image_mean, proc.image_std

    train_tf = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ])
    eval_tf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ])

    # ---- loaders ----
    train_ds = BilateralImageDataset(df_tr, whole_root, train_tf)
    val_ds = BilateralImageDataset(df_val, whole_root, eval_tf)
    test_ds = BilateralImageDataset(df_te, whole_root, eval_tf)

    num_workers = int(os.environ.get("NUM_WORKERS", "8"))

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=4,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        prefetch_factor=4,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        prefetch_factor=4,
    )

    log(f"Bilateral images: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ---- model ----
    init_mode = "random" if args.random_init else "pretrained"
    encoder = VisionEncoder(model_name=args.dinov3_model_name, device=DEVICE, init_mode=init_mode).to(DEVICE)

    if not args.random_init:
        load_student_from_ckpt(encoder, args.fm_ckpt_path)
        log("Encoder initialized from foundation checkpoint (ema_state).")
    else:
        log("Encoder randomly initialized.")

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        log("Encoder frozen (head-only).")
    else:
        log("Encoder will be fine-tuned.")

    model = BilateralSVHModel(encoder).to(DEVICE)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    loss_main = MaskedBCELoss(pos_weight=args.pos_weight_main)
    scaler = GradScaler() if DEVICE == "cuda" else None

    best_val_auc = -1.0

    for ep in range(0, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_ld, optimizer, loss_main, target_idx, scaler)

        tr_m = eval_metrics(model, train_ld, target_idx)
        va_m = eval_metrics(model, val_ld, target_idx)
        te_m = eval_metrics(model, test_ld, target_idx)

        log_binary_metrics(f"Train {task.capitalize()} (bilateral)", tr_m)
        log_binary_metrics(f"Val   {task.capitalize()} (bilateral)", va_m)
        log_binary_metrics(f"Test  {task.capitalize()} (bilateral)", te_m)

        if args.use_conformal:
            val_logits, val_scores = eval_loader_collect(model, val_ld, target_idx)
            val_mask = val_scores > -0.5
            val_logits = val_logits[val_mask]
            val_true_bin = (val_scores[val_mask] > 0).astype(int)

            cal_scores = fit_split_conformal_scores(val_logits, val_true_bin)

            val_sets, val_stats = conformal_predict_sets(val_logits, cal_scores, args.conformal_alpha)
            val_cov = conformal_coverage(val_sets, val_true_bin)
            log_conformal_stats("VAL(calib-self)", args.conformal_alpha, val_cov, val_stats)

            test_logits, test_scores = eval_loader_collect(model, test_ld, target_idx)
            test_mask = test_scores > -0.5
            test_logits = test_logits[test_mask]
            test_true_bin = (test_scores[test_mask] > 0).astype(int)

            test_sets, test_stats = conformal_predict_sets(test_logits, cal_scores, args.conformal_alpha)
            test_cov = conformal_coverage(test_sets, test_true_bin)
            log_conformal_stats("TEST", args.conformal_alpha, test_cov, test_stats)

        log(
            f"Epoch {ep:03d}/{args.epochs} | TrainLoss={train_loss:.4f} | "
            f"Val AUC={va_m['auc']:.4f} | Time {time.time()-t0:.1f}s"
        )

        val_auc = va_m["auc"]
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = os.path.join(args.output_dir, f"bilateral_{task}_binary_ep{ep}_valauc{best_val_auc:.4f}.pt")
            torch.save(model.state_dict(), save_path)
            log(f"✅ Saved best (by VAL {task.capitalize()} AUC) to {save_path}")
            if args.use_conformal:
                log("🧷 (Saved best) Conformal stats above correspond to this epoch.")

    log("Training finished.")


if __name__ == "__main__":
    main()

# Example:
# python3 svh_bilateral_binary_conformal.py \
#   --task jsn \
#   --csv_path /path/to/RA_joint_score_raw.csv \
#   --whole_img_root /path/to/bilateral_images \
#   --fm_ckpt_path /path/to/foundation_ckpt.pt \
#   --output_dir ./outputs/svh_bilateral_binary \
#   --use_conformal
