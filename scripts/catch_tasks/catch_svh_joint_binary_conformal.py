#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint-level SvH Score Prediction on CATCH dataset with DINOv3 FM backbone (Binary 0 vs >0)

Changes in this version:
1) Patient-level split: train/val/test = 0.8/0.1/0.1
2) Add Split Conformal Prediction (calibration = val split)
3) Metrics: AUC / ACC / AUPR / F1
   - When saving best ckpt, additionally report per joint type metrics (PIP/MCP/Wrist/Ulna/Radius)
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from catch_split_utils import shared_patient_level_split_3way
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
# Hardcoded paths & hparams
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------- Task switch --------
TASK = "erosion"
# TASK = "jsn"
assert TASK in ("erosion", "jsn"), "TASK must be 'erosion' or 'jsn'"

# -------- Foundation Model checkpoint (reads ema_state) --------
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# -------- CATCH joint-level csv --------
CSV_PATH = "/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_joint_score_raw.csv"

# NOTE: keep the original space in the path
IMG_ROOT = "/home/UWO/ylong66/data/RA/RA/Joint Detection /yolov5/data/extracted_joint_images"

# HF model
DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# random init encoder
RANDOM_INIT = False

# hparams
IMG_SIZE       = 224
BATCH          = 64
EPOCHS         = 50
LR             = 1e-5
FREEZE_ENCODER = False      # True=head-only

# imbalance
USE_WEIGHTED_SAMPLER = True
POS_WEIGHT_MAIN      = 5.0

# Conformal
USE_CONFORMAL   = True
CONFORMAL_ALPHA = 0.10  # 0.10 -> target 90% marginal coverage

# Save
MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


# =========================
# Image mean/std from HF processor
# =========================
_processor_for_stats = AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME)
IMAGE_MEAN = _processor_for_stats.image_mean
IMAGE_STD  = _processor_for_stats.image_std


# =========================
# Dataset & Transforms
# =========================
train_tf = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

eval_tf = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])


def clean_label(v, nan_val=-1):
    """Clean Erosion_score / JSN_score to int. Non-numeric -> -1"""
    if pd.isna(v):
        return nan_val
    s = str(v).strip()
    if s == "":
        return nan_val
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return nan_val


def infer_joint_type_from_filename(file_name: str) -> str:
    """
    Infer joint type from filename string.
    Expected tokens often include: PIP, MCP, WRIST, ULNA, RADIUS, DIP...
    We only report: PIP/MCP/Wrist/Ulna/Radius. Others -> "Other".
    """
    s = str(file_name).upper()
    if "PIP" in s:
        return "PIP"
    if "MCP" in s:
        return "MCP"
    if "WRIST" in s:
        return "Wrist"
    if "ULNA" in s:
        return "Ulna"
    if "RADIUS" in s:
        return "Radius"
    return "Other"


class JointDataset(Dataset):
    """Returns: image, labels_tensor([erosion, jsn]), joint_type(str)"""
    def __init__(self, df: pd.DataFrame, root: str, tf):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row.file_name
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        eros = int(row.Erosion_clean)
        jsn  = int(row.JSN_clean)
        labels = torch.tensor([eros, jsn], dtype=torch.long)

        jt = row.joint_type if "joint_type" in self.df.columns else infer_joint_type_from_filename(row.file_name)
        return img, labels, jt


def collate_with_joint_type(batch):
    imgs, labels, jts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    return imgs, labels, list(jts)


# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
    """
    - AutoModel loads DINOv3 backbone
    - Input already normalized [B,3,H,W]
    - Remove register tokens if exist; keep [CLS + patches]
    """
    def __init__(self, model_name: str = DINOV3_MODEL_NAME,
                 device=None, init_mode: str = "pretrained"):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
            tokens = torch.cat([cls_tok, patches], dim=1)  # [B, 1+P, D]

        return tokens


def load_student_from_ckpt(encoder: VisionEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(ckpt["ema_state"], strict=False)
    log(f"Loaded ema_state from {ckpt_path}")
    if missing:
        log(f"  Missing keys: {len(missing)} (first 5) {missing[:5]}")
    if unexpected:
        log(f"  Unexpected keys: {len(unexpected)} (first 5) {unexpected[:5]}")


# =========================
# Model (LN + Linear head) —— binary logit
# =========================
class JointSVHModel(nn.Module):
    def __init__(self, encoder: VisionEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size

        self.head_e = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        self.head_j = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

    def forward(self, x):
        feat = self.encoder(x)         # [B, 1+P, D]
        pooled = feat.mean(dim=1)      # [B, D]
        logit_e = self.head_e(pooled).squeeze(1)  # [B]
        logit_j = self.head_j(pooled).squeeze(1)  # [B]
        return logit_e, logit_j


# =========================
# Loss (masked BCE) for binary 0 vs >0
# =========================
class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        valid_pred = pred[mask]
        valid_tgt  = (target[mask] > 0.0).float()
        if valid_tgt.numel() == 0:
            return pred.new_tensor(0.0)

        base_loss = F.binary_cross_entropy_with_logits(valid_pred, valid_tgt, reduction="none")
        if self.pos_weight is not None and self.pos_weight != 1.0:
            weights = torch.ones_like(base_loss)
            weights[valid_tgt > 0.5] = self.pos_weight
            base_loss = base_loss * weights

        if self.reduction == "mean":
            return base_loss.mean()
        if self.reduction == "sum":
            return base_loss.sum()
        return base_loss


# =========================
# Metrics — binary (AUC/ACC/AUPR/F1)
# =========================
@torch.no_grad()
def compute_binary_metrics_from_logits(pred_logits: np.ndarray,
                                       target_scores: np.ndarray) -> Dict[str, float]:
    """
    pred_logits: [N] raw logits
    target_scores: [N] original integer SvH scores (>=0 valid; -1 invalid)
    """
    mask = target_scores > -0.5
    if mask.sum() == 0:
        return dict(auc=np.nan, acc=np.nan, aupr=np.nan, f1=np.nan)

    y_scores = target_scores[mask]
    y_true = (y_scores > 0).astype(int)

    logits = pred_logits[mask]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # AUC needs both classes
    try:
        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan

    # AUPR (Average Precision) also usually needs positives
    try:
        aupr = float(average_precision_score(y_true, probs)) if np.sum(y_true) > 0 else np.nan
    except Exception:
        aupr = np.nan

    return dict(auc=auc, acc=acc, aupr=aupr, f1=float(f1))


def log_binary_metrics(title: str, m: Dict[str, float]):
    log(f"📊 {title} | AUC={m['auc']:.3f} ACC={m['acc']:.3f} AUPR={m['aupr']:.3f} F1={m['f1']:.3f}")


# =========================
# Conformal prediction (split conformal for classification)
# =========================
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_split_conformal_scores(cal_logits: np.ndarray, cal_targets_bin: np.ndarray) -> np.ndarray:
    """
    Nonconformity scores: s_i = 1 - p_{y_i}(x_i)
    cal_logits: [N] logits
    cal_targets_bin: [N] in {0,1}
    """
    probs_pos = sigmoid_np(cal_logits)
    p_true = np.where(cal_targets_bin == 1, probs_pos, 1.0 - probs_pos)
    scores = 1.0 - p_true
    return scores.astype(np.float64)


def conformal_predict_sets(test_logits: np.ndarray,
                           cal_scores: np.ndarray,
                           alpha: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    For each sample x, compute p-values for labels y in {0,1}:
      pval(y) = ( #{i: cal_scores[i] >= 1 - p_y(x)} + 1 ) / (n_cal + 1)
    Include y if pval(y) > alpha.

    Returns:
      pred_sets: [N,2] bool array, [:,0] for label0, [:,1] for label1
      stats: avg_set_size, empty_rate
    """
    n_cal = len(cal_scores)
    probs_pos = sigmoid_np(test_logits)
    p0 = 1.0 - probs_pos
    p1 = probs_pos

    # thresholds t_y = 1 - p_y(x)
    t0 = 1.0 - p0
    t1 = 1.0 - p1

    # vectorized p-value computation via broadcasting
    # pval_y = (count(cal_scores >= t_y) + 1) / (n_cal + 1)
    cal_scores_col = cal_scores.reshape(1, -1)  # [1, n_cal]
    t0_col = t0.reshape(-1, 1)                  # [N, 1]
    t1_col = t1.reshape(-1, 1)                  # [N, 1]

    pval0 = (np.sum(cal_scores_col >= t0_col, axis=1) + 1.0) / (n_cal + 1.0)
    pval1 = (np.sum(cal_scores_col >= t1_col, axis=1) + 1.0) / (n_cal + 1.0)

    in0 = pval0 > alpha
    in1 = pval1 > alpha
    pred_sets = np.stack([in0, in1], axis=1)  # [N,2]

    set_sizes = pred_sets.sum(axis=1)
    stats = {
        "avg_set_size": float(np.mean(set_sizes)),
        "empty_rate": float(np.mean(set_sizes == 0)),
        "both_rate": float(np.mean(set_sizes == 2)),
        "singleton_rate": float(np.mean(set_sizes == 1)),
    }
    return pred_sets, stats


def conformal_coverage(pred_sets: np.ndarray, true_bin: np.ndarray) -> float:
    """
    Coverage = P(true label is in prediction set)
    pred_sets: [N,2] bool
    true_bin: [N] 0/1
    """
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

    for img, labels, _jts in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        eros_t = labels[:, 0].float()
        jsn_t  = labels[:, 1].float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and DEVICE == "cuda":
            with autocast():
                logit_e, logit_j = model(img)
                if target_idx == 0:
                    tgt, pred = eros_t, logit_e
                else:
                    tgt, pred = jsn_t, logit_j
                mask = tgt >= 0.0
                loss = loss_main(pred, tgt, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logit_e, logit_j = model(img)
            if target_idx == 0:
                tgt, pred = eros_t, logit_e
            else:
                tgt, pred = jsn_t, logit_j
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
    """
    Collect logits/targets/joint_types for current task.
    Returns:
      pred_logits [N]
      tgt_scores  [N] (original integer scores)
      joint_types list[str] length N
    """
    model.eval()
    pred_list, tgt_list, jt_list = [], [], []

    for img, labels, jts in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        eros_t = labels[:, 0].float()
        jsn_t  = labels[:, 1].float()

        if DEVICE == "cuda":
            with autocast():
                logit_e, logit_j = model(img)
        else:
            logit_e, logit_j = model(img)

        if target_idx == 0:
            pred, tgt = logit_e, eros_t
        else:
            pred, tgt = logit_j, jsn_t

        pred_list.append(pred.detach().cpu().numpy())
        tgt_list.append(tgt.detach().cpu().numpy())
        jt_list.extend(jts)

    pred_logits = np.concatenate(pred_list, axis=0)
    tgt_scores  = np.concatenate(tgt_list,  axis=0)
    return pred_logits, tgt_scores, jt_list


def eval_metrics(model, loader, target_idx: int) -> Dict[str, float]:
    pred_logits, tgt_scores, _ = eval_loader_collect(model, loader, target_idx)
    return compute_binary_metrics_from_logits(pred_logits, tgt_scores)


def eval_metrics_by_joint_type(model, loader, target_idx: int, joint_types_of_interest=None) -> Dict[str, Dict[str, float]]:
    if joint_types_of_interest is None:
        joint_types_of_interest = ["PIP", "MCP", "Wrist", "Ulna", "Radius"]

    pred_logits, tgt_scores, jt_list = eval_loader_collect(model, loader, target_idx)
    jt_arr = np.array(jt_list, dtype=object)

    out = {}
    for jt in joint_types_of_interest:
        m = (jt_arr == jt)
        if np.sum(m) == 0:
            out[jt] = dict(auc=np.nan, acc=np.nan, aupr=np.nan, f1=np.nan)
            continue
        out[jt] = compute_binary_metrics_from_logits(pred_logits[m], tgt_scores[m])
    return out


def log_metrics_by_joint_type(title: str, metrics_dict: Dict[str, Dict[str, float]]):
    log(f"🔎 {title} | Per-joint-type metrics:")
    for jt, m in metrics_dict.items():
        log(f"  - {jt:6s}: AUC={m['auc']:.3f} ACC={m['acc']:.3f} AUPR={m['aupr']:.3f} F1={m['f1']:.3f}")


# =========================
# Patient-level split helper (0.8/0.1/0.1)
# =========================
def patient_level_split_3way(df: pd.DataFrame,
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             seed: int = 3407):
    return shared_patient_level_split_3way(
        df,
        patient_col="patient_id",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        source_csv_path=CSV_PATH,
    )


# =========================
# Weighted Sampler (binary 0 vs >0)
# =========================
def build_weighted_sampler(df_train: pd.DataFrame, label_col: str):
    raw_labels = df_train[label_col].values.astype(int)
    bin_labels = (raw_labels > 0).astype(int)
    counts = np.bincount(bin_labels, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts  # [w0, w1]
    sample_weights = class_weights[bin_labels]
    sample_weights = torch.from_numpy(sample_weights).double()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def describe_label_dist(df: pd.DataFrame, col: str, name: str):
    vals = df[col].values
    unique, counts = np.unique(vals, return_counts=True)
    info = ", ".join([f"{int(k)}:{int(v)}" for k, v in zip(unique, counts)])
    log(f"[Label Dist] {name} {col}: {info}")


# =========================
# Main
# =========================
def main():
    log(f"Running task = {TASK} (Binary: 0 vs >0)")
    log(f"Split = train/val/test = 0.8/0.1/0.1 (patient-level)")
    log(f"Metrics = AUC/ACC/AUPR/F1 | Conformal={USE_CONFORMAL} alpha={CONFORMAL_ALPHA}")

    # ---- load csv & clean labels ----
    df_all = pd.read_csv(CSV_PATH)
    required_cols = ["patient_id", "file_name", "Erosion_score", "JSN_score"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df_all["Erosion_clean"] = df_all["Erosion_score"].apply(clean_label)
    df_all["JSN_clean"]     = df_all["JSN_score"].apply(clean_label)

    # add joint_type column
    df_all["joint_type"] = df_all["file_name"].apply(infer_joint_type_from_filename)

    # select task
    if TASK == "erosion":
        label_col = "Erosion_clean"
        target_idx = 0
    else:
        label_col = "JSN_clean"
        target_idx = 1

    df_label = df_all[df_all[label_col] >= 0].reset_index(drop=True)
    log(f"Total samples with valid {TASK} label: {len(df_label)}")

    # patient-level split 0.8/0.1/0.1
    df_tr, df_val, df_test = patient_level_split_3way(df_label, train_ratio=0.8, val_ratio=0.1, seed=3407)
    log(
        f"Patient split: train={df_tr['patient_id'].nunique()} pts, "
        f"val={df_val['patient_id'].nunique()} pts, "
        f"test={df_test['patient_id'].nunique()} pts"
    )
    log(f"Samples: train={len(df_tr)} val={len(df_val)} test={len(df_test)}")

    describe_label_dist(df_tr,  "Erosion_clean", "Train")
    describe_label_dist(df_tr,  "JSN_clean",     "Train")
    describe_label_dist(df_val, "Erosion_clean", "Val")
    describe_label_dist(df_val, "JSN_clean",     "Val")
    describe_label_dist(df_test,"Erosion_clean", "Test")
    describe_label_dist(df_test,"JSN_clean",     "Test")

    # ---- datasets & loaders ----
    train_ds = JointDataset(df_tr,   IMG_ROOT, train_tf)
    val_ds   = JointDataset(df_val,  IMG_ROOT, eval_tf)
    test_ds  = JointDataset(df_test, IMG_ROOT, eval_tf)

    NUM_WORKERS = 8

    if USE_WEIGHTED_SAMPLER:
        train_sampler = build_weighted_sampler(df_tr, label_col=label_col)
        train_ld = DataLoader(
            train_ds, batch_size=BATCH,
            sampler=train_sampler, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True, drop_last=True,
            prefetch_factor=4,
            collate_fn=collate_with_joint_type,
        )
    else:
        train_ld = DataLoader(
            train_ds, batch_size=BATCH,
            shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True, drop_last=True,
            prefetch_factor=4,
            collate_fn=collate_with_joint_type,
        )

    val_ld = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
        collate_fn=collate_with_joint_type,
    )
    test_ld = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
        collate_fn=collate_with_joint_type,
    )

    # ---- encoder & model ----
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(device=DEVICE, init_mode=init_mode).to(DEVICE)

    if not RANDOM_INIT:
        load_student_from_ckpt(encoder, CKPT_PATH)
        log("Encoder initialized from pretrained FM (ema_state).")
    else:
        log("Encoder randomly initialized (no pretrained weights).")

    if FREEZE_ENCODER:
        for p in encoder.parameters():
            p.requires_grad = False
        log("Encoder frozen (only heads will be trained).")
    else:
        log("Encoder will be fine-tuned.")

    model = JointSVHModel(encoder).to(DEVICE)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    loss_main = MaskedBCELoss(pos_weight=POS_WEIGHT_MAIN)
    scaler = GradScaler() if DEVICE == "cuda" else None

    # best by VAL AUC (recommended; avoids test leakage)
    best_val_auc = -1.0
    best_ckpt_path = None

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_ld, optimizer, loss_main, target_idx, scaler)

        # Train/Val metrics only; test is reserved for final reporting.
        tr_m = eval_metrics(model, train_ld, target_idx)
        va_m = eval_metrics(model, val_ld, target_idx)

        log_binary_metrics(f"Train {TASK.capitalize()}", tr_m)
        log_binary_metrics(f"Val   {TASK.capitalize()}", va_m)

        # Conformal: fit on VAL (calibration) and check calibration split during training.
        if USE_CONFORMAL:
            val_logits, val_scores, _ = eval_loader_collect(model, val_ld, target_idx)
            val_mask = val_scores > -0.5
            val_logits = val_logits[val_mask]
            val_true_bin = (val_scores[val_mask] > 0).astype(int)

            cal_scores = fit_split_conformal_scores(val_logits, val_true_bin)
            val_sets, val_stats = conformal_predict_sets(val_logits, cal_scores, CONFORMAL_ALPHA)
            val_cov = conformal_coverage(val_sets, val_true_bin)
            log_conformal_stats("VAL(calib-self)", CONFORMAL_ALPHA, val_cov, val_stats)

        log(
            f"Epoch {ep:03d}/{EPOCHS} | TrainLoss={train_loss:.4f} | "
            f"Val AUC={va_m['auc']:.4f} | Time {time.time()-t0:.1f}s"
        )

        val_auc = va_m["auc"]
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = os.path.join(
                MODEL_SAVE_DIR,
                f"catch_svh_{TASK}_binary_dinov3_amp_ep{ep}_valauc{best_val_auc:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            best_ckpt_path = save_path
            log(f"Saved best (by VAL {TASK.capitalize()} AUC) to {save_path}")

            va_by_jt = eval_metrics_by_joint_type(
                model, val_ld, target_idx,
                joint_types_of_interest=["PIP", "MCP", "Wrist", "Ulna", "Radius"],
            )
            log_metrics_by_joint_type(f"VAL  {TASK.capitalize()}", va_by_jt)

    if best_ckpt_path is None:
        log("No best checkpoint saved.")
        return

    log("======= Final: Load best ckpt and report VAL/TEST + Conformal =======")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    log(f"Loaded best ckpt: {best_ckpt_path}")

    va_m = eval_metrics(model, val_ld, target_idx)
    te_m = eval_metrics(model, test_ld, target_idx)
    log_binary_metrics(f"Val   {TASK.capitalize()}", va_m)
    log_binary_metrics(f"Test  {TASK.capitalize()}", te_m)

    va_by_jt = eval_metrics_by_joint_type(
        model, val_ld, target_idx,
        joint_types_of_interest=["PIP", "MCP", "Wrist", "Ulna", "Radius"],
    )
    te_by_jt = eval_metrics_by_joint_type(
        model, test_ld, target_idx,
        joint_types_of_interest=["PIP", "MCP", "Wrist", "Ulna", "Radius"],
    )
    log_metrics_by_joint_type(f"VAL  {TASK.capitalize()}", va_by_jt)
    log_metrics_by_joint_type(f"TEST {TASK.capitalize()}", te_by_jt)

    if USE_CONFORMAL:
        val_logits, val_scores, _ = eval_loader_collect(model, val_ld, target_idx)
        val_mask = val_scores > -0.5
        val_logits = val_logits[val_mask]
        val_true_bin = (val_scores[val_mask] > 0).astype(int)
        cal_scores = fit_split_conformal_scores(val_logits, val_true_bin)

        val_sets, val_stats = conformal_predict_sets(val_logits, cal_scores, CONFORMAL_ALPHA)
        val_cov = conformal_coverage(val_sets, val_true_bin)
        log_conformal_stats("VAL(calib-self)", CONFORMAL_ALPHA, val_cov, val_stats)

        test_logits, test_scores, _ = eval_loader_collect(model, test_ld, target_idx)
        test_mask = test_scores > -0.5
        test_logits = test_logits[test_mask]
        test_true_bin = (test_scores[test_mask] > 0).astype(int)
        test_sets, test_stats = conformal_predict_sets(test_logits, cal_scores, CONFORMAL_ALPHA)
        test_cov = conformal_coverage(test_sets, test_true_bin)
        log_conformal_stats("TEST", CONFORMAL_ALPHA, test_cov, test_stats)

    log("Training finished.")


if __name__ == "__main__":
    main()

