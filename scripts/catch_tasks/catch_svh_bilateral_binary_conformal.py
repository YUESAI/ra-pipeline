#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilateral (two-hand) SvH Binary Classification on CATCH dataset with DINOv3 FM backbone

Binary task (0 vs >0):
- TASK="erosion":
    * One bilateral whole image must have exactly 26 joints with valid Erosion scores
    * Erosion_sum = sum over 26 joints (0~130)
    * Binary label: Erosion_sum==0 -> 0, else 1
- TASK="jsn":
    * One bilateral whole image must have exactly 18 joints with valid JSN scores
    * JSN_sum = sum over 18 joints (0~72)
    * Binary label: JSN_sum==0 -> 0, else 1

This version changes:
1) patient-level split: train/val/test = 0.8/0.1/0.1, seed=3407
2) add split conformal prediction (calibration = val)
3) metrics: AUC / ACC / AUPR / F1
4) save best by VAL AUC (avoid test leakage); when saving, also print conformal stats (VAL/TEST)
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
# Hardcoded paths & hparams
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------- Task switch --------
# TASK = "erosion"
TASK = "jsn"
assert TASK in ("erosion", "jsn"), "TASK must be 'erosion' or 'jsn'"

EXPECTED_JOINTS_EROSION = 26
EXPECTED_JOINTS_JSN = 18

# -------- Foundation Model ckpt (ema_state) --------
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# -------- joint-level SvH CSV & whole image dir --------
CSV_PATH = "/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_joint_score_raw.csv"
WHOLE_IMG_ROOT = Path("/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_data/all_RA_update")

DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RANDOM_INIT = False

IMG_SIZE       = 224
BATCH          = 64
EPOCHS         = 200
LR             = 1e-5
FREEZE_ENCODER = False

POS_WEIGHT_MAIN = 5.0

# Split settings
SPLIT_SEED  = 3407
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1  # test = 0.1

# Conformal
USE_CONFORMAL   = True
CONFORMAL_ALPHA = 0.10

MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema_bilateral"
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
    if base.lower().endswith(".tif"):
        base_no_ext = base[:-4]
    else:
        base_no_ext = Path(base).stem

    parts = base_no_ext.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected joint file name pattern: {file_name}")
    whole_base = "_".join(parts[:-3])
    return whole_base + ".tif"


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
        jsn_sum  = float(row.JSN_sum)
        labels = torch.tensor([eros_sum, jsn_sum], dtype=torch.float32)
        return img, labels


# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
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
            tokens = torch.cat([cls_tok, patches], dim=1)

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
        valid_tgt  = (target[mask] > 0.0).float()
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
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def compute_binary_metrics_from_logits(pred_logits: np.ndarray,
                                       target_scores: np.ndarray) -> Dict[str, float]:
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


def describe_label_dist(df: pd.DataFrame, col: str, name: str):
    vals = df[col].values
    unique, counts = np.unique(vals, return_counts=True)
    info = ", ".join([f"{int(k)}:{int(v)}" for k, v in zip(unique, counts)])
    log(f"[Label Dist] {name} {col}: {info}")


# =========================
# Conformal prediction (split conformal for binary classification)
# =========================
def fit_split_conformal_scores(cal_logits: np.ndarray, cal_targets_bin: np.ndarray) -> np.ndarray:
    probs_pos = sigmoid_np(cal_logits)
    p_true = np.where(cal_targets_bin == 1, probs_pos, 1.0 - probs_pos)
    scores = 1.0 - p_true
    return scores.astype(np.float64)


def conformal_predict_sets(test_logits: np.ndarray,
                           cal_scores: np.ndarray,
                           alpha: float) -> Tuple[np.ndarray, Dict[str, float]]:
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
def train_epoch(model, loader, optimizer, loss_main, target_idx: int,
                scaler: Optional[GradScaler]) -> float:
    model.train()
    total, n = 0.0, 0

    for img, labels in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        eros_t = labels[:, 0]
        jsn_t  = labels[:, 1]

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
    model.eval()
    pred_list, tgt_list = [], []

    for img, labels in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        eros_t = labels[:, 0]
        jsn_t  = labels[:, 1]

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

    pred_logits = np.concatenate(pred_list, axis=0)
    tgt_scores  = np.concatenate(tgt_list,  axis=0)
    return pred_logits, tgt_scores


def eval_metrics(model, loader, target_idx: int) -> Dict[str, float]:
    pred_logits, tgt_scores = eval_loader_collect(model, loader, target_idx)
    return compute_binary_metrics_from_logits(pred_logits, tgt_scores)


# =========================
# Patient-level split helper (0.8/0.1/0.1)
# =========================
def patient_level_split_3way(df: pd.DataFrame,
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             seed: int = 3407):
    assert 0.0 < train_ratio < 1.0
    assert 0.0 <= val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0

    rng = np.random.RandomState(seed)
    patients = df["patient_id"].astype(str).unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_p = set(patients[:n_train])
    val_p   = set(patients[n_train:n_train + n_val])
    test_p  = set(patients[n_train + n_val:])

    df_train = df[df["patient_id"].astype(str).isin(train_p)].reset_index(drop=True)
    df_val   = df[df["patient_id"].astype(str).isin(val_p)].reset_index(drop=True)
    df_test  = df[df["patient_id"].astype(str).isin(test_p)].reset_index(drop=True)

    return df_train, df_val, df_test


# =========================
# Build bilateral dataframe (per task)
# =========================
def build_bilateral_df(task: str) -> pd.DataFrame:
    df_all = pd.read_csv(CSV_PATH)

    required_cols = ["patient_id", "timepoint", "side", "joint_type",
                     "index", "Erosion_score", "JSN_score", "file_name"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df_all["Erosion_clean"] = df_all["Erosion_score"].apply(clean_label)
    df_all["JSN_clean"]     = df_all["JSN_score"].apply(clean_label)
    df_all["whole_image"]   = df_all["file_name"].apply(extract_whole_image_name)

    if task == "erosion":
        df_valid = df_all[(df_all["Erosion_clean"] >= 0) & (df_all["Erosion_clean"] <= 5)].copy()
        expected_joints = EXPECTED_JOINTS_EROSION
        log(f"[erosion] joint samples with valid erosion scores: {len(df_valid)}")
    elif task == "jsn":
        df_valid = df_all[(df_all["JSN_clean"] >= 0) & (df_all["JSN_clean"] <= 4)].copy()
        expected_joints = EXPECTED_JOINTS_JSN
        log(f"[jsn] joint samples with valid JSN scores: {len(df_valid)}")
    else:
        raise ValueError(f"Unknown task: {task}")

    log(f"Unique whole_image candidates from CSV for {task}: {df_valid['whole_image'].nunique()}")

    group_cols = ["patient_id", "timepoint", "whole_image"]
    group_list = []

    for (pid, tp, wimg), g in df_valid.groupby(group_cols):
        img_path = WHOLE_IMG_ROOT / wimg
        if not img_path.is_file():
            continue

        if len(g) != expected_joints:
            continue

        if task == "erosion":
            eros_sum = int(g["Erosion_clean"].sum())
            jsn_sum  = -1
        else:
            eros_sum = -1
            jsn_sum  = int(g["JSN_clean"].sum())

        group_list.append({
            "patient_id": pid,
            "timepoint": tp,
            "whole_image": wimg,
            "Erosion_sum": eros_sum,
            "JSN_sum": jsn_sum,
            "num_joints": len(g),
        })

    df_bi = pd.DataFrame(group_list)
    log(f"Built bilateral DataFrame for task={task}: {len(df_bi)} images with exactly {expected_joints} valid joints")
    return df_bi


# =========================
# Main
# =========================
def main():
    log(f"Running bilateral task = {TASK} (Binary 0 vs >0)")
    log(f"Split = train/val/test = {TRAIN_RATIO:.1f}/{VAL_RATIO:.1f}/{1-TRAIN_RATIO-VAL_RATIO:.1f} (patient-level), seed={SPLIT_SEED}")
    log(f"Metrics = AUC/ACC/AUPR/F1 | Conformal={USE_CONFORMAL} alpha={CONFORMAL_ALPHA}")

    df_bi = build_bilateral_df(task=TASK)
    if df_bi.empty:
        raise RuntimeError(f"No valid bilateral images found for task={TASK}.")

    if TASK == "erosion":
        label_col = "Erosion_sum"
        target_idx = 0
    else:
        label_col = "JSN_sum"
        target_idx = 1

    # ---- split 0.8/0.1/0.1 ----
    df_tr, df_val, df_te = patient_level_split_3way(
        df_bi, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SPLIT_SEED
    )
    log(
        f"Patient split: train={df_tr['patient_id'].nunique()} pts, "
        f"val={df_val['patient_id'].nunique()} pts, "
        f"test={df_te['patient_id'].nunique()} pts"
    )
    log(f"Samples: train={len(df_tr)} val={len(df_val)} test={len(df_te)}")

    describe_label_dist(df_tr,  label_col, "Train")
    describe_label_dist(df_val, label_col, "Val")
    describe_label_dist(df_te,  label_col, "Test")

    # ---- loaders ----
    train_ds = BilateralImageDataset(df_tr,  WHOLE_IMG_ROOT, train_tf)
    val_ds   = BilateralImageDataset(df_val, WHOLE_IMG_ROOT, eval_tf)
    test_ds  = BilateralImageDataset(df_te,  WHOLE_IMG_ROOT, eval_tf)

    NUM_WORKERS = 8

    train_ld = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, drop_last=True, prefetch_factor=4,
    )
    val_ld = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, drop_last=False, prefetch_factor=4,
    )
    test_ld = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, drop_last=False, prefetch_factor=4,
    )

    log(f"Bilateral images: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ---- model ----
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(device=DEVICE, init_mode=init_mode).to(DEVICE)

    if not RANDOM_INIT:
        load_student_from_ckpt(encoder, CKPT_PATH)
        log("Encoder initialized from pretrained FM (ema_state).")
    else:
        log("Encoder randomly initialized.")

    if FREEZE_ENCODER:
        for p in encoder.parameters():
            p.requires_grad = False
        log("Encoder frozen (head-only).")
    else:
        log("Encoder will be fine-tuned.")

    model = BilateralSVHModel(encoder).to(DEVICE)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    loss_main = MaskedBCELoss(pos_weight=POS_WEIGHT_MAIN)
    scaler = GradScaler() if DEVICE == "cuda" else None

    # save best by VAL AUC
    best_val_auc = -1.0

    for ep in range(0, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_ld, optimizer, loss_main, target_idx, scaler)

        tr_m = eval_metrics(model, train_ld, target_idx)
        va_m = eval_metrics(model, val_ld,   target_idx)
        te_m = eval_metrics(model, test_ld,  target_idx)

        log_binary_metrics(f"Train {TASK.capitalize()} (bilateral)", tr_m)
        log_binary_metrics(f"Val   {TASK.capitalize()} (bilateral)", va_m)
        log_binary_metrics(f"Test  {TASK.capitalize()} (bilateral)", te_m)

        # conformal (calibration = val)
        if USE_CONFORMAL:
            val_logits, val_scores = eval_loader_collect(model, val_ld, target_idx)
            val_mask = val_scores > -0.5
            val_logits = val_logits[val_mask]
            val_true_bin = (val_scores[val_mask] > 0).astype(int)

            cal_scores = fit_split_conformal_scores(val_logits, val_true_bin)

            val_sets, val_stats = conformal_predict_sets(val_logits, cal_scores, CONFORMAL_ALPHA)
            val_cov = conformal_coverage(val_sets, val_true_bin)
            log_conformal_stats("VAL(calib-self)", CONFORMAL_ALPHA, val_cov, val_stats)

            test_logits, test_scores = eval_loader_collect(model, test_ld, target_idx)
            test_mask = test_scores > -0.5
            test_logits = test_logits[test_mask]
            test_true_bin = (test_scores[test_mask] > 0).astype(int)

            test_sets, test_stats = conformal_predict_sets(test_logits, cal_scores, CONFORMAL_ALPHA)
            test_cov = conformal_coverage(test_sets, test_true_bin)
            log_conformal_stats("TEST", CONFORMAL_ALPHA, test_cov, test_stats)

        log(
            f"Epoch {ep:03d}/{EPOCHS} | TrainLoss={train_loss:.4f} | "
            f"Val AUC={va_m['auc']:.4f} | Time {time.time()-t0:.1f}s"
        )

        val_auc = va_m["auc"]
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = os.path.join(
                MODEL_SAVE_DIR,
                f"catch_bilateral_{TASK}_binary_dinov3_amp_ep{ep}_valauc{best_val_auc:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            log(f"✅ Saved best (by VAL {TASK.capitalize()} AUC) to {save_path}")
            if USE_CONFORMAL:
                log("🧷 (Saved best) Conformal stats above correspond to this epoch.")

    log("Training finished.")


if __name__ == "__main__":
    main()
