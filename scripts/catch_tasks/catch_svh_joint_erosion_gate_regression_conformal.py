#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CATCH Joint-level SvH Score Prediction (Erosion/JSN) with:
✅ 5 joints (MCP/PIP/Wrist/Ulna/Radius) joint-specific heads
✅ Wrist/Ulna/Radius: keep ONLY labels {0,1,2} (K=3)
✅ MCP/PIP: full labels (Erosion K=6 / JSN K=5)
✅ Gate head = binary-head style (LayerNorm + Linear(D->1)) so it can LOAD the corresponding binary checkpoint
✅ Mean pooling (same as the corresponding binary script)
✅ Warm-start: load encoder + gate from binary ckpt, with logs to confirm success
✅ Two-stage training:
    - First WARMUP_FREEZE_EPOCHS: freeze encoder+gate, train ordinal only
    - Then unfreeze encoder+gate and finetune (encoder lr=0.3x)
✅ Select best ckpt by Val(SOFT) PCC (NO test leakage for selection)
✅ Also report SCC/QWK/RMSE/MAE/R2/ACC
✅ Add Split Conformal Prediction (calibration = val split): regression interval coverage
✅ Train metrics computed on a stable loader (no sampler, no drop_last), like the JSN script
✅ When saving best ckpt: print per-split (Train/Val/Test) overall + per-joint metrics (SOFT),
   like the JSN script.

Run:
  CUDA_VISIBLE_DEVICES=7 nohup python catch_svh_joint_erosion_gate_regression_conformal.py \
    > train_log/catch_svh_joint_erosion_gate_regression_conformal.log 2>&1 &
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catch_split_utils import shared_patient_level_split_3way
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoModel, AutoImageProcessor, AutoConfig
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, cohen_kappa_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr


# =========================
# 1) Global config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------- Task switch --------
TASK = "erosion"  # "erosion" or "jsn"
assert TASK in ["erosion", "jsn"]

# -------- Paths --------
CSV_PATH = "/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_joint_score_raw.csv"
IMG_ROOT = "/home/UWO/ylong66/data/RA/RA/Joint Detection /yolov5/data/extracted_joint_images"
MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/catch_5joints_jointK_warmstart_bestPCC_conformal"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Best binary checkpoint for the erosion gate
BINARY_CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema/catch_svh_erosion_binary_dinov3_amp.pt" 

# -------- Data columns --------
FILE_COL = "file_name"
PATIENT_COL = "patient_id"
REGION_COL = "joint_type"

# Keep 5 joints
KEEP_ONLY = {"mcp", "pip", "wrist", "ulna", "radius"}
REGION_LIST = ["MCP", "PIP", "Wrist", "Ulna", "Radius"]

# Small joints constraint
SMALL_JOINTS = {"wrist", "ulna", "radius"}
SMALL_JOINT_MAX_LABEL = 2  # keep only 0/1/2

# -------- Hyperparams --------
IMG_SIZE = 224
PAD_SIZE = 256

BATCH = 720
EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 1e-4

# Warmup freeze
WARMUP_FREEZE_EPOCHS = 10

# Loss weights
LAMBDA_GATE = 1.0
LAMBDA_ORD = 1.0

# Focal loss on gate
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# LDS
USE_LDS = True
LDS_SIGMA = 2.0

# Sampler
USE_WEIGHTED_SAMPLER = True
REGION_BOOST = {"mcp": 1.0, "pip": 1.6, "wrist": 1.0, "ulna": 1.0, "radius": 1.0}
NONZERO_BOOST = 1.15
HIGH_GRADE_BOOST_POWER = 0.35
SAMPLE_W_CLIP = (1e-4, 10.0)  # align w/ the JSN sampler-clamping setup

# Gate threshold tuning on Val (optional HARD reporting)
USE_HARD_GATE_AT_REPORT = True
THRESH_GRID = np.linspace(0.05, 0.95, 19)
THRESH_CRITERION = "pcc"  # "pcc" or "rmse"

# Conformal (regression interval)
USE_CONFORMAL = True
CONFORMAL_ALPHA = 0.10  # target 90% marginal coverage

# Model
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RANDOM_INIT = False


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


# =========================
# 2) Distortion-free transforms + class-conditional aug
# =========================
class KeepRatioPadToSquare:
    def __init__(self, size: int, fill=(0, 0, 0), random_center: bool = False, center_jitter: float = 0.06):
        self.size = int(size)
        self.fill = fill
        self.random_center = bool(random_center)
        self.center_jitter = float(center_jitter)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.random_center:
            j = self.center_jitter
            cx = float(np.random.uniform(0.5 - j, 0.5 + j))
            cy = float(np.random.uniform(0.5 - j, 0.5 + j))
            centering = (cx, cy)
        else:
            centering = (0.5, 0.5)

        return ImageOps.pad(
            img,
            (self.size, self.size),
            method=Image.BICUBIC,
            color=self.fill,
            centering=centering,
        )


class AddGaussianNoise:
    def __init__(self, p: float = 0.25, std: Tuple[float, float] = (0.01, 0.03)):
        self.p = float(p)
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return x
        s = float(np.random.uniform(self.std[0], self.std[1]))
        return torch.clamp(x + torch.randn_like(x) * s, 0.0, 1.0)


tf_geom_train = T.Compose(
    [
        KeepRatioPadToSquare(PAD_SIZE, fill=(0, 0, 0), random_center=True, center_jitter=0.06),
        T.RandomRotation(degrees=6, fill=(0, 0, 0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(IMG_SIZE),
    ]
)
tf_geom_eval = T.Compose(
    [
        KeepRatioPadToSquare(PAD_SIZE, fill=(0, 0, 0), random_center=False),
        T.CenterCrop(IMG_SIZE),
    ]
)

tf_photo_zero = T.Compose([T.ColorJitter(brightness=0.05, contrast=0.05)])
tf_photo_pos = T.Compose(
    [
        T.ColorJitter(brightness=0.12, contrast=0.12),
        T.RandomAutocontrast(p=0.25),
        T.RandomEqualize(p=0.05),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.20),
        T.GaussianBlur(kernel_size=3, sigma=(0.05, 0.3)),
    ]
)

train_tf_zero = T.Compose([tf_geom_train, tf_photo_zero, T.ToTensor()])
train_tf_pos = T.Compose([tf_geom_train, tf_photo_pos, T.ToTensor(), AddGaussianNoise(p=0.25, std=(0.01, 0.03))])
eval_tf = T.Compose([tf_geom_eval, T.ToTensor()])


# =========================
# 3) Utils
# =========================
def clean_label(v, nan_val=-1) -> int:
    if pd.isna(v):
        return nan_val
    s = str(v).strip()
    if s == "":
        return nan_val
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return nan_val


def norm_region(s: str) -> str:
    return str(s).strip().lower()


def filter_keep_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_region_norm"] = df[REGION_COL].astype(str).map(norm_region)
    df = df[df["_region_norm"].isin(set(KEEP_ONLY))].reset_index(drop=True)
    return df


def patient_level_split_three_way(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 3407,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    return shared_patient_level_split_3way(
        df,
        patient_col=PATIENT_COL,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        source_csv_path=CSV_PATH,
    )


def filter_by_vocab(df: pd.DataFrame, joint2id: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["_region_norm"] = df[REGION_COL].astype(str).map(norm_region)
    df = df[df["_region_norm"].isin(set(joint2id.keys()))].reset_index(drop=True)
    return df


def joint_num_classes(task: str, region_norm: str) -> int:
    if region_norm in SMALL_JOINTS:
        return 3
    if task == "erosion":
        return 6
    return 5


# =========================
# 4) Dataset
# =========================
class JointDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, tf_zero, tf_pos, joint2id: dict):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.tf_zero = tf_zero
        self.tf_pos = tf_pos
        self.joint2id = joint2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / str(row[FILE_COL])
        img = Image.open(img_path).convert("RGB")

        y_int = int(row["_y"])
        tf = self.tf_pos if y_int > 0 else self.tf_zero
        img = tf(img)

        rn = norm_region(row[REGION_COL])
        jid = self.joint2id.get(rn, -1)
        jid_t = torch.tensor(jid, dtype=torch.long)

        y_t = torch.tensor(y_int, dtype=torch.long)
        return img, y_t, jid_t


# =========================
# 5) Per-joint LDS weights
# =========================
def compute_lds_weights_from_labels(labels: np.ndarray, num_classes: int, sigma: float) -> torch.Tensor:
    labels = labels.astype(int)
    labels = labels[(labels >= 0) & (labels < num_classes)]
    if len(labels) == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    smoothed = gaussian_filter1d(counts, sigma=sigma)
    w = 1.0 / (smoothed + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def compute_per_joint_lds(df_tr: pd.DataFrame, jointid2K: Dict[int, int], sigma: float) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for jid in sorted(df_tr["_joint_id"].unique().tolist()):
        jid = int(jid)
        K = int(jointid2K[jid])
        sub = df_tr[df_tr["_joint_id"] == jid]
        y = sub["_y"].values
        out[jid] = compute_lds_weights_from_labels(y, num_classes=K, sigma=sigma)
    return out


# =========================
# 6) Model: DINOv3 + mean pooling + per-joint heads (joint-specific K)
# =========================
class VisionEncoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, device=None, init_mode: str = "pretrained"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.device = device or DEVICE

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
            or 768
        )
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    @torch.no_grad()
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.processor(images=list(x), return_tensors="pt", do_rescale=False)
        return inp["pixel_values"].to(self.device)

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)
        pixel_values = self._process(x)
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D]
        if self.num_register_tokens > 0:
            cls_tok = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_register_tokens:, :]
            tokens = torch.cat([cls_tok, patches], dim=1)  # [B,1+P,D]
        return tokens


class GateOrdinalHead(nn.Module):
    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.K = int(K)
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1),
        )
        self.ordinal = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(True),
            nn.Linear(in_dim // 2, self.K - 1),
        )

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logit = self.gate(pooled).squeeze(1)
        ord_logits = self.ordinal(pooled)
        return gate_logit, ord_logits


class JointSVHGateOrdinalModel(nn.Module):
    def __init__(self, encoder: VisionEncoder, jointid2K: Dict[int, int]):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size
        self.jointid2K = {int(k): int(v) for k, v in jointid2K.items()}
        self.maxK = int(max(self.jointid2K.values()))
        # IMPORTANT: assumes joint_id is 0..N-1 (joint2id is constructed this way)
        self.heads = nn.ModuleList([GateOrdinalHead(D, self.jointid2K[jid]) for jid in sorted(self.jointid2K.keys())])

    def forward(self, x, joint_id) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(x)
        pooled = tokens.mean(dim=1)

        B = pooled.size(0)
        gate_out = pooled.new_zeros((B,), dtype=pooled.dtype)
        ord_out = pooled.new_zeros((B, self.maxK - 1), dtype=pooled.dtype)

        unique = torch.unique(joint_id)
        for jid in unique.tolist():
            jid = int(jid)
            if jid < 0 or jid >= len(self.heads):
                continue
            m = (joint_id == jid)
            if not m.any():
                continue
            g, o = self.heads[jid](pooled[m])
            gate_out[m] = g.to(gate_out.dtype)
            ord_out[m, : o.size(1)] = o.to(ord_out.dtype)
        return gate_out, ord_out


# =========================
# 7) Losses (variable-K ordinal)
# =========================
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_ordinal_targets_and_mask(y: torch.Tensor, K_per_sample: torch.Tensor, maxK: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B = y.size(0)
    Kps = K_per_sample.long().clamp(min=2, max=maxK)
    thresh = torch.arange(0, maxK - 1, device=y.device).view(1, -1).repeat(B, 1)
    yv = y.view(-1, 1).float()
    t = (yv > thresh.float()).float()
    valid = thresh < (Kps.view(-1, 1) - 1)
    return t, valid


class OrdinalBCELossVarKPerJoint(nn.Module):
    def __init__(self, jointid2K: Dict[int, int], per_joint_weights: Optional[Dict[int, torch.Tensor]] = None):
        super().__init__()
        self.jointid2K = {int(k): int(v) for k, v in jointid2K.items()}
        self.maxK = int(max(self.jointid2K.values()))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.per_joint_weights = per_joint_weights or {}

    def forward(self, ord_logits_pad: torch.Tensor, y: torch.Tensor, joint_id: torch.Tensor) -> torch.Tensor:
        Kps = torch.empty_like(joint_id, dtype=torch.long)
        for jid in torch.unique(joint_id).tolist():
            jid = int(jid)
            m = (joint_id == jid)
            if m.any():
                Kps[m] = int(self.jointid2K.get(jid, self.maxK))

        t, valid = make_ordinal_targets_and_mask(y, Kps, self.maxK)
        loss_mat = self.bce(ord_logits_pad, t)
        loss_mat = loss_mat * valid.float()
        denom = valid.float().sum(dim=1).clamp(min=1.0)
        loss = loss_mat.sum(dim=1) / denom

        if len(self.per_joint_weights) > 0:
            w_all = loss.new_ones(loss.shape)
            for jid in torch.unique(joint_id).tolist():
                jid = int(jid)
                if jid not in self.per_joint_weights:
                    continue
                m = (joint_id == jid)
                if not m.any():
                    continue
                w = self.per_joint_weights[jid].to(loss.device)
                yi = torch.clamp(y[m].long(), 0, len(w) - 1)
                w_all[m] = w[yi]
            loss = loss * w_all

        return loss.mean()


@torch.no_grad()
def ordinal_expected_value_raw_varK(ord_logits_pad: torch.Tensor, joint_id: torch.Tensor, jointid2K: Dict[int, int]) -> torch.Tensor:
    B, _ = ord_logits_pad.shape
    expv = ord_logits_pad.new_zeros((B,), dtype=torch.float32)

    j_np = joint_id.detach().cpu().numpy().astype(int)
    Kps = np.array([int(jointid2K.get(int(j), max(jointid2K.values()))) for j in j_np], dtype=np.int32)
    Kps_t = torch.tensor(Kps, device=ord_logits_pad.device, dtype=torch.long)

    for K in torch.unique(Kps_t).tolist():
        K = int(K)
        m = (Kps_t == K)
        if not m.any():
            continue
        logits = ord_logits_pad[m, : K - 1]
        p = torch.sigmoid(logits)
        b = p.size(0)
        probs = p.new_zeros((b, K))
        probs[:, 0] = 1 - p[:, 0]
        for c in range(1, K - 1):
            probs[:, c] = p[:, c - 1] - p[:, c]
        probs[:, K - 1] = p[:, K - 2]
        scores = torch.arange(0, K, device=p.device).float().view(1, -1)
        expv[m] = (probs * scores).sum(dim=1)

    return expv.to(ord_logits_pad.dtype)


# =========================
# 8) Metrics
# =========================
@torch.no_grad()
def compute_metrics(pred: np.ndarray, target: np.ndarray, pred_round_max: Optional[np.ndarray] = None) -> Dict[str, float]:
    mask = target > -0.5
    if mask.sum() == 0:
        return {}

    y = target[mask].astype(float)
    p = pred[mask].astype(float)

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = float(r2_score(y, p))
    mae = float(mean_absolute_error(y, p))

    pcc = float(pearsonr(y, p)[0]) if len(y) > 1 else 0.0
    scc = float(spearmanr(y, p)[0]) if len(y) > 1 else 0.0

    y_true_cls = y.astype(int)
    y_pred_cls = np.rint(p).astype(int)

    if pred_round_max is None:
        y_pred_cls = np.clip(y_pred_cls, 0, int(np.max(y_true_cls)))
    else:
        mmax = pred_round_max[mask].astype(int)
        y_pred_cls = np.clip(y_pred_cls, 0, mmax)

    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    try:
        qwk = float(cohen_kappa_score(y_true_cls, y_pred_cls, weights="quadratic"))
        if np.isnan(qwk) or np.isinf(qwk):
            qwk = 0.0
    except Exception:
        qwk = 0.0

    return {"rmse": rmse, "r2": r2, "pcc": pcc, "scc": scc, "mae": mae, "acc": acc, "qwk": qwk}


# =========================
# 9) Prediction / Eval (SOFT + optional HARD)
# =========================
@torch.no_grad()
def predict_with_thresholds(
    model,
    loader,
    jointid2K: Dict[int, int],
    thresholds: Optional[Dict[int, float]] = None,
):
    model.eval()
    preds, tgts, jids = [], [], []
    pred_round_max = []

    for img, y, joint_id in loader:
        img = img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        joint_id = joint_id.to(DEVICE, non_blocking=True)

        with autocast():
            gate_logit, ord_logits_pad = model(img, joint_id)
            expv = ordinal_expected_value_raw_varK(ord_logits_pad, joint_id, jointid2K)
            gate_prob = torch.sigmoid(gate_logit)

            if thresholds is None:
                pred = gate_prob * expv
            else:
                pred = expv.clone()
                for jid in torch.unique(joint_id).tolist():
                    jid = int(jid)
                    t = thresholds.get(jid, 0.5)
                    m = (joint_id == jid)
                    pred[m] = pred[m] * (gate_prob[m] > t).float()

        preds.append(pred.detach().cpu().numpy())
        tgts.append(y.detach().cpu().numpy())
        jids.append(joint_id.detach().cpu().numpy())

        jid_np = joint_id.detach().cpu().numpy().astype(int)
        mmax = np.array([jointid2K[int(j)] - 1 for j in jid_np], dtype=np.int32)
        pred_round_max.append(mmax)

    preds = np.concatenate(preds) if preds else np.array([])
    tgts = np.concatenate(tgts) if tgts else np.array([])
    jids = np.concatenate(jids) if jids else np.array([])
    pred_round_max = np.concatenate(pred_round_max) if pred_round_max else np.array([])
    return preds, tgts, jids, pred_round_max


@torch.no_grad()
def eval_loader(model, loader, jointid2K: Dict[int, int], thresholds: Optional[Dict[int, float]] = None) -> Dict[str, float]:
    pred, tgt, _, pred_round_max = predict_with_thresholds(model, loader, jointid2K, thresholds)
    if len(pred) == 0:
        return {}
    return compute_metrics(pred, tgt, pred_round_max=pred_round_max)


@torch.no_grad()
def eval_by_region(
    model,
    df_split: pd.DataFrame,
    joint2id: dict,
    jointid2K: Dict[int, int],
    thresholds: Optional[Dict[int, float]] = None,
    split_name: str = "SPLIT",
):
    ds_all = JointDataset(df_split, IMG_ROOT, eval_tf, eval_tf, joint2id)
    ld_all = DataLoader(ds_all, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    m_all = eval_loader(model, ld_all, jointid2K, thresholds)

    gate_mode = "HARD" if thresholds is not None else "SOFT"
    if m_all:
        log(
            f"[Eval-{gate_mode}] {split_name} ALL -> "
            f"SCC {m_all['scc']:.3f} | PCC {m_all['pcc']:.3f} | QWK {m_all['qwk']:.3f} | "
            f"RMSE {m_all['rmse']:.3f} | MAE {m_all['mae']:.3f} | R2 {m_all['r2']:.3f} | ACC {m_all['acc']:.3f}"
        )
    else:
        log(f"[Eval-{gate_mode}] {split_name} ALL -> no valid labels")

    for region in REGION_LIST:
        sub = df_split[df_split[REGION_COL].astype(str).str.lower() == region.lower()].reset_index(drop=True)
        if len(sub) == 0:
            log(f"  [{split_name}][{region}] n=0 -> skip")
            continue
        ds_r = JointDataset(sub, IMG_ROOT, eval_tf, eval_tf, joint2id)
        ld_r = DataLoader(ds_r, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
        m_r = eval_loader(model, ld_r, jointid2K, thresholds)
        if not m_r:
            log(f"  [{split_name}][{region}] n={len(sub)} -> no valid labels, skip")
            continue
        log(
            f"  [{split_name}][{region}][{gate_mode}] n={len(sub)} -> "
            f"SCC {m_r['scc']:.3f} | PCC {m_r['pcc']:.3f} | QWK {m_r['qwk']:.3f} | "
            f"RMSE {m_r['rmse']:.3f} | MAE {m_r['mae']:.3f} | R2 {m_r['r2']:.3f} | ACC {m_r['acc']:.3f}"
        )


# =========================
# 10) Sampler
# =========================
def build_sample_weights(
    df_tr: pd.DataFrame,
    per_joint_lds: Optional[Dict[int, torch.Tensor]],
    jointid2K: Dict[int, int]
) -> torch.Tensor:
    w = np.ones(len(df_tr), dtype=np.float32)

    regions = df_tr["_region_norm"].values
    for i, r in enumerate(regions):
        w[i] *= float(REGION_BOOST.get(str(r), 1.0))

    if per_joint_lds is not None and len(per_joint_lds) > 0:
        y = df_tr["_y"].values.astype(int)
        jid = df_tr["_joint_id"].values.astype(int)
        for i in range(len(df_tr)):
            jj = int(jid[i])
            yy = int(y[i])
            K = int(jointid2K[jj])
            if yy < 0 or yy >= K:
                continue
            if jj in per_joint_lds:
                w[i] *= float(per_joint_lds[jj][yy].item())

    y = df_tr["_y"].values.astype(int)
    w *= np.where(y > 0, NONZERO_BOOST, 1.0).astype(np.float32)
    w *= np.power((1.0 + y.clip(min=0).astype(np.float32)), HIGH_GRADE_BOOST_POWER)

    w = w / (w.mean() + 1e-8)
    w = np.clip(w, SAMPLE_W_CLIP[0], SAMPLE_W_CLIP[1])
    return torch.tensor(w, dtype=torch.float32)


# =========================
# 11) Warm-start from binary ckpt (encoder + gate)
# =========================
def warmstart_from_binary_ckpt(model: JointSVHGateOrdinalModel, binary_ckpt_path: str):
    sd = torch.load(binary_ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    # ---- encoder: keys like "encoder.encoder.*"
    enc_map = {}
    for k, v in sd.items():
        if k.startswith("encoder.encoder."):
            nk = k.replace("encoder.encoder.", "encoder.", 1)
            enc_map[nk] = v

    missing, unexpected = model.encoder.load_state_dict(enc_map, strict=False)
    log("[WarmStart] Encoder load from binary ckpt:")
    log(f"  missing={len(missing)} (first5={missing[:5]})")
    log(f"  unexpected={len(unexpected)} (first5={unexpected[:5]})")

    # ---- gate: erosion binary head is head_e = LN + Linear
    need = ["head_e.0.weight", "head_e.0.bias", "head_e.1.weight", "head_e.1.bias"]
    if not all(k in sd for k in need):
        log("[WarmStart] WARNING: binary ckpt missing head_e.* keys -> skip gate init.")
        return

    ln_w = sd["head_e.0.weight"]
    ln_b = sd["head_e.0.bias"]
    fc_w = sd["head_e.1.weight"]
    fc_b = sd["head_e.1.bias"]

    ok = True
    for head in model.heads:
        if head.gate[0].weight.shape != ln_w.shape or head.gate[1].weight.shape != fc_w.shape:
            ok = False
            break
    if not ok:
        log("[WarmStart] WARNING: gate shape mismatch -> skip gate init.")
        return

    with torch.no_grad():
        for head in model.heads:
            head.gate[0].weight.copy_(ln_w)
            head.gate[0].bias.copy_(ln_b)
            head.gate[1].weight.copy_(fc_w)
            head.gate[1].bias.copy_(fc_b)

    diff = float(torch.norm(model.heads[0].gate[1].weight.detach().float().cpu() - fc_w.float()).item())
    log(f"[WarmStart] Gate init from binary head_e applied to ALL joints. ||W_loaded - W_src||={diff:.6f}")


def set_requires_grad(mod: nn.Module, flag: bool):
    for p in mod.parameters():
        p.requires_grad = flag


# =========================
# 12) Conformal prediction for regression (split conformal intervals)
# =========================
def conformal_q_from_calibration(abs_residuals: np.ndarray, alpha: float) -> float:
    r = abs_residuals.astype(np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    n = r.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    q = float(np.partition(r, k - 1)[k - 1])
    return q


def conformal_interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> Dict[str, float]:
    y = y_true.astype(np.float64)
    p = y_pred.astype(np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0 or not np.isfinite(q):
        return {"coverage": float("nan"), "avg_width": float("nan"), "n": int(m.sum())}
    lo = p[m] - q
    hi = p[m] + q
    ok = (y[m] >= lo) & (y[m] <= hi)
    return {"coverage": float(np.mean(ok)), "avg_width": float(2.0 * q), "n": int(m.sum())}


def log_conformal_reg(title: str, alpha: float, q: float, stats: Dict[str, float]):
    log(
        f"🧪 Conformal {title} | alpha={alpha:.2f} "
        f"q={q:.4f} coverage={stats['coverage']:.3f} avg_width={stats['avg_width']:.3f} n={stats['n']}"
    )


# =========================
# 13) Train
# =========================
def train_epoch(model, loader, optimizer, gate_loss_fn, ord_loss_fn, scaler):
    model.train()
    total, n = 0.0, 0
    for img, y, joint_id in loader:
        img = img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        joint_id = joint_id.to(DEVICE, non_blocking=True)

        y_gate = (y > 0).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            gate_logit, ord_logits_pad = model(img, joint_id)
            lg = gate_loss_fn(gate_logit, y_gate)
            lo = ord_loss_fn(ord_logits_pad, y, joint_id)
            loss = LAMBDA_GATE * lg + LAMBDA_ORD * lo

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item()) * img.size(0)
        n += img.size(0)
    return total / max(1, n)


# =========================
# 14) Threshold tuning (optional HARD report)
# =========================
@torch.no_grad()
def tune_thresholds_on_val(model, df_val: pd.DataFrame, joint2id: Dict[str, int], jointid2K: Dict[int, int]) -> Dict[int, float]:
    ds = JointDataset(df_val, IMG_ROOT, eval_tf, eval_tf, joint2id)
    ld = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_gate, all_expv, all_y, all_j, all_mmax = [], [], [], [], []

    for img, y, joint_id in ld:
        img = img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        joint_id = joint_id.to(DEVICE, non_blocking=True)
        with autocast():
            gate_logit, ord_logits_pad = model(img, joint_id)
            gate_prob = torch.sigmoid(gate_logit)
            expv = ordinal_expected_value_raw_varK(ord_logits_pad, joint_id, jointid2K)

        all_gate.append(gate_prob.detach().cpu().numpy())
        all_expv.append(expv.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        all_j.append(joint_id.detach().cpu().numpy())

        jid_np = joint_id.detach().cpu().numpy().astype(int)
        mmax = np.array([jointid2K[int(j)] - 1 for j in jid_np], dtype=np.int32)
        all_mmax.append(mmax)

    gate = np.concatenate(all_gate)
    expv = np.concatenate(all_expv)
    y = np.concatenate(all_y)
    j = np.concatenate(all_j).astype(int)
    mmax_all = np.concatenate(all_mmax).astype(int)

    thresholds: Dict[int, float] = {}
    for region in REGION_LIST:
        rn = region.lower()
        if rn not in joint2id:
            continue
        jid = int(joint2id[rn])
        m = (j == jid)
        if m.sum() < 2:
            thresholds[jid] = 0.5
            continue

        best_t = 0.5
        best_score = -1e9 if THRESH_CRITERION == "pcc" else 1e9

        for t in THRESH_GRID:
            pred = expv[m] * (gate[m] > t).astype(np.float32)
            met = compute_metrics(pred, y[m], pred_round_max=mmax_all[m])
            if not met:
                continue
            if THRESH_CRITERION == "pcc":
                score = met["pcc"]
                if score > best_score:
                    best_score = score
                    best_t = float(t)
            else:
                score = met["rmse"]
                if score < best_score:
                    best_score = score
                    best_t = float(t)

        thresholds[jid] = best_t

    return thresholds


# =========================
# 15) Main
# =========================
def main():
    log(f"🚀 CATCH | TASK={TASK.upper()} | 5 joints | joint-specific K | mean pool | gate=binary-head | warmstart binary | best by Val(SOFT) PCC | + Conformal")

    # task setup
    label_col = "Erosion_score" if TASK == "erosion" else "JSN_score"

    df_all = pd.read_csv(CSV_PATH)
    for c in [FILE_COL, PATIENT_COL, REGION_COL, "Erosion_score", "JSN_score"]:
        assert c in df_all.columns, f"CSV missing {c}"

    df_all = filter_keep_only(df_all)

    df_all["_y"] = df_all[label_col].apply(clean_label).astype(int)
    valid_ratio = float((df_all["_y"] >= 0).mean())
    log(f"After keep-only: N={len(df_all)} | valid label ratio for {label_col} = {valid_ratio:.3f}")
    df_all = df_all[df_all["_y"] >= 0].reset_index(drop=True)
    log(f"After dropping invalid labels: N={len(df_all)}")

    # enforce small joints label cap
    df_all["_region_norm"] = df_all[REGION_COL].astype(str).map(norm_region)
    m_small = df_all["_region_norm"].isin(SMALL_JOINTS)
    before = int(m_small.sum())
    df_all = df_all[~m_small | (df_all["_y"] <= SMALL_JOINT_MAX_LABEL)].reset_index(drop=True)
    after = int(df_all["_region_norm"].isin(SMALL_JOINTS).sum())
    log(f"Small joints filter (keep 0/1/2): before={before} after={after}")

    # split
    df_tr, df_va, df_te = patient_level_split_three_way(df_all, 0.80, 0.10, 0.10, seed=3407)
    log(f"Split (patient-level) -> Train: {len(df_tr)} | Val: {len(df_va)} | Test: {len(df_te)}")

    # vocab (train only)
    joints_train = sorted(df_tr[REGION_COL].astype(str).map(norm_region).unique().tolist())
    joint2id = {j: i for i, j in enumerate(joints_train)}
    log(f"Joint vocab (train) size={len(joint2id)} -> {joints_train}")

    # joint-specific K
    jointid2K: Dict[int, int] = {}
    for jn, jid in joint2id.items():
        jointid2K[int(jid)] = int(joint_num_classes(TASK, jn))

    # annotate ids
    for df_ in [df_tr, df_va, df_te]:
        df_["_region_norm"] = df_[REGION_COL].astype(str).map(norm_region)
        df_["_joint_id"] = df_["_region_norm"].map(lambda x: joint2id.get(x, -1)).astype(int)

    df_va = filter_by_vocab(df_va, joint2id)
    df_te = filter_by_vocab(df_te, joint2id)

    # LDS
    per_joint_lds: Optional[Dict[int, torch.Tensor]] = None
    if USE_LDS:
        per_joint_lds = compute_per_joint_lds(df_tr, jointid2K=jointid2K, sigma=LDS_SIGMA)
        for region in REGION_LIST:
            r = region.lower()
            if r in joint2id:
                jid = joint2id[r]
                K = jointid2K[jid]
                sub = df_tr[df_tr["_joint_id"] == jid]
                counts = np.bincount(sub["_y"].values.astype(int), minlength=K).astype(int).tolist()
                w = per_joint_lds.get(jid, torch.ones(K))
                log(f"--- {region} K={K} LDS sigma={LDS_SIGMA} counts={counts} weights={np.round(w.numpy(), 3).tolist()}")

    # datasets
    train_ds = JointDataset(df_tr, IMG_ROOT, train_tf_zero, train_tf_pos, joint2id)
    val_ds = JointDataset(df_va, IMG_ROOT, eval_tf, eval_tf, joint2id)
    test_ds = JointDataset(df_te, IMG_ROOT, eval_tf, eval_tf, joint2id)

    # loaders
    if USE_WEIGHTED_SAMPLER:
        sample_w = build_sample_weights(df_tr, per_joint_lds, jointid2K)
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
        train_ld = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
        log(f"WeightedRandomSampler enabled | mean_w={float(sample_w.mean()):.3f} | max_w={float(sample_w.max()):.3f}")
    else:
        train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # ✅ stable train metrics loader (NO sampler, NO drop_last), like JSN script
    train_eval_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # model
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(model_name=MODEL_NAME, device=DEVICE, init_mode=init_mode).to(DEVICE)
    model = JointSVHGateOrdinalModel(encoder, jointid2K=jointid2K).to(DEVICE)

    # warmstart (encoder + gate)
    if (BINARY_CKPT_PATH is not None) and os.path.exists(BINARY_CKPT_PATH):
        warmstart_from_binary_ckpt(model, BINARY_CKPT_PATH)
    else:
        log(f"[WarmStart] WARNING: BINARY_CKPT_PATH not found -> {BINARY_CKPT_PATH}")

    # losses
    gate_loss_fn = FocalLossWithLogits(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction="mean")
    ord_loss_fn = OrdinalBCELossVarKPerJoint(jointid2K=jointid2K, per_joint_weights=per_joint_lds)

    # warmup freeze: encoder + gate frozen, only train ordinal
    set_requires_grad(model.encoder, False)
    for h in model.heads:
        set_requires_grad(h.gate, False)
    log(f"[Warmup] Freeze encoder+gate for first {WARMUP_FREEZE_EPOCHS} epochs (train ordinal only).")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()

    best_val_pcc = -1e9
    best_ckpt_path = None

    for ep in range(1, EPOCHS + 1):
        # unfreeze at boundary + rebuild optimizer
        if ep == WARMUP_FREEZE_EPOCHS + 1:
            set_requires_grad(model.encoder, True)
            for h in model.heads:
                set_requires_grad(h.gate, True)

            optimizer = optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": LR * 0.3},
                    {"params": model.heads.parameters(),   "lr": LR},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            log("[Warmup] Unfroze encoder+gate. Rebuilt optimizer (encoder lr=0.3x).")

        t0 = time.time()
        tr_loss = train_epoch(model, train_ld, optimizer, gate_loss_fn, ord_loss_fn, scaler)

        # ✅ stable train metrics (no sampler) to reduce noise
        tr_m = eval_loader(model, train_eval_ld, jointid2K, thresholds=None)
        va_m = eval_loader(model, val_ld,        jointid2K, thresholds=None)

        curr_val_pcc = va_m.get("pcc", -1e9)

        # ✅ Conformal (VAL as calibration), report on VAL during training (SOFT)
        if USE_CONFORMAL:
            val_pred, val_y, _, _ = predict_with_thresholds(model, val_ld, jointid2K, thresholds=None)
            q = conformal_q_from_calibration(np.abs(val_y.astype(np.float64) - val_pred.astype(np.float64)), CONFORMAL_ALPHA)
            log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, conformal_interval_coverage(val_y, val_pred, q))


        log(
            f"Ep {ep:03d} | Loss {tr_loss:.4f} | "
            f"Tr  SCC {tr_m.get('scc',0):.3f} | PCC {tr_m.get('pcc',0):.3f} | QWK {tr_m.get('qwk',0):.3f} | "
            f"RMSE {tr_m.get('rmse',0):.3f} | MAE {tr_m.get('mae',0):.3f} | R2 {tr_m.get('r2',0):.3f} | ACC {tr_m.get('acc',0):.3f} || "
            f"Val(SOFT) SCC {va_m.get('scc',0):.3f} | PCC {va_m.get('pcc',0):.3f} | QWK {va_m.get('qwk',0):.3f} | "
            f"RMSE {va_m.get('rmse',0):.3f} | MAE {va_m.get('mae',0):.3f} | R2 {va_m.get('r2',0):.3f} | ACC {va_m.get('acc',0):.3f} || "
            f"time {(time.time()-t0):.1f}s"
        )

        # save best by Val PCC only
        if curr_val_pcc > best_val_pcc:
            best_val_pcc = curr_val_pcc
            save_name = f"catch_{TASK}_5joints_jointK_warmstart_meanpool_bestValSoftPCC{best_val_pcc:.4f}.pt"
            best_ckpt_path = os.path.join(MODEL_SAVE_DIR, save_name)

            torch.save(
                {
                    "model": model.state_dict(),
                    "joint2id": joint2id,
                    "jointid2K": jointid2K,
                    "task": TASK,
                    "epoch": ep,
                    "best_val_soft_pcc": best_val_pcc,
                    "val_metrics_soft": va_m,
                    "config": {
                        "MODEL_NAME": MODEL_NAME,
                        "BINARY_CKPT_PATH": BINARY_CKPT_PATH,
                        "WARMUP_FREEZE_EPOCHS": WARMUP_FREEZE_EPOCHS,
                        "IMG_SIZE": IMG_SIZE,
                        "PAD_SIZE": PAD_SIZE,
                        "BATCH": BATCH,
                        "EPOCHS": EPOCHS,
                        "LR": LR,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "FOCAL_GAMMA": FOCAL_GAMMA,
                        "FOCAL_ALPHA": FOCAL_ALPHA,
                        "USE_LDS": USE_LDS,
                        "LDS_SIGMA": LDS_SIGMA,
                        "USE_WEIGHTED_SAMPLER": USE_WEIGHTED_SAMPLER,
                        "REGION_BOOST": REGION_BOOST,
                        "NONZERO_BOOST": NONZERO_BOOST,
                        "HIGH_GRADE_BOOST_POWER": HIGH_GRADE_BOOST_POWER,
                        "SAMPLE_W_CLIP": list(SAMPLE_W_CLIP),
                        "USE_HARD_GATE_AT_REPORT": USE_HARD_GATE_AT_REPORT,
                        "THRESH_GRID": THRESH_GRID.tolist(),
                        "THRESH_CRITERION": THRESH_CRITERION,
                        "SMALL_JOINTS": sorted(list(SMALL_JOINTS)),
                        "SMALL_JOINT_MAX_LABEL": SMALL_JOINT_MAX_LABEL,
                        "POOLING": "mean",
                        "GATE": "LN+Linear (binary-head style)",
                        "AUG": "pos-only photometric+noise; zeros very light jitter",
                        "BEST_SELECT": "Val(SOFT) PCC",
                        "USE_CONFORMAL": USE_CONFORMAL,
                        "CONFORMAL_ALPHA": CONFORMAL_ALPHA,
                        "CONFORMAL_TYPE": "split conformal regression intervals on |y-pred| using VAL as calib",
                    },
                },
                best_ckpt_path,
            )
            log(f"  ✅ Saved Best (Val SOFT PCC) -> {save_name}")

            # ✅ Like JSN: print per-split + per-joint metrics at the moment best improves
            log("  📌 Breakdown (SOFT) validation per-joint metrics at BEST checkpoint:")
            eval_by_region(model, df_tr, joint2id, jointid2K, thresholds=None, split_name="TRAIN")
            eval_by_region(model, df_va, joint2id, jointid2K, thresholds=None, split_name="VAL")

    if best_ckpt_path is None:
        log("No best checkpoint found.")
        return

    # -------- Final evaluate best --------
    log("======= Final: Evaluate Best ckpt on Val/Test (SOFT) + Conformal, then tune HARD thresholds and report HARD on Test =======")
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    log(f"Loaded best ckpt epoch={ckpt.get('epoch', -1)} | best Val(SOFT) PCC={ckpt.get('best_val_soft_pcc', -1):.4f}")

    log("---- SOFT gate point metrics ----")
    eval_by_region(model, df_va, joint2id, jointid2K, thresholds=None, split_name="VAL")
    eval_by_region(model, df_te, joint2id, jointid2K, thresholds=None, split_name="TEST")

    if USE_CONFORMAL:
        val_pred, val_y, _, _ = predict_with_thresholds(model, val_ld, jointid2K, thresholds=None)
        q = conformal_q_from_calibration(np.abs(val_y.astype(np.float64) - val_pred.astype(np.float64)), CONFORMAL_ALPHA)
        log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, conformal_interval_coverage(val_y, val_pred, q))

        te_pred, te_y, _, _ = predict_with_thresholds(model, test_ld, jointid2K, thresholds=None)
        log_conformal_reg("TEST", CONFORMAL_ALPHA, q, conformal_interval_coverage(te_y, te_pred, q))

    if USE_HARD_GATE_AT_REPORT:
        thresholds = tune_thresholds_on_val(model, df_va, joint2id, jointid2K)
        th_str = []
        for r in REGION_LIST:
            rn = r.lower()
            if rn in joint2id:
                jid = int(joint2id[rn])
                th_str.append(f"{r}:{thresholds.get(jid,0.5):.2f}")
        log("Val-tuned gate thresholds -> " + " | ".join(th_str))

        log("---- HARD gate point metrics on Test ----")
        eval_by_region(model, df_te, joint2id, jointid2K, thresholds=thresholds, split_name="TEST")


if __name__ == "__main__":
    main()
