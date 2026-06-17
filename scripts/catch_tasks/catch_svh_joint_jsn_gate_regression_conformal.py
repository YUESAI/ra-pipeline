#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CATCH Joint-level SvH Score Prediction (JSN) with:
✅ ONLY 2 joints: MCP/PIP (joint-specific heads)
✅ JSN-friendly augmentation: keep aspect ratio (letterbox), no geometric distortion
✅ Gate head = binary-head style (LayerNorm + Linear(D->1)) so it can LOAD your JSN binary ckpt
✅ Mean pooling (same as your binary script)
✅ Warm-start: load encoder + gate from JSN binary ckpt, with logs to confirm success
✅ Two-stage training:
    - First WARMUP_FREEZE_EPOCHS: freeze encoder+gate, train ordinal only
    - Then unfreeze encoder+gate and finetune (encoder lr=0.3x)
✅ Select best ckpt ONLY by Val(SOFT) PCC
✅ Also report SCC/QWK/RMSE/MAE/R2/ACC
✅ Add Split Conformal Prediction (calibration = val split): regression interval coverage

Run:
  CUDA_VISIBLE_DEVICES=7 nohup python catch_svh_joint_jsn_gate_regression_conformal.py \
    > train_log/catch_svh_joint_jsn_gate_regression_conformal_extra8.py.log 2>&1 &
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
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

TASK = "jsn"
assert TASK == "jsn"

CSV_PATH = "/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_joint_score_raw.csv"
IMG_ROOT = "/home/UWO/ylong66/data/RA/RA/Joint Detection /yolov5/data/extracted_joint_images"
MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/catch_jsn_pip_mcp_warmstart_conformal_bestPCC"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ✅ Your best JSN binary ckpt (encoder + head_j)
BINARY_CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema/catch_svh_jsn_binary_dinov3_amp.pt" 

FILE_COL = "file_name"
PATIENT_COL = "patient_id"
REGION_COL = "joint_type"

# ✅ JSN only: PIP + MCP
KEEP_ONLY = {"mcp", "pip"}
REGION_LIST = ["MCP", "PIP"]

# JSN score: 0..4
NUM_CLASSES = 5  # K=5

# Image / training
IMG_SIZE = 224
PAD_SIZE = 224  # letterbox target square size

BATCH = 720
EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 1e-4

# two-stage
WARMUP_FREEZE_EPOCHS = 10  # freeze encoder+gate, train ordinal only

# Loss weights
LAMBDA_GATE = 1.0
LAMBDA_ORD  = 1.0

# Gate focal
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# LDS (optional)
USE_LDS = True
LDS_SIGMA = 2.0

# Sampler
USE_WEIGHTED_SAMPLER = True
REGION_BOOST = {"mcp": 1.0, "pip": 1.10}
NONZERO_BOOST = 1.05
HIGH_GRADE_BOOST_POWER = 0.15
SAMPLE_W_CLIP = (1e-4, 10.0)

# Conformal (regression interval)
USE_CONFORMAL = True
CONFORMAL_ALPHA = 0.10  # target 90% marginal coverage

# Model
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RANDOM_INIT = False


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


# =========================
# 2) JSN-friendly transforms (keep aspect ratio)
# =========================
class KeepRatioPadToSquare:
    """
    Resize + pad to square while keeping aspect ratio (letterbox).
    This avoids geometric distortion (important for JSN gap geometry).
    """
    def __init__(self, size: int, fill=(0, 0, 0), random_center: bool = False, center_jitter: float = 0.03):
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


# Light photometric only (no random crop / no resize distortion)
tf_geom_train = T.Compose([
    KeepRatioPadToSquare(PAD_SIZE, fill=(0, 0, 0), random_center=True, center_jitter=0.03),
    T.RandomRotation(degrees=4, fill=(0, 0, 0)),
    T.RandomHorizontalFlip(p=0.5),
])

tf_geom_eval = T.Compose([
    KeepRatioPadToSquare(PAD_SIZE, fill=(0, 0, 0), random_center=False),
])

tf_photo = T.Compose([
    T.ColorJitter(brightness=0.08, contrast=0.08),
    T.RandomAutocontrast(p=0.20),
])

train_tf = T.Compose([tf_geom_train, tf_photo, T.ToTensor()])
eval_tf  = T.Compose([tf_geom_eval,  T.ToTensor()])


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
    rng = np.random.RandomState(seed)
    patients = df[PATIENT_COL].astype(str).unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    df_tr = df[df[PATIENT_COL].astype(str).isin(train_p)].reset_index(drop=True)
    df_va = df[df[PATIENT_COL].astype(str).isin(val_p)].reset_index(drop=True)
    df_te = df[df[PATIENT_COL].astype(str).isin(test_p)].reset_index(drop=True)
    return df_tr, df_va, df_te


def filter_by_vocab(df: pd.DataFrame, joint2id: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["_region_norm"] = df[REGION_COL].astype(str).map(norm_region)
    df = df[df["_region_norm"].isin(set(joint2id.keys()))].reset_index(drop=True)
    return df


# =========================
# 4) Dataset
# =========================
class JointDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, tf, joint2id: dict):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.tf = tf
        self.joint2id = joint2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / str(row[FILE_COL])
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        rn = norm_region(row[REGION_COL])
        jid = self.joint2id.get(rn, -1)
        jid_t = torch.tensor(jid, dtype=torch.long)

        y = int(row["_y"])
        y_t = torch.tensor(y, dtype=torch.long)
        return img, y_t, jid_t


# =========================
# 5) LDS weights (optional)
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


def compute_per_joint_lds(df_tr: pd.DataFrame, num_classes: int, sigma: float) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for jid in sorted(df_tr["_joint_id"].unique().tolist()):
        sub = df_tr[df_tr["_joint_id"] == int(jid)]
        y = sub["_y"].values
        out[int(jid)] = compute_lds_weights_from_labels(y, num_classes=num_classes, sigma=sigma)
    return out


# =========================
# 6) Model: DINOv3 + mean pooling + per-joint heads
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
        # x is list/stack of tensors in [0,1]; do_rescale=False is correct
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
            tokens = torch.cat([cls_tok, patches], dim=1)  # [B, 1+P, D]
        return tokens


class GateOrdinalHead(nn.Module):
    """
    Gate = binary-head style: LayerNorm + Linear(D->1)  ✅ loads from binary ckpt head_j
    Ordinal = MLP -> (K-1)
    """
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
    def __init__(self, encoder: VisionEncoder, joint_vocab: int, K: int):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size
        self.K = int(K)
        self.heads = nn.ModuleList([GateOrdinalHead(D, self.K) for _ in range(joint_vocab)])

    def forward(self, x, joint_id) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(x)         # [B, N, D]
        pooled = tokens.mean(dim=1)      # ✅ mean pooling (same as binary)

        B = pooled.size(0)
        gate_out = pooled.new_zeros((B,), dtype=pooled.dtype)
        ord_out  = pooled.new_zeros((B, self.K - 1), dtype=pooled.dtype)

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
            ord_out[m]  = o.to(ord_out.dtype)
        return gate_out, ord_out


# =========================
# 7) Losses + Ordinal utilities
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


def make_ordinal_targets(y: torch.Tensor, K: int) -> torch.Tensor:
    thresh = torch.arange(0, K - 1, device=y.device).view(1, -1)
    yv = y.view(-1, 1).float()
    return (yv > thresh.float()).float()


class OrdinalBCELoss(nn.Module):
    def __init__(self, K: int, per_joint_weights: Optional[Dict[int, torch.Tensor]] = None):
        super().__init__()
        self.K = int(K)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.per_joint_weights = per_joint_weights or {}

    def forward(self, ord_logits: torch.Tensor, y: torch.Tensor, joint_id: torch.Tensor) -> torch.Tensor:
        t = make_ordinal_targets(y, self.K)             # [B,K-1]
        loss_mat = self.bce(ord_logits, t).mean(dim=1)  # [B]

        # per-joint LDS weights (by class y)
        if len(self.per_joint_weights) > 0:
            w_all = loss_mat.new_ones(loss_mat.shape)
            for jid in torch.unique(joint_id).tolist():
                jid = int(jid)
                if jid not in self.per_joint_weights:
                    continue
                m = (joint_id == jid)
                if not m.any():
                    continue
                w = self.per_joint_weights[jid].to(loss_mat.device)  # len=K
                yi = torch.clamp(y[m].long(), 0, len(w) - 1)
                w_all[m] = w[yi]
            loss_mat = loss_mat * w_all

        return loss_mat.mean()


@torch.no_grad()
def ordinal_expected_value_raw(ord_logits: torch.Tensor, K: int) -> torch.Tensor:
    """
    CORAL expected value from cumulative logits (K-1)
    """
    p = torch.sigmoid(ord_logits)  # [B,K-1]
    B = p.size(0)
    probs = p.new_zeros((B, K))
    probs[:, 0] = 1 - p[:, 0]
    for c in range(1, K - 1):
        probs[:, c] = p[:, c - 1] - p[:, c]
    probs[:, K - 1] = p[:, K - 2]
    scores = torch.arange(0, K, device=p.device).float().view(1, -1)
    return (probs * scores).sum(dim=1)


# =========================
# 8) Metrics (PCC main selection + others)
# =========================
def _safe_pearson(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) <= 1:
        return 0.0
    try:
        r = pearsonr(y, p)[0]
        if np.isnan(r) or np.isinf(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0


def _safe_spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) <= 1:
        return 0.0
    try:
        r = spearmanr(y, p).correlation
        if r is None or np.isnan(r) or np.isinf(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        k = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        if np.isnan(k) or np.isinf(k):
            return 0.0
        return float(k)
    except Exception:
        return 0.0


@torch.no_grad()
def compute_metrics(pred: np.ndarray, target: np.ndarray, K: int) -> Dict[str, float]:
    mask = target > -0.5
    if mask.sum() == 0:
        return {}

    y = target[mask].astype(float)
    p = pred[mask].astype(float)

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = float(r2_score(y, p))
    mae = float(mean_absolute_error(y, p))
    pcc = _safe_pearson(y, p)
    scc = _safe_spearman(y, p)

    y_true_cls = y.astype(int)
    y_pred_cls = np.clip(np.rint(p).astype(int), 0, K - 1)
    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    qwk = _qwk(y_true_cls, y_pred_cls)

    nz = y_true_cls > 0
    if nz.sum() >= 2:
        pcc_nz = _safe_pearson(y[nz], p[nz])
        scc_nz = _safe_spearman(y[nz], p[nz])
        qwk_nz = _qwk(y_true_cls[nz], y_pred_cls[nz])
        rmse_nz = float(np.sqrt(mean_squared_error(y[nz], p[nz])))
        mae_nz = float(mean_absolute_error(y[nz], p[nz]))
        r2_nz = float(r2_score(y[nz], p[nz]))
        acc_nz = float(accuracy_score(y_true_cls[nz], y_pred_cls[nz]))
    else:
        pcc_nz = scc_nz = qwk_nz = 0.0
        rmse_nz = mae_nz = r2_nz = acc_nz = 0.0

    return {
        "pcc": pcc, "scc": scc, "qwk": qwk,
        "rmse": rmse, "mae": mae, "r2": r2, "acc": acc,
        "pcc_nz": pcc_nz, "scc_nz": scc_nz, "qwk_nz": qwk_nz,
        "rmse_nz": rmse_nz, "mae_nz": mae_nz, "r2_nz": r2_nz, "acc_nz": acc_nz,
        "nz_n": int(nz.sum()),
    }


# =========================
# 9) Prediction / Eval (SOFT)
# =========================
@torch.no_grad()
def predict_soft(model, loader, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, tgts, jids = [], [], []
    for img, y, joint_id in loader:
        img = img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        joint_id = joint_id.to(DEVICE, non_blocking=True)

        with autocast():
            gate_logit, ord_logits = model(img, joint_id)
            expv = ordinal_expected_value_raw(ord_logits, K)
            gate_prob = torch.sigmoid(gate_logit)
            pred = gate_prob * expv

        preds.append(pred.detach().cpu().numpy())
        tgts.append(y.detach().cpu().numpy())
        jids.append(joint_id.detach().cpu().numpy())

    return np.concatenate(preds), np.concatenate(tgts), np.concatenate(jids)


@torch.no_grad()
def eval_soft(model, loader, K: int) -> Dict[str, float]:
    p, y, _ = predict_soft(model, loader, K)
    return compute_metrics(p, y, K)


@torch.no_grad()
def eval_by_region_soft(model, df: pd.DataFrame, joint2id: Dict[str, int], K: int, split_name: str):
    ds_all = JointDataset(df, IMG_ROOT, eval_tf, joint2id)
    ld_all = DataLoader(ds_all, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    m_all = eval_soft(model, ld_all, K)
    if m_all:
        log(
            f"[Eval-SOFT] {split_name} ALL -> "
            f"PCC {m_all['pcc']:.3f} | SCC {m_all['scc']:.3f} | QWK {m_all['qwk']:.3f} | "
            f"RMSE {m_all['rmse']:.3f} | MAE {m_all['mae']:.3f} | R2 {m_all['r2']:.3f} | ACC {m_all['acc']:.3f} | "
            f"PCC(nz) {m_all['pcc_nz']:.3f} | SCC(nz) {m_all['scc_nz']:.3f} | QWK(nz) {m_all['qwk_nz']:.3f} | "
            f"nz_n {m_all['nz_n']}"
        )
    for region in REGION_LIST:
        sub = df[df[REGION_COL].astype(str).str.lower() == region.lower()].reset_index(drop=True)
        if len(sub) == 0:
            continue
        ds_r = JointDataset(sub, IMG_ROOT, eval_tf, joint2id)
        ld_r = DataLoader(ds_r, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
        m_r = eval_soft(model, ld_r, K)
        if not m_r:
            continue
        log(
            f"  [{region}][SOFT] n={len(sub)} -> "
            f"PCC {m_r['pcc']:.3f} | SCC {m_r['scc']:.3f} | QWK {m_r['qwk']:.3f} | "
            f"RMSE {m_r['rmse']:.3f} | MAE {m_r['mae']:.3f} | R2 {m_r['r2']:.3f} | ACC {m_r['acc']:.3f}"
        )


# =========================
# 10) Weighted sampler
# =========================
def build_sample_weights(df_tr: pd.DataFrame, per_joint_lds: Optional[Dict[int, torch.Tensor]], K: int) -> torch.Tensor:
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
            if 0 <= yy < K and jj in per_joint_lds:
                w[i] *= float(per_joint_lds[jj][yy].item())

    y = df_tr["_y"].values.astype(int)
    w *= np.where(y > 0, NONZERO_BOOST, 1.0).astype(np.float32)
    w *= np.power((1.0 + y.clip(min=0).astype(np.float32)), HIGH_GRADE_BOOST_POWER)

    w = w / (w.mean() + 1e-8)
    w = np.clip(w, SAMPLE_W_CLIP[0], SAMPLE_W_CLIP[1])
    return torch.tensor(w, dtype=torch.float32)


# =========================
# 11) Warm-start from JSN binary ckpt (encoder + gate from head_j)
# =========================
def warmstart_from_jsn_binary_ckpt(model: JointSVHGateOrdinalModel, binary_ckpt_path: str):
    sd = torch.load(binary_ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    # ---- encoder: binary keys like "encoder.encoder.*"
    enc_map = {}
    for k, v in sd.items():
        if k.startswith("encoder.encoder."):
            nk = k.replace("encoder.encoder.", "encoder.", 1)  # VisionEncoder expects "encoder.*"
            enc_map[nk] = v

    missing, unexpected = model.encoder.load_state_dict(enc_map, strict=False)
    log("[WarmStart] Encoder load from JSN binary ckpt:")
    log(f"  missing={len(missing)} (first5={missing[:5]})")
    log(f"  unexpected={len(unexpected)} (first5={unexpected[:5]})")

    # ---- gate: JSN binary head is head_j = LN + Linear
    need = ["head_j.0.weight", "head_j.0.bias", "head_j.1.weight", "head_j.1.bias"]
    if not all(k in sd for k in need):
        log("[WarmStart] WARNING: binary ckpt missing head_j.* keys -> skip gate init.")
        return

    ln_w = sd["head_j.0.weight"]
    ln_b = sd["head_j.0.bias"]
    fc_w = sd["head_j.1.weight"]
    fc_b = sd["head_j.1.bias"]

    # shape check
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
    log(f"[WarmStart] Gate init from binary head_j applied to ALL joints. ||W_loaded - W_src||={diff:.6f}")


def set_requires_grad(mod: nn.Module, flag: bool):
    for p in mod.parameters():
        p.requires_grad = flag


# =========================
# 12) Conformal prediction for regression (split conformal intervals)
# =========================
def conformal_q_from_calibration(abs_residuals: np.ndarray, alpha: float) -> float:
    """
    Split conformal for regression:
      q = quantile_{ceil((n+1)*(1-alpha))/n} of |y - pred|
    """
    r = abs_residuals.astype(np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    n = r.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))  # 1..n+1
    k = min(max(k, 1), n)
    # k-th smallest (1-index) => quantile
    q = float(np.partition(r, k - 1)[k - 1])
    return q


def conformal_interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> Dict[str, float]:
    """
    Interval: [pred-q, pred+q]
    Also report avg width = 2q
    """
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
            gate_logit, ord_logits = model(img, joint_id)
            lg = gate_loss_fn(gate_logit, y_gate)
            lo = ord_loss_fn(ord_logits, y, joint_id)
            loss = LAMBDA_GATE * lg + LAMBDA_ORD * lo

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item()) * img.size(0)
        n += img.size(0)

    return total / max(1, n)


# =========================
# 14) Main
# =========================
def main():
    log("🚀 CATCH | TASK=JSN | ONLY PIP/MCP | keep-ratio letterbox | mean pool | gate from JSN-binary ckpt | best by Val(SOFT) PCC | + Conformal")

    df_all = pd.read_csv(CSV_PATH)
    for c in [FILE_COL, PATIENT_COL, REGION_COL, "JSN_score"]:
        assert c in df_all.columns, f"CSV missing {c}"

    df_all = filter_keep_only(df_all)

    df_all["_y"] = df_all["JSN_score"].apply(clean_label).astype(int)
    valid_ratio = float((df_all["_y"] >= 0).mean())
    log(f"After keep-only (PIP/MCP): N={len(df_all)} | valid label ratio for JSN_score = {valid_ratio:.3f}")
    df_all = df_all[df_all["_y"] >= 0].reset_index(drop=True)
    log(f"After dropping invalid labels: N={len(df_all)}")

    # split
    df_tr, df_va, df_te = patient_level_split_three_way(df_all, 0.80, 0.10, 0.10, seed=3407)
    log(f"Split (patient-level) -> Train: {len(df_tr)} | Val: {len(df_va)} | Test: {len(df_te)}")

    # vocab (train only)
    joints_train = sorted(df_tr[REGION_COL].astype(str).map(norm_region).unique().tolist())
    joint2id = {j: i for i, j in enumerate(joints_train)}
    log(f"Joint vocab (train) size={len(joint2id)} -> {joints_train}")

    # annotate ids
    for df_ in [df_tr, df_va, df_te]:
        df_["_region_norm"] = df_[REGION_COL].astype(str).map(norm_region)
        df_["_joint_id"] = df_["_region_norm"].map(lambda x: joint2id.get(x, -1)).astype(int)

    df_va = filter_by_vocab(df_va, joint2id)
    df_te = filter_by_vocab(df_te, joint2id)

    # LDS
    per_joint_lds: Optional[Dict[int, torch.Tensor]] = None
    if USE_LDS:
        per_joint_lds = compute_per_joint_lds(df_tr, num_classes=NUM_CLASSES, sigma=LDS_SIGMA)
        for region in REGION_LIST:
            rn = region.lower()
            if rn in joint2id:
                jid = joint2id[rn]
                sub = df_tr[df_tr["_joint_id"] == jid]
                counts = np.bincount(sub["_y"].values.astype(int), minlength=NUM_CLASSES).astype(int).tolist()
                w = per_joint_lds.get(jid, torch.ones(NUM_CLASSES))
                log(f"--- {region} LDS sigma={LDS_SIGMA} counts={counts} weights={np.round(w.numpy(), 3).tolist()}")

    # datasets
    train_ds = JointDataset(df_tr, IMG_ROOT, train_tf, joint2id)
    val_ds   = JointDataset(df_va, IMG_ROOT, eval_tf,  joint2id)
    test_ds  = JointDataset(df_te, IMG_ROOT, eval_tf,  joint2id)

    # loaders
    if USE_WEIGHTED_SAMPLER:
        sample_w = build_sample_weights(df_tr, per_joint_lds, NUM_CLASSES)
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
        train_ld = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
        log(f"WeightedRandomSampler enabled | mean_w={float(sample_w.mean()):.3f} | max_w={float(sample_w.max()):.3f}")
    else:
        train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # eval loaders (no sampler) for stable metrics
    train_eval_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    val_ld  = DataLoader(val_ds,  batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # model
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(model_name=MODEL_NAME, device=DEVICE, init_mode=init_mode).to(DEVICE)
    model = JointSVHGateOrdinalModel(encoder, joint_vocab=len(joint2id), K=NUM_CLASSES).to(DEVICE)

    # warmstart (encoder + gate from binary head_j)
    if (BINARY_CKPT_PATH is not None) and os.path.exists(BINARY_CKPT_PATH):
        warmstart_from_jsn_binary_ckpt(model, BINARY_CKPT_PATH)
    else:
        log(f"[WarmStart] WARNING: BINARY_CKPT_PATH not found -> {BINARY_CKPT_PATH}")

    # losses
    gate_loss_fn = FocalLossWithLogits(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction="mean")
    ord_loss_fn  = OrdinalBCELoss(K=NUM_CLASSES, per_joint_weights=per_joint_lds)

    # warmup freeze: encoder + gate frozen, only train ordinal
    set_requires_grad(model.encoder, False)
    for h in model.heads:
        set_requires_grad(h.gate, False)
    log(f"[Warmup] Freeze encoder+gate for first {WARMUP_FREEZE_EPOCHS} epochs (train ordinal only).")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()

    best_val_pcc = -1e9
    best_ckpt_path = None

    for ep in range(EPOCHS + 1):
        # unfreeze at boundary + rebuild optimizer
        if ep == WARMUP_FREEZE_EPOCHS:
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

        tr_m = eval_soft(model, train_eval_ld, NUM_CLASSES)
        va_m = eval_soft(model, val_ld,        NUM_CLASSES)
        te_m = eval_soft(model, test_ld,       NUM_CLASSES)  # monitoring only

        curr_val_pcc = va_m.get("pcc", -1e9)

        # Conformal (regression interval): fit on VAL (calib), report on VAL/TES
        if USE_CONFORMAL:
            # val as calibration
            val_pred, val_y, _ = predict_soft(model, val_ld, NUM_CLASSES)
            cal_abs = np.abs(val_y.astype(np.float64) - val_pred.astype(np.float64))
            q = conformal_q_from_calibration(cal_abs, CONFORMAL_ALPHA)

            val_stats = conformal_interval_coverage(val_y, val_pred, q)
            log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, val_stats)

            # test
            te_pred, te_y, _ = predict_soft(model, test_ld, NUM_CLASSES)
            te_stats = conformal_interval_coverage(te_y, te_pred, q)
            log_conformal_reg("TEST", CONFORMAL_ALPHA, q, te_stats)

        log(
            f"Ep {ep:03d} | Loss {tr_loss:.4f} | "
            f"Tr  PCC {tr_m.get('pcc',0):.3f} | SCC {tr_m.get('scc',0):.3f} | QWK {tr_m.get('qwk',0):.3f} | "
            f"RMSE {tr_m.get('rmse',0):.3f} | MAE {tr_m.get('mae',0):.3f} | R2 {tr_m.get('r2',0):.3f} | ACC {tr_m.get('acc',0):.3f} || "
            f"Val PCC {va_m.get('pcc',0):.3f} | SCC {va_m.get('scc',0):.3f} | QWK {va_m.get('qwk',0):.3f} | "
            f"RMSE {va_m.get('rmse',0):.3f} | MAE {va_m.get('mae',0):.3f} | R2 {va_m.get('r2',0):.3f} | ACC {va_m.get('acc',0):.3f} || "
            f"Te  PCC {te_m.get('pcc',0):.3f} | SCC {te_m.get('scc',0):.3f} | QWK {te_m.get('qwk',0):.3f} | "
            f"RMSE {te_m.get('rmse',0):.3f} | MAE {te_m.get('mae',0):.3f} | R2 {te_m.get('r2',0):.3f} | ACC {te_m.get('acc',0):.3f} | "
            f"time {(time.time()-t0):.1f}s"
        )

        # save best by Val PCC only
        if curr_val_pcc > best_val_pcc:
            best_val_pcc = curr_val_pcc
            save_name = f"catch_jsn_pipmcp_warmstart_meanpool_bestValSoftPCC{best_val_pcc:.4f}.pt"
            best_ckpt_path = os.path.join(MODEL_SAVE_DIR, save_name)

            torch.save(
                {
                    "model": model.state_dict(),
                    "joint2id": joint2id,
                    "task": TASK,
                    "K": NUM_CLASSES,
                    "epoch": ep,
                    "best_val_soft_pcc": best_val_pcc,
                    "val_metrics_soft": va_m,
                    "test_metrics_soft": te_m,
                    "config": {
                        "MODEL_NAME": MODEL_NAME,
                        "BINARY_CKPT_PATH": BINARY_CKPT_PATH,
                        "KEEP_ONLY": sorted(list(KEEP_ONLY)),
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
                        "POOLING": "mean",
                        "GATE": "LN+Linear (loaded from binary head_j)",
                        "AUG": "keep-ratio letterbox + light rotation/flip + light photometric",
                        "BEST_SELECT": "Val(SOFT) PCC",
                        "USE_CONFORMAL": USE_CONFORMAL,
                        "CONFORMAL_ALPHA": CONFORMAL_ALPHA,
                        "CONFORMAL_TYPE": "split conformal regression intervals on |y-pred| using VAL as calib",
                    },
                },
                best_ckpt_path,
            )
            log(f"  ✅ Saved Best (Val SOFT PCC) -> {save_name}")
            # optional breakdown (monitor)
            eval_by_region_soft(model, df_va, joint2id, NUM_CLASSES, split_name="VAL")
            eval_by_region_soft(model, df_te, joint2id, NUM_CLASSES, split_name="TEST")

    if best_ckpt_path is None:
        log("No best checkpoint found.")
        return

    # Final evaluate best
    log("======= Final: Evaluate Best ckpt on Val/Test (SOFT) + Conformal =======")
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    log(f"Loaded best ckpt epoch={ckpt.get('epoch', -1)} | best Val(SOFT) PCC={ckpt.get('best_val_soft_pcc', -1):.4f}")

    eval_by_region_soft(model, df_va, joint2id, NUM_CLASSES, split_name="VAL")
    eval_by_region_soft(model, df_te, joint2id, NUM_CLASSES, split_name="TEST")

    if USE_CONFORMAL:
        val_pred, val_y, _ = predict_soft(model, val_ld, NUM_CLASSES)
        q = conformal_q_from_calibration(np.abs(val_y - val_pred), CONFORMAL_ALPHA)
        log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, conformal_interval_coverage(val_y, val_pred, q))

        te_pred, te_y, _ = predict_soft(model, test_ld, NUM_CLASSES)
        log_conformal_reg("TEST", CONFORMAL_ALPHA, q, conformal_interval_coverage(te_y, te_pred, q))


if __name__ == "__main__":
    main()
