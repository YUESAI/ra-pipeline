#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CATCH Bilateral (two-hand) JSN Regression with Warm-start from Bilateral JSN Binary CKPT

Task (regression):
- One bilateral whole image must have exactly 18 joints with valid JSN scores
- JSN_sum = sum over 18 joints (0~72)
- Predict JSN_sum (soft regression)

Warm-start:
- Load encoder + head_j (gate) from the best bilateral JSN binary checkpoint

Model:
- Encoder: DINOv3 ViT-B/16
- Gate head (compatible with binary): mean pooling -> LN -> Linear(1)  (head_j)
- Reg head: mean pooling -> LN -> MLP -> Linear(1)
- Prediction (bounded, non-negative):
    gate_prob = sigmoid(gate_logit)
    reg_val   = MAX_SUM_JSN * sigmoid(reg_raw)   in [0,72]
    pred_soft = gate_prob * reg_val              in [0,72]

Loss:
- L = λ_gate * BCEWithLogits(gate_logit, y>0) + λ_reg * SmoothL1(pred_soft, y)

Selection:
- Save best ONLY by Val(SOFT) PCC (no test leakage)

Extras:
- Stable Train(SOFT) metrics loader: no shuffle, no drop_last
- Split conformal regression interval (calibration=VAL): report VAL(calib-self) and TEST coverage

Run:
  CUDA_VISIBLE_DEVICES=2 nohup python catch_svh_bilateral_jsn_gate_regression_conformal.py \
    > train_log/catch_svh_bilateral_jsn_gate_regression_conformal.log 2>&1 &
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catch_split_utils import shared_patient_level_split_3way
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoModel, AutoImageProcessor, AutoConfig
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, cohen_kappa_score
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

EXPECTED_JOINTS_JSN = 18
MAX_SUM_JSN = 72

# -------- Paths --------
CSV_PATH = "/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_joint_score_raw.csv"
WHOLE_IMG_ROOT = Path("/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_data/all_RA_update")

# ✅ best bilateral JSN binary ckpt (encoder + head_j)
BINARY_CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema_bilateral/catch_bilateral_jsn_binary_dinov3_amp.pt"

# Optional fallback FM ema_state (only used if binary ckpt missing)
FM_CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema_bilateral_jsn_regression"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RANDOM_INIT = False

# -------- Train hparams --------
IMG_SIZE = 224
BATCH = 64
EPOCHS = 120
LR = 1e-5
WEIGHT_DECAY = 1e-4

# two-stage
WARMUP_FREEZE_EPOCHS = 5
ENCODER_LR_MULT = 0.3

# loss weights
LAMBDA_GATE = 1.0
LAMBDA_REG = 1.0

# gate imbalance (consistent with the corresponding binary model)
POS_WEIGHT_GATE = 5.0

# split
SPLIT_SEED = 3407
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# conformal regression
USE_CONFORMAL = True
CONFORMAL_ALPHA = 0.10


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


# =========================
# 2) Image mean/std + transforms
# =========================
_processor_for_stats = AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME)
IMAGE_MEAN = _processor_for_stats.image_mean
IMAGE_STD = _processor_for_stats.image_std

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


# =========================
# 3) Build bilateral df (jsn)
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


def build_bilateral_df_jsn() -> pd.DataFrame:
    df_all = pd.read_csv(CSV_PATH)
    required_cols = ["patient_id", "timepoint", "side", "joint_type",
                     "index", "JSN_score", "file_name"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df_all["JSN_clean"] = df_all["JSN_score"].apply(clean_label)
    df_all["whole_image"] = df_all["file_name"].apply(extract_whole_image_name)

    df_valid = df_all[(df_all["JSN_clean"] >= 0) & (df_all["JSN_clean"] <= 4)].copy()
    log(f"[jsn] joint samples with valid JSN scores: {len(df_valid)}")
    log(f"Unique whole_image candidates: {df_valid['whole_image'].nunique()}")

    group_cols = ["patient_id", "timepoint", "whole_image"]
    group_list = []

    for (pid, tp, wimg), g in df_valid.groupby(group_cols):
        img_path = WHOLE_IMG_ROOT / wimg
        if not img_path.is_file():
            continue
        if len(g) != EXPECTED_JOINTS_JSN:
            continue

        jsn_sum = int(g["JSN_clean"].sum())  # 0..72
        group_list.append({
            "patient_id": pid,
            "timepoint": tp,
            "whole_image": wimg,
            "JSN_sum": jsn_sum,
            "num_joints": len(g),
        })

    df_bi = pd.DataFrame(group_list)
    log(f"Built bilateral JSN DF: {len(df_bi)} images with exactly {EXPECTED_JOINTS_JSN} valid joints")
    return df_bi


class BilateralJSNDataset(Dataset):
    """Returns: image, y_sum(float)"""
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
        y = float(row.JSN_sum)
        return img, torch.tensor(y, dtype=torch.float32)


# =========================
# 4) Model (compatible with bilateral binary head_j)
# =========================
class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = DINOV3_MODEL_NAME, device=None, init_mode: str = "pretrained"):
        super().__init__()
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

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        out = self.encoder(pixel_values=x, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D]
        if self.num_register_tokens > 0:
            cls_tok = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_register_tokens:, :]
            tokens = torch.cat([cls_tok, patches], dim=1)
        return tokens


def load_student_from_fm_ckpt(encoder: VisionEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(ckpt["ema_state"], strict=False)
    log(f"Loaded ema_state from {ckpt_path}")
    if missing:
        log(f"  Missing keys: {len(missing)} (first 5) {missing[:5]}")
    if unexpected:
        log(f"  Unexpected keys: {len(unexpected)} (first 5) {unexpected[:5]}")


class BilateralJSNGateRegModel(nn.Module):
    """
    Keep head_j identical name/shape to bilateral binary (LN+Linear),
    so we can load binary ckpt (encoder + head_j) with strict=False.
    """
    def __init__(self, encoder: VisionEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size

        # ✅ same as bilateral binary head_j
        self.head_j = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

        # regression head (new)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(x)
        pooled = tokens.mean(dim=1)
        gate_logit = self.head_j(pooled).squeeze(1)
        reg_raw = self.reg_head(pooled).squeeze(1)
        return gate_logit, reg_raw


def warmstart_from_bilateral_jsn_binary_ckpt(model: BilateralJSNGateRegModel, binary_ckpt_path: str):
    sd = torch.load(binary_ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError("Binary ckpt is not a state_dict dict.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"[WarmStart] Loaded from bilateral JSN binary ckpt: {binary_ckpt_path}")
    log(f"  missing={len(missing)} (first5={missing[:5]})")
    log(f"  unexpected={len(unexpected)} (first5={unexpected[:5]})")

    if "head_j.1.weight" in sd:
        diff = float(torch.norm(model.head_j[1].weight.detach().cpu().float() - sd["head_j.1.weight"].float()).item())
        log(f"  check ||head_j.W_loaded - head_j.W_src|| = {diff:.6f}")


def set_requires_grad(mod: nn.Module, flag: bool):
    for p in mod.parameters():
        p.requires_grad = flag


# =========================
# 5) Losses + metrics
# =========================
class WeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_sum: torch.Tensor) -> torch.Tensor:
        y_bin = (y_sum > 0).float()
        loss = F.binary_cross_entropy_with_logits(logits, y_bin, reduction="none")
        if self.pos_weight != 1.0:
            w = torch.ones_like(loss)
            w[y_bin > 0.5] = self.pos_weight
            loss = loss * w
        return loss.mean() if self.reduction == "mean" else loss.sum()


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
def compute_metrics_reg(y_pred: np.ndarray, y_true: np.ndarray, max_label: int) -> Dict[str, float]:
    y = y_true.astype(np.float64)
    p = y_pred.astype(np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return {}

    y = y[m]
    p = p[m]

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p)) if len(y) > 1 else 0.0
    pcc = _safe_pearson(y, p)
    scc = _safe_spearman(y, p)

    y_true_cls = y.astype(int)
    y_pred_cls = np.clip(np.rint(p).astype(int), 0, int(max_label))
    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    qwk = _qwk(y_true_cls, y_pred_cls)

    nz = y_true_cls > 0
    if nz.sum() >= 2:
        pcc_nz = _safe_pearson(y[nz], p[nz])
        scc_nz = _safe_spearman(y[nz], p[nz])
    else:
        pcc_nz = 0.0
        scc_nz = 0.0

    return {
        "pcc": pcc, "scc": scc, "qwk": qwk,
        "rmse": rmse, "mae": mae, "r2": r2, "acc": acc,
        "pcc_nz": pcc_nz, "scc_nz": scc_nz, "nz_n": int(nz.sum()),
    }


def log_metrics(title: str, m: Dict[str, float]):
    if not m:
        log(f"{title} -> no valid")
        return
    log(
        f"{title} -> "
        f"PCC {m['pcc']:.3f} | SCC {m['scc']:.3f} | QWK {m['qwk']:.3f} | "
        f"RMSE {m['rmse']:.3f} | MAE {m['mae']:.3f} | R2 {m['r2']:.3f} | ACC {m['acc']:.3f} | "
        f"PCC(nz) {m['pcc_nz']:.3f} | SCC(nz) {m['scc_nz']:.3f} | nz_n {m['nz_n']}"
    )


# =========================
# 6) Predict / Eval (SOFT)
# =========================
@torch.no_grad()
def predict_soft(model: BilateralJSNGateRegModel, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pred_soft in [0,72], y_true, gate_prob
    """
    model.eval()
    preds, ys, gates = [], [], []

    for img, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast(enabled=(DEVICE == "cuda")):
            gate_logit, reg_raw = model(img)
            gate_prob = torch.sigmoid(gate_logit)
            reg_val = float(MAX_SUM_JSN) * torch.sigmoid(reg_raw)  # [0,72]
            pred = gate_prob * reg_val

        preds.append(pred.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        gates.append(gate_prob.detach().cpu().numpy())

    return np.concatenate(preds), np.concatenate(ys), np.concatenate(gates)


@torch.no_grad()
def eval_split(model: BilateralJSNGateRegModel, loader: DataLoader) -> Dict[str, float]:
    pred, y, _ = predict_soft(model, loader)
    return compute_metrics_reg(pred, y, max_label=MAX_SUM_JSN)


# =========================
# 7) Conformal regression interval
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
# 8) Train
# =========================
def train_epoch(model: BilateralJSNGateRegModel,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                gate_loss_fn: nn.Module,
                reg_loss_fn: nn.Module,
                scaler: GradScaler) -> float:
    model.train()
    total, n = 0.0, 0

    for img, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(DEVICE == "cuda")):
            gate_logit, reg_raw = model(img)
            gate_prob = torch.sigmoid(gate_logit)
            reg_val = float(MAX_SUM_JSN) * torch.sigmoid(reg_raw)
            pred = gate_prob * reg_val

            lg = gate_loss_fn(gate_logit, y)
            lr = reg_loss_fn(pred, y)
            loss = LAMBDA_GATE * lg + LAMBDA_REG * lr

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = img.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


# =========================
# 9) Main
# =========================
def main():
    log("🚀 CATCH | BILATERAL | TASK=JSN regression | warmstart bilateral JSN binary ckpt | pred=sigmoid(gate)*72*sigmoid(reg) | best by Val(SOFT) PCC | + Conformal")

    df_bi = build_bilateral_df_jsn()
    if df_bi.empty:
        raise RuntimeError("No valid bilateral JSN images found.")

    df_tr, df_va, df_te = patient_level_split_3way(df_bi, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
    log(
        f"Patient split: train={df_tr['patient_id'].nunique()} pts, "
        f"val={df_va['patient_id'].nunique()} pts, "
        f"test={df_te['patient_id'].nunique()} pts"
    )
    log(f"Samples: train={len(df_tr)} val={len(df_va)} test={len(df_te)}")
    log(f"Nonzero rate: train={float((df_tr['JSN_sum']>0).mean()):.3f} val={float((df_va['JSN_sum']>0).mean()):.3f} test={float((df_te['JSN_sum']>0).mean()):.3f}")

    train_ds = BilateralJSNDataset(df_tr, WHOLE_IMG_ROOT, train_tf)
    val_ds = BilateralJSNDataset(df_va, WHOLE_IMG_ROOT, eval_tf)
    test_ds = BilateralJSNDataset(df_te, WHOLE_IMG_ROOT, eval_tf)

    NUM_WORKERS = 8
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=True, persistent_workers=True, drop_last=True, prefetch_factor=4)

    # ✅ stable train metrics loader
    train_eval_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True, persistent_workers=True, drop_last=False, prefetch_factor=4)

    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=True, drop_last=False, prefetch_factor=4)

    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True, persistent_workers=True, drop_last=False, prefetch_factor=4)

    log(f"Bilateral images: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # model
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(device=DEVICE, init_mode=init_mode).to(DEVICE)
    model = BilateralJSNGateRegModel(encoder).to(DEVICE)

    # warmstart
    if BINARY_CKPT_PATH and os.path.exists(BINARY_CKPT_PATH):
        warmstart_from_bilateral_jsn_binary_ckpt(model, BINARY_CKPT_PATH)
    else:
        log(f"[WarmStart] WARNING: binary ckpt not found: {BINARY_CKPT_PATH}")
        if (not RANDOM_INIT) and FM_CKPT_PATH and os.path.exists(FM_CKPT_PATH):
            load_student_from_fm_ckpt(model.encoder, FM_CKPT_PATH)

    gate_loss_fn = WeightedBCEWithLogits(pos_weight=POS_WEIGHT_GATE, reduction="mean")
    reg_loss_fn = nn.SmoothL1Loss(reduction="mean")
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # warmup freeze: encoder + gate frozen, only train reg_head
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.head_j, False)
    set_requires_grad(model.reg_head, True)
    log(f"[Warmup] Freeze encoder+gate for first {WARMUP_FREEZE_EPOCHS} epochs (train reg_head only).")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_pcc = -1e9
    best_ckpt_path = None

    for ep in range(1, EPOCHS + 1):
        if ep == WARMUP_FREEZE_EPOCHS + 1:
            set_requires_grad(model.encoder, True)
            set_requires_grad(model.head_j, True)
            set_requires_grad(model.reg_head, True)

            optimizer = optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": LR * ENCODER_LR_MULT},
                    {"params": model.head_j.parameters(), "lr": LR},
                    {"params": model.reg_head.parameters(), "lr": LR},
                ],
                weight_decay=WEIGHT_DECAY
            )
            log(f"[Warmup] Unfroze encoder+gate. Rebuilt optimizer (encoder lr={ENCODER_LR_MULT}x).")

        t0 = time.time()
        tr_loss = train_epoch(model, train_ld, optimizer, gate_loss_fn, reg_loss_fn, scaler)

        tr_m = eval_split(model, train_eval_ld)
        va_m = eval_split(model, val_ld)

        log(f"Ep {ep:03d}/{EPOCHS:03d} | Loss {tr_loss:.4f} | time {time.time()-t0:.1f}s")
        log_metrics("  [SOFT] TRAIN", tr_m)
        log_metrics("  [SOFT] VAL  ", va_m)

        if USE_CONFORMAL:
            val_pred, val_y, _ = predict_soft(model, val_ld)
            q = conformal_q_from_calibration(np.abs(val_y.astype(np.float64) - val_pred.astype(np.float64)), CONFORMAL_ALPHA)
            log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, conformal_interval_coverage(val_y, val_pred, q))


        curr_val_pcc = va_m.get("pcc", -1e9)
        if np.isfinite(curr_val_pcc) and curr_val_pcc > best_val_pcc:
            best_val_pcc = curr_val_pcc
            save_name = f"catch_bilateral_jsn_warmstart_reg_bestValSoftPCC{best_val_pcc:.4f}_ep{ep}.pt"
            best_ckpt_path = os.path.join(MODEL_SAVE_DIR, save_name)

            torch.save(
                {
                    "model": model.state_dict(),
                    "task": "bilateral_jsn_regression",
                    "epoch": ep,
                    "best_val_soft_pcc": best_val_pcc,
                    "val_metrics_soft": va_m,
                    "config": {
                        "BINARY_CKPT_PATH": BINARY_CKPT_PATH,
                        "DINOV3_MODEL_NAME": DINOV3_MODEL_NAME,
                        "PRED": "sigmoid(gate) * (72 * sigmoid(reg_raw))",
                        "LOSS": "BCE(gate, y>0) + SmoothL1(pred, y)",
                        "WARMUP_FREEZE_EPOCHS": WARMUP_FREEZE_EPOCHS,
                        "ENCODER_LR_MULT": ENCODER_LR_MULT,
                        "CONFORMAL_ALPHA": CONFORMAL_ALPHA,
                    },
                },
                best_ckpt_path,
            )
            log(f"✅ Saved BEST (Val SOFT PCC={best_val_pcc:.4f}) -> {best_ckpt_path}")

    if best_ckpt_path is None:
        log("No best checkpoint saved.")
        return

    log("======= Final: Load best ckpt and report SOFT + Conformal =======")
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    log(f"Loaded best: epoch={ckpt.get('epoch', -1)} | best Val(SOFT) PCC={ckpt.get('best_val_soft_pcc', -1):.4f}")

    log_metrics("  [SOFT] VAL ", eval_split(model, val_ld))
    log_metrics("  [SOFT] TEST", eval_split(model, test_ld))

    if USE_CONFORMAL:
        val_pred, val_y, _ = predict_soft(model, val_ld)
        q = conformal_q_from_calibration(np.abs(val_y.astype(np.float64) - val_pred.astype(np.float64)), CONFORMAL_ALPHA)
        log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, conformal_interval_coverage(val_y, val_pred, q))

        te_pred, te_y, _ = predict_soft(model, test_ld)
        log_conformal_reg("TEST", CONFORMAL_ALPHA, q, conformal_interval_coverage(te_y, te_pred, q))


if __name__ == "__main__":
    main()
