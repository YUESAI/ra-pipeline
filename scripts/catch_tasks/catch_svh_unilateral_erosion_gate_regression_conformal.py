
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CATCH Hand-level SvH Erosion Regression with Warm-start from Hand-level Binary CKPT

- Build hand-level unilateral images (exactly 13 valid erosion joints)
- Target: Erosion_sum in [0,65]
- Model:
    encoder: DINOv3 ViT-B/16
    gate head (erosion): mean pool -> LN -> Linear(1)  (same as binary head_e, can load ckpt)
    reg head:            mean pool -> LN -> MLP -> Linear(1) -> softplus (non-negative)
  pred_soft = sigmoid(gate_logit) * softplus(reg_raw)

- Loss:
    L = lambda_gate * BCE(gate_logit, y>0) + lambda_reg * SmoothL1(pred_soft, y)

- Best selection: Val(SOFT) PCC only (no test leakage)
- Stable train metrics loader: no shuffle, no drop_last
- Conformal prediction: split conformal regression intervals (calibration = VAL residuals)
- Optional HARD gate threshold tuning on VAL for reporting only

Run:
  CUDA_VISIBLE_DEVICES=1 nohup python catch_svh_hand_erosion_gate_regression_conformal.py \
    > train_log/catch_svh_hand_erosion_gate_regression_conformal.log 2>&1 &
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

TASK = "erosion"  # this script focuses on erosion hand-level regression

# Hand-level definition for erosion
EXPECTED_JOINTS_EROSION = 13
MAX_SUM_EROSION = 65

# -------- Paths --------
CSV_PATH = "/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_joint_score_raw.csv"
WHOLE_IMG_ROOT = Path("/home/UWO/ylong66/data/RA/RA/SvHScorePrediction/RA_data/all_RA_update")

# FM pretrain (ema_state) - optional if you want to init from ema_state when no binary ckpt
FM_CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# ✅ Best binary ckpt you gave
BINARY_CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema_handlevel/catch_hand_erosion_binary_dinov3_amp.pt"

MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema_handlevel_regression"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# -------- HF model --------
DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RANDOM_INIT = False

# -------- Train hparams --------
IMG_SIZE = 224
BATCH = 64
EPOCHS = 80
LR = 1e-5
WEIGHT_DECAY = 1e-4

# warmup freeze
WARMUP_FREEZE_EPOCHS = 5   # freeze encoder+gate, train reg_head only for first N epochs
ENCODER_LR_MULT = 0.3      # when unfreezing, encoder lr = LR*0.3

# loss weights
LAMBDA_GATE = 1.0
LAMBDA_REG = 1.0

# gate imbalance weight (like your binary)
POS_WEIGHT_GATE = 5.0

# split
SPLIT_SEED = 3407
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# conformal regression
USE_CONFORMAL = True
CONFORMAL_ALPHA = 0.10

# optional hard gate reporting
USE_HARD_GATE_AT_REPORT = True
THRESH_GRID = np.linspace(0.05, 0.95, 19)
THRESH_CRITERION = "pcc"  # "pcc" or "rmse"


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


# =========================
# 2) Mean/std + transforms
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
# 3) Build hand-level df (same logic as your binary)
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


def extract_hand_image_name(file_name: str) -> str:
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
    hand_base = "_".join(parts[:-3])
    return hand_base + ".tif"


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


def build_hand_level_df_erosion() -> pd.DataFrame:
    """
    Keep unilateral hand images for erosion:
      - whole image has exactly 13 joints with valid erosion scores
      - and the side/hand group also has exactly 13 joints
    """
    df_all = pd.read_csv(CSV_PATH)
    required_cols = ["patient_id", "timepoint", "side", "joint_type",
                     "index", "Erosion_score", "file_name"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df_all["Erosion_clean"] = df_all["Erosion_score"].apply(clean_label)
    df_all["hand_image"] = df_all["file_name"].apply(extract_hand_image_name)

    # valid erosion joint labels
    df_valid = df_all[(df_all["Erosion_clean"] >= 0) & (df_all["Erosion_clean"] <= 5)].copy()
    log(f"[erosion] joint samples with valid erosion scores: {len(df_valid)}")
    log(f"Unique hand_image candidates from CSV (erosion): {df_valid['hand_image'].nunique()}")

    # whole-image level: exactly 13 valid joints
    img_group_cols = ["patient_id", "timepoint", "hand_image"]
    img_counts = df_valid.groupby(img_group_cols).size().rename("num_joints_image").reset_index()
    single_img = img_counts[img_counts["num_joints_image"] == EXPECTED_JOINTS_EROSION]
    log(f"[erosion] whole images with exactly {EXPECTED_JOINTS_EROSION} valid joints: {len(single_img)}")

    single_img_keys = set((r["patient_id"], r["timepoint"], r["hand_image"]) for _, r in single_img.iterrows())

    # side-level grouping (unilateral consistency)
    hand_group_cols = ["patient_id", "timepoint", "side", "hand_image"]
    group_list = []
    for (pid, tp, sd, hand_img), g in df_valid.groupby(hand_group_cols):
        if (pid, tp, hand_img) not in single_img_keys:
            continue
        img_path = WHOLE_IMG_ROOT / hand_img
        if not img_path.is_file():
            continue
        if len(g) != EXPECTED_JOINTS_EROSION:
            continue

        eros_sum = int(g["Erosion_clean"].sum())  # 0..65
        group_list.append({
            "patient_id": pid,
            "timepoint": tp,
            "side": sd,
            "hand_image": hand_img,
            "Erosion_sum": eros_sum,
            "num_joints": len(g),
        })

    df_hand = pd.DataFrame(group_list)
    log(f"Built hand-level erosion DataFrame: {len(df_hand)} hands (exactly 13 joints, unilateral).")
    return df_hand


class HandImageDataset(Dataset):
    """Returns: image, y_sum(float)"""
    def __init__(self, df: pd.DataFrame, root: Path, tf):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row.hand_image
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        y = float(row.Erosion_sum)
        return img, torch.tensor(y, dtype=torch.float32)


# =========================
# 4) Model (compatible with binary ckpt head_e)
# =========================
class VisionEncoder(nn.Module):
    """
    Same wrapper style as your binary scripts:
      VisionEncoder.encoder = AutoModel(...)
    """
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
            tokens = torch.cat([cls_tok, patches], dim=1)  # [B, 1+P, D]
        return tokens


def load_student_from_fm_ckpt(encoder: VisionEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(ckpt["ema_state"], strict=False)
    log(f"Loaded ema_state from {ckpt_path}")
    if missing:
        log(f"  Missing keys: {len(missing)} (first 5) {missing[:5]}")
    if unexpected:
        log(f"  Unexpected keys: {len(unexpected)} (first 5) {unexpected[:5]}")


class HandGateRegModel(nn.Module):
    """
    Keep head_e name identical to binary ckpt (LN+Linear),
    so we can directly load warmstart from binary state_dict with strict=False.
    """
    def __init__(self, encoder: VisionEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size

        # ✅ same as binary head_e
        self.head_e = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

        # regression head (new)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(x)      # [B, 1+P, D]
        pooled = tokens.mean(dim=1)   # [B, D]
        gate_logit = self.head_e(pooled).squeeze(1)         # [B]
        reg_raw = self.reg_head(pooled).squeeze(1)          # [B]
        return gate_logit, reg_raw


def warmstart_from_binary_ckpt(model: HandGateRegModel, binary_ckpt_path: str):
    sd = torch.load(binary_ckpt_path, map_location="cpu")
    # your binary saves: torch.save(model.state_dict(), path)
    if not isinstance(sd, dict):
        raise ValueError("Binary ckpt is not a state_dict dict.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"[WarmStart] Loaded from binary ckpt: {binary_ckpt_path}")
    log(f"  missing={len(missing)} (first5={missing[:5]})")
    log(f"  unexpected={len(unexpected)} (first5={unexpected[:5]})")
    # quick sanity: ensure head_e loaded (weights not default)
    if "head_e.1.weight" in sd:
        diff = float(torch.norm(model.head_e[1].weight.detach().cpu().float() - sd["head_e.1.weight"].float()).item())
        log(f"  check ||W_loaded - W_src|| = {diff:.6f}")


def set_requires_grad(mod: nn.Module, flag: bool):
    for p in mod.parameters():
        p.requires_grad = flag


# =========================
# 5) Losses + metrics
# =========================
class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target_sum: torch.Tensor):
        # target_sum >=0 always here
        y_bin = (target_sum > 0).float()
        base = F.binary_cross_entropy_with_logits(pred_logits, y_bin, reduction="none")
        if self.pos_weight is not None and self.pos_weight != 1.0:
            w = torch.ones_like(base)
            w[y_bin > 0.5] = float(self.pos_weight)
            base = base * w
        if self.reduction == "mean":
            return base.mean()
        if self.reduction == "sum":
            return base.sum()
        return base


@torch.no_grad()
def compute_metrics_reg(y_pred: np.ndarray, y_true: np.ndarray, max_label: int) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return {}

    y = y_true[m]
    p = y_pred[m]

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p)) if len(y) > 1 else 0.0
    pcc = float(pearsonr(y, p)[0]) if len(y) > 1 else 0.0
    scc = float(spearmanr(y, p)[0]) if len(y) > 1 else 0.0

    # "ACC/QWK" here follows your joint script style: round->clip->exact match
    y_true_cls = y.astype(int)
    y_pred_cls = np.rint(p).astype(int)
    y_pred_cls = np.clip(y_pred_cls, 0, int(max_label))

    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    try:
        qwk = float(cohen_kappa_score(y_true_cls, y_pred_cls, weights="quadratic"))
        if np.isnan(qwk) or np.isinf(qwk):
            qwk = 0.0
    except Exception:
        qwk = 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "pcc": pcc, "scc": scc, "acc": acc, "qwk": qwk}


def log_metrics(title: str, m: Dict[str, float]):
    if not m:
        log(f"{title} -> no valid")
        return
    log(
        f"{title} -> "
        f"SCC {m['scc']:.3f} | PCC {m['pcc']:.3f} | QWK {m['qwk']:.3f} | "
        f"RMSE {m['rmse']:.3f} | MAE {m['mae']:.3f} | R2 {m['r2']:.3f} | ACC {m['acc']:.3f}"
    )


# =========================
# 6) Predict / Eval (SOFT & HARD)
# =========================
@torch.no_grad()
def predict_soft(model: HandGateRegModel, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pred_soft [N]
      y_true    [N]
      gate_prob [N]  (for optional thresholding)
    """
    model.eval()
    preds, ys, gates = [], [], []

    for img, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast(enabled=(DEVICE == "cuda")):
            gate_logit, reg_raw = model(img)
            gate_prob = torch.sigmoid(gate_logit)
            reg_val = F.softplus(reg_raw)  # >=0
            pred = gate_prob * reg_val

        preds.append(pred.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        gates.append(gate_prob.detach().cpu().numpy())

    return np.concatenate(preds), np.concatenate(ys), np.concatenate(gates)


@torch.no_grad()
def eval_split(model: HandGateRegModel, loader: DataLoader, max_label: int, hard_t: Optional[float] = None) -> Dict[str, float]:
    pred, y, gate = predict_soft(model, loader)
    if hard_t is not None:
        # hard gate: keep reg_val when gate_prob > t, else 0
        # We recompute reg_val cheaply by using pred/(gate_prob+eps) as approximation is not safe.
        # So do a proper pass:
        model.eval()
        preds_h, ys_h = [], []
        for img, yy in loader:
            img = img.to(DEVICE, non_blocking=True)
            yy = yy.to(DEVICE, non_blocking=True)
            with autocast(enabled=(DEVICE == "cuda")):
                gate_logit, reg_raw = model(img)
                gate_prob = torch.sigmoid(gate_logit)
                reg_val = F.softplus(reg_raw)
                pred_h = reg_val * (gate_prob > float(hard_t)).float()
            preds_h.append(pred_h.detach().cpu().numpy())
            ys_h.append(yy.detach().cpu().numpy())
        pred = np.concatenate(preds_h)
        y = np.concatenate(ys_h)

    return compute_metrics_reg(pred, y, max_label=max_label)


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
def train_epoch(model: HandGateRegModel,
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
            reg_val = F.softplus(reg_raw)
            pred_soft = gate_prob * reg_val

            lg = gate_loss_fn(gate_logit, y)
            lr = reg_loss_fn(pred_soft, y)
            loss = LAMBDA_GATE * lg + LAMBDA_REG * lr

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = img.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


@torch.no_grad()
def tune_threshold_on_val(model: HandGateRegModel, val_loader: DataLoader, max_label: int) -> float:
    best_t = 0.5
    best = -1e9 if THRESH_CRITERION == "pcc" else 1e9

    for t in THRESH_GRID:
        m = eval_split(model, val_loader, max_label=max_label, hard_t=float(t))
        if not m:
            continue
        if THRESH_CRITERION == "pcc":
            score = m["pcc"]
            if score > best:
                best = score
                best_t = float(t)
        else:
            score = m["rmse"]
            if score < best:
                best = score
                best_t = float(t)
    return best_t


# =========================
# 9) Main
# =========================
def main():
    log("🚀 CATCH | HAND-LEVEL | TASK=EROSION | warmstart binary ckpt -> regression (gate * softplus) | best by Val(SOFT) PCC | + Conformal")

    # build df
    df_hand = build_hand_level_df_erosion()
    if df_hand.empty:
        raise RuntimeError("No valid hand-level erosion images found.")

    # split
    df_tr, df_va, df_te = patient_level_split_3way(df_hand, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
    log(
        f"Patient split: train={df_tr['patient_id'].nunique()} pts, "
        f"val={df_va['patient_id'].nunique()} pts, "
        f"test={df_te['patient_id'].nunique()} pts"
    )
    log(f"Samples: train={len(df_tr)} val={len(df_va)} test={len(df_te)}")
    log(f"Nonzero rate: train={float((df_tr['Erosion_sum']>0).mean()):.3f} val={float((df_va['Erosion_sum']>0).mean()):.3f} test={float((df_te['Erosion_sum']>0).mean()):.3f}")

    # datasets & loaders
    train_ds = HandImageDataset(df_tr, WHOLE_IMG_ROOT, train_tf)
    val_ds = HandImageDataset(df_va, WHOLE_IMG_ROOT, eval_tf)
    test_ds = HandImageDataset(df_te, WHOLE_IMG_ROOT, eval_tf)

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
    log(f"Hand images: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # model
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(device=DEVICE, init_mode=init_mode).to(DEVICE)
    model = HandGateRegModel(encoder).to(DEVICE)

    # warmstart priority: binary ckpt (encoder+head_e)
    if BINARY_CKPT_PATH and os.path.exists(BINARY_CKPT_PATH):
        warmstart_from_binary_ckpt(model, BINARY_CKPT_PATH)
    else:
        log(f"[WarmStart] WARNING: binary ckpt not found: {BINARY_CKPT_PATH}")
        # optional: fallback to FM ema_state
        if (not RANDOM_INIT) and FM_CKPT_PATH and os.path.exists(FM_CKPT_PATH):
            load_student_from_fm_ckpt(model.encoder, FM_CKPT_PATH)

    # losses
    gate_loss_fn = MaskedBCELoss(pos_weight=POS_WEIGHT_GATE, reduction="mean")
    reg_loss_fn = nn.SmoothL1Loss(reduction="mean")  # stable vs MSE for skewed sums

    # warmup freeze: encoder+gate frozen, train reg_head only
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.head_e, False)
    set_requires_grad(model.reg_head, True)
    log(f"[Warmup] Freeze encoder+gate for first {WARMUP_FREEZE_EPOCHS} epochs (train reg_head only).")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_val_pcc = -1e9
    best_ckpt_path = None

    for ep in range(1, EPOCHS + 1):
        # unfreeze at boundary, rebuild optimizer with param groups
        if ep == WARMUP_FREEZE_EPOCHS + 1:
            set_requires_grad(model.encoder, True)
            set_requires_grad(model.head_e, True)
            set_requires_grad(model.reg_head, True)
            optimizer = optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": LR * ENCODER_LR_MULT},
                    {"params": model.head_e.parameters(),  "lr": LR},
                    {"params": model.reg_head.parameters(), "lr": LR},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            log(f"[Warmup] Unfroze encoder+gate. Rebuilt optimizer (encoder lr={ENCODER_LR_MULT}x).")

        t0 = time.time()
        tr_loss = train_epoch(model, train_ld, optimizer, gate_loss_fn, reg_loss_fn, scaler)

        # metrics (SOFT)
        tr_m = eval_split(model, train_eval_ld, max_label=MAX_SUM_EROSION, hard_t=None)
        va_m = eval_split(model, val_ld,        max_label=MAX_SUM_EROSION, hard_t=None)

        log(f"Ep {ep:03d}/{EPOCHS} | Loss {tr_loss:.4f} | time {time.time()-t0:.1f}s")
        log_metrics("  [SOFT] TRAIN", tr_m)
        log_metrics("  [SOFT] VAL  ", va_m)

        # conformal (VAL as calibration)
        if USE_CONFORMAL:
            val_pred, val_y, _ = predict_soft(model, val_ld)
            q = conformal_q_from_calibration(np.abs(val_y.astype(np.float64) - val_pred.astype(np.float64)), CONFORMAL_ALPHA)
            log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, conformal_interval_coverage(val_y, val_pred, q))


        # save best by Val PCC
        curr_val_pcc = va_m.get("pcc", -1e9)
        if curr_val_pcc > best_val_pcc:
            best_val_pcc = curr_val_pcc
            save_name = f"catch_hand_erosion_warmstart_reg_softgate_bestValPCC{best_val_pcc:.4f}_ep{ep}.pt"
            best_ckpt_path = os.path.join(MODEL_SAVE_DIR, save_name)
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": "erosion_hand_regression",
                    "epoch": ep,
                    "best_val_soft_pcc": best_val_pcc,
                    "config": {
                        "BINARY_CKPT_PATH": BINARY_CKPT_PATH,
                        "DINOV3_MODEL_NAME": DINOV3_MODEL_NAME,
                        "WARMUP_FREEZE_EPOCHS": WARMUP_FREEZE_EPOCHS,
                        "LR": LR,
                        "ENCODER_LR_MULT": ENCODER_LR_MULT,
                        "LOSS": "BCE(gate) + SmoothL1(pred_soft, y)",
                        "PRED": "sigmoid(gate) * softplus(reg_raw)",
                        "CONFORMAL_ALPHA": CONFORMAL_ALPHA,
                    },
                    "val_metrics_soft": va_m,
                },
                best_ckpt_path,
            )
            log(f"✅ Saved BEST (Val SOFT PCC={best_val_pcc:.4f}) -> {best_ckpt_path}")

    if best_ckpt_path is None:
        log("No best checkpoint saved.")
        return

    # ===== Final evaluation best ckpt =====
    log("======= Final: Load best ckpt and report SOFT + (optional) HARD + Conformal =======")
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)

    log(f"Loaded best: epoch={ckpt.get('epoch', -1)} | best Val(SOFT) PCC={ckpt.get('best_val_soft_pcc', -1):.4f}")

    # SOFT
    log_metrics("  [SOFT] VAL ", eval_split(model, val_ld,  max_label=MAX_SUM_EROSION, hard_t=None))
    log_metrics("  [SOFT] TEST", eval_split(model, test_ld, max_label=MAX_SUM_EROSION, hard_t=None))

    # conformal
    if USE_CONFORMAL:
        val_pred, val_y, _ = predict_soft(model, val_ld)
        q = conformal_q_from_calibration(np.abs(val_y.astype(np.float64) - val_pred.astype(np.float64)), CONFORMAL_ALPHA)
        log_conformal_reg("VAL(calib-self)", CONFORMAL_ALPHA, q, conformal_interval_coverage(val_y, val_pred, q))

        te_pred, te_y, _ = predict_soft(model, test_ld)
        log_conformal_reg("TEST", CONFORMAL_ALPHA, q, conformal_interval_coverage(te_y, te_pred, q))

    # HARD threshold (report only)
    if USE_HARD_GATE_AT_REPORT:
        best_t = tune_threshold_on_val(model, val_ld, max_label=MAX_SUM_EROSION)
        log(f"Val-tuned HARD gate threshold = {best_t:.2f} (criterion={THRESH_CRITERION})")
        log_metrics("  [HARD] TEST", eval_split(model, test_ld, max_label=MAX_SUM_EROSION, hard_t=float(best_t)))


if __name__ == "__main__":
    main()
