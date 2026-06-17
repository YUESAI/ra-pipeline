#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint-level SvH Score Prediction (JSN only in current setting)
Multi-seed version
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import AutoModel, AutoImageProcessor, AutoConfig
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

# =========================
# Hardcoded paths & hparams
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

TRAIN_CSV   = "/data/lab_ph/shared/RA/external_data/AIRINIIReumHands/pocessed_scores_same_side.csv"
IMG_ROOT    = "/home/UWO/ylong66/data/RA/RA/external_data/AIRINIIReumHands/extracted_joint_image/"
SPLIT_JSON  = "/home/UWO/ylong66/data/RA/RA/external_data/AIRINIIReumHands/train_val_test_split.json"

RANDOM_INIT = False

IMG_SIZE       = 224
BATCH          = 64
EPOCHS         = 50
LR             = 1e-4
FREEZE_ENCODER = False

MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema_multiseed"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===== multi-seed settings =====
SEEDS = [0, 1, 2, 42, 3407]

def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Dataset & Transforms
# =========================
train_tf = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
])

eval_tf = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
])

class JointDataset(Dataset):
    """Returns: image, labels_tensor([erosion, jsn])"""
    def __init__(self, df: pd.DataFrame, root: str, tf):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row.file_name).convert("RGB")
        img = self.tf(img)

        def _to_int(v, nan_val=-1):
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return nan_val

        eros = _to_int(row.Erosion_score)
        jsn  = _to_int(row.JSN_score)
        labels = torch.tensor([eros, jsn], dtype=torch.long)
        return img, labels

# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", device=None, init_mode='pretrained'):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if init_mode == "random":
            cfg = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(cfg)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        self.encoder.to(self.device)

        cfg = self.encoder.config
        self.hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "hidden_dim", None) \
                           or getattr(cfg, "embed_dim", None) or getattr(cfg, "width", None)
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
# Model
# =========================
class JointSVHModel(nn.Module):
    def __init__(self, encoder: VisionEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size
        self.head_e = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=True),
            nn.Linear(D // 2, 1)
        )
        self.head_j = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=True),
            nn.Linear(D // 2, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        pooled = feat.mean(dim=1)
        pred_e = self.head_e(pooled)
        pred_j = self.head_j(pooled)
        return pred_e.squeeze(1), pred_j.squeeze(1)

# =========================
# Loss
# =========================
class MaskedRegressionLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.crit = nn.SmoothL1Loss(reduction="none")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.sum() == 0:
            return pred.new_tensor(0.0)
        loss = self.crit(pred[mask], target[mask])
        return loss.mean() if self.reduction == "mean" else loss.sum()

# =========================
# Metrics
# =========================
@torch.no_grad()
def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    mask = target > -0.5
    if mask.sum() == 0:
        return dict(rmse=np.nan, r2=np.nan, pcc=np.nan, mae=np.nan)
    y = target[mask]
    p = pred[mask]
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae  = float(mean_absolute_error(y, p))
    r2   = float(r2_score(y, p)) if len(y) > 1 else np.nan
    pcc  = float(pearsonr(y, p)[0]) if len(y) > 1 else np.nan
    return dict(rmse=rmse, r2=r2, pcc=pcc, mae=mae)

def log_split_metrics(title: str, m_j: Dict[str, float]):
    log(
        f"📊 {title}\n"
        f"  JSN    : RMSE={m_j['rmse']:.3f}, R2={m_j['r2']:.3f}, PCC={m_j['pcc']:.3f}, MAE={m_j['mae']:.3f}"
    )

# =========================
# Train / Eval
# =========================
def train_epoch(model, loader, optimizer, loss_j) -> float:
    model.train()
    total, n = 0.0, 0
    for img, labels in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        jsn_t  = labels[:, 1].float()
        mask_j = jsn_t >= 0.0

        optimizer.zero_grad(set_to_none=True)
        _, pred_j = model(img)

        loss = loss_j(pred_j, jsn_t, mask_j)
        loss.backward()
        optimizer.step()

        bs = img.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)

@torch.no_grad()
def eval_loader(model, loader) -> Dict[str, Dict[str, float]]:
    model.eval()
    pred_j_list, tgt_j_list = [], []

    for img, labels in loader:
        img = img.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        jsn_t  = labels[:, 1].float()
        _, pred_j = model(img)

        pred_j_list.append(pred_j.detach().cpu().numpy())
        tgt_j_list.append(jsn_t.detach().cpu().numpy())

    pred_j = np.concatenate(pred_j_list, axis=0)
    tgt_j  = np.concatenate(tgt_j_list, axis=0)

    m_j = compute_metrics(pred_j, tgt_j)
    return {"jsn": m_j}

# =========================
# One-seed run
# =========================
def run_one_seed(seed: int):
    log("=" * 80)
    log(f"Start seed {seed}")
    set_seed(seed)

    # ---- split ----
    with open(SPLIT_JSON, "r") as f:
        split_data = json.load(f)

    df_all  = pd.read_csv(TRAIN_CSV)
    df_tr   = df_all.iloc[split_data["train"]].reset_index(drop=True)
    val_key = "val" if "val" in split_data else "valid"
    if val_key not in split_data:
        raise KeyError("SPLIT_JSON must contain a validation split under 'val' or 'valid'.")
    df_val  = df_all.iloc[split_data[val_key]].reset_index(drop=True)
    df_test = df_all.iloc[split_data["test"]].reset_index(drop=True)

    # ---- datasets & loaders ----
    train_ds = JointDataset(df_tr, IMG_ROOT, train_tf)
    val_ds   = JointDataset(df_val, IMG_ROOT, eval_tf)
    test_ds  = JointDataset(df_test, IMG_ROOT, eval_tf)

    g = torch.Generator()
    g.manual_seed(seed)

    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        generator=g,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    log(f"Seed {seed} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")

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
        log("Encoder frozen (feature extractor).")
    else:
        log("Encoder will be fine-tuned.")

    model = JointSVHModel(encoder).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_j = MaskedRegressionLoss()

    best_val_pcc = -1e9
    best_val_metrics = None
    best_epoch = -1
    best_path = None

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_ld, optimizer, loss_j)

        train_metrics = eval_loader(model, train_ld)
        tr_j = train_metrics["jsn"]
        log_split_metrics(f"Train (seed={seed}, epoch={ep})", tr_j)

        val_metrics = eval_loader(model, val_ld)
        va_j = val_metrics["jsn"]
        log_split_metrics(f"Val (seed={seed}, epoch={ep})", va_j)

        val_pcc = va_j["pcc"]

        log(
            f"Seed {seed} | Epoch {ep:03d}/{EPOCHS} | "
            f"TrainLoss={train_loss:.4f} | Val PCC={val_pcc:.4f} | "
            f"Time {time.time()-t0:.1f}s"
        )

        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            best_val_metrics = va_j.copy()
            best_epoch = ep

            save_path = os.path.join(
                MODEL_SAVE_DIR,
                f"seed{seed}_airi_svh_joint_jsn_ep{ep}_valpcc{best_val_pcc:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            best_path = save_path
            log(f"Saved best-by-val seed-{seed} model to {save_path}")

    if best_path is None:
        raise RuntimeError(f"No checkpoint was selected for seed {seed}.")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = eval_loader(model, test_ld)
    test_j = test_metrics["jsn"]
    log_split_metrics(f"Test (seed={seed}, best_epoch={best_epoch})", test_j)

    log(f"Finished seed {seed} | Best epoch={best_epoch} | Best val metrics={best_val_metrics} | Test metrics={test_j}")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_j,
    }

# =========================
# Main
# =========================
def summarize_results(results):
    metric_names = ["rmse", "r2", "pcc", "mae"]
    summary = {}

    for m in metric_names:
        vals = [r["test_metrics"][m] for r in results]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals]
        }

    log("=" * 80)
    log("Multi-seed summary (test metrics from validation-selected checkpoints)")
    log(
        "JSN: "
        f"RMSE={summary['rmse']['mean']:.4f}±{summary['rmse']['std']:.4f}, "
        f"R2={summary['r2']['mean']:.4f}±{summary['r2']['std']:.4f}, "
        f"PCC={summary['pcc']['mean']:.4f}±{summary['pcc']['std']:.4f}, "
        f"MAE={summary['mae']['mean']:.4f}±{summary['mae']['std']:.4f}"
    )
    return summary

def main():
    all_results = []
    for seed in SEEDS:
        result = run_one_seed(seed)
        all_results.append(result)

    summary = summarize_results(all_results)

    out_path = os.path.join(MODEL_SAVE_DIR, "multi_seed_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "per_seed": all_results,
                "summary": summary,
            },
            f,
            indent=2
        )
    log(f"Saved multi-seed results to {out_path}")

if __name__ == "__main__":
    main()
