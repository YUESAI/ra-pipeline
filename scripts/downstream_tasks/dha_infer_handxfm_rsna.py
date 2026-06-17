#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DHA external inference for HandXFM RSNA-trained checkpoint
"""

import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from scipy.stats import pearsonr

# =========================
# Hardcoded paths
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema_rsna_select/best_rsna_meta_ep100_RSNA.pt"
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

DHA_ROOT = "/home/UWO/ylong66/data/RA/RA/external_data/Digital Hand Atlas/"
DHA_SAMPLES = os.path.join(DHA_ROOT, "annotations/samples.csv")

IMG_SIZE = 224
BATCH = 32
NUM_WORKERS = 4
USE_META = True

def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)

# =========================
# Transforms
# =========================
val_tf = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
])

# =========================
# Dataset
# =========================
class DHADataset(Dataset):
    """
    DHA:
    - image path from samples.csv
    - label in years -> normalized by /18
    - gender inferred from cate ending M/F
    """
    def __init__(self, data_dir: str, df: pd.DataFrame, transform=None):
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, r["image_path"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        years = float(r["boneage"])
        y_norm = torch.tensor(years / 18.0, dtype=torch.float32)

        val = str(r["cate"])[-1] if "cate" in r else "M"
        is_male = 1.0 if val == "M" else 0.0
        is_male = torch.tensor(is_male, dtype=torch.float32)

        return image, y_norm, is_male

# =========================
# Vision Encoder
# =========================
class VisionEncoder(nn.Module):
    """
    - AutoImageProcessor does normalization
    - input tensors already in [0,1], so do_rescale=False
    - remove register tokens, keep [CLS + patches]
    """
    def __init__(self, model_name=MODEL_NAME, device=None):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        cfg = self.encoder.config
        self.hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "hidden_dim", None)
            or getattr(cfg, "embed_dim", None)
            or getattr(cfg, "width", None)
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
        tokens = out.last_hidden_state

        if self.num_register_tokens > 0:
            cls_tok = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_register_tokens:, :]
            tokens = torch.cat([cls_tok, patches], dim=1)

        return tokens

# =========================
# Heads
# =========================
class BoneAgeModel(nn.Module):
    def __init__(self, encoder: VisionEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size
        self.head = nn.Sequential(
            nn.Linear(D, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        tokens = self.encoder(x)
        feat = tokens[:, 0]
        y = self.head(feat).squeeze(1)
        return y

class BoneAgeModelWithMeta(nn.Module):
    def __init__(self, encoder: VisionEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.hidden_size
        self.meta_proj = nn.Linear(1, D)
        self.head = nn.Sequential(
            nn.Linear(D * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x, gender):
        tokens = self.encoder(x)
        feat_img = tokens[:, 0]
        feat_meta = self.meta_proj(gender.view(-1, 1))
        feat = torch.cat([feat_img, feat_meta], dim=1)
        y = self.head(feat).squeeze(1)
        return y

# =========================
# Metrics
# =========================
def eval_metrics(norm_preds, norm_targets, name="Eval"):
    preds = np.asarray(norm_preds) * 18.0
    targets = np.asarray(norm_targets) * 18.0

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mad = median_absolute_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        pcc = pearsonr(targets, preds)[0]
    else:
        pcc = float("nan")

    log(f"📊 {name}: RMSE={rmse:.3f}y  MAD={mad:.3f}y  MAE={mae:.3f}y  R2={r2:.3f}  PCC={pcc:.3f}")
    return rmse, mad, mae, r2, pcc

@torch.no_grad()
def evaluate(model, loader, name="DHA External"):
    model.eval()
    preds, targets = [], []

    for img, y_norm, gender in loader:
        img = img.to(DEVICE, non_blocking=True)
        y_norm = y_norm.to(DEVICE, non_blocking=True)
        gender = gender.to(DEVICE, non_blocking=True)

        pred = model(img, gender) if USE_META else model(img)

        preds.extend(pred.detach().cpu().tolist())
        targets.extend(y_norm.detach().cpu().tolist())

    return eval_metrics(preds, targets, name=name)

# =========================
# Main
# =========================
def main():
    log(f"DEVICE = {DEVICE}")
    log(f"CKPT_PATH = {CKPT_PATH}")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    dha_df = pd.read_csv(DHA_SAMPLES)
    if "boneage" not in dha_df.columns:
        dha_df["boneage"] = dha_df["age_id"].apply(lambda x: int(str(x)[-2:]))

    dha_ds = DHADataset(DHA_ROOT, dha_df, transform=val_tf)
    dha_loader = DataLoader(
        dha_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    log(f"DHA samples: {len(dha_ds)}")

    encoder = VisionEncoder(device=DEVICE).to(DEVICE)
    model = (BoneAgeModelWithMeta(encoder) if USE_META else BoneAgeModel(encoder)).to(DEVICE)

    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    log("Loaded HandXFM downstream checkpoint successfully.")

    rmse, mad, mae, r2, pcc = evaluate(model, dha_loader, name="DHA External FINAL (years)")

    log("=" * 80)
    log("FINAL DHA RESULTS")
    log(f"Checkpoint: {CKPT_PATH}")
    log(f"RMSE={rmse:.3f}y | MAD={mad:.3f}y | MAE={mae:.3f}y | R2={r2:.3f} | PCC={pcc:.3f}")
    log("=" * 80)

if __name__ == "__main__":
    main()

