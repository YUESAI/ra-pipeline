#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA Bone Age Prediction (Downstream)
- Image encoder: DINOv3 student from your pretrained FM (load ema_state)
- Token usage: CLS token for regression head
- Transforms: No Normalize here (processor handles it); do_rescale=False in forward
"""

import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import AutoModel, AutoImageProcessor
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from scipy.stats import pearsonr

# =========================
# Hardcoded paths & hparams
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

# 你的预训练 FM（multi-expert）的 checkpoint（只读 ema_state）
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# RSNA 数据（保持硬编码，不从命令行传入）
TRAIN_IMG_DIR = "/data/lab_ph/shared/RA/external_data/RSNA/training_set/boneage-training-dataset"
TRAIN_CSV     = "/data/lab_ph/shared/RA/external_data/RSNA/training_set/train.csv"
VAL_IMG_DIR   = "/data/lab_ph/shared/RA/external_data/RSNA/val_set/Bone Age Validation Set/val_set/boneage-validation-dataset"
VAL_CSV       = "/data/lab_ph/shared/RA/external_data/RSNA/val_set/Validation Dataset.csv"

# 训练超参（如需改，直接改这里的常量）
IMG_SIZE = 224
BATCH    = 32
EPOCHS   = 100
LR       = 1e-4         # 你之前写的是 1e-6，容易训练很慢/不稳定；这里用 1e-4 更常见
USE_META = True          # 是否使用性别元信息
FREEZE_ENCODER = False    # True=只训头；False=端到端微调
MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_multiexpert"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)

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

val_tf = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
])

class RSNADataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # 兼容 RSNA csv 的列名：id / boneage / male
        # male 可能是 0/1 或 True/False；做个兜底
        for col in ["id", "boneage", "male"]:
            if col not in self.df.columns:
                raise RuntimeError(f"CSV 缺少列: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = f"{int(row['id'])}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 归一化 bone age 到 0~1（以 240 月为基准）
        bone_age = float(row["boneage"]) / 240.0
        bone_age = torch.tensor(bone_age, dtype=torch.float32)

        # 性别布尔/数值 to {0,1} float
        val = row["male"]
        if pd.isna(val):
            val = 0
        is_male = float(val)
        if isinstance(val, str):
            is_male = 1.0 if val.lower() in ["1", "true", "male", "m"] else 0.0
        is_male = torch.tensor(is_male, dtype=torch.float32)

        return image, bone_age, is_male

# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
    """
    与预训练阶段保持一致：
    - AutoImageProcessor 做归一化/resize；这里 images=list(x), do_rescale=False
    - 去掉 DINOv3 的 register tokens，保留 [CLS+patches]
    """
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", device=None):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device).eval()

        cfg = self.encoder.config
        self.hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "hidden_dim", None) \
                           or getattr(cfg, "embed_dim", None) or getattr(cfg, "width", None)
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    @torch.no_grad()
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] in [0,1]
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

def load_student_from_ckpt(encoder: VisionEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(ckpt["ema_state"], strict=False)
    log(f"Loaded ema_state from {ckpt_path}")
    if missing:
        log(f"  Missing keys: {len(missing)} (showing first 5) {missing[:5]}")
    if unexpected:
        log(f"  Unexpected keys: {len(unexpected)} (showing first 5) {unexpected[:5]}")

# =========================
# Heads
# =========================
class BoneAgeModel(nn.Module):
    """Image → bone age"""
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
        tokens = self.encoder(x)         # [B, 1+P, D]
        feat = tokens[:, 0]              # CLS
        out = self.head(feat).squeeze(1) # [B]
        return out

class BoneAgeModelWithMeta(nn.Module):
    """Image + gender → bone age"""
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
        tokens = self.encoder(x)             # [B, 1+P, D]
        feat_img = tokens[:, 0]              # CLS
        feat_meta = self.meta_proj(gender.view(-1, 1))  # [B, D]
        feat = torch.cat([feat_img, feat_meta], dim=1)  # [B, 2D]
        out = self.head(feat).squeeze(1)     # [B]
        return out

# =========================
# Metrics
# =========================
def eval_metrics(preds, targets, name="Eval"):
    preds = np.asarray(preds) * 240.0
    targets = np.asarray(targets) * 240.0

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mad  = median_absolute_error(targets, preds)
    mae  = mean_absolute_error(targets, preds)
    r2   = r2_score(targets, preds)
    pcc  = pearsonr(targets, preds)[0]

    log(f"📊 {name}: RMSE={rmse:.3f}  MAD={mad:.3f}  MAE={mae:.3f}  R2={r2:.3f}  PCC={pcc:.3f}")
    return rmse, mad, mae, r2, pcc

# =========================
# Train / Val loops
# =========================
def train_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    n = 0
    # for img, age, gender in tqdm(loader, leave=False):
    for img, age, gender in loader:
        img = img.to(DEVICE, non_blocking=True)
        age = age.to(DEVICE, non_blocking=True)
        gender = gender.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if USE_META:
            pred = model(img, gender)
        else:
            pred = model(img)

        loss = loss_fn(pred, age)
        loss.backward()
        optimizer.step()

        total += loss.item() * img.size(0)
        n += img.size(0)
    return total / max(1, n)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    # for img, age, gender in tqdm(loader, leave=False):
    for img, age, gender in loader:
        img = img.to(DEVICE, non_blocking=True)
        age = age.to(DEVICE, non_blocking=True)
        gender = gender.to(DEVICE, non_blocking=True)
        if USE_META:
            pred = model(img, gender)
        else:
            pred = model(img)
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(age.detach().cpu().tolist())
    return eval_metrics(preds, targets, name="Validation")

# =========================
# Main
# =========================
def main():
    # Data
    train_ds = RSNADataset(TRAIN_IMG_DIR, TRAIN_CSV, transform=train_tf)
    val_ds   = RSNADataset(VAL_IMG_DIR,   VAL_CSV,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # Encoder & load FM student
    encoder = VisionEncoder(device=DEVICE).to(DEVICE)
    load_student_from_ckpt(encoder, CKPT_PATH)

    # 冻结/解冻
    if FREEZE_ENCODER:
        for p in encoder.parameters():
            p.requires_grad = False
        log("Encoder frozen (feature extractor).")
    else:
        log("Encoder will be fine-tuned.")

    # Model
    model = (BoneAgeModelWithMeta(encoder) if USE_META else BoneAgeModel(encoder)).to(DEVICE)

    # # model_ckpt_path = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_multiexpert/rsna_boneage_meta_10_ep43_MAD5.6434_PCC0.9690.pt"
    # model_ckpt_path = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_muon/rsna_boneage_meta_20_ep417_MAD10.3797_PCC0.8662.pt"
    # model.load_state_dict(torch.load(model_ckpt_path, map_location=DEVICE))
    # print(f"[MODEL] Loaded model state from: {model_ckpt_path}")

    # Optim / Loss
    # 仅训练 head 时，较大学习率问题不大；端到端微调时可把 LR 调小一些（如 1e-5）
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="mean")

    # best_mad = 5.6434
    best_mad = 10
    # best_mad = float("inf")

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        rmse, mad, mae, r2, pcc = evaluate(model, val_loader)

        log(f"Epoch {ep:03d}/{EPOCHS} | TrainLoss={train_loss:.4f} | "
            f"Val MAD={mad:.3f} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f} PCC={pcc:.3f} | "
            f"Time {time.time()-t0:.1f}s")

        # 保存最好模型（按 MAD）
        if mad < best_mad:
            best_mad = mad
            tag = "meta" if USE_META else "img"
            save_path = os.path.join(
                MODEL_SAVE_DIR,
                f"rsna_boneage_{tag}_20_ep{ep}_MAD{best_mad:.4f}_PCC{pcc:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            log(f"✅ Saved best model to {save_path}")

    log("Training finished.")

if __name__ == "__main__":
    main()






