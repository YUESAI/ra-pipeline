#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand-level SvH Score Prediction (Anhui) — Clean Baseline (Single-Pipeline, CLS Pooling)
- 关键修复：
  1) 只做一次图像几何/规范化处理（转到 transforms，去掉前向里的 AutoImageProcessor）
  2) CLS pooling 替代 patch-mean
  3) 从头训练：不加载旧的下游 ckpt，best=-inf
  4) AdamW(weight_decay=1e-2) + SmoothL1Loss()
"""

import os
import time
from pathlib import Path
import math

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


# 数据路径
IMAGE_DIR  = "/home/UWO/ylong66/data/RA/RA/external_data/anhui/PreparedSharpDataForStudy/"
LABEL_CSV  = "/home/UWO/ylong66/data/RA/RA/external_data/anhui/PreparedSharpDataForStudy/data/data.csv"

# 划分规模
N_TRAIN = 2700
N_VAL   = 760
N_TEST  = 158
SEED    = 42

# 视觉骨干（优先本地目录）
VISION_MODEL_LOCAL_PATH = "/home/UWO/ylong66/data/RA/LLM/hf_model/dinov3-vitb16"
VISION_MODEL_NAME       = "facebook/dinov3-vitb16"  # 建议与本地模型架构一致（vitb16）

FOUNDATION_CKPT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/"

FOUNDATION_CKPT_FILE = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# 超参
IMG_SIZE   = 224
BATCH      = 64
EPOCHS     = 100
LR         = 1e-5
WD         = 0          # ← 加权重衰减
# HUBER_BETA = 0.02         # 旧：很 L1；现统一用 SmoothL1Loss 默认 β=1.0

# 训练策略
FREEZE_ENCODER = False     # True=只训头；False=端到端
USE_META       = False     # [gender, age]

# ==== 对比实验关键开关 ====
RANDOM_INIT    = False     # True=随机初始化 encoder；False=加载 HF 预训练（并尽量加载你的 foundation ema_state）

# 输出
# MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_muon"

# MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema"
# MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_biomedclip_muon"
# MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_chest_muon"
# MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_biomedclip_chest_alignonly"
# MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/recon_only"
# MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/student_only"
MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/mask_random"


os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)

# =========================
# Build transforms (单次处理，归一化与骨干一致)
# =========================
def build_transforms():
    src = VISION_MODEL_LOCAL_PATH if os.path.isdir(VISION_MODEL_LOCAL_PATH) else VISION_MODEL_NAME
    proc = AutoImageProcessor.from_pretrained(src)
    mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
    std  = getattr(proc, "image_std",  [0.229, 0.224, 0.225])

    train_tf = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),                     # [0,1]
        T.Normalize(mean=mean, std=std),  # 与骨干一致
    ])
    eval_tf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf

train_tf, eval_tf = build_transforms()

# =========================
# Dataset & Split
# =========================
class AnhuiDatasetWithMeta(Dataset):
    def __init__(self, df, image_dir, tf, use_meta=False):
        self.df = df.reset_index(drop=True)
        self.dir = Path(image_dir)
        self.tf = tf
        self.use_meta = use_meta

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(self.dir / f"DX{r['exam_number']}.jpg").convert("RGB")
        img = self.tf(img)

        y = torch.tensor(float(r["score_avg"]), dtype=torch.float32)
        if self.use_meta:
            gender = 0.0 if str(r["gender"]).upper() == "F" else 1.0
            age = float(r["age"])
            meta = torch.tensor([gender, age], dtype=torch.float32)
        else:
            meta = torch.empty(0)

        return img, meta, y

def split_anhui_df(df: pd.DataFrame, image_dir: str, seed=42,
                   n_train=2700, n_val=760, n_test=158):
    df = df.copy()
    df["img_path"] = df["exam_number"].apply(lambda x: os.path.join(image_dir, f"DX{x}.jpg"))
    df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    tr = df.iloc[:n_train].reset_index(drop=True)
    va = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    te = df.iloc[n_train+n_val:n_train+n_val+n_test].reset_index(drop=True)
    return tr, va, te

# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
    """
    - RANDOM_INIT=True -> AutoConfig + AutoModel.from_config 随机初始化
    - RANDOM_INIT=False -> AutoModel.from_pretrained（并可选加载你的 ema_state）
    - 前向不再做任何 resize/crop/normalize；这些已在 transforms 完成
    - 输出 [B, 1+P(+R), D]，后续用 CLS token
    """
    def __init__(self, device=None, init_mode: str = "pretrained"):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        src = VISION_MODEL_LOCAL_PATH if os.path.isdir(VISION_MODEL_LOCAL_PATH) else VISION_MODEL_NAME

        if init_mode == "random":
            cfg = AutoConfig.from_pretrained(src)
            self.encoder = AutoModel.from_config(cfg)
            log("⚙️ VisionEncoder: initialized RANDOM weights from config.")
        else:
            self.encoder = AutoModel.from_pretrained(src)
            log("⚙️ VisionEncoder: loaded HF pretrained weights.")

        self.encoder.to(self.device)

        cfg = self.encoder.config
        self.hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "hidden_dim", None) \
                           or getattr(cfg, "embed_dim", None) or getattr(cfg, "width", None)
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 已是 [B,3,H,W]、已 Normalize 到骨干期望
        x = x.to(self.device, dtype=torch.float32)
        out = self.encoder(pixel_values=x, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D]
        return tokens

def load_student_from_ckpt(encoder: VisionEncoder):
    ck = Path(FOUNDATION_CKPT_FILE)
    if not ck.exists():
        log(f"⚠️ 未找到预训练 ema_state：{ck}；继续使用 HF 权重。")
        return
    state = torch.load(ck, map_location="cpu")
    # key = "ema_state" if "ema_state" in state else None
    key = "ema_state" if "ema_state" in state else None
    if key is None:
        log(f"⚠️ ckpt 中未找到 'ema_state'，忽略加载。")
        return
    res = encoder.load_state_dict(state[key], strict=False)
    log(f"Loaded ema_state from {ck}")
    if hasattr(res, "missing_keys") and hasattr(res, "unexpected_keys"):
        miss, unexp = res.missing_keys, res.unexpected_keys
        if len(miss):  log(f"  Missing keys: {len(miss)} (first 5) {miss[:5]}")
        if len(unexp): log(f"  Unexpected keys: {len(unexp)} (first 5) {unexp[:5]}")

# =========================
# Model (CLS → regression head)
# =========================
class WholeSVHModel(nn.Module):
    def __init__(self, encoder: VisionEncoder, use_meta: bool = False):
        super().__init__()
        self.encoder = encoder
        self.use_meta = use_meta
        D = encoder.hidden_size

        if use_meta:
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(inplace=True), nn.Linear(64, D)
            )
            head_in = 2 * D
        else:
            head_in = D

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # 显式初始化回归头（Xavier）
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if use_meta:
            for m in self.meta_fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.encoder(x)               # [B, 1+R+P, D]
        feat = tokens[:, 0, :]                 # ← 使用 CLS
        if self.use_meta:
            meta_feat = self.meta_fc(meta)
            feat = torch.cat([feat, meta_feat], dim=1)
        out = self.head(feat).squeeze(1)
        return out

# =========================
# Loss & Metrics
# =========================
class SmoothL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.SmoothL1Loss()  # β=1.0

    def forward(self, pred, target):
        return self.crit(pred, target)

@torch.no_grad()
def compute_metrics(pred: np.ndarray, target: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(target, pred)))
    mae  = float(mean_absolute_error(target, pred))
    r2   = float(r2_score(target, pred)) if len(target) > 1 else np.nan
    pcc  = float(pearsonr(target, pred)[0]) if len(target) > 1 else np.nan
    return dict(rmse=rmse, r2=r2, pcc=pcc, mae=mae)

def log_metrics(title: str, m: dict):
    log(f"📊 {title}: RMSE={m['rmse']:.3f}, R2={m['r2']:.3f}, PCC={m['pcc']:.3f}, MAE={m['mae']:.3f}")

# =========================
# Train / Eval
# =========================
def eval_loader(model, loader) -> dict:
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for img, meta, y in loader:
            img = img.to(DEVICE, non_blocking=True)
            y   = y.to(DEVICE, non_blocking=True)
            if USE_META:
                meta = meta.to(DEVICE, non_blocking=True)
                out = model(img, meta)
            else:
                out = model(img)
            preds.append(out.detach().cpu().numpy())
            targs.append(y.detach().cpu().numpy())
    p = np.concatenate(preds, 0)
    t = np.concatenate(targs, 0)
    return compute_metrics(p, t)

def train_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    tot, n = 0.0, 0
    for img, meta, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE, non_blocking=True)
        if USE_META:
            meta = meta.to(DEVICE, non_blocking=True)
            out = model(img, meta)
        else:
            out = model(img)
        loss = loss_fn(out, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = img.size(0)
        tot += loss.item() * bs
        n   += bs
    return tot / max(1, n)

# =========================
# Main
# =========================
def main():
    # 固定随机性
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)

    # 读取并划分
    df_all = pd.read_csv(LABEL_CSV)
    tr_df, va_df, te_df = split_anhui_df(df_all, IMAGE_DIR, seed=SEED,
                                          n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
    log(f"Samples -> Train: {len(tr_df)} | Valid: {len(va_df)} | Test: {len(te_df)}")

    # Datasets & Loaders（不丢样本）
    train_ds = AnhuiDatasetWithMeta(tr_df, IMAGE_DIR, train_tf, use_meta=USE_META)
    valid_ds = AnhuiDatasetWithMeta(va_df, IMAGE_DIR, eval_tf,   use_meta=USE_META)
    test_ds  = AnhuiDatasetWithMeta(te_df, IMAGE_DIR, eval_tf,   use_meta=USE_META)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)  # drop_last=False
    valid_ld = DataLoader(valid_ds, batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    # Encoder & Model
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(device=DEVICE, init_mode=init_mode).to(DEVICE)

    if not RANDOM_INIT:
        load_student_from_ckpt(encoder)   # 若找不到或没有 ema_state，会自动忽略
        log("Encoder initialized from pretrained FM (ema_state if available).")
    else:
        log("Encoder randomly initialized (no pretrained weights).")

    if FREEZE_ENCODER:
        for p in encoder.parameters(): p.requires_grad = False
        log("Encoder frozen (feature extractor).")
    else:
        log("Encoder will be fine-tuned.")

    model = WholeSVHModel(encoder, use_meta=USE_META).to(DEVICE)
    start_ep = 0
    best_val_pcc = -math.inf

    # model_ckpt_path = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema/anhui_svh_hand_pre_v1_ep99_pcc_0.8912.pt"
    # model.load_state_dict(torch.load(model_ckpt_path, map_location=DEVICE))
    # log(f"Model initialized from {model_ckpt_path}")
    # start_epoch = 99
    # best_val_pcc = 0.8912

    
    # Optim & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn   = SmoothL1()

    # 训练循环（从 0 开始；best = -inf）
    best_path = None

    for ep in range(start_ep, start_ep + EPOCHS):
        t0 = time.time()

        train_loss = train_epoch(model, train_ld, optimizer, loss_fn)
        m_tr = eval_loader(model, train_ld);  log_metrics("Train", m_tr)
        m_va = eval_loader(model, valid_ld);  log_metrics("Valid", m_va)

        # 保存：以验证集 PCC 为准
        if (m_va["pcc"] > best_val_pcc) or (best_path is None):
            best_val_pcc = m_va["pcc"]
            tag = "rand" if RANDOM_INIT else "pre"
            best_path = os.path.join(
                MODEL_SAVE_DIR, f"anhui_svh_hand_{tag}_v1_ep{ep}_pcc_{best_val_pcc:.4f}.pt"
            )
            torch.save(model.state_dict(), best_path)
            log(f"✅ Saved best-by-Valid-PCC to {best_path}")

        log(f"Epoch {ep:03d}/{start_ep + EPOCHS - 1} | TrainLoss={train_loss:.4f} | "
            f"Val PCC={m_va['pcc']:.4f} | Time {time.time()-t0:.1f}s")

    # Final Test
    if best_path and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        log(f"Loaded best checkpoint: {best_path}")
    m_te = eval_loader(model, test_ld)
    log_metrics("Test (Best-by-Valid-PCC)", m_te)
    log("Training finished.")

if __name__ == "__main__":
    main()

