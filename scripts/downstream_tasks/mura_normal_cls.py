#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MURA XR_HAND Normal/Abnormal Classification (DINOv3, CLS-only Head, safer LR)
- RANDOM_INIT=True → 随机初始化 DINOv3（不加载任何预训练/旧下游权重）
- RANDOM_INIT=False → 从 HF + 你的 multi-expert ema_state 加载
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoImageProcessor, AutoConfig

# ============== 可配置项 ==============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

DATA_ROOT = "/home/UWO/ylong66/data/RA/RA/external_data/mura/muramskxrays/"
TRAIN_LABEL_CSV = f"{DATA_ROOT}/MURA-v1.1/train_labeled_studies.csv"
VAL_LABEL_CSV   = f"{DATA_ROOT}/MURA-v1.1/valid_labeled_studies.csv"
TRAIN_IMAGE_CSV = f"{DATA_ROOT}/MURA-v1.1/train_image_paths.csv"
VAL_IMAGE_CSV   = f"{DATA_ROOT}/MURA-v1.1/valid_image_paths.csv"
POSITION_FILTER = "XR_HAND"  # None 则不按部位过滤

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

CKPT_EPOCH = 20
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"
# CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/dinov3_biomedclip_muon/handx_pretrain_dinov3_biomedclip_only_224_10.pt"
# CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/dinov3_chest_muon/handx_pretrain_dinov3_chest_10.pt"
# CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_align_only/handx_pretrain_multiexpert_align_10.pt"
# CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/recon_only/handx_pretrain_multiexpert_recon_224_10.pt"
# CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/dinov3_only/handx_pretrain_student_only_224_10.pt"
# CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert/adam_mask_random/handx_pretrain_multiexpert_224_10.pt"


# OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema"
# OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_biomedclip_muon"
# OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_chest_muon"
# OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/dinov3_biomedclip_chest_alignonly"
# OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/recon_only"
# OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/student_only"
OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/mask_random"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 32
ENC_LR = 1e-5
HEAD_LR = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# 训练稳态
AMP = True
EARLY_STOP = 0
GRAD_CLIP_NORM = 0
WARMUP_FREEZE_EPOCHS = 0

# 类别不平衡设置
POS_WEIGHT = 0.0              # <=0 自动计算
WEIGHTED_SAMPLER = True

# 数据增强（仅几何，保持 PIL；不 ToTensor/不 Normalize）
IMG_SIZE = 224
TRAIN_AUG = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10, fill=0),
    T.CenterCrop(IMG_SIZE),
])
VAL_AUG = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
])

# ==== 关键对比开关 ====
RANDOM_INIT = False  # True=随机初始化 encoder；False=加载你预训练 ema_state
# =====================================


# --------- 数据处理 ---------
def load_mura_df(label_path, image_path, position_filter: Optional[str] = "XR_HAND"):
    label_df = pd.read_csv(label_path, header=None, names=["study", "label"])
    image_df = pd.read_csv(image_path, header=None, names=["image"])
    image_df["study"] = image_df["image"].apply(lambda x: "/".join(x.split("/")[:-1]) + "/")
    image_df["position"] = image_df["image"].apply(lambda x: x.split("/")[2])
    merged = pd.merge(image_df, label_df, on="study")
    if position_filter:
        merged = merged[merged["position"] == position_filter]
    return merged.reset_index(drop=True)

class MuraDataset(Dataset):
    def __init__(self, root, df, transform=None):
        self.root = Path(root)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["image"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return img, label


# --------- 指标 ---------
def compute_pos_weight(labels: np.ndarray) -> float:
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    return float(neg / max(pos, 1)) if pos > 0 else 1.0

@torch.no_grad()
def eval_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().numpy().astype(int)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


# --------- 编码器（支持随机初始化） ---------
class VisionEncoder(nn.Module):
    """
    - RANDOM_INIT=True: AutoConfig + AutoModel.from_config（随机初始化）
    - RANDOM_INIT=False: AutoModel.from_pretrained(MODEL_NAME) 再可选加载 ema_state
    - forward: processor(images=list_of_PIL)
    - 去掉 register tokens → 返回 [CLS + patches]
    """
    def __init__(self, model_name: str, device: Optional[str] = None, init_mode: str = "pretrained"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if init_mode == "random":
            cfg = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(cfg)
            print("⚙️ VisionEncoder: RANDOM init from config.")
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
            print("⚙️ VisionEncoder: loaded HF pretrained weights.")

        self.encoder.to(self.device)  # 不 .eval()，保持可训练
        # self.encoder.to(self.device).eval()  # 不可训练

        cfg = self.encoder.config
        self.hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "hidden_dim", None) \
                           or getattr(cfg, "embed_dim", None) or getattr(cfg, "width", None)
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    @torch.no_grad()
    def _process(self, images) -> torch.Tensor:
        batch = self.processor(images=images, return_tensors="pt")
        return batch["pixel_values"].to(self.device)

    def forward(self, x) -> torch.Tensor:
        pixel_values = self._process(list(x))
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D]
        if self.num_register_tokens and self.num_register_tokens > 0:
            cls_tok = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_register_tokens:, :]
            tokens = torch.cat([cls_tok, patches], dim=1)  # [B, 1+P, D]
        return tokens


def load_student_from_ckpt(encoder: VisionEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 注意：StateDictMapping 返回 (missing, unexpected)
    res = encoder.load_state_dict(ckpt["ema_state"], strict=False)

    print(f"[CKPT] Loaded ema_state from: {ckpt_path}")
    if getattr(res, "missing_keys", None):
        print(f"[CKPT] missing_keys: {len(res.missing_keys)} (first 5) {res.missing_keys[:5]}")
    if getattr(res, "unexpected_keys", None):
        print(f"[CKPT] unexpected_keys: {len(res.unexpected_keys)} (first 5) {res.unexpected_keys[:5]}")


# --------- Head（CLS-only） ----------
class CLSHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, linear: bool = False, dropout: float = 0.2):
        super().__init__()
        if linear:
            self.net = nn.Linear(dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, 384),
                nn.ReLU(inplace=True),
                # nn.Dropout(dropout),
                nn.Linear(384, 1),
            )

        # # 显式初始化（对随机 encoder 更稳，对预训练也安全）
        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = tokens[:, 0, :]                 # [B, D]
        return self.net(cls).squeeze(-1)      # [B]


class MURAHandClassifier(nn.Module):
    def __init__(self, encoder: VisionEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, x):
        tokens = self.encoder(x)  # [B, 1+P, D]
        return self.head(tokens)  # [B]


# --------- 训练主逻辑 ---------
def main():
    print("Device:", DEVICE)

    train_df = load_mura_df(TRAIN_LABEL_CSV, TRAIN_IMAGE_CSV, POSITION_FILTER)
    val_df   = load_mura_df(VAL_LABEL_CSV,   VAL_IMAGE_CSV,   POSITION_FILTER)
    print(f"Train={len(train_df)} | Val={len(val_df)}")

    train_set = MuraDataset(DATA_ROOT, train_df, TRAIN_AUG)
    val_set   = MuraDataset(DATA_ROOT, val_df,   VAL_AUG)

    sampler = None
    if WEIGHTED_SAMPLER:
        y = train_df["label"].values.astype(int)
        class_counts = np.bincount(y, minlength=2)
        weights = 1.0 / (class_counts + 1e-6)
        sample_weights = weights[y]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def collate_pil(batch):
        imgs, labels = zip(*batch)
        labels = torch.stack(labels, dim=0)
        return list(imgs), labels

    train_loader = DataLoader(
        train_set, BATCH_SIZE, shuffle=(sampler is None), sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        collate_fn=collate_pil,
    )
    val_loader = DataLoader(
        val_set, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,drop_last=True, pin_memory=True, persistent_workers=True,
        collate_fn=collate_pil,
    )

    # Encoder & 上游权重
    init_mode = "random" if RANDOM_INIT else "pretrained"
    encoder = VisionEncoder(MODEL_NAME, device=DEVICE, init_mode=init_mode).to(DEVICE)

    if not RANDOM_INIT:
        load_student_from_ckpt(encoder, CKPT_PATH)
        print("Encoder initialized from pretrained FM (ema_state).")
    else:
        print("Encoder randomly initialized (no pretrained weights).")

    head = CLSHead(dim=encoder.hidden_size, hidden=512, linear=False, dropout=0.2).to(DEVICE)
    model = MURAHandClassifier(encoder, head).to(DEVICE)

    # ⚠️ 随机初始化对比：不要加载任何旧的下游 ckpt
    # model_ckpt_path = "..."  # ← 不要加载

    # 分组学习率
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENC_LR,  "weight_decay": WEIGHT_DECAY},
        {"params": model.head.parameters(),    "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
    ])

    # # 正负样本权重
    # if POS_WEIGHT > 0:
    #     pos_w = torch.tensor([POS_WEIGHT], device=DEVICE)
    # else:
    #     w = compute_pos_weight(train_df["label"].values)
    #     pos_w = torch.tensor([w], device=DEVICE)
    #     print(f"Auto pos_weight={w:.4f}")
    # # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler('cuda', enabled=AMP)

    # 随机初始化对比：从 0 开始，best_auc = -inf
    start_epoch = 0 if RANDOM_INIT else 1
    best_auc = float("-inf")
    # best_auc = 0.8
    no_improve = 0

    for ep in range(start_epoch, EPOCHS + 1):
        # ---- 可选热身冻结 ----
        if WARMUP_FREEZE_EPOCHS > 0:
            if ep < WARMUP_FREEZE_EPOCHS:
                for p in model.encoder.parameters(): p.requires_grad = False
                enc_frozen = True
            else:
                if any(not p.requires_grad for p in model.encoder.parameters()):
                    for p in model.encoder.parameters(): p.requires_grad = True
                enc_frozen = False
        else:
            enc_frozen = False

        model.train()
        total_loss = 0.0
        logits_all, labels_all = [], []

        for imgs, labs in train_loader:
            labs = labs.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=AMP):
                logits = model(imgs)
                loss = criterion(logits, labs)

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labs)
            logits_all.append(logits.detach().cpu())
            labels_all.append(labs.detach().cpu())

        train_loss = total_loss / len(train_loader.dataset)
        train_metrics = eval_metrics_from_logits(torch.cat(logits_all), torch.cat(labels_all))

        # ---- 验证 ----
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for imgs, labs in val_loader:
                labs = labs.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=AMP):
                    logit = model(imgs)
                val_logits.append(logit.cpu())
                val_labels.append(labs.cpu())
        val_metrics = eval_metrics_from_logits(torch.cat(val_logits), torch.cat(val_labels))

        print(
            f"\nEpoch {ep}/{EPOCHS} "
            f"{'(ENC FROZEN)' if enc_frozen else ''}"
            f"\n  Train — Loss {train_loss:.4f} | AUC {train_metrics['auc']:.4f} | "
            f"ACC {train_metrics['acc']:.4f} | F1 {train_metrics['f1']:.4f} | "
            f"P {train_metrics['precision']:.4f} | R {train_metrics['recall']:.4f}"
            f"\n  Val   — AUC {val_metrics['auc']:.4f} | ACC {val_metrics['acc']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | P {val_metrics['precision']:.4f} | R {val_metrics['recall']:.4f}"
        )

        cur_auc = val_metrics["auc"]
        if np.isnan(cur_auc): cur_auc = -1.0
        if cur_auc > best_auc:
            best_auc = cur_auc
            no_improve = 0
            tag = "rand" if RANDOM_INIT else f"ckpt{CKPT_EPOCH}"
            ckpt_path = Path(OUTPUT_DIR) / f"mura_normal_v3_{tag}_ep{ep}_auc{best_auc:.4f}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"☆ 新最佳！Val AUC={best_auc:.4f} 已保存: {ckpt_path}")
        else:
            no_improve += 1
            if EARLY_STOP > 0 and no_improve >= EARLY_STOP:
                print(f"早停触发（{EARLY_STOP} epochs 未提升）。最佳 AUC={best_auc:.4f}")
                break

    print(f"训练完成。最佳 Val AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
