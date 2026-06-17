#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedPix VQA (Downstream)
- Image encoder: DINOv3 student from the pretrained FM (load ema_state)
- Token usage: CLS token + patches from image encoder (register tokens removed)
- Text encoder: BiomedCLIP (PubMedBERT) from open_clip
- Fusion: Cross-attention (Text queries -> Image keys/values), masked mean pooling, MLP classifier
- Transforms: No Normalize here (processor handles it); do_rescale=False in VisionEncoder.forward
- Logging: Same style as the RSNA script; save best by Top-5 accuracy
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from torchvision import transforms as T
from transformers import AutoModel, AutoImageProcessor

from open_clip import get_tokenizer, create_model_from_pretrained

# =========================
# Hardcoded paths & hparams
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- FM checkpoint (ema_state) ----
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# ---- MedPix data ----
JSON_PATH = "/home/UWO/ylong66/data/RA/RA/external_data/MedPix/hand_jsn.json"
IMAGE_DIR = "/home/UWO/ylong66/data/RA/RA/external_data/MedPix/image/"

# ---- Training hparams ----
IMG_SIZE = 224
BATCH    = 16
EPOCHS   = 1000
LR       = 1e-4
WEIGHT_DECAY = 1e-4

FREEZE_VISION = False   # True: freeze the vision encoder and train only the fusion/classification heads
FREEZE_TEXT   = True    # True: freeze the text encoder

MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

SEED = 42
VAL_SPLIT = 0.1

def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)

# =========================
# Transforms (no Normalize)
# =========================
train_tf = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),            # keep [0,1], normalization inside processor
])

val_tf = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
])

# =========================
# Vision Encoder (DINOv3)
# =========================
class VisionEncoder(nn.Module):
    """
    - Accept [B,3,H,W] float in [0,1]
    - Internally uses AutoImageProcessor (do_rescale=False) to normalize/resize
    - Returns [B, 1+P, D] (CLS + patches), with register tokens removed if present
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
    log(f"[CKPT] Loaded ema_state from: {ckpt_path}")
    if missing:
        log(f"[CKPT] missing_keys: {len(missing)} (first 8): {missing[:8]}")
    if unexpected:
        log(f"[CKPT] unexpected_keys: {len(unexpected)} (first 8): {unexpected[:8]}")

# =========================
# Dataset & Collate
# =========================
class MedPixVQADataset(Dataset):
    """
    Each sample: (image, caption) -> label (title)
    JSON schema assumed like the expected data-preparation code:
      e["image"]["id"]   -> jpg filename (with .jpg)
      e["image"]["caption"]
      e["title"]         -> class label
    """
    def __init__(self, json_path: str, image_dir: str, transform, tokenizer, label2id=None, max_text_len=64):
        with open(json_path, "r") as f:
            entries = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer  # open_clip tokenizer wrapper (hf-based)
        self.max_text_len = max_text_len

        if label2id is None:
            label_set = sorted({(e.get("title") or "").strip() for e in entries if (e.get("title") or "").strip()})
            self.label2id = {lab: i for i, lab in enumerate(label_set)}
        else:
            self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.samples = []
        kept, skipped = 0, 0
        for e in entries:
            try:
                img_name = e["image"]["id"] + ".jpg"
                img_path = os.path.join(self.image_dir, img_name)
                if not os.path.exists(img_path):
                    skipped += 1
                    continue

                caption = (e["image"].get("caption") or "").strip()
                title = (e.get("title") or "").strip()
                if not caption or title not in self.label2id:
                    skipped += 1
                    continue

                label = self.label2id[title]
                self.samples.append((img_path, caption, label))
                kept += 1
            except Exception:
                skipped += 1
                continue

        log(f"Loaded {kept} VQA samples | {len(self.label2id)} unique labels | skipped={skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Use underlying HF tokenizer via open_clip wrapper
        toks = self.tokenizer.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_text_len,
            padding=False,
            return_tensors="pt"
        )
        # squeeze batch dim from HF
        input_ids = toks["input_ids"].squeeze(0)
        attention_mask = toks["attention_mask"].squeeze(0)

        return {
            "image": image,
            "text": {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            },
            "label": torch.tensor(label, dtype=torch.long)
        }

def vqa_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = [b["image"] for b in batch]
    labels = [b["label"] for b in batch]
    input_ids = [b["text"]["input_ids"] for b in batch]
    attention = [b["text"]["attention_mask"] for b in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention = pad_sequence(attention, batch_first=True, padding_value=0)

    return {
        "image": torch.stack(images, dim=0),
        "text": {
            "input_ids": input_ids,
            "attention_mask": attention
        },
        "label": torch.stack(labels, dim=0)
    }

# =========================
# Model (Cross-Attention)
# =========================
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q, kdim=dim_kv, vdim=dim_kv,
            num_heads=num_heads, batch_first=False
        )

    def forward(self, query, key_value):
        # query: [B, Lq, Dq], key_value: [B, Lkv, Dkv]
        q = query.transpose(0, 1)       # [Lq, B, Dq]
        kv = key_value.transpose(0, 1)  # [Lkv, B, Dkv]
        out, _ = self.attn(q, kv, kv)   # [Lq, B, Dq]
        return out.transpose(0, 1)      # [B, Lq, Dq]

def masked_mean_pooling(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    z:    [B, L, D]
    mask: [B, L] (1=valid, 0=pad)
    """
    mask = mask.unsqueeze(-1).float()           # [B, L, 1]
    summed = torch.sum(z * mask, dim=1)         # [B, D]
    counts = mask.sum(dim=1).clamp(min=1e-6)    # [B, 1]
    return summed / counts

class MedPixVQAModel(nn.Module):
    def __init__(self, vision_encoder: VisionEncoder, num_classes: int,
                 freeze_text: bool = True, freeze_vision: bool = False, num_heads: int = 8):
        super().__init__()
        self.vision_encoder = vision_encoder

        # Load BiomedCLIP (PubMedBERT text)
        clip_model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.text_encoder = clip_model.text  # has .transformer for HF encoder

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.hidden_dim = vision_encoder.hidden_size
        self.cross_attn = CrossAttention(dim_q=self.hidden_dim, dim_kv=self.hidden_dim, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, num_classes)
        )

    def forward(self, image: torch.Tensor, text_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        image: [B,3,224,224]  in [0,1]
        text_dict: {'input_ids':[B,L], 'attention_mask':[B,L]}
        """
        input_ids = text_dict["input_ids"]
        attention_mask = text_dict["attention_mask"]

        # Vision
        z_img = self.vision_encoder(image)  # [B, 1+P, D]

        # Text (HF transformer inside open_clip.text)
        out = self.text_encoder.transformer(input_ids=input_ids, attention_mask=attention_mask)
        z_txt = out.last_hidden_state       # [B, L, D]

        # Cross-attention: text queries attend over image tokens
        z_cross = self.cross_attn(z_txt, z_img)            # [B, L, D]
        z = masked_mean_pooling(z_cross, attention_mask)   # [B, D]

        return self.classifier(z)                          # [B, C]

# =========================
# Metrics
# =========================
@torch.no_grad()
def topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    topk = logits.topk(k, dim=1).indices
    match = (topk == labels.unsqueeze(1))
    return match.any(dim=1).float().mean().item()

# =========================
# Train / Eval loops
# =========================
def train_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        img = batch["image"].to(DEVICE, non_blocking=True)
        lbl = batch["label"].to(DEVICE, non_blocking=True)
        text_dict = {
            "input_ids": batch["text"]["input_ids"].to(DEVICE, non_blocking=True),
            "attention_mask": batch["text"]["attention_mask"].to(DEVICE, non_blocking=True)
        }

        optimizer.zero_grad(set_to_none=True)
        logits = model(img, text_dict)
        loss = loss_fn(logits, lbl)
        loss.backward()
        optimizer.step()

        total += loss.item() * img.size(0)
        n += img.size(0)
    return total / max(1, n)

@torch.no_grad()
def evaluate(model, loader) -> Tuple[float, float, float]:
    model.eval()
    total, sum_acc1, sum_acc5, sum_loss = 0, 0.0, 0.0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        img = batch["image"].to(DEVICE, non_blocking=True)
        lbl = batch["label"].to(DEVICE, non_blocking=True)
        text_dict = {
            "input_ids": batch["text"]["input_ids"].to(DEVICE, non_blocking=True),
            "attention_mask": batch["text"]["attention_mask"].to(DEVICE, non_blocking=True)
        }
        logits = model(img, text_dict)
        loss = loss_fn(logits, lbl)
        bsz = img.size(0)
        total += bsz
        sum_loss += loss.item() * bsz
        sum_acc1 += topk_acc(logits, lbl, k=1) * bsz
        sum_acc5 += topk_acc(logits, lbl, k=5) * bsz
    avg_loss = sum_loss / max(1, total)
    acc1 = sum_acc1 / max(1, total)
    acc5 = sum_acc5 / max(1, total)
    return avg_loss, acc1, acc5

# =========================
# Main
# =========================
def main():
    # ---- Data & tokenizer ----
    tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    full_ds = MedPixVQADataset(
        json_path=JSON_PATH,
        image_dir=IMAGE_DIR,
        transform=val_tf,      # Optional: use train_tf for training; keep val_tf here for consistency with the original run
        tokenizer=tokenizer
    )

    # Fixed split
    generator = torch.Generator().manual_seed(SEED)
    total_len = len(full_ds)
    val_len = round(VAL_SPLIT * total_len)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=generator)
    log(f"Split dataset: {train_len} train / {val_len} val")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=vqa_collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=vqa_collate_fn)

    # ---- Vision encoder & load FM ----
    encoder = VisionEncoder(device=DEVICE).to(DEVICE)
    load_student_from_ckpt(encoder, CKPT_PATH)

    # Freeze policy
    if FREEZE_VISION:
        for p in encoder.parameters():
            p.requires_grad = False
        log("Vision encoder frozen.")
    else:
        log("Vision encoder will be fine-tuned.")

    # ---- Model ----
    num_classes = len(full_ds.label2id)
    model = MedPixVQAModel(
        vision_encoder=encoder,
        num_classes=num_classes,
        freeze_text=FREEZE_TEXT,
        freeze_vision=FREEZE_VISION,
        num_heads=8
    ).to(DEVICE)

    # ---- Optim / Loss ----
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_top5 = -1.0
    best_path = None

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, acc1, acc5 = evaluate(model, val_loader)

        log(f"Epoch {ep:03d}/{EPOCHS} | "
            f"TrainLoss={tr_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"Top1={acc1:.4f} Top5={acc5:.4f} | Time {time.time()-t0:.1f}s")

        if acc5 > best_top5:
            best_top5 = acc5
            save_path = os.path.join(
                MODEL_SAVE_DIR,
                f"medpix_vqa_close_30_ep{ep}_top1_{acc1:.4f}_top5_{acc5:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            best_path = save_path
            log(f"Saved best model to {save_path}")

    log(f"Training finished. Best Top-5={best_top5:.4f} | ckpt={best_path}")

if __name__ == "__main__":
    main()

