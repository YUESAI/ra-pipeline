#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RHPE Bone Age Prediction (Downstream, EMA-as-encoder, Muon-ready)
- Encoder: EMA weights from your multi-expert pretrain FM
- Token: selectable -> 'cls' | 'patch_mean' | 'cls_patch_cat'
- Transforms: ToTensor(); processor do_rescale=False
- Optim: Muon (if available) + Aux AdamW, fallback to AdamW
- Logging: epoch-level only
"""

import os
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from scipy.stats import pearsonr

# ====== Optional Muon ======
_USE_MUON = True
try:
    from muon import MuonWithAuxAdam
except Exception:
    _USE_MUON = False

# =========================
# Paths & hparams
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

# Foundation model checkpoint. Prefer ema_state for the paper setting.
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

DINO_LOCAL_PATH = "/home/UWO/ylong66/data/RA/LLM/hf_model/dinov3-vitb16"
LOAD_EMA = True

# RHPE data paths.
TRAIN_IMG_DIR = "/home/UWO/ylong66/data/RA/RA/external_data/RHPE/RHPE_train"
VAL_IMG_DIR   = "/home/UWO/ylong66/data/RA/RA/external_data/RHPE/RHPE_val"
TRAIN_CSV     = "/home/UWO/ylong66/data/RA/RA/external_data/RHPE/RHPE_Annotations/RHPE_Boneage_train.csv"
VAL_CSV       = "/home/UWO/ylong66/data/RA/RA/external_data/RHPE/RHPE_Annotations/RHPE_Boneage_val.csv"

# Training hyperparameters.
IMG_SIZE = 224
BATCH    = 64
EPOCHS   = 100
LR       = 1e-5
WEIGHT_DECAY = 1e-5
USE_META = True            # Use the sex covariate.
FREEZE_ENCODER = True      # Train only the task head when True.
USE_AMP = True             # Mixed precision.
USE_MUON = True            # Use Muon if available.
NUM_WORKERS = 6

# Representation mode: 'cls' | 'patch_mean' | 'cls_patch_cat'.
TOKEN_MODE = "cls"

MODEL_SAVE_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)

# =========================
# Transforms
# =========================
train_tf = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),          # [0,1] tensor; processor will NOT rescale
])

# val_tf = T.Compose([
#     T.Resize((256, 256)),
#     T.CenterCrop(IMG_SIZE),
#     T.ToTensor(),
# ])

val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])


# =========================
# RHPE Dataset
# =========================
class RHPEDataset(Dataset):
    """
    Read RHPE annotations with columns ID, Male, Boneage, and Chronological.
      - ID: linked to image filenames; supports recursive .png/.jpg/.jpeg search.
      - Male: True/False, also compatible with 0/1 and M/F values.
      - Boneage: months, normalized by /240 to [0,1].
      - Chronological: retained for future extensions but not used here.
    """
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

    def __init__(self, image_dir: str, csv_path: str, transform=None, verbose=True):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        # Standardize column names.
        def norm(c: str) -> str:
            c = c.strip().lower().replace("_", " ").replace("（", "(").replace("）", ")")
            c = " ".join(c.split())
            return c
        df.rename(columns={c: norm(c) for c in df.columns}, inplace=True)

        # Expected fields, for example: ID,Male,Boneage,Chronological.
        name_map = {}
        for c in df.columns:
            if c in ["id"]:
                name_map[c] = "id"
            elif c in ["male", "sex", "gender"]:
                name_map[c] = "male"
            elif c in ["boneage", "bone age", "bone age (months)", "bone age months"]:
                name_map[c] = "boneage"
            elif c in ["chronological", "chronological age", "chronological months"]:
                name_map[c] = "chronological"
        df.rename(columns=name_map, inplace=True)

        for req in ["id", "male", "boneage"]:
            if req not in df.columns:
                raise RuntimeError(f"CSV is missing required column '{req}'. Detected columns: {list(df.columns)}")

        # Convert IDs to extension-free basenames.
        def to_id_str(v):
            s = str(v).strip()
            # Example: "123.png" -> "123".
            for ext in self.IMG_EXTS:
                if s.endswith(ext):
                    s = s[: -len(ext)]
                    break
            return s

        df["id_str"] = df["id"].apply(to_id_str)

        def to_male01(v):
            if pd.isna(v):
                return 0.0
            if isinstance(v, (int, float, np.number)):
                # Numeric 0/1 fallback.
                return float(v)
            s = str(v).strip().lower()
            if s in ["1", "true", "t", "yes", "y", "male", "m"]:
                return 1.0
            if s in ["0", "false", "f", "no", "n", "female", "f"]:
                return 0.0
            return 1.0 if s.startswith("t") or s.startswith("m") else 0.0

        def to_age01(v):
            # Months to [0,1], using 240 months (20 years) as the upper bound.
            return float(v) / 240.0

        df["male01"] = df["male"].apply(to_male01)
        df["boneage01"] = df["boneage"].apply(to_age01)

        # Optionally keep chronological age for future use.
        if "chronological" in df.columns:
            df["chron01"] = df["chronological"].apply(lambda x: float(x) / 240.0)
        else:
            df["chron01"] = 0.0

        self.df = df.reset_index(drop=True)
        self._image_cache: Dict[str, str] = {}

        if verbose:
            head_cols = ["id", "id_str", "male", "male01", "boneage", "boneage01"]
            if "chronological" in df.columns:
                head_cols += ["chronological", "chron01"]
            head = self.df.head(5)[head_cols]
            log(f"[RHPEDataset] Parsed columns successfully. Sample:\n{head}")

    def __len__(self):
        return len(self.df)

    def _build_index_if_needed(self):
        if self._image_cache:
            return
        for root, _, files in os.walk(self.image_dir):
            for fn in files:
                lower = fn.lower()
                if lower.endswith((".png", ".jpg", ".jpeg")):
                    stem = os.path.splitext(fn)[0]
                    self._image_cache[stem] = os.path.join(root, fn)

    def _resolve_image_path(self, id_str: str) -> Optional[str]:
        # Try the direct path first.
        for ext in self.IMG_EXTS:
            p = os.path.join(self.image_dir, id_str + ext)
            if os.path.exists(p):
                return p
        # Build a global filename index.
        self._build_index_if_needed()
        if id_str in self._image_cache:
            return self._image_cache[id_str]
        # Then try zero-padded variants, e.g. 1 -> 0001, for widths 2 to 6.
        for width in range(2, 7):
            key = str(id_str).zfill(width)
            if key in self._image_cache:
                return self._image_cache[key]
        return None

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(str(row["id_str"]))
        if img_path is None:
            raise FileNotFoundError(f"Image not found for id='{row['id_str']}' under {self.image_dir}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # [3,H,W] in [0,1]

        bone_age = torch.tensor(float(row["boneage01"]), dtype=torch.float32)
        is_male  = torch.tensor(float(row["male01"]),    dtype=torch.float32)

        return image, bone_age, is_male

# =========================
# Vision encoder, consistent with the RSNA version.
# =========================
try:
    from transformers import AutoFeatureExtractor as _AutoFE
except Exception:
    _AutoFE = None

class VisionEncoder(nn.Module):
    """
    - Force local loading with local_files_only=True.
    - Fall back to AutoFeatureExtractor if preprocessor_config.json is unavailable.
    - Inputs are in [0,1]; the processor runs on CPU and then moves tensors to the encoder device.
    - Remove register tokens and keep only [CLS + patches].
    """
    def __init__(self, model_path_or_name=DINO_LOCAL_PATH, local_files_only=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path_or_name, local_files_only=local_files_only
        ).eval()

        self.processor = None
        _proc_err = None
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_path_or_name, local_files_only=local_files_only
            )
        except Exception as e:
            _proc_err = e
            if _AutoFE is not None:
                try:
                    self.processor = _AutoFE.from_pretrained(
                        model_path_or_name, local_files_only=local_files_only
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load image processor from local path '{model_path_or_name}'. "
                        f"AutoImageProcessor err: {repr(_proc_err)} | AutoFeatureExtractor err: {repr(e2)}"
                    )
            else:
                raise RuntimeError(
                    f"Failed to load image processor from local path '{model_path_or_name}'. "
                    f"AutoImageProcessor err: {repr(_proc_err)} and AutoFeatureExtractor is unavailable."
                )

        cfg = getattr(self.encoder, "config", None)
        self.hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "hidden_dim", None)
            or getattr(cfg, "embed_dim", None)
            or getattr(cfg, "width", None)
        )
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    @torch.no_grad()
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        try:
            inp = self.processor(images=[xi.cpu() for xi in x], return_tensors="pt", do_rescale=False)
        except TypeError:
            inp = self.processor(images=[xi.cpu() for xi in x], return_tensors="pt")
        dev = next(self.encoder.parameters()).device
        return inp["pixel_values"].to(dev, non_blocking=True)

    def forward(self, x) -> torch.Tensor:
        pixel_values = self._process(x)
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D]
        if self.num_register_tokens and self.num_register_tokens > 0:
            cls_tok = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_register_tokens:, :]
            tokens = torch.cat([cls_tok, patches], dim=1)  # [B, 1+P, D]
        return tokens

def load_foundation_encoder(encoder: VisionEncoder, ckpt_path: str, prefer_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = None
    if prefer_ema and ("ema_state" in ckpt) and (ckpt["ema_state"] is not None):
        state = ckpt["ema_state"]
        log(f"[CKPT] Using ema_state from: {ckpt_path}")
    else:
        state = ckpt.get("student_state", None)
        log(f"[CKPT] Using student_state from: {ckpt_path}")
    if state is None:
        raise RuntimeError("Neither ema_state nor student_state found in checkpoint.")

    # Strip the optional 'encoder.' prefix.
    if any(k.startswith("encoder.") for k in state.keys()):
        state = {k.replace("encoder.", "", 1): v for k, v in state.items()}

    missing, unexpected = encoder.encoder.load_state_dict(state, strict=False)
    if missing:
        log(f"[CKPT] missing_keys: {len(missing)} (first 8) {missing[:8]}")
    if unexpected:
        log(f"[CKPT] unexpected_keys: {len(unexpected)} (first 8) {unexpected[:8]}")

# =========================
# Heads (token mode aware)
# =========================
def _feat_from_tokens(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    cls = tokens[:, 0]
    patches = tokens[:, 1:]
    if mode == "cls":
        return cls
    elif mode == "patch_mean":
        return patches.mean(dim=1)
    elif mode == "cls_patch_cat":
        return torch.cat([cls, patches.mean(dim=1)], dim=1)
    else:
        raise ValueError(f"Unsupported TOKEN_MODE: {mode}")

class BoneAgeModel(nn.Module):
    """Image (+ optional gender) → bone age"""
    def __init__(self, encoder: VisionEncoder, token_mode: str = "cls", use_meta: bool = True):
        super().__init__()
        self.encoder = encoder
        self.token_mode = token_mode
        D = encoder.hidden_size
        in_dim = D if token_mode in ("cls", "patch_mean") else (2 * D)
        if use_meta:
            self.meta_proj = nn.Linear(1, D)
            head_in = in_dim + D
        else:
            self.meta_proj = None
            head_in = in_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x, gender=None):
        tokens = self.encoder(x)                       # [B, 1+P, D]
        feat_img = _feat_from_tokens(tokens, self.token_mode)
        if self.meta_proj is not None and gender is not None:
            feat_meta = self.meta_proj(gender.view(-1, 1))
            feat = torch.cat([feat_img, feat_meta], dim=1)
        else:
            feat = feat_img
        out = self.head(feat).squeeze(1)              # [B]
        return out

# =========================
# Metrics reported in months.
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
# Optim helpers (Muon/AdamW).
# =========================
def _split_params_for_muon(m: nn.Module):
    hidden_weights, hidden_gains_biases = [], []
    for _, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            hidden_weights.append(p)
        else:
            hidden_gains_biases.append(p)
    return hidden_weights, hidden_gains_biases

def build_optimizer(model: nn.Module):
    if USE_MUON and _USE_MUON:
        hw, hb = _split_params_for_muon(model)
        param_groups = []
        if len(hw) > 0:
            param_groups.append(dict(params=hw, use_muon=True, lr=LR, weight_decay=WEIGHT_DECAY))
        if len(hb) > 0:
            param_groups.append(dict(params=hb, use_muon=False, lr=LR, betas=(0.9, 0.95),
                                     weight_decay=WEIGHT_DECAY))
        opt = MuonWithAuxAdam(param_groups)
        log(f"[OPT] Muon enabled: muon_tensors={len(hw)} aux_tensors={len(hb)}")
        return opt
    else:
        if USE_MUON and not _USE_MUON:
            log("[WARN] Muon not available; fallback to AdamW.")
        return optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =========================
# Train / Val loops
# =========================
from torch.amp import GradScaler, autocast

def train_epoch(model, loader, optimizer, loss_fn, scaler: GradScaler | None) -> float:
    model.train()
    total = 0.0
    n = 0
    use_amp = (USE_AMP and DEVICE == "cuda")
    for img, age, gender in loader:
        img = img.to(DEVICE, non_blocking=True)
        age = age.to(DEVICE, non_blocking=True)
        gender = gender.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast("cuda", dtype=torch.bfloat16, enabled=True):
                pred = model(img, gender if USE_META else None)
                loss = loss_fn(pred, age)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(img, gender if USE_META else None)
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
    use_amp = (USE_AMP and DEVICE == "cuda")
    for img, age, gender in loader:
        img = img.to(DEVICE, non_blocking=True)
        age = age.to(DEVICE, non_blocking=True)
        gender = gender.to(DEVICE, non_blocking=True)
        if use_amp:
            with autocast("cuda", dtype=torch.bfloat16, enabled=True):
                pred = model(img, gender if USE_META else None)
        else:
            pred = model(img, gender if USE_META else None)
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(age.detach().cpu().tolist())
    return eval_metrics(preds, targets, name="Validation")

# =========================
# Main
# =========================
def main():
    # Data
    train_ds = RHPEDataset(TRAIN_IMG_DIR, TRAIN_CSV, transform=train_tf)
    val_ds   = RHPEDataset(VAL_IMG_DIR,   VAL_CSV,   transform=val_tf)

    pin_mem = (DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem,
                              persistent_workers=(NUM_WORKERS > 0), drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem,
                              persistent_workers=(NUM_WORKERS > 0))

    # Encoder & load FM
    encoder = VisionEncoder(model_path_or_name=DINO_LOCAL_PATH, local_files_only=True)
    load_foundation_encoder(encoder, CKPT_PATH, prefer_ema=LOAD_EMA)

    # Model & device
    model = BoneAgeModel(encoder, token_mode=TOKEN_MODE, use_meta=USE_META).to(DEVICE)
    start_epoch = 0
    best_mad = float("inf")


    # model_ckpt_path = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema/rhpe_boneage_meta_ema_cls_ep95_MAD6.6562_PCC0.9603.pt"
    # model.load_state_dict(torch.load(model_ckpt_path, map_location=DEVICE))
    # log(f"Model initialized from {model_ckpt_path}")
    # start_epoch = 95
    # best_mad = 0.4835

    # Freeze strategy.
    if FREEZE_ENCODER:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()
        log("Encoder frozen (feature extractor).")
    else:
        log("Encoder will be fine-tuned.")

    # Optim / Loss
    optimizer = build_optimizer(model)
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    scaler = GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))


    for ep in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler)
        rmse, mad, mae, r2, pcc = evaluate(model, val_loader)

        log(f"Epoch {ep:03d}/{EPOCHS} | TrainLoss={train_loss:.4f} | "
            f"Val MAD={mad:.3f} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f} PCC={pcc:.3f} | "
            f"Time {time.time()-t0:.1f}s")

        if mad < best_mad:
            best_mad = mad
            tag = ("meta" if USE_META else "img")
            ema_tag = ("ema" if LOAD_EMA else "student")
            save_path = os.path.join(
                MODEL_SAVE_DIR,
                f"rhpe_boneage_{tag}_{ema_tag}_{TOKEN_MODE}_ep{ep}_MAD{best_mad:.4f}_PCC{pcc:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            log(f"✅ Saved best model to {save_path}")

    log("Training finished.")

if __name__ == "__main__":
    print("Torch:", torch.__version__, " CUDA:", torch.cuda.is_available())
    main()
