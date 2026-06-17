#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HandX Whole-Image Masked Latent Alignment Pretraining (multi-expert, EMA, edge-aware masking)
Fixed version:
- Projection heads are pre-registered (LazyLinear) so they are optimized.
- Per-expert losses normalized by number of experts.
- Two LR groups: encoder vs (regressor + proj-heads); optional warmup-freeze.
- Avoid double rescale: pass do_rescale=False to HF processors; keep input x in [0,1].
- More transparent logging of loss components.
"""

import os
import sys
import copy
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
from timm.models.layers import DropPath, trunc_normal_

sys.path.append('/home/UWO/ylong66/workplace/RA/repo/Muon/')

torch.set_num_threads(2)

# ========== optional Muon ==========
_USE_MUON = True
try:
    from muon import MuonWithAuxAdam
    print("use muon")
except Exception:
    _USE_MUON = False
    print("use adam")


# =========================
# Data & Augment
# =========================
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10, fill=0),
    transforms.CenterCrop(224),
    transforms.ToTensor(),               # [0,1]
])

class WholeHandXDataset(Dataset):
    def __init__(self, roots: List[Path], tfm):
        self.transforms = tfm
        self.paths = []
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for r in roots:
            r = Path(r)
            if not r.exists():
                print(f"[WARN] Missing root: {r}", file=sys.stderr)
                continue
            self.paths += sorted([
                p for p in r.rglob("*")
                if p.is_file()
                and p.suffix.lower() in exts
                and "foot" not in p.name.lower()
                and "feet" not in p.name.lower()
            ])
        if not self.paths:
            raise RuntimeError("No valid hand images found under data_roots")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read {path}: {e}")
        img = self.transforms(img)  # [3,H,W] in [0,1]
        return img


# =========================
# Student (DINOv3 HF)
# =========================
class VisionEncoder(nn.Module):
    """
    Wrapper for HF ViT (e.g., DINOv3).
    - Uses AutoImageProcessor for normalization
    - Output [B,197,D] (CLS + patches), strip any register tokens.
    """
    def __init__(self, model_path_or_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
                 local_files_only=False, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(
            model_path_or_name, local_files_only=local_files_only, use_fast=True
        )
        self.encoder = AutoModel.from_pretrained(
            model_path_or_name, local_files_only=local_files_only
        ).to(self.device)

        cfg = getattr(self.encoder, "config", None)
        self.hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "hidden_dim", None)
            or getattr(cfg, "embed_dim", None)
            or getattr(cfg, "width", None)
        )
        name_lower = str(model_path_or_name).lower()
        self.is_vit = ("vit" in name_lower)
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0) if cfg is not None else 0

    def forward(self, x, return_pooled=False):
        """
        x: [B,3,H,W] in [0,1]
        """
        assert x.dim() == 4 and x.size(1) == 3
        # IMPORTANT: avoid double rescale. We pass do_rescale=False explicitly.
        inputs = self.processor(images=[xi.cpu() for xi in x], return_tensors="pt", do_rescale=False)
        dev = next(self.encoder.parameters()).device
        pixel_values = inputs["pixel_values"].to(dev)

        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=False)

        if return_pooled:
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                last_hidden = outputs.last_hidden_state
                if self.is_vit and last_hidden.size(1) >= 2:
                    pooled = last_hidden[:, 1:, :].mean(dim=1)  # drop CLS
                else:
                    pooled = last_hidden.mean(dim=1)
            return pooled  # [B, D]

        tokens = outputs.last_hidden_state  # [B, N_tokens, D]
        n_reg = self.num_register_tokens if self.num_register_tokens is not None else 0
        cls = tokens[:, :1, :]
        patches = tokens[:, (1 + n_reg):, :]
        tokens_197 = torch.cat([cls, patches], dim=1)  # [B, 197, D]
        return tokens_197


# =========================
# Teachers (BiomedCLIP + Chest HF local)
# =========================
from open_clip import create_model_and_transforms

class CLIPTeacher(nn.Module):
    """
    BiomedCLIP visual trunk as teacher; uses its preprocess.
    Returns [B,197,D_t] (or longer; caller will trim to 196 patches).
    """
    def __init__(self, cache_dir: str = None, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            cache_dir=cache_dir
        )
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        imgs = [self.preprocess(transforms.ToPILImage()(xi.cpu())).unsqueeze(0) for xi in x]
        px = torch.cat(imgs, dim=0)
        dev = next(self.model.parameters()).device
        px = px.to(dev)
        out = self.model.visual.trunk.forward_features(px)
        if isinstance(out, dict):
            feats = out.get('x', out.get('tokens', None))
            if feats is None:
                feats = self.model.encode_image(px, proj=False).unsqueeze(1)
        elif isinstance(out, torch.Tensor):
            feats = out
        else:
            raise RuntimeError("Unsupported output from CLIP visual trunk.")
        return feats  # [B, 197?, D_t]


class ChestXrayHFExpert(nn.Module):
    """
    Local HF ViT (chest), outputs [B,1+P,D]; caller will drop CLS.
    """
    def __init__(self, model_path: str, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep do_rescale=False to avoid double scaling.
        inputs = self.processor(images=[xi.cpu() for xi in x], return_tensors="pt", do_rescale=False)
        dev = next(self.encoder.parameters()).device
        pixel_values = inputs["pixel_values"].to(dev)
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        return outputs.last_hidden_state  # [B, 1+P, D]


# =========================
# Masking + Regressor
# =========================
class PatchMasker(nn.Module):
    """
    mask_type: 'random' or 'edge'
    """
    def __init__(self, num_patches=196, mask_ratio=0.5, hw=14,
                 mask_type: str = "random", mask_edge_alpha: float = 0.5):
        super().__init__()
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.hw = hw
        self.mask_type = mask_type
        self.mask_edge_alpha = mask_edge_alpha

        # Sobel kernels for edge mode
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    def _random_mask_batch(self, B: int, device) -> torch.Tensor:
        num_mask = int(self.num_patches * self.mask_ratio)
        masks = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        for i in range(B):
            perm = torch.randperm(self.num_patches, device=device)
            masks[i, perm[:num_mask]] = 1
        return masks

    @torch.no_grad()
    def _edge_mask_batch(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        B = x.size(0)
        num_mask = int(self.num_patches * self.mask_ratio)

        gray = x.mean(1, keepdim=True)  # [B,1,H,W]
        gx = F.conv2d(gray, self.kx.to(device), padding=1)
        gy = F.conv2d(gray, self.ky.to(device), padding=1)
        mag = (gx**2 + gy**2).sqrt()    # [B,1,H,W]

        mag_grid = F.adaptive_avg_pool2d(mag, (self.hw, self.hw)).view(B, -1)  # [B,196]
        prob_edge = torch.softmax(mag_grid, dim=1)

        masks = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        uni = torch.ones(self.num_patches, device=device) / self.num_patches
        for i in range(B):
            prob = self.mask_edge_alpha * uni + (1 - self.mask_edge_alpha) * prob_edge[i]
            idx = torch.multinomial(prob, num_samples=num_mask, replacement=False)
            masks[i, idx] = 1
        return masks

    def forward(self, B: int, x: torch.Tensor = None) -> torch.Tensor:
        if self.mask_type == "edge":
            if x is None:
                raise ValueError("edge-aware masking requires input x")
            return self._edge_mask_batch(x)
        else:
            device = x.device if x is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            return self._random_mask_batch(B, device=device)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x_q, k, v, key_padding_mask=None):
        B, N_q, C = x_q.shape
        _, N_kv, _ = k.shape
        q = self.q(x_q).reshape(B, N_q, self.num_heads, -1).transpose(1, 2)
        k = self.k(k).reshape(B, N_kv, self.num_heads, -1).transpose(1, 2)
        v = self.v(v).reshape(B, N_kv, self.num_heads, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.proj(out)
        return self.proj_drop(out)

class RegressorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=1e-5, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
    def forward(self, x_q, x_kv, pos_q, pos_k, key_padding_mask=None):
        x = x_q + self.drop_path(self.gamma_1 * self.cross_attn(
            self.norm1_q(x_q + pos_q),
            k=self.norm1_k(x_kv + pos_k),
            v=self.norm1_v(x_kv),
            key_padding_mask=key_padding_mask
        ))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class MaskedTokenRegressor(nn.Module):
    def __init__(self, dim=768, depth=6, num_heads=8, mlp_ratio=4., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, init_values=1e-5, init_std=0.02, hw=14):
        super().__init__()
        self.hw = hw
        self.dim = dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.mask_token, std=init_std)
        pos = self.build_sincos_2d_position_embedding(dim=dim, hw=hw)  # [196,D]
        self.register_buffer("pos_embed", pos, persistent=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            RegressorBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)
        ])
        self.norm = norm_layer(dim)

    @staticmethod
    def build_sincos_2d_position_embedding(dim=768, hw=14, temperature=10000.):
        grid_w = torch.arange(hw, dtype=torch.float32)
        grid_h = torch.arange(hw, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert dim % 4 == 0
        pos_dim = dim // 4
        omega = 1. / (temperature ** (torch.arange(pos_dim).float() / pos_dim))
        out_w = torch.einsum('m,d->md', grid_w.flatten(), omega)
        out_h = torch.einsum('m,d->md', grid_h.flatten(), omega)
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w),
                             torch.sin(out_h), torch.cos(out_h)], dim=1)
        return pos_emb  # [196, D]

    def forward(self, z_visible: torch.Tensor, idx_mask: torch.Tensor, idx_vis: torch.Tensor, len_vis: torch.Tensor) -> torch.Tensor:
        B, Lv, D = z_visible.shape
        Lm = idx_mask.size(1)
        pos_masked = torch.stack([ self.pos_embed[idx_mask[b]] for b in range(B) ], dim=0)  # [B,Lm,D]
        pos_visible = torch.stack([ self.pos_embed[idx_vis[b]] for b in range(B) ], dim=0)  # [B,Lv,D]
        masked_tokens = self.mask_token.expand(B, Lm, D).clone()
        arange_lv = torch.arange(Lv, device=z_visible.device)[None, :].expand(B, -1)
        kv_pad_mask = arange_lv >= len_vis[:, None]  # [B,Lv]
        for blk in self.blocks:
            masked_tokens = blk(masked_tokens, z_visible, pos_masked, pos_visible, key_padding_mask=kv_pad_mask)
        return self.norm(masked_tokens)  # [B,Lm,D]


# =========================
# Loss
# =========================
class MaskedLatentAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    def forward(self, z_student_vis, z_student_mask, z_teacher_vis, z_teacher_mask):
        return self.criterion(z_student_vis, z_teacher_vis) + self.criterion(z_student_mask, z_teacher_mask)


# =========================
# Top model (multi-expert + EMA + CLS distill)
# =========================
class MaskedPretrainingModel(nn.Module):
    def __init__(self, encoder, experts: List[nn.Module], regressor,
                 mask_ratio=0.5, mask_type="random", mask_edge_alpha=0.5,
                 use_ema=True, ema_m=0.996, ema_loss_weight=0.5,
                 cls_loss_weight=0.1, normalize_per_expert=True):
        super().__init__()
        self.encoder = encoder
        self.experts = nn.ModuleList(experts)
        self.regressor = regressor

        self.masker = PatchMasker(mask_ratio=mask_ratio, mask_type=mask_type,
                                  mask_edge_alpha=mask_edge_alpha)

        self.loss_fn = MaskedLatentAlignmentLoss()

        # === Projection heads (pre-registered!) ===
        # Use LazyLinear to ensure params exist & are optimized even before first forward.
        D_s = encoder.hidden_size
        self._proj_heads = nn.ModuleList([
            nn.Identity() if isinstance(exp, nn.Identity) else nn.LazyLinear(D_s)
            for exp in self.experts
        ])

        # EMA (optional)
        self.use_ema = use_ema
        self.ema_m = ema_m
        self.ema_loss_weight = ema_loss_weight
        if self.use_ema:
            self.encoder_m = copy.deepcopy(encoder).eval()
            for p in self.encoder_m.parameters(): p.requires_grad = False

        # CLS distill
        self.cls_loss_weight = cls_loss_weight

        # normalize multi-expert losses
        self.normalize_per_expert = normalize_per_expert

    @torch.no_grad()
    def _ema_update(self):
        if not self.use_ema:
            return
        m = self.ema_m
        for p, pm in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            pm.data.mul_(m).add_(p.data, alpha=(1 - m))

    def _gather_visible(self, z: torch.Tensor, mask_bool: torch.Tensor):
        B, N, D = z.shape
        vis_seqs, idx_vis_list, len_vis = [], [], []
        for b in range(B):
            idx_vis = (~mask_bool[b]).nonzero(as_tuple=True)[0]
            vis_seqs.append(z[b][idx_vis])
            idx_vis_list.append(idx_vis)
            len_vis.append(idx_vis.numel())
        z_vis = nn.utils.rnn.pad_sequence(vis_seqs, batch_first=True)            # [B,Lv,D]
        idx_vis_pad = nn.utils.rnn.pad_sequence(idx_vis_list, batch_first=True)  # [B,Lv]
        len_vis = torch.tensor(len_vis, device=z.device, dtype=torch.long)       # [B]
        return z_vis, idx_vis_pad, len_vis

    def _gather_masked(self, mask_bool: torch.Tensor) -> torch.Tensor:
        idx_mask_list = [(mask_bool[b]).nonzero(as_tuple=True)[0] for b in range(mask_bool.size(0))]
        idx_mask_pad = nn.utils.rnn.pad_sequence(idx_mask_list, batch_first=True)  # [B,Lm]
        return idx_mask_pad

    def forward(self, x) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        num_experts = max(1, len(self.experts))
        norm_scale = (1.0 / num_experts) if self.normalize_per_expert else 1.0

        # sample mask
        mask_bool = self.masker(B, x)  # [B,196] True=masked

        # student tokens
        z_all = self.encoder(x)             # [B,197,D_s]
        z_s = z_all[:, 1:, :]               # [B,196,D_s]

        idx_mask = self._gather_masked(mask_bool)                      # [B,Lm]
        z_s_vis, idx_vis, len_vis = self._gather_visible(z_s, mask_bool)  # [B,Lv,D_s],...

        # predict masked tokens once
        z_s_mask_pred = self.regressor(z_visible=z_s_vis, idx_mask=idx_mask, idx_vis=idx_vis, len_vis=len_vis)

        # losses
        total_loss = 0.0
        log_terms: Dict[str, float] = {}

        # EMA consistency
        if self.use_ema and self.ema_loss_weight > 0:
            with torch.no_grad():
                z_all_m = self.encoder_m(x)
                z_m = z_all_m[:, 1:, :]
                z_m_vis, _, _ = self._gather_visible(z_m, mask_bool)
                z_m_mask_list = [z_m[b][idx_mask[b]] for b in range(B)]
                z_m_mask = nn.utils.rnn.pad_sequence(z_m_mask_list, batch_first=True)
            ema_loss = self.loss_fn(z_s_vis, z_s_mask_pred, z_m_vis, z_m_mask)
            total_loss = total_loss + self.ema_loss_weight * ema_loss
            log_terms["ema"] = float(ema_loss.detach().mean().cpu())

        # per-expert losses + CLS
        cls_s = z_all[:, :1, :]  # student's CLS
        for i, teacher in enumerate(self.experts):
            with torch.no_grad():
                z_t_all = teacher(x)  # [B,1+P,D_t] or [B,197,D_t]
                if z_t_all.dim() == 3 and z_t_all.size(1) >= 197:
                    z_t = z_t_all[:, 1:197, :]
                elif z_t_all.dim() == 3 and z_t_all.size(1) >= 2:
                    z_t = z_t_all[:, 1:, :]
                else:
                    raise RuntimeError("Teacher output shape unexpected; need token sequence.")
                z_t_vis, _, _ = self._gather_visible(z_t, mask_bool)
                z_t_mask_list = [z_t[b][idx_mask[b]] for b in range(B)]
                z_t_mask = nn.utils.rnn.pad_sequence(z_t_mask_list, batch_first=True)
                cls_t = z_t_all[:, :1, :] if z_t_all.size(1) >= 1 else None

            proj = self._proj_heads[i]  # LazyLinear or Identity
            z_t_vis_proj  = proj(z_t_vis)
            z_t_mask_proj = proj(z_t_mask)

            align_loss = self.loss_fn(z_s_vis, z_s_mask_pred, z_t_vis_proj, z_t_mask_proj)
            total_loss = total_loss + norm_scale * align_loss
            log_terms[f"exp{i}_align"] = float(align_loss.detach().mean().cpu())

            if self.cls_loss_weight > 0 and cls_t is not None:
                cls_t_proj = proj(cls_t) if not isinstance(proj, nn.Identity) else cls_t
                cls_loss = F.mse_loss(cls_s, cls_t_proj)
                total_loss = total_loss + norm_scale * self.cls_loss_weight * cls_loss
                log_terms[f"exp{i}_cls"] = float(cls_loss.detach().mean().cpu())

        # expose small log dict for outer loop
        if not hasattr(self, "_last_logs"):
            self._last_logs = {}
        self._last_logs = log_terms

        return total_loss

    def last_logs(self) -> Dict[str, float]:
        return getattr(self, "_last_logs", {})


# =========================
# Training
# =========================
def build_dataloader(roots: List[str], batch_size: int, num_workers: int, pin_memory: bool):
    ds = WholeHandXDataset([Path(r) for r in roots], IMG_TRANSFORMS)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    return loader

def group_parameters_for_opt(model: MaskedPretrainingModel):
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("experts.") or n.startswith("encoder_m."):
            continue
        if n.startswith("encoder."):
            enc_params.append(p)
        else:
            head_params.append(p)
    return enc_params, head_params

def save_checkpoint(save_dir: Path, epoch: int, encoder: nn.Module,
                    encoder_m: Optional[nn.Module], regressor: nn.Module, proj_heads: nn.ModuleList,
                    optimizer, avg_loss: float):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"handx_pretrain_multiexpert_224_{epoch}.pt"
    torch.save({
        "student_state": encoder.state_dict(),
        "ema_state": (encoder_m.state_dict() if encoder_m is not None else None),
        "regressor_state": regressor.state_dict(),
        "proj_heads_state": proj_heads.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "avg_loss": avg_loss
    }, ckpt_path)
    print(f"[CKPT] Saved: {ckpt_path}")
    if encoder_m is not None:
        ema_export = save_dir / f"foundation_encoderM_epoch{epoch}.pth"
        torch.save(encoder_m.state_dict(), ema_export)
        print(f"[EXPORT] EMA encoder saved for downstream: {ema_export}")

def train(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print("="*60)
    print("🔧  Training Configuration Summary")
    print("- Device:", device)
    print(f"- use_ema: {getattr(args, 'use_ema', False)}")
    print(f"- ema_loss_weight: {getattr(args, 'ema_loss_weight', 0.0)}")
    print(f"- cls_loss_weight: {getattr(args, 'cls_loss_weight', 0.0)}")
    print(f"- use_muon: {getattr(args, 'use_muon', False)}")
    print(f"- mask_type: {getattr(args, 'mask_type', 'random')} (ratio={getattr(args, 'mask_ratio', 0.5)}, edge_alpha={getattr(args, 'mask_edge_alpha', 0.5)})")
    print(f"- lr (enc/head): {getattr(args, 'enc_lr', None)} / {getattr(args, 'head_lr', None)}")
    print(f"- freeze_epochs: {getattr(args, 'freeze_epochs', 0)}")
    print(f"- resume_ckpt: {getattr(args, 'resume_ckpt', None)}")
    print("="*60)
    
    # Data
    loader = build_dataloader(args.data_roots, args.batch_size, args.num_workers, pin_memory)

    # Student
    vit_src = args.dinov3_path if args.dinov3_path is not None else args.dinov3_name
    local_only = True if args.dinov3_path is not None else args.local_files_only
    encoder = VisionEncoder(model_path_or_name=vit_src, local_files_only=local_only, device=device)

    # Teachers
    experts = [
        CLIPTeacher(cache_dir=args.hf_cache_dir, device=device),
        ChestXrayHFExpert(model_path=args.chest_local_path, device=device),
    ]

    # Regressor
    regressor = MaskedTokenRegressor(dim=args.reg_dim, depth=args.reg_depth,
                                     num_heads=args.reg_heads, mlp_ratio=args.reg_mlp_ratio,
                                     drop_path_rate=args.reg_drop_path, hw=14).to(device)

    # Model
    model = MaskedPretrainingModel(
        encoder=encoder, experts=experts, regressor=regressor,
        mask_ratio=args.mask_ratio, mask_type=args.mask_type, mask_edge_alpha=args.mask_edge_alpha,
        use_ema=args.use_ema, ema_m=args.ema_m, ema_loss_weight=args.ema_loss_weight,
        cls_loss_weight=args.cls_loss_weight, normalize_per_expert=True
    ).to(device)

        # =========================
    # (可选) 从指定 checkpoint 加载继续训练
    # =========================
    start_epoch = 0
    if getattr(args, "resume_ckpt", None):
        ckpt_path = Path(args.resume_ckpt)
        if ckpt_path.exists():
            print(f"[RESUME] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            # 学生编码器
            if "student_state" in ckpt:
                model.encoder.load_state_dict(ckpt["student_state"], strict=True)
                print("  ✓ loaded student_state")

            # EMA teacher
            if model.use_ema and ckpt.get("ema_state") is not None:
                try:
                    model.encoder_m.load_state_dict(ckpt["ema_state"], strict=True)
                    print("  ✓ loaded ema_state")
                except Exception as e:
                    print(f"  ⚠️ EMA state load failed: {e}")

            # Regressor
            if "regressor_state" in ckpt:
                model.regressor.load_state_dict(ckpt["regressor_state"], strict=True)
                print("  ✓ loaded regressor_state")

            # Projection heads
            if "proj_heads_state" in ckpt:
                model._proj_heads.load_state_dict(ckpt["proj_heads_state"], strict=False)
                print("  ✓ loaded proj_heads_state")

            # Optimizer (optional)
            if "optimizer" in ckpt and not args.reset_optimizer:
                try:
                    optimizer_state = ckpt["optimizer"]
                    if optimizer_state:
                        print("  ✓ will restore optimizer state after build.")
                except Exception:
                    optimizer_state = None
            else:
                optimizer_state = None

            # epoch
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"  → Resume from epoch {start_epoch}")

            del ckpt
            import gc; gc.collect()
        else:
            print(f"[WARN] resume_ckpt not found: {ckpt_path}")
            optimizer_state = None
    else:
        optimizer_state = None


    # Optional freeze encoder for warmup
    if args.freeze_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # Optimizer with two LR groups
    enc_params, head_params = group_parameters_for_opt(model)
    param_groups = []
    if len(enc_params) > 0:
        param_groups.append(dict(params=enc_params, lr=args.enc_lr, weight_decay=args.weight_decay))
    if len(head_params) > 0:
        if _USE_MUON and args.use_muon:
            # Muon for "hidden weights" only is nice, but here we apply Muon wrapper at the optimizer level.
            optimizer = MuonWithAuxAdam([
                dict(params=head_params, use_muon=True,  lr=args.head_lr, weight_decay=args.weight_decay),
                dict(params=enc_params,  use_muon=False, lr=args.enc_lr,  weight_decay=args.weight_decay, betas=(0.9,0.95)),
            ])
        else:
            optimizer = optim.Adam([
                dict(params=enc_params,  lr=args.enc_lr,  weight_decay=args.weight_decay, betas=(0.9,0.999)),
                dict(params=head_params, lr=args.head_lr, weight_decay=args.weight_decay, betas=(0.9,0.999)),
            ])
    else:
        # fallback single group
        if _USE_MUON and args.use_muon:
            optimizer = MuonWithAuxAdam([dict(params=enc_params, use_muon=True, lr=args.enc_lr, weight_decay=args.weight_decay)])
        else:
            optimizer = optim.Adam(enc_params, lr=args.enc_lr, betas=(0.9,0.999), weight_decay=args.weight_decay)

    if args.use_muon and not _USE_MUON:
        print("[WARN] Muon not available, fallback to Adam.")

            # 恢复 optimizer 状态
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("  ✓ optimizer state restored")
        except Exception as e:
            print(f"  ⚠️ failed to restore optimizer state: {e}")

    # Train loop
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == (args.freeze_epochs + 1):
            # unfreeze encoder
            for p in model.encoder.parameters():
                p.requires_grad = True
            # rebuild optimizer so encoder params pick correct lr
            enc_params, head_params = group_parameters_for_opt(model)
            if _USE_MUON and args.use_muon:
                optimizer = MuonWithAuxAdam([
                    dict(params=head_params, use_muon=True,  lr=args.head_lr, weight_decay=args.weight_decay),
                    dict(params=enc_params,  use_muon=False, lr=args.enc_lr,  weight_decay=args.weight_decay, betas=(0.9,0.95)),
                ])
            else:
                optimizer = optim.Adam([
                    dict(params=enc_params,  lr=args.enc_lr,  weight_decay=args.weight_decay, betas=(0.9,0.999)),
                    dict(params=head_params, lr=args.head_lr, weight_decay=args.weight_decay, betas=(0.9,0.999)),
                ])
            print(f"[OPT] Encoder unfrozen at epoch {epoch}.")

        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            loss = model(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model._ema_update()

            total_loss += loss.item()
            steps += 1
            global_step += 1
            if steps % args.log_every == 0:
                logs = model.last_logs()
                extras = " | ".join([f"{k}:{v:.4f}" for k,v in logs.items()])
                print(f"[Epoch {epoch}] Step {steps}  Loss: {loss.item():.4f}" + (f" | {extras}" if extras else ""))

        avg_loss = total_loss / max(steps, 1)
        print(f"===> Epoch {epoch} Avg Loss: {avg_loss:.4f}")

        if (epoch % args.save_every) == 0 or (epoch == args.epochs):
            save_checkpoint(
                Path(args.save_dir), epoch,
                encoder=model.encoder,
                encoder_m=(model.encoder_m if model.use_ema else None),
                regressor=model.regressor,
                proj_heads=model._proj_heads,
                optimizer=optimizer,
                avg_loss=avg_loss
            )


# =========================
# CLI
# =========================
def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="HandX whole-image masked latent alignment pretraining (multi-expert, EMA, edge-aware masking) — fixed")
    # data
    p.add_argument("--data_roots", type=str, nargs="+", required=True, help="one or more data roots")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    # student
    p.add_argument("--dinov3_path", type=str, default=None, help="local DINOv3 directory (preferred if provided)")
    p.add_argument("--dinov3_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="HF model name (used if no local path)")
    p.add_argument("--local_files_only", action="store_true", help="HF local-only")
    p.add_argument("--hf_cache_dir", type=str, default=None, help="optional cache dir for open_clip/HF")
    # experts
    p.add_argument("--chest_local_path", type=str, required=True,
                   help="Chest X-ray HF local path")
    # regressor
    p.add_argument("--reg_dim", type=int, default=768)
    p.add_argument("--reg_depth", type=int, default=6)
    p.add_argument("--reg_heads", type=int, default=8)
    p.add_argument("--reg_mlp_ratio", type=float, default=4.0)
    p.add_argument("--reg_drop_path", type=float, default=0.1)
    # train
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--enc_lr", type=float, default=3e-5, help="LR for encoder")
    p.add_argument("--head_lr", type=float, default=1e-4, help="LR for regressor + proj-heads")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--mask_type", type=str, default="random", choices=["random","edge"], help="masking strategy")
    p.add_argument("--mask_edge_alpha", type=float, default=0.5, help="edge-aware vs uniform mix (0=all edge,1=all uniform)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--freeze_epochs", type=int, default=0, help="freeze encoder for first N epochs")
    # EMA / losses
    p.add_argument("--use_ema", action="store_true", help="enable EMA consistency")
    p.add_argument("--ema_m", type=float, default=0.996, help="EMA momentum")
    p.add_argument("--ema_loss_weight", type=float, default=0.0, help="EMA consistency loss weight")
    p.add_argument("--cls_loss_weight", type=float, default=0.0, help="CLS distillation weight per expert (pre-normalization)")
    # optimizer
    p.add_argument("--use_muon", action="store_true", help="use MuonWithAuxAdam if available")
    # save
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=10)

    # resume
    p.add_argument("--resume_ckpt", type=str, default=None, help="path to checkpoint to resume from")
    p.add_argument("--reset_optimizer", action="store_true", help="do not load optimizer state even if present")

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    print("Torch:", torch.__version__, " CUDA:", torch.cuda.is_available())
    train(args)
