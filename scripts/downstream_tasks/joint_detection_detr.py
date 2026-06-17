#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint Detection with DINOv3 FM + DETR-style Decoder (optimized version)
- 真正可训练的 ViT encoder（不再一直 eval）
- 用纯 tensor 的预处理，去掉 list(images) 的慢路径
- 固定可学习 2D pos，并支持插值
- encoder 和其余部分分组学习率
- 可选 AMP
- eval 阶段加 NMS，让观感/指标更接近 YOLO 系
- imgsz 保持 224
"""

import os, sys, time
from pathlib import Path
from argparse import Namespace
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoImageProcessor
from torchvision.ops import batched_nms

# ==== 本地仓库路径（保持和你原来一致） ====
sys.path.append(os.path.abspath('../../repo/yolov12/'))
sys.path.append(os.path.abspath('../../repo/thop/'))
sys.path.append(os.path.abspath('../../repo/detr'))
sys.path.append(os.path.abspath('../../repo/torchmetrics/src/'))
sys.path.append(os.path.abspath('../../repo/utilities/src/'))
sys.path.append(os.path.abspath('../../repo/cocoapi/PythonAPI/'))

from ultralytics.data import build, utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from matcher import HungarianMatcher
from detr import SetCriterion, PostProcess


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

# 预训练权重
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# Joint detection dataset (ultralytics)
IMG_SIZE = 224                # 保持不变
# BATCH_SIZE = 128
BATCH_SIZE = 512
EPOCHS = 1000

# Detection heads
NUM_CLASSES = 17              # 不含背景
NUM_QUERIES = 50
EOS_COEF = 0.1

# Optim
LR = 1e-4                     # decoder / heads
ENCODER_LR = 1e-5             # encoder 更小一点
WEIGHT_DECAY = 1e-5
# WEIGHT_DECAY = 1e-4

# 训练策略
FREEZE_ENCODER = True         # 前几轮先冻住 encoder
ENCODER_FREEZE_EPOCHS = 1000    # 前 10 epoch 不动 encoder
USE_AMP = True                # 建议开 AMP

# Model save
OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] "), msg, flush=True)


# =========================
# Vision Encoder (DINOv3)
# =========================
class VisionEncoder(nn.Module):
    """
    - 输入：batch tensor in [0,1], shape [B, 3, H, W]
    - 内部：只做 normalize，不再走 list(images)
    - 输出：patch tokens [B, P, D]
    """
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", device=None):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # 我们自己 /255，所以这里不需要再 rescale
        if hasattr(self.processor, "do_rescale"):
            self.processor.do_rescale = False

        self.encoder = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

        # 把 mean/std 缓存成 buffer，走纯 tensor
        image_mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        image_std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean, persistent=False)
        self.register_buffer("image_std", image_std, persistent=False)

        cfg = self.encoder.config
        self.hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "hidden_dim", None)
            or getattr(cfg, "embed_dim", None)
            or getattr(cfg, "width", None)
        )
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] ∈ [0,1]
        return (x - self.image_mean) / self.image_std

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)
        pixel_values = self._process(x)
        # 非冻结阶段这里会被 .train() 掉
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D] or [B, 1+P, D]
        if self.num_register_tokens > 0:
            patches = tokens[:, 1 + self.num_register_tokens :, :]  # [B, P, D]
        else:
            patches = tokens[:, 1:, :]  # [B, P, D]
        return patches

    def train(self, mode: bool = True):
        # 覆盖一下，确保内部 encoder 也切换
        super().train(mode)
        self.encoder.train(mode)
        return self


def load_foundation_encoder(encoder: VisionEncoder, ckpt_path: str, prefer_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if prefer_ema and ("ema_state" in ckpt) and (ckpt["ema_state"] is not None):
        state = ckpt["ema_state"]
        log(f"[CKPT] Using ema_state from: {ckpt_path}")
    else:
        state = ckpt.get("student_state", None)
        log(f"[CKPT] Using student_state from: {ckpt_path}")

    if state is None:
        raise RuntimeError("Neither ema_state nor student_state found in checkpoint.")

    def _strip_prefix(sd, prefixes=("encoder.", "module.", "model.")):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("processor."):
                continue
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[nk] = v
        return new_sd

    state = _strip_prefix(state)
    missing, unexpected = encoder.encoder.load_state_dict(state, strict=False)
    if missing:
        log(f"[CKPT(post-fix)] missing_keys: {len(missing)} (first 8) {missing[:8]}")
    if unexpected:
        log(f"[CKPT(post-fix)] unexpected_keys: {len(unexpected)} (first 8) {unexpected[:8]}")


# =========================
# Decoder-only Head
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_d = input_dim if i == 0 else hidden_dim
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class JointDETRUpdated(nn.Module):
    """
    - ViT patch tokens → proj → +2D pos → TransformerDecoder
    - queries: learnable
    - aux: per-layer outputs
    - 这里固定了一个 224×224 / 16×16 patch 的长度 14×14=196，如果 encoder 输出别的长度会插值
    """
    def __init__(
        self,
        encoder: VisionEncoder,
        num_classes: int,
        num_queries: int = 100,
        d_model: int = 256,
        nheads: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_patches: int = 196,   # 224 / 16 = 14 → 14*14
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        D = encoder.hidden_size

        self.input_proj = nn.Linear(D, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # 固定长度的可学习 2D pos
        self.memory_pos = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.memory_pos, std=0.02)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0.0)

    def forward(self, imgs: torch.Tensor):
        # 1) encoder
        patches = self.encoder(imgs)                 # [B,P,D]
        B, P, _ = patches.shape

        # 2) proj
        memory = self.input_proj(patches)            # [B,P,256]

        # 3) 2D pos for memory，若 P 变化则插值
        if P != self.memory_pos.shape[1]:
            mem_pos = F.interpolate(
                self.memory_pos.transpose(1, 2),     # [1,256,P0]
                size=P,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)                         # [1,P,256]
        else:
            mem_pos = self.memory_pos

        memory = (memory + mem_pos[:, :P, :]).transpose(0, 1)  # [P,B,256]

        # 4) queries
        tgt = torch.zeros(self.num_queries, B, self.d_model, device=imgs.device)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q,B,256]

        # 5) decoder (手动拿中间层输出 → aux loss)
        hs = []
        out = tgt
        for layer in self.decoder.layers:
            out = layer(out + query_pos, memory)     # [Q,B,256]
            hs.append(out.transpose(0, 1))           # [B,Q,256]
        hs = torch.stack(hs, dim=0)                  # [L,B,Q,256]

        # 6) heads
        outputs_class = self.class_embed(hs[-1])     # [B,Q,C+1]
        outputs_coord = self.bbox_embed(hs[-1]).sigmoid()

        aux_outputs = [
            {
                "pred_logits": self.class_embed(hs[l]),
                "pred_boxes": self.bbox_embed(hs[l]).sigmoid(),
            }
            for l in range(hs.shape[0] - 1)
        ]

        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "aux_outputs": aux_outputs,
        }


# =========================
# Eval + NMS
# =========================
def nms_postprocess(results, iou_th=0.6):
    final = []
    for r in results:
        boxes = r["boxes"]
        scores = r["scores"]
        labels = r["labels"]
        keep = batched_nms(boxes, scores, labels, iou_th)
        final.append(
            {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            }
        )
    return final


@torch.no_grad()
def evaluate_detr_like(
    model,
    val_loader,
    device,
    matcher,
    img_size=224,
    num_classes=17,
    nms_iou=0.6,
):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    postprocessor = PostProcess()

    for batch in val_loader:
        imgs = batch["img"].to(device, dtype=torch.float32) / 255.0
        B = imgs.size(0)

        cls_ = batch["cls"]
        bboxes = batch["bboxes"]
        batch_idx = batch["batch_idx"]

        targets = []
        for i in range(B):
            labels_i = cls_[batch_idx == i].long().squeeze(-1).to(device)
            boxes_i = bboxes[batch_idx == i].to(device)  # cxcywh normalized
            targets.append({"labels": labels_i, "boxes": boxes_i})

        outputs = model(imgs)

        # 原图尺寸（ultralytics 通常会给 ori_shape，是 (h, w)）
        if "ori_shape" in batch:
            orig_sizes = []
            for s in batch["ori_shape"]:
                if isinstance(s, (list, tuple)):
                    h, w = s[0], s[1]
                else:
                    h, w = img_size, img_size
                orig_sizes.append([h, w])
            orig_target_sizes = torch.tensor(orig_sizes, device=device, dtype=torch.long)
        else:
            orig_target_sizes = torch.tensor([[img_size, img_size]] * B, device=device, dtype=torch.long)

        results = postprocessor(outputs, orig_target_sizes)
        # NMS 一下，和 YOLO 观感更像
        results = nms_postprocess(results, iou_th=nms_iou)

        preds_for_map, gts_for_map = [], []
        for i in range(B):
            result = results[i]
            preds_for_map.append(
                {
                    "boxes": result["boxes"].cpu(),
                    "scores": result["scores"].cpu(),
                    "labels": result["labels"].cpu(),
                }
            )

            cxcywh = targets[i]["boxes"]
            x_c, y_c, w, h = cxcywh.unbind(-1)
            img_h, img_w = orig_target_sizes[i]
            x0 = (x_c - 0.5 * w) * img_w
            y0 = (y_c - 0.5 * h) * img_h
            x1 = (x_c + 0.5 * w) * img_w
            y1 = (y_c + 0.5 * h) * img_h
            gts_for_map.append(
                {
                    "boxes": torch.stack([x0, y0, x1, y1], dim=-1).cpu(),
                    "labels": targets[i]["labels"].cpu(),
                }
            )

        metric.update(preds_for_map, gts_for_map)

    res = metric.compute()
    log(f"📊 Detection mAP: @0.5:0.95={res['map']:.4f}, @0.5={res['map_50']:.4f}, @0.75={res['map_75']:.4f}")
    return float(res["map"])


# =========================
# Data
# =========================
def build_dataloaders():
    train_dataset = build.YOLODataset(
        task="detect",
        img_path="/home/UWO/ylong66/data/RA/RA/external_data/det_17_wo_catch/images/train",
        data=utils.check_det_dataset("/home/UWO/ylong66/data/RA/RA/external_data/det_17_wo_catch/data.yaml"),
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augment=True,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        hyp=Namespace(
            degrees=10,
            deterministic=True,
            overlap_mask=True,
            mask_ratio=1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.2,
            scale=0.2,
            shear=10,
            perspective=0.0005,
            flipud=0.5,
            fliplr=0.5,
            bgr=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            copy_paste_mode="flip",
            auto_augment="randaugment",
            erasing=0.4,
            crop_fraction=1.0,
        ),
        fraction=1.0,
    )
    val_dataset = build.YOLODataset(
        task="detect",
        img_path="/home/UWO/ylong66/data/RA/RA/external_data/det_17_wo_catch/images/val",
        data=utils.check_det_dataset("/home/UWO/ylong66/data/RA/RA/external_data/det_17_wo_catch/data.yaml"),
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augment=False,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        fraction=1.0,
    )

    train_loader = build.build_dataloader(train_dataset, batch=BATCH_SIZE, workers=10, shuffle=True)
    val_loader = build.build_dataloader(val_dataset, batch=BATCH_SIZE, workers=10, shuffle=False)
    return train_loader, val_loader


# =========================
# Main
# =========================
def main():
    # data
    train_loader, val_loader = build_dataloaders()

    # encoder
    encoder = VisionEncoder(device=DEVICE).to(DEVICE)
    load_foundation_encoder(encoder, CKPT_PATH, prefer_ema=True)

    # model
    model = JointDETRUpdated(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        d_model=256,
        nheads=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_patches=196,  # 224 -> 14x14
    ).to(DEVICE)

    start_epoch = 0
    best_map = 0.3

    # criterion
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=3)  # giou 稍微拉高
    losses = ["labels", "boxes", "cardinality"]

    weight_dict = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    aux_weight_dict = {}
    for i in range(6 - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=EOS_COEF,
        losses=losses,
    ).to(DEVICE)

    # 参数分组：真正的 ViT backbone 用更小 lr
    backbone_prefix = "encoder.encoder"

    enc_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(backbone_prefix):
            enc_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": LR},
            {"params": enc_params, "lr": ENCODER_LR},
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    postprocessor = PostProcess()

    for epoch in range(1 + start_epoch, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss, n_samples = 0.0, 0

        # 冻住/解冻 encoder（注意要切 train(False/True)）
        if FREEZE_ENCODER and epoch <= ENCODER_FREEZE_EPOCHS:
            for p in model.encoder.parameters():
                p.requires_grad = False
            model.encoder.train(False)
        elif FREEZE_ENCODER and epoch == ENCODER_FREEZE_EPOCHS + 1:
            log(f"🔓 Unfreeze encoder from epoch {epoch}")
            for p in model.encoder.parameters():
                p.requires_grad = True
            model.encoder.train(True)

        for batch in train_loader:
            imgs = batch["img"].to(DEVICE, dtype=torch.float32) / 255.0
            B = imgs.size(0)

            cls_ = batch["cls"]
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"]

            targets = [
                {
                    "labels": cls_[batch_idx == i].long().squeeze(-1).to(DEVICE),
                    "boxes": bboxes[batch_idx == i].to(DEVICE),
                }
                for i in range(B)
            ]

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(imgs)
                loss_dict = criterion(outputs, targets)
                weight = criterion.weight_dict
                loss = sum(loss_dict[k] * weight[k] for k in loss_dict.keys() if k in weight)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * B
            n_samples += B

        scheduler.step()

        train_loss = running_loss / max(1, n_samples)
        log(f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | time={time.time() - t0:.1f}s")

        # eval
        val_map = evaluate_detr_like(
            model,
            val_loader,
            DEVICE,
            matcher,
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            nms_iou=0.6,
        )
        if val_map > best_map:
            best_map = val_map
            save_path = os.path.join(
                OUTPUT_DIR, f"joint_det_detr_updated_v1_ep{epoch}_map{best_map:.4f}.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_map": best_map,
                },
                save_path,
            )
            log(f"✅ Saved best to: {save_path}")

    log("Training finished.")


if __name__ == "__main__":
    main()

