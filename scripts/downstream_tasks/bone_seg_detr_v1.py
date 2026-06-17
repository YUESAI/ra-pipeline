#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bone Segmentation with DINOv3 FM (DETR-style, joint seg + det heads)

Implementation notes:
- Segmentation head upsamples 14x14 -> 56x56 -> 224x224 and outputs at input resolution
- Segmentation loss = weighted CE + 0.5 * Dice to mitigate foreground/background imbalance
- The segmentation head injects global decoder semantics from the final query features
"""

import os, sys, time
from pathlib import Path
from argparse import Namespace
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from transformers import AutoModel, AutoImageProcessor

# ==== Local repository paths ====
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
from transformer import Transformer
from position_encoding import PositionEmbeddingSine  # can be switched to PositionEmbeddingLearned

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

# -- Foundation model checkpoint; prefer ema_state--
CKPT_PATH = "/home/UWO/ylong66/data/RA/LLM/ckpt/pretrain/multi_expert_v1/handx_pretrain_multiexpert_224_10.pt"

# -- Dataset; Ultralytics segment task--
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 1000

# -- Classes/queries; NUM_CLASSES excludes the background class
NUM_CLASSES = 19
NUM_QUERIES = 100
EOS_COEF    = 0.1

# -- Optimizer/training --
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 0.1

# -- Model output --
OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Whether to freeze the encoder --
FREEZE_ENCODER = False


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] "), msg, flush=True)


# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
    """
    - Input: batch tensor in [0,1], shape [B, 3, H, W]
    - Internally uses AutoImageProcessor for normalization/resize (do_rescale=False)
    - Output: patch tokens [B, P, D] only, with CLS/register tokens removed
    """
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", device=None):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if hasattr(self.processor, 'do_rescale'):
            self.processor.do_rescale = False
        self.encoder = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device).eval()

        cfg = self.encoder.config
        self.hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "hidden_dim", None) \
                           or getattr(cfg, "embed_dim", None) or getattr(cfg, "width", None)
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    @torch.no_grad()
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] ∈ [0,1]
        inp = self.processor(images=list(x), return_tensors="pt")
        return inp["pixel_values"].to(self.device)

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)
        pixel_values = self._process(x)
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1+R+P, D] or [B, 1+P, D]
        if self.num_register_tokens > 0:
            patches = tokens[:, 1 + self.num_register_tokens:, :]  # [B, P, D]
        else:
            patches = tokens[:, 1:, :]  # [B, P, D]
        return patches


def load_foundation_encoder(encoder: VisionEncoder, ckpt_path: str, prefer_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None
    if prefer_ema and ("ema_state" in ckpt) and (ckpt["ema_state"] is not None):
        state = ckpt["ema_state"]; log(f"[CKPT] Using ema_state from: {ckpt_path}")
    else:
        state = ckpt.get("student_state", None); log(f"[CKPT] Using student_state from: {ckpt_path}")
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
# Seg Head (new)
# =========================
class SegHead(nn.Module):
    """
    Simple semantic segmentation head: 14x14 -> 56x56 -> 224x224
    Input: spatial features after encoder+transformer [B, d_model, S, S]
    Output: semantic logits [B, C+1, H, W]
    """
    def __init__(self, d_model=256, num_classes=19 + 1, out_size=(224, 224)):
        super().__init__()
        self.out_size = out_size
        self.conv1 = nn.Conv2d(d_model, d_model, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, d_model)
        self.conv2 = nn.Conv2d(d_model, d_model // 2, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, d_model // 2)
        self.out = nn.Conv2d(d_model // 2, num_classes, 1)

    def forward(self, x):
        # x: [B, d_model, 14, 14]
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)  # 14 -> 56
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)  # 56 -> 224
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = self.out(x)  # [B, C+1, 224, 224]
        if self.out_size is not None and x.shape[-1] != self.out_size[0]:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


# =========================
# Segmentation Model
# =========================
class SegmentationModel(nn.Module):
    """
    - ViT patch tokens -> reshape into a 2D feature map (Hf x Wf), then project to d_model with a 1x1 conv
    - Transformer: encoder/decoder
    - Outputs:
        * pred_logits: [B, Q, C+1]
        * pred_boxes:  [B, Q, 4] (cxcywh in [0,1])
        * pred_masks:  [B, C+1, H, W] (input resolution)
    """
    def __init__(self, encoder: VisionEncoder, num_queries=100, d_model=256, nhead=8, num_classes=19):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_size
        self.d_model = d_model

        self.transformer = Transformer(
            d_model=d_model,
            dropout=0.1,
            nhead=nhead,
            dim_feedforward=2048,
            num_encoder_layers=3,
            num_decoder_layers=3,
            normalize_before=True,
            return_intermediate_dec=True,
        )

        self.input_proj = nn.Conv2d(self.hidden_dim, d_model, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes + 1)   # + background
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4)
        )

        # Segmentation head
        self.seg_head = SegHead(d_model=d_model, num_classes=num_classes + 1, out_size=(IMG_SIZE, IMG_SIZE))

        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.num_queries = num_queries

    def forward(self, imgs):  # imgs: [B, 3, H, W] in [0,1]
        patches = self.encoder(imgs)                      # [B, P, D]
        B, P, D = patches.shape
        S = int(P ** 0.5)                                 # e.g., 14x14 for 224/16
        feats = patches.view(B, S, S, D).permute(0, 3, 1, 2).contiguous()  # [B, D, S, S]

        mask = torch.zeros((B, S, S), dtype=torch.bool, device=imgs.device)
        src_proj = self.input_proj(feats)                 # [B, d_model, S, S]

        pos = self.position_embedding(feats, mask)        # [B, d_model, S, S]
        hs, memory = self.transformer(src_proj, mask, self.query_embed.weight, pos)  # hs: [layers, B, Q, d_model]

        outputs_class = self.class_head(hs)               # [layers, B, Q, C+1]
        outputs_coord = self.bbox_head(hs).sigmoid()      # [layers, B, Q, 4]

        # Light coupling: add global semantics from the final decoder layer to the feature map
        dec_feat = hs[-1].mean(1)                         # [B, d_model]
        dec_feat = dec_feat[:, :, None, None].expand(-1, -1, src_proj.shape[2], src_proj.shape[3])
        seg_in = src_proj + dec_feat

        outputs_masks = self.seg_head(seg_in)             # [B, C+1, H, W]

        return {
            'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
            'aux_outputs': _set_aux_loss(outputs_class, outputs_coord),
            'pred_masks': outputs_masks
        }


@torch.jit.unused
def _set_aux_loss(outputs_class, outputs_coord):
    return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


# =========================
# Data
# =========================
def build_dataloaders():
    train_dataset = build.YOLODataset(
        task='segment',
        img_path="/home/UWO/ylong66/data/RA/RA/external_data/seg_19/images/train",
        data=utils.check_det_dataset('/home/UWO/ylong66/data/RA/RA/external_data/seg_19/data.yaml'),
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augment=True,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        hyp=Namespace(
            degrees=180,
            deterministic=True,
            overlap_mask=True, mask_ratio=1,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.2, scale=0.2, shear=10, perspective=0.0005,
            flipud=0.5, fliplr=0.5,
            bgr=0.0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, copy_paste_mode="flip",
            auto_augment="randaugment", erasing=0.4,
            crop_fraction=1.0
        ),
        fraction=1.0
    )
    val_dataset = build.YOLODataset(
        task='segment',
        img_path="/home/UWO/ylong66/data/RA/RA/external_data/seg_19/images/val",
        data=utils.check_det_dataset('/home/UWO/ylong66/data/RA/RA/external_data/seg_19/data.yaml'),
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augment=False,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        hyp=Namespace(overlap_mask=True, mask_ratio=1),
        fraction=1.0
    )
    train_loader = build.build_dataloader(train_dataset, batch=BATCH_SIZE, workers=10, shuffle=True)
    val_loader   = build.build_dataloader(val_dataset,   batch=BATCH_SIZE, workers=10, shuffle=False)
    return train_loader, val_loader


# =========================
# Utils
# =========================
def cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    x0 = x_c - 0.5 * w
    y0 = y_c - 0.5 * h
    x1 = x_c + 0.5 * w
    y1 = y_c + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def dice_loss(inputs, targets, num_classes, eps=1e-6):
    """
    inputs: [B, C, H, W] (logits)
    targets: [B, H, W] (int)
    """
    inputs = inputs.softmax(dim=1)
    targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersect = (inputs * targets_onehot).sum(dims)
    cardinal = (inputs + targets_onehot).sum(dims)
    dice = 1 - (2. * intersect + eps) / (cardinal + eps)
    return dice.mean()


# =========================
# Evaluate
# =========================
@torch.no_grad()
def evaluate(model, val_loader, device, matcher, img_size=224, num_classes=NUM_CLASSES):
    """
    - Classification metrics based on Hungarian matching
    - Detection mAP @0.5:0.95, @0.5, @0.75
    - Pixel-level segmentation ACC / P / R / F1
    """
    model.eval()
    matched_preds, matched_labels = [], []
    preds_for_map, gts_for_map = [], []

    pixel_accs, pixel_ps, pixel_rs, pixel_f1s = [], [], [], []

    postprocessor = PostProcess()
    metric = MeanAveragePrecision(iou_type="bbox")

    for batch in val_loader:
        imgs = batch['img'].to(device, dtype=torch.float32) / 255.0
        B = imgs.size(0)

        cls_ = batch['cls']
        bboxes = batch['bboxes']
        masks = batch['masks'].long().to(device)
        batch_idx = batch['batch_idx']

        targets = [
            {
                'labels': cls_[batch_idx == i].long().squeeze(-1).to(device),
                'boxes':  bboxes[batch_idx == i].to(device)  # normalized cxcywh
            }
            for i in range(B)
        ]

        outputs = model(imgs)
        orig_target_sizes = torch.tensor([[img_size, img_size]] * B).to(device, dtype=torch.long)
        results = postprocessor(outputs, orig_target_sizes)

        # -- Classification based on matching--
        indices = matcher(outputs, targets)
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            logits_matched = outputs['pred_logits'][i, pred_idx]
            pclasses = logits_matched.argmax(-1).cpu().tolist()
            tclasses = targets[i]['labels'][tgt_idx].cpu().tolist()
            matched_preds  += pclasses
            matched_labels += tclasses

        # -- Segmentation metrics; pred_masks are already 224x224
        seg_logits = outputs['pred_masks']  # [B, C+1, H, W]
        pmasks = seg_logits.argmax(dim=1).cpu().flatten().tolist()
        tmasks = masks.flatten().cpu().tolist()

        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        pixel_accs.append(accuracy_score(tmasks, pmasks))
        pixel_ps.append(precision_score(tmasks, pmasks, average='macro', zero_division=0))
        pixel_rs.append(recall_score(tmasks, pmasks, average='macro', zero_division=0))
        pixel_f1s.append(f1_score(tmasks, pmasks, average='macro', zero_division=0))

        # -- mAP inputs --
        for i in range(B):
            result = results[i]
            preds_for_map.append({
                "boxes":  result['boxes'].cpu(),    # xyxy absolute
                "scores": result['scores'].cpu(),
                "labels": result['labels'].cpu(),
            })
            gt_xyxy = cxcywh_to_xyxy(targets[i]['boxes']) * img_size
            gts_for_map.append({
                "boxes":  gt_xyxy.cpu(),
                "labels": targets[i]['labels'].cpu(),
            })

    # Classification
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    if matched_preds:
        prec = precision_score(matched_labels, matched_preds, average='macro', zero_division=0)
        rec  = recall_score(   matched_labels, matched_preds, average='macro', zero_division=0)
        f1   = f1_score(       matched_labels, matched_preds, average='macro', zero_division=0)
        acc  = accuracy_score( matched_labels, matched_preds)
        log(f"📊 Classification: ACC={acc:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    else:
        log(" No matched prediction -> skip classification metrics")

    # Detection mAP
    metric.update(preds_for_map, gts_for_map)
    res = metric.compute()
    log(f"📊 Detection mAP: @0.5:0.95={res['map']:.4f}, @0.5={res['map_50']:.4f}, @0.75={res['map_75']:.4f}")

    # Segmentation
    log(f"📊 Segmentation: ACC={sum(pixel_accs)/len(pixel_accs):.4f}, "
        f"P={sum(pixel_ps)/len(pixel_ps):.4f}, R={sum(pixel_rs)/len(pixel_rs):.4f}, "
        f"F1={sum(pixel_f1s)/len(pixel_f1s):.4f}")

    return float(res['map'])


# =========================
# Train
# =========================
def main():
    # Data
    train_loader, val_loader = build_dataloaders()

    # Encoder
    encoder = VisionEncoder(device=DEVICE).to(DEVICE)
    load_foundation_encoder(encoder, CKPT_PATH, prefer_ema=True)
    if FREEZE_ENCODER:
        for p in encoder.parameters():
            p.requires_grad = False
        log("Encoder frozen.")

    # Model
    model = SegmentationModel(
        encoder=encoder,
        num_queries=NUM_QUERIES,
        d_model=256,
        nhead=8,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    best_map = -1.0
    start_epoch = 0

    # model_ckpt_path = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema/bone_seg_dinov3_ema_ep901_map0.4690.pt"
    
    # model.load_state_dict(torch.load(model_ckpt_path, map_location=DEVICE))
    # log(f"Model initialized from {model_ckpt_path}")
    # start_epoch = 901
    # best_map = 0.469

    # Hungarian + DETR-style criterion（labels / boxes / cardinality）
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    aux_weight_dict = {}
    for i in range(3 - 1):  # num_decoder_layers - 1; here 3 - 1
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=EOS_COEF,
        losses=losses
    ).to(DEVICE)

    # Optimizer; smaller LR for encoder
    param_main = [{"params": [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")],
                   "lr": LR}]
    if not FREEZE_ENCODER:
        enc_lr = LR * 0.5  # can be smaller, e.g. 0.25
        enc_params = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
        param_main.append({"params": enc_params, "lr": enc_lr})
    optimizer = torch.optim.AdamW(param_main, lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(start_epoch + 1, start_epoch + EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, n_imgs = 0.0, 0

        for batch in train_loader:
            imgs = batch['img'].to(DEVICE, dtype=torch.float32) / 255.0
            B = imgs.size(0)

            cls_ = batch['cls']
            bboxes = batch['bboxes']
            masks = batch['masks'].long().to(DEVICE)          # [B, H, W] pixel classes, including background
            batch_idx = batch['batch_idx']

            targets = [
                {
                    'labels': cls_[batch_idx == i].long().squeeze(-1).to(DEVICE),
                    'boxes':  bboxes[batch_idx == i].to(DEVICE)  # normalized cxcywh
                }
                for i in range(B)
            ]

            outputs = model(imgs)

            # DETR-style losses
            loss_dict = criterion(outputs, targets)
            wd = criterion.weight_dict
            det_loss = sum(loss_dict[k] * wd[k] for k in loss_dict.keys() if k in wd)

            # Segmentation loss directly at 224x224
            seg_logits = outputs['pred_masks']  # [B, C+1, H, W]

            # class weights: lower background weight
            class_weights = torch.ones(NUM_CLASSES + 1, device=DEVICE)
            class_weights[0] = 0.3
            seg_ce = F.cross_entropy(seg_logits, masks, weight=class_weights)
            seg_dice = dice_loss(seg_logits, masks, num_classes=NUM_CLASSES + 1)

            seg_loss = seg_ce + 0.5 * seg_dice

            loss = det_loss + seg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += float(loss.item()) * B
            n_imgs += B

        avg_loss = total_loss / max(1, n_imgs)
        log(f"Epoch {epoch:03d} | TrainLoss={avg_loss:.4f} | time={time.time()-t0:.1f}s")

        # ==== Small sanity check before eval ====
        # Check the number of GT instances in the first validation batch
        with torch.no_grad():
            for batch in val_loader:
                bi = batch['batch_idx']
                cls_ = batch['cls']
                gt_obj_cnt = 0
                for i in range(batch['img'].shape[0]):
                    gt_obj_cnt += int((bi == i).sum().item())
                log(f"[DEBUG] first validation batch GT instances: {gt_obj_cnt}")
                break

        # Eval
        val_map = evaluate(model, val_loader, DEVICE, matcher, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
        if val_map > best_map:
            best_map = val_map
            save_path = os.path.join(
                OUTPUT_DIR, f"bone_seg_dinov3_ema_ep{epoch}_map{best_map:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            log(f"Saved best to: {save_path}")

    log("Training finished.")


if __name__ == "__main__":
    main()



