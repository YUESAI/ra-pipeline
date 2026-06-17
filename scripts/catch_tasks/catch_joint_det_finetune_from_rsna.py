#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CATCH Joint Detection Finetune (DETR-style) from RSNA-trained checkpoint

Requirements:
1) LOAD RSNA detector ckpt (full model.state_dict, includes encoder)
2) Read CATCH data from:
   - root: /home/UWO/ylong66/data/RA/RA/external_data/catch_joint_17
   - split yaml: /home/UWO/ylong66/data/RA/RA/external_data/catch_joint_17/data_split.yaml
3) Splits usage:
   - 80% train: used for training
   - 10% val: used for checkpoint selection; downstream visualization code performs conformal calibration
   - 10% test: final evaluation
4) Final reporting from the best validation checkpoint:
   - overall mAP (@0.5:0.95, @0.5, @0.75)
   - per joint-type group metrics for: DIP / PIP / MCP / Wrist / Ulna / Radius
     based on 17-class names:
     ['DIP_5','DIP_4','DIP_3','DIP_2','PIP_5','PIP_4','PIP_3','PIP_2','PIP_1',
      'MCP_5','MCP_4','MCP_3','MCP_2','MCP_1','Radius','Ulna','Wrist']

Notes:
- We do NOT read data.yaml; we read data_split.yaml only.
- Ultralytics YOLODataset can take img_path as a .txt list of image paths.
- Patient-level grouping: patient_id = image_name.split('_')[0]
"""

import os, sys, time, glob, random
from pathlib import Path
from argparse import Namespace
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoImageProcessor

# Local repository paths.
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

# ---- CATCH paths ----
CATCH_ROOT = "/home/UWO/ylong66/data/RA/RA/external_data/catch_joint_17"
DATA_SPLIT_YAML = "/home/UWO/ylong66/data/RA/RA/external_data/catch_joint_17/data_split.yaml"  # IMPORTANT

# ---- RSNA trained detector ckpt (best) ----
RSNA_DET_CKPT = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/multi_expert_ema/joint_det_detr_updated_ep856_map0.7636.pt"

# Train
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 200            # fine-tuning epochs
SEED = 3407

# Detection heads
NUM_CLASSES = 17        # no background
NUM_QUERIES = 100
EOS_COEF = 0.1

# Optim
LR = 1e-4
ENCODER_LR = 1e-5       # finetune encoder slightly smaller often helps
WEIGHT_DECAY = 1e-4

# Train strategy
FREEZE_ENCODER = True
ENCODER_FREEZE_EPOCHS = 5

# Model save
OUTPUT_DIR = "/home/UWO/ylong66/data/RA/LLM/ckpt/train/catch_finetune_from_rsna"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Logging
# =========================
def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] "), msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Joint type grouping
# =========================
CLASS_NAMES_17 = [
    'DIP_5', 'DIP_4', 'DIP_3', 'DIP_2',
    'PIP_5', 'PIP_4', 'PIP_3', 'PIP_2', 'PIP_1',
    'MCP_5', 'MCP_4', 'MCP_3', 'MCP_2', 'MCP_1',
    'Radius', 'Ulna', 'Wrist'
]

# 6 reported joint-type groups
JOINT_GROUPS: Dict[str, List[int]] = {
    "DIP":    [0, 1, 2, 3],
    "PIP":    [4, 5, 6, 7, 8],
    "MCP":    [9, 10, 11, 12, 13],
    "Radius": [14],
    "Ulna":   [15],
    "Wrist":  [16],
}


# =========================
# Vision Encoder (DINOv3 student)
# =========================
class VisionEncoder(nn.Module):
    """
    - Input: batch tensor in [0,1], shape [B, 3, H, W]
    - Internal: AutoImageProcessor preprocess (do_rescale=False)
    - Output: patch tokens [B, P, D]
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
            patches = tokens[:, 1 + self.num_register_tokens:, :]
        else:
            patches = tokens[:, 1:, :]
        return patches


# =========================
# Decoder Head
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


class JointDEIMv2Updated(nn.Module):
    """
    - ViT patch tokens -> proj -> + learnable memory pos -> TransformerDecoder
    - queries: learnable
    - aux outputs: per-layer
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
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        D = encoder.hidden_size

        self.input_proj = nn.Linear(D, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nheads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)  # + background
        self.bbox_embed  = MLP(d_model, d_model, 4, num_layers=3)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.)
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0.)

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # learnable memory pos (created lazily)
        self.register_buffer("memory_pos", None, persistent=False)

    def forward(self, imgs: torch.Tensor):
        patches = self.encoder(imgs)                 # [B,P,D]
        B, P, _ = patches.shape

        memory = self.input_proj(patches)            # [B,P,256]

        # learnable memory pos
        if (self.memory_pos is None) or (self.memory_pos.shape[1] != P):
            pos = torch.zeros(1, P, self.d_model, device=memory.device)
            nn.init.trunc_normal_(pos, std=0.02)
            self.memory_pos = pos
        memory = (memory + self.memory_pos).transpose(0, 1)   # [P,B,256]

        tgt = torch.zeros(self.num_queries, B, self.d_model, device=imgs.device)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q,B,256]

        hs = []
        out = tgt
        for layer in self.decoder.layers:
            out = layer(out + query_pos, memory)     # [Q,B,256]
            hs.append(out.transpose(0, 1))           # [B,Q,256]
        hs = torch.stack(hs, dim=0)                  # [L,B,Q,256]

        outputs_class = self.class_embed(hs[-1])     # [B,Q,C+1]
        outputs_coord = self.bbox_embed(hs[-1]).sigmoid()

        aux_outputs = [
            {"pred_logits": self.class_embed(hs[l]), "pred_boxes": self.bbox_embed(hs[l]).sigmoid()}
            for l in range(hs.shape[0] - 1)
        ]

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord, "aux_outputs": aux_outputs}


# =========================
# Eval: overall + per-group mAP
# =========================
def _filter_by_allowed_labels(pred: Dict, gt: Dict, allowed: set) -> Tuple[Dict, Dict]:
    # pred: boxes/scores/labels (cpu)
    # gt: boxes/labels (cpu)
    pl = pred["labels"]
    gl = gt["labels"]

    pm = torch.tensor([int(x.item()) in allowed for x in pl], dtype=torch.bool)
    gm = torch.tensor([int(x.item()) in allowed for x in gl], dtype=torch.bool)

    pred_f = {
        "boxes":  pred["boxes"][pm],
        "scores": pred["scores"][pm],
        "labels": pred["labels"][pm],
    }
    gt_f = {
        "boxes":  gt["boxes"][gm],
        "labels": gt["labels"][gm],
    }
    return pred_f, gt_f


@torch.no_grad()
def evaluate_detr_like_with_groups(
    model,
    loader,
    device,
    img_size=224,
    groups: Optional[Dict[str, List[int]]] = None,
    tag: str = "VAL"
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        "ALL": {"map":..., "map50":..., "map75":...},
        "DIP": {...}, ...
      }
    """
    model.eval()
    postprocessor = PostProcess()

    metric_all = MeanAveragePrecision(iou_type="bbox")
    metric_groups = {}
    if groups:
        for g in groups:
            metric_groups[g] = MeanAveragePrecision(iou_type="bbox")

    for batch in loader:
        imgs = batch['img'].to(device, dtype=torch.float32) / 255.0
        B = imgs.size(0)

        cls_ = batch['cls']
        bboxes = batch['bboxes']
        batch_idx = batch['batch_idx']

        targets = []
        for i in range(B):
            labels_i = cls_[batch_idx == i].long().squeeze(-1).to(device)
            boxes_i  = bboxes[batch_idx == i].to(device)  # cxcywh normalized
            targets.append({"labels": labels_i, "boxes": boxes_i})

        outputs = model(imgs)

        # original sizes
        if 'ori_shape' in batch:
            orig_sizes = []
            for s in batch['ori_shape']:
                if isinstance(s, (list, tuple)):
                    h, w = int(s[0]), int(s[1])
                else:
                    h, w = img_size, img_size
                orig_sizes.append([h, w])
            orig_target_sizes = torch.tensor(orig_sizes, device=device, dtype=torch.long)
        else:
            orig_target_sizes = torch.tensor([[img_size, img_size]] * B, device=device, dtype=torch.long)

        results = postprocessor(outputs, orig_target_sizes)

        preds_for_map, gts_for_map = [], []
        for i in range(B):
            result = results[i]
            pred_i = {
                "boxes":  result['boxes'].cpu(),
                "scores": result['scores'].cpu(),
                "labels": result['labels'].cpu(),
            }

            cxcywh = targets[i]['boxes']
            x_c, y_c, w, h = cxcywh.unbind(-1)
            img_h, img_w = orig_target_sizes[i]
            x0 = (x_c - 0.5 * w) * img_w
            y0 = (y_c - 0.5 * h) * img_h
            x1 = (x_c + 0.5 * w) * img_w
            y1 = (y_c + 0.5 * h) * img_h
            gt_i = {
                "boxes":  torch.stack([x0, y0, x1, y1], dim=-1).cpu(),
                "labels": targets[i]['labels'].cpu(),
            }

            preds_for_map.append(pred_i)
            gts_for_map.append(gt_i)

        metric_all.update(preds_for_map, gts_for_map)

        if groups:
            for gname, cls_ids in groups.items():
                allowed = set(cls_ids)
                preds_g, gts_g = [], []
                for pi, gi in zip(preds_for_map, gts_for_map):
                    p_f, g_f = _filter_by_allowed_labels(pi, gi, allowed)
                    preds_g.append(p_f)
                    gts_g.append(g_f)
                metric_groups[gname].update(preds_g, gts_g)

    res_all = metric_all.compute()
    out = {
        "ALL": {
            "map": float(res_all["map"]),
            "map50": float(res_all["map_50"]),
            "map75": float(res_all["map_75"]),
        }
    }

    if groups:
        for gname, m in metric_groups.items():
            r = m.compute()
            out[gname] = {
                "map": float(r["map"]),
                "map50": float(r["map_50"]),
                "map75": float(r["map_75"]),
            }

    # pretty print
    log(f"📊 [{tag}] mAP: @0.5:0.95={out['ALL']['map']:.4f}  @0.5={out['ALL']['map50']:.4f}  @0.75={out['ALL']['map75']:.4f}")
    if groups:
        parts = []
        for g in ["DIP", "PIP", "MCP", "Wrist", "Ulna", "Radius"]:
            if g in out:
                parts.append(f"{g}={out[g]['map']:.4f}")
        log(f"    [{tag}] per-type mAP: " + " | ".join(parts))

    return out


# =========================
# Data
# =========================
def build_dataloaders():
    """
    Uses DATA_SPLIT_YAML:
      - data['train'] : 80% patient train list
      - data['val']   : 10% patient validation list for checkpoint selection
      - data['test']  : 10% patient test list
    """
    data = utils.check_det_dataset(DATA_SPLIT_YAML)

    assert "train" in data and "val" in data and "test" in data, "data_split.yaml must define train/val/test"
    train_txt = data["train"]
    val_txt = data["val"]
    test_txt  = data["test"]

    # ---- datasets ----
    train_dataset = build.YOLODataset(
        task='detect',
        img_path=train_txt,
        data=data,
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augment=True,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        hyp=Namespace(
            degrees=10, deterministic=True,
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
        task='detect',
        img_path=val_txt,
        data=data,
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augment=False,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        fraction=1.0
    )

    # test
    test_dataset = build.YOLODataset(
        task='detect',
        img_path=test_txt,
        data=data,
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augment=False,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        fraction=1.0
    )

    # ---- loaders ----
    train_loader = build.build_dataloader(train_dataset, batch=BATCH_SIZE, workers=10, shuffle=True)
    val_loader   = build.build_dataloader(val_dataset,   batch=BATCH_SIZE, workers=10, shuffle=False)
    test_loader  = build.build_dataloader(test_dataset,  batch=BATCH_SIZE, workers=10, shuffle=False)

    log(f"[Data] Using split yaml: {DATA_SPLIT_YAML}")
    log(f"[Data] train: {train_txt}")
    log(f"[Data] val: {val_txt} (checkpoint selection)")
    log(f"[Data] test: {test_txt}")

    return train_loader, val_loader, test_loader


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    # data
    train_loader, val_loader, test_loader = build_dataloaders()

    # model
    encoder = VisionEncoder(device=DEVICE).to(DEVICE)
    model = JointDEIMv2Updated(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        d_model=256,
        nheads=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(DEVICE)

    # ---- load RSNA detector ckpt (full model state_dict) ----
    sd = torch.load(RSNA_DET_CKPT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"[CKPT] Loaded RSNA detector ckpt: {RSNA_DET_CKPT}")
    if missing:
        log(f"[CKPT] missing_keys: {len(missing)} (first 10) {missing[:10]}")
    if unexpected:
        log(f"[CKPT] unexpected_keys: {len(unexpected)} (first 10) {unexpected[:10]}")

    # criterion
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    losses = ['labels', 'boxes', 'cardinality']

    weight_dict = {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 2}
    aux_weight_dict = {}
    for i in range(6 - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=EOS_COEF,
        losses=losses
    ).to(DEVICE)

    # optim (encoder vs other)
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]

    optimizer = torch.optim.AdamW(
        [{"params": other_params, "lr": LR},
         {"params": enc_params,   "lr": ENCODER_LR}],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_map = -1.0
    best_ckpt_path = None

    # ---- initial eval (optional) ----
    log("Initial evaluation on validation set:")
    _ = evaluate_detr_like_with_groups(model, val_loader, DEVICE, img_size=IMG_SIZE, groups=JOINT_GROUPS, tag="VAL(init)")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss, n_samples = 0.0, 0

        # freeze encoder for first few epochs
        if FREEZE_ENCODER and epoch <= ENCODER_FREEZE_EPOCHS:
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif FREEZE_ENCODER and epoch == ENCODER_FREEZE_EPOCHS + 1:
            log(f"🔓 Unfreeze encoder from epoch {epoch}")
            for p in model.encoder.parameters():
                p.requires_grad = True

        for batch in train_loader:
            imgs = batch['img'].to(DEVICE, dtype=torch.float32) / 255.0
            B = imgs.size(0)

            cls_ = batch['cls']
            bboxes = batch['bboxes']
            batch_idx = batch['batch_idx']

            targets = [
                {
                    'labels': cls_[batch_idx == i].long().squeeze(-1).to(DEVICE),
                    'boxes':  bboxes[batch_idx == i].to(DEVICE)
                }
                for i in range(B)
            ]

            outputs = model(imgs)
            loss_dict = criterion(outputs, targets)
            weight = criterion.weight_dict
            losses_sum = sum(loss_dict[k] * weight[k] for k in loss_dict.keys() if k in weight)

            optimizer.zero_grad(set_to_none=True)
            losses_sum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            running_loss += float(losses_sum.item()) * B
            n_samples += B

        scheduler.step()

        train_loss = running_loss / max(1, n_samples)
        log(f"Epoch {epoch:03d}/{EPOCHS} | TrainLoss={train_loss:.4f} | time={time.time()-t0:.1f}s")

        # ---- validation eval for selecting best ----
        val_res = evaluate_detr_like_with_groups(
            model, val_loader, DEVICE, img_size=IMG_SIZE, groups=JOINT_GROUPS, tag=f"VAL(ep{epoch})"
        )
        val_map = val_res["ALL"]["map"]

        if val_map > best_val_map:
            best_val_map = val_map
            save_path = os.path.join(OUTPUT_DIR, f"catch_det_from_rsna_best_ep{epoch}_valmap{best_val_map:.4f}.pt")
            torch.save(model.state_dict(), save_path)
            best_ckpt_path = save_path
            log(f"Saved BEST (by VAL mAP) to: {save_path}")

    log("Training finished.")
    log(f"Best VAL mAP = {best_val_map:.4f}")

    if best_ckpt_path is None:
        log("No best checkpoint saved.")
        return

    log("======= Final: Load best validation checkpoint and report VAL/TEST =======")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    log(f"Loaded best ckpt: {best_ckpt_path}")
    _ = evaluate_detr_like_with_groups(
        model, val_loader, DEVICE, img_size=IMG_SIZE, groups=JOINT_GROUPS, tag="VAL(best)"
    )
    _ = evaluate_detr_like_with_groups(
        model, test_loader, DEVICE, img_size=IMG_SIZE, groups=JOINT_GROUPS, tag="TEST(best)"
    )


if __name__ == "__main__":
    main()
