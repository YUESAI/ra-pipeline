#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint Detection Finetuning (DETR-style) with a ViT encoder.

Selection protocol:
- Best checkpoint selected ONLY by ES mAP@0.5:0.95.
- When saving best, print per-type mAP@0.75 (and other mAPs).
"""

import os, sys, time, random
from pathlib import Path
from argparse import Namespace, ArgumentParser
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn

from transformers import AutoModel, AutoImageProcessor

# ==== local repos (relative to this script; adjust as needed) ====
# Tip: if you package these as a python module, you can remove sys.path hacks.
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
# Logging / seed
# =========================
def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] "), msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Joint grouping
# =========================
CLASS_NAMES_17 = [
    'DIP_5', 'DIP_4', 'DIP_3', 'DIP_2',
    'PIP_5', 'PIP_4', 'PIP_3', 'PIP_2', 'PIP_1',
    'MCP_5', 'MCP_4', 'MCP_3', 'MCP_2', 'MCP_1',
    'Radius', 'Ulna', 'Wrist'
]

JOINT_GROUPS: Dict[str, List[int]] = {
    "DIP":    [0, 1, 2, 3],
    "PIP":    [4, 5, 6, 7, 8],
    "MCP":    [9, 10, 11, 12, 13],
    "Radius": [14],
    "Ulna":   [15],
    "Wrist":  [16],
}

TYPE_ORDER = ["DIP", "PIP", "MCP", "Wrist", "Ulna", "Radius"]


# =========================
# Vision Encoder
# =========================
class VisionEncoder(nn.Module):
    def __init__(self, model_name: str, device: str):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if hasattr(self.processor, "do_rescale"):
            self.processor.do_rescale = False
        self.encoder = AutoModel.from_pretrained(model_name)
        self.device = device
        self.encoder.to(self.device).eval()

        cfg = self.encoder.config
        self.hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "hidden_dim", None) \
                           or getattr(cfg, "embed_dim", None) or getattr(cfg, "width", None)
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    @torch.no_grad()
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.processor(images=list(x), return_tensors="pt")
        return inp["pixel_values"].to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)
        pixel_values = self._process(x)
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state
        if self.num_register_tokens > 0:
            patches = tokens[:, 1 + self.num_register_tokens:, :]
        else:
            patches = tokens[:, 1:, :]
        return patches


# =========================
# DETR-like head
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_d = input_dim if i == 0 else hidden_dim
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class JointDEIMv2Updated(nn.Module):
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

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.)
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0.)

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        self.register_buffer("memory_pos", None, persistent=False)

    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        patches = self.encoder(imgs)  # [B,P,D]
        B, P, _ = patches.shape

        memory = self.input_proj(patches)  # [B,P,256]

        if (self.memory_pos is None) or (self.memory_pos.shape[1] != P):
            pos = torch.zeros(1, P, self.d_model, device=memory.device)
            nn.init.trunc_normal_(pos, std=0.02)
            self.memory_pos = pos

        memory = (memory + self.memory_pos).transpose(0, 1)  # [P,B,256]

        tgt = torch.zeros(self.num_queries, B, self.d_model, device=imgs.device)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q,B,256]

        hs = []
        out = tgt
        for layer in self.decoder.layers:
            out = layer(out + query_pos, memory)
            hs.append(out.transpose(0, 1))
        hs = torch.stack(hs, dim=0)  # [L,B,Q,256]

        outputs_class = self.class_embed(hs[-1])
        outputs_coord = self.bbox_embed(hs[-1]).sigmoid()

        aux_outputs = [
            {"pred_logits": self.class_embed(hs[l]), "pred_boxes": self.bbox_embed(hs[l]).sigmoid()}
            for l in range(hs.shape[0] - 1)
        ]

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord, "aux_outputs": aux_outputs}


# =========================
# Split utils
# =========================
def _read_list_file(p: str) -> List[str]:
    with open(p, "r") as f:
        lines = [x.strip() for x in f.readlines()]
    return [x for x in lines if x]


def _write_list_file(p: str, items: List[str]):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for x in items:
            f.write(x + "\n")


def make_patient_level_split_from_train_txt(
    train_txt: str,
    out_dir: str,
    seed: int,
    es_ratio: float,
) -> Tuple[str, str]:
    imgs = _read_list_file(train_txt)
    assert len(imgs) > 0, f"Empty train list: {train_txt}"

    groups = defaultdict(list)
    for p in imgs:
        name = os.path.basename(p)
        pid = name.split("_")[0]
        groups[pid].append(p)

    pids = sorted(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(pids)

    n = len(pids)
    n_es = max(1, int(es_ratio * n))
    es_pids = set(pids[:n_es])

    tr_imgs, es_imgs = [], []
    for pid, ps in groups.items():
        if pid in es_pids:
            es_imgs.extend(ps)
        else:
            tr_imgs.extend(ps)

    tr_imgs = sorted(tr_imgs)
    es_imgs = sorted(es_imgs)

    tr_txt = os.path.join(out_dir, f"train_tr_seed{seed}.txt")
    es_txt = os.path.join(out_dir, f"train_es_seed{seed}.txt")
    _write_list_file(tr_txt, tr_imgs)
    _write_list_file(es_txt, es_imgs)

    log(f"[Split] internal train-> train_tr/train_es by patient | patients={n} es_patients={len(es_pids)}")
    log(f"[Split] images: train_total={len(imgs)} train_tr={len(tr_imgs)} train_es={len(es_imgs)}")
    return tr_txt, es_txt


# =========================
# Eval (overall + per-group map + per-group map75)
# =========================
def _filter_by_allowed_labels(pred: Dict, gt: Dict, allowed: set) -> Tuple[Dict, Dict]:
    pl = pred["labels"]
    gl = gt["labels"]

    pm = torch.tensor([int(x.item()) in allowed for x in pl], dtype=torch.bool)
    gm = torch.tensor([int(x.item()) in allowed for x in gl], dtype=torch.bool)

    pred_f = {
        "boxes": pred["boxes"][pm],
        "scores": pred["scores"][pm],
        "labels": pred["labels"][pm],
    }
    gt_f = {
        "boxes": gt["boxes"][gm],
        "labels": gt["labels"][gm],
    }
    return pred_f, gt_f


@torch.no_grad()
def evaluate_detr_like_with_groups(
    model,
    loader,
    device: str,
    img_size: int,
    groups: Optional[Dict[str, List[int]]] = None,
    tag: str = "VAL",
) -> Dict[str, Dict[str, float]]:
    model.eval()
    postprocessor = PostProcess()

    metric_all = MeanAveragePrecision(iou_type="bbox")
    metric_groups = {g: MeanAveragePrecision(iou_type="bbox") for g in groups} if groups else {}

    for batch in loader:
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

        if "ori_shape" in batch:
            orig_sizes = []
            for s in batch["ori_shape"]:
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
                "boxes": result["boxes"].cpu(),
                "scores": result["scores"].cpu(),
                "labels": result["labels"].cpu(),
            }

            cxcywh = targets[i]["boxes"]
            x_c, y_c, w, h = cxcywh.unbind(-1)
            img_h, img_w = orig_target_sizes[i]
            x0 = (x_c - 0.5 * w) * img_w
            y0 = (y_c - 0.5 * h) * img_h
            x1 = (x_c + 0.5 * w) * img_w
            y1 = (y_c + 0.5 * h) * img_h
            gt_i = {
                "boxes": torch.stack([x0, y0, x1, y1], dim=-1).cpu(),
                "labels": targets[i]["labels"].cpu(),
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

    log(f"📊 [{tag}] mAP: @0.5:0.95={out['ALL']['map']:.4f}  @0.5={out['ALL']['map50']:.4f}  @0.75={out['ALL']['map75']:.4f}")

    if groups:
        parts_map, parts_75 = [], []
        for g in TYPE_ORDER:
            if g in out:
                parts_map.append(f"{g}={out[g]['map']:.4f}")
                parts_75.append(f"{g}={out[g]['map75']:.4f}")
        log(f"    [{tag}] per-type mAP@0.5:0.95: " + " | ".join(parts_map))
        log(f"    [{tag}] per-type mAP@0.75:    " + " | ".join(parts_75))

    return out


# =========================
# Data
# =========================
def build_dataloaders(
    data_split_yaml: str,
    output_dir: str,
    seed: int,
    train_es_ratio: float,
    img_size: int,
    batch_size: int,
    workers: int,
):
    data = utils.check_det_dataset(data_split_yaml)
    assert "train" in data and "val" in data and "test" in data, "split yaml must define train/val/test"

    train_txt = data["train"]
    calib_txt = data["val"]
    test_txt = data["test"]

    internal_dir = os.path.join(output_dir, "internal_splits")
    train_tr_txt, train_es_txt = make_patient_level_split_from_train_txt(
        train_txt=train_txt,
        out_dir=internal_dir,
        seed=seed,
        es_ratio=train_es_ratio,
    )

    train_dataset = build.YOLODataset(
        task="detect",
        img_path=train_tr_txt,
        data=data,
        imgsz=img_size,
        batch_size=batch_size,
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
            crop_fraction=1.0,
        ),
        fraction=1.0,
    )

    es_dataset = build.YOLODataset(
        task="detect",
        img_path=train_es_txt,
        data=data,
        imgsz=img_size,
        batch_size=batch_size,
        augment=False,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        fraction=1.0,
    )

    calib_dataset = build.YOLODataset(
        task="detect",
        img_path=calib_txt,
        data=data,
        imgsz=img_size,
        batch_size=batch_size,
        augment=False,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        fraction=1.0,
    )

    test_dataset = build.YOLODataset(
        task="detect",
        img_path=test_txt,
        data=data,
        imgsz=img_size,
        batch_size=batch_size,
        augment=False,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        fraction=1.0,
    )

    train_loader = build.build_dataloader(train_dataset, batch=batch_size, workers=workers, shuffle=True)
    es_loader = build.build_dataloader(es_dataset, batch=batch_size, workers=workers, shuffle=False)
    calib_loader = build.build_dataloader(calib_dataset, batch=batch_size, workers=workers, shuffle=False)
    test_loader = build.build_dataloader(test_dataset, batch=batch_size, workers=workers, shuffle=False)

    log(f"[Data] split yaml: {data_split_yaml}")
    log(f"[Data] train_tr list: {train_tr_txt}")
    log(f"[Data] train_es list: {train_es_txt} (selection ONLY)")
    log(f"[Data] calib(val) list: {calib_txt} (calibration ONLY)")
    log(f"[Data] test list: {test_txt}")

    return train_loader, es_loader, calib_loader, test_loader


# =========================
# Args
# =========================
def parse_args():
    p = ArgumentParser()

    # paths (anonymous)
    p.add_argument("--data_split_yaml", type=str, default=os.environ.get("DATA_SPLIT_YAML", ""),
                   help="Path to dataset split YAML that defines train/val/test.")
    p.add_argument("--resume_ckpt", type=str, default=os.environ.get("RESUME_CKPT", ""),
                   help="Path to a model state_dict checkpoint to resume/initialize from (e.g., RSNA-pretrained or a continued finetune ckpt).")
    p.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "./outputs/joint_det"),
                   help="Directory to save logs/checkpoints/splits.")

    # training schedule
    p.add_argument("--start_epoch", type=int, default=int(os.environ.get("START_EPOCH", "0")),
                   help="Epoch number for logging/filenames (no effect on optimizer state).")
    p.add_argument("--extra_epochs", type=int, default=int(os.environ.get("EXTRA_EPOCHS", "500")),
                   help="Number of additional epochs to train.")
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3407")))

    # model/data
    p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "facebook/dinov3-vitb16-pretrain-lvd1689m"))
    p.add_argument("--img_size", type=int, default=int(os.environ.get("IMG_SIZE", "224")))
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "360")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "10")))

    # detection head
    p.add_argument("--num_classes", type=int, default=17)
    p.add_argument("--num_queries", type=int, default=100)
    p.add_argument("--eos_coef", type=float, default=0.1)

    # optim
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    p.add_argument("--encoder_lr", type=float, default=float(os.environ.get("ENCODER_LR", "1e-5")))
    p.add_argument("--weight_decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "1e-4")))

    # strategy
    p.add_argument("--freeze_encoder", action="store_true", help="Optionally freeze encoder for initial epochs.")
    p.add_argument("--encoder_freeze_epochs", type=int, default=0)
    p.add_argument("--train_es_ratio", type=float, default=float(os.environ.get("TRAIN_ES_RATIO", "0.05")))

    args = p.parse_args()

    if not args.data_split_yaml:
        raise ValueError("Missing --data_split_yaml (or set $DATA_SPLIT_YAML).")
    if not args.resume_ckpt:
        raise ValueError("Missing --resume_ckpt (or set $RESUME_CKPT).")

    return args


# =========================
# Main
# =========================
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_loader, es_loader, calib_loader, test_loader = build_dataloaders(
        data_split_yaml=args.data_split_yaml,
        output_dir=args.output_dir,
        seed=args.seed,
        train_es_ratio=args.train_es_ratio,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    encoder = VisionEncoder(model_name=args.model_name, device=device).to(device)
    model = JointDEIMv2Updated(
        encoder=encoder,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        d_model=256,
        nheads=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ).to(device)

    # ---- load checkpoint (RSNA-pretrained OR continued finetune) ----
    sd = torch.load(args.resume_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"[CKPT] Loaded state_dict from: {args.resume_ckpt}")
    if missing:
        log(f"[CKPT] missing_keys: {len(missing)} (first 10) {missing[:10]}")
    if unexpected:
        log(f"[CKPT] unexpected_keys: {len(unexpected)} (first 10) {unexpected[:10]}")

    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    losses = ["labels", "boxes", "cardinality"]

    weight_dict = {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}
    aux_weight_dict = {}
    for i in range(6 - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
    ).to(device)

    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]

    optimizer = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr},
         {"params": enc_params, "lr": args.encoder_lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.extra_epochs)

    start_epoch = int(args.start_epoch)
    total_epochs = start_epoch + int(args.extra_epochs)

    # baseline ES evaluation
    log("Initial evaluation on ES to set baseline best:")
    es0 = evaluate_detr_like_with_groups(
        model, es_loader, device, img_size=args.img_size, groups=JOINT_GROUPS, tag=f"ES(init@ep{start_epoch})"
    )
    best_es_map = es0["ALL"]["map"]
    best_epoch = start_epoch
    log(f"[Init] best_es_map={best_es_map:.4f} at epoch={best_epoch}")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_samples = 0.0, 0

        if args.freeze_encoder and (epoch <= start_epoch + args.encoder_freeze_epochs):
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif args.freeze_encoder and (epoch == start_epoch + args.encoder_freeze_epochs + 1):
            log(f"🔓 Unfreeze encoder from epoch {epoch}")
            for p in model.encoder.parameters():
                p.requires_grad = True

        for batch in train_loader:
            imgs = batch["img"].to(device, dtype=torch.float32) / 255.0
            B = imgs.size(0)

            cls_ = batch["cls"]
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"]

            targets = [
                {
                    "labels": cls_[batch_idx == i].long().squeeze(-1).to(device),
                    "boxes": bboxes[batch_idx == i].to(device),
                }
                for i in range(B)
            ]

            outputs = model(imgs)
            loss_dict = criterion(outputs, targets)
            weight = criterion.weight_dict
            loss_sum = sum(loss_dict[k] * weight[k] for k in loss_dict.keys() if k in weight)

            optimizer.zero_grad(set_to_none=True)
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            running_loss += float(loss_sum.item()) * B
            n_samples += B

        scheduler.step()

        train_loss = running_loss / max(1, n_samples)
        log(f"Epoch {epoch}/{total_epochs} | TrainLoss={train_loss:.4f} | time={time.time()-t0:.1f}s")

        # ES eval (selection only)
        es_res = evaluate_detr_like_with_groups(
            model, es_loader, device, img_size=args.img_size, groups=JOINT_GROUPS, tag=f"ES(ep{epoch})"
        )
        es_map = es_res["ALL"]["map"]

        if es_map > best_es_map:
            best_es_map = es_map
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, f"joint_det_best_ep{epoch}_esmap{best_es_map:.4f}.pt")
            torch.save(model.state_dict(), save_path)
            log(f"✅ Saved BEST (by ES mAP) to: {save_path}")

            log("---- Evaluate on CALIB (val, calibration-only split) ----")
            _ = evaluate_detr_like_with_groups(
                model, calib_loader, device, img_size=args.img_size, groups=JOINT_GROUPS, tag=f"CALIB(best@ep{epoch})"
            )

            log("---- Evaluate on TEST (final split) ----")
            _ = evaluate_detr_like_with_groups(
                model, test_loader, device, img_size=args.img_size, groups=JOINT_GROUPS, tag=f"TEST(best@ep{epoch})"
            )

    log("Training finished.")
    log(f"Best ES mAP = {best_es_map:.4f} at epoch={best_epoch}")


if __name__ == "__main__":
    main()

# Example:
# python3 catch_joint_detection.py \
#   --data_split_yaml /path/to/data_split.yaml \
#   --resume_ckpt /path/to/rsna_or_resume_ckpt.pt \
#   --output_dir ./outputs/joint_det \
#   --start_epoch 0 --extra_epochs 500
