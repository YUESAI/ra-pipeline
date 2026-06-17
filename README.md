# RADx / HandXFM Code Release

This repository contains the training and evaluation scripts used for the RADx / HandXFM study on hand radiographs and automated Sharp-van der Heijde (SvH) assessment. The code is organized around the manuscript workflow:

1. HandXFM foundation-model pretraining.
2. Public downstream transfer tasks.
3. CATCH joint detection and SvH prediction experiments.

The scripts are research scripts with hardcoded dataset and checkpoint paths from the original compute environment. Before running, update the path constants at the top of each script to match the local data and checkpoint locations.

## Requirements

The code was developed and run in a conda environment with Python 3.11.13 and CUDA-enabled PyTorch. The main Python dependencies are:

- `torch`, `torchvision`
- `transformers`, `timm`
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `Pillow`
- `tqdm`
- `torchmetrics`
- `ultralytics`
- `open_clip_torch` / `open_clip`
- `opencv-python` and standard image-processing utilities as needed by the detection and segmentation scripts

Several detection, segmentation, CLIP, and optimizer dependencies were imported from local source checkouts in the original environment rather than installed as standard site-packages. The relevant scripts append these local repositories to `sys.path`, for example:

```text
../../repo/yolov12/
../../repo/thop/
../../repo/detr/
../../repo/torchmetrics/src/
../../repo/utilities/src/
../../repo/cocoapi/PythonAPI/
```

The pretraining script also exposes an optional Muon optimizer path. Muon is not required for the final reported model; if Muon is unavailable, the script falls back to Adam-based optimization.

To save a minimal conda environment specification from the original environment, run:

```bash
conda env export --from-history > environment.yml
```

## Repository Layout

```text
scripts/
  pretrain/
    handx_pretrain_multi_expert.py
  downstream_tasks/
    airi_svh_joint_erosion_multiseed.py
    airi_svh_joint_jsn_multiseed.py
    anhui_svh_hand_v1.py
    bone_seg_detr_v1.py
    dha_infer_handxfm_rsna.py
    joint_detection_detr.py
    medpix_vqa_close.py
    mura_normal_cls.py
    rhpe_boneage_month.py
    rsna_boneage_month.py
  catch_tasks/
    catch_split_utils.py
    catch_joint_det_finetune_from_rsna.py
    catch_svh_joint_binary_conformal.py
    catch_svh_joint_erosion_gate_regression_conformal.py
    catch_svh_joint_jsn_gate_regression_conformal.py
    catch_svh_unilateral_binary_conformal.py
    catch_svh_unilateral_erosion_gate_regression_conformal.py
    catch_svh_unilateral_jsn_gate_regression_conformal.py
    catch_svh_bilateral_binary_conformal.py
    catch_svh_bilateral_erosion_regression_conformal.py
    catch_svh_bilateral_jsn_regression_conformal.py
```

## HandXFM Pretraining

Script: `scripts/pretrain/handx_pretrain_multi_expert.py`

This script implements the HandXFM pretraining setup described in the manuscript:

- DINOv3 ViT-B/16 student encoder.
- Frozen expert supervision from BioMedCLIP and a radiographic ViT expert.
- Masked latent alignment / reconstruction over visible and masked patch tokens.
- EMA consistency using the student EMA encoder.
- CLS-token distillation from the expert encoders.
- Edge-aware masking to favor high-gradient anatomical regions.
- Optional Muon optimizer with Adam fallback.

Default manuscript-aligned settings include:

```text
batch size: 512
epochs: 10
image size: 224
mask ratio: 0.5
edge-mask alpha: 0.8
encoder LR: 3e-5
head/projection LR: 1e-4
weight decay: 1e-4
EMA loss weight: 0.01
CLS loss weight: 0.01
optimizer: MuonWithAuxAdam if available, otherwise Adam
```

Example:

```bash
python scripts/pretrain/handx_pretrain_multi_expert.py \
  --data_roots /path/to/public_hand_xray_root_1 /path/to/public_hand_xray_root_2 \
  --out_dir /path/to/output/pretrain
```

The exported `ema_state` is used as the default HandXFM backbone for downstream scripts.

## Public Downstream Transfer Tasks

Scripts in `scripts/downstream_tasks/` correspond to the public transfer experiments and auxiliary supervised tasks reported in the manuscript appendices.

| Script | Task | Selection / reporting |
| --- | --- | --- |
| `airi_svh_joint_erosion_multiseed.py` | AIRINII ReumHands joint-level erosion regression | Select checkpoint by validation PCC; report test metrics after loading the best validation checkpoint. |
| `airi_svh_joint_jsn_multiseed.py` | AIRINII ReumHands joint-level JSN regression | Select checkpoint by validation PCC; report test metrics after loading the best validation checkpoint. |
| `anhui_svh_hand_v1.py` | Anhui hand-level SvH regression | Select by validation PCC; report final test metrics. |
| `rsna_boneage_month.py` | RSNA bone age prediction | Select by validation MAD. |
| `rhpe_boneage_month.py` | RHPE bone age prediction | Loads the HandXFM EMA checkpoint by default; select by validation MAD. |
| `dha_infer_handxfm_rsna.py` | Digital Hand Atlas inference using the RSNA-trained model | Evaluation/inference script. |
| `mura_normal_cls.py` | MURA XR_HAND normal/abnormal classification | Select by validation AUC. |
| `joint_detection_detr.py` | Public joint detection with a DETR-style head | Fine-tune the HandXFM encoder with a smaller encoder LR; select by validation mAP. |
| `bone_seg_detr_v1.py` | 19-class bone segmentation with matched DETR-style design | Select by validation mAP. |
| `medpix_vqa_close.py` | MedPix hand-X-ray VQA / text-image transfer | Select by validation top-5 accuracy. |

These scripts load the pretraining checkpoint through `ema_state` unless a random-initialization ablation switch is explicitly enabled.

## CATCH Experiments

Scripts in `scripts/catch_tasks/` implement the CATCH analyses described in the main paper and appendix.

### Shared Patient Split

All CATCH SvH scripts call `catch_split_utils.shared_patient_level_split_3way`, which creates or reuses one persistent patient-level split table. This keeps the train/validation/test partition shared across tasks, matching the manuscript design and preventing leakage across patient IDs.

By default, the split file is written next to the CATCH scripts as:

```text
scripts/catch_tasks/catch_patient_split_seed3407.csv
```

To force all scripts to use a pre-existing split, set:

```bash
export CATCH_PATIENT_SPLIT_CSV=/path/to/catch_patient_split_seed3407.csv
```

### Joint Detection

Script: `scripts/catch_tasks/catch_joint_det_finetune_from_rsna.py`

- Initializes from the RSNA-trained detector checkpoint.
- Uses the official train/validation/test entries from `data_split.yaml`.
- Freezes the encoder for the first 5 epochs, then unfreezes it.
- Uses AdamW with separate head and encoder learning rates.
- Selects the best checkpoint by validation mAP.
- Reports validation and test detection metrics only after loading the best validation checkpoint.

Detection conformal prediction is not implemented in this training script. In the current code organization, conformal localization analysis is handled by the visualization/analysis code.

### Joint-Level SvH Binary Classification

Script: `scripts/catch_tasks/catch_svh_joint_binary_conformal.py`

Run once with `TASK = "erosion"` and once with `TASK = "jsn"`.

- Uses HandXFM encoder initialized from the pretrained EMA checkpoint.
- Uses BCE-with-logits for score 0 vs score > 0.
- Uses positive-class loss reweighting and optional weighted sampling.
- Selects checkpoints by validation ROC-AUC.
- Fits split conformal prediction sets on validation logits.
- Reports held-out test metrics only after loading the best validation checkpoint.

### Joint-Level SvH Regression

Scripts:

- `catch_svh_joint_erosion_gate_regression_conformal.py`
- `catch_svh_joint_jsn_gate_regression_conformal.py`

These implement the gated ordinal regression setup:

- Gate head warm-started from the corresponding binary classifier.
- Joint-specific ordinal heads.
- Erosion uses joint-specific class counts for MCP/PIP and wrist-region targets.
- JSN is restricted to MCP/PIP joints.
- Warm-up freezes the encoder and gate, then fine-tunes all components with a reduced encoder LR.
- Model selection uses validation soft-gated PCC only.
- Split conformal regression intervals are calibrated on validation residuals and evaluated on the held-out test set after loading the best checkpoint.

### Unilateral- and Bilateral-Hand SvH Models

Scripts:

- `catch_svh_unilateral_binary_conformal.py`
- `catch_svh_unilateral_erosion_gate_regression_conformal.py`
- `catch_svh_unilateral_jsn_gate_regression_conformal.py`
- `catch_svh_bilateral_binary_conformal.py`
- `catch_svh_bilateral_erosion_regression_conformal.py`
- `catch_svh_bilateral_jsn_regression_conformal.py`

The hand-level scripts aggregate complete observed joint sets:

- Unilateral erosion: 13 valid joints, score range 0-65.
- Unilateral JSN: 9 valid MCP/PIP joints, score range 0-36.
- Bilateral erosion: 26 valid joints, score range 0-130.
- Bilateral JSN: 18 valid MCP/PIP joints, score range 0-72.

Binary models select by validation AUC. Regression models warm-start from their task-matched binary checkpoints and select by validation soft-gated PCC. Test metrics and conformal test coverage are reported only after loading the best validation checkpoint.

## Reproducibility Notes

- The repository intentionally keeps each experiment as a separate script to mirror the manuscript experiments and make individual components easy to rerun.
- Most scripts define data paths, checkpoint paths, and output directories as constants near the top of the file.
- CATCH SvH scripts use a shared persistent patient split; keep the generated split CSV fixed once produced.
- Validation is used for checkpoint selection and conformal calibration. Test splits are reserved for final reporting.
- No CATCH images are used in HandXFM pretraining.
- Random-initialization ablations are controlled by `RANDOM_INIT` flags in the relevant downstream scripts.

## Minimal Execution Order

A typical reproduction flow is:

```bash
# 1) Pretrain HandXFM or point downstream scripts to an existing checkpoint.
python scripts/pretrain/handx_pretrain_multi_expert.py --data_roots /path/to/public/data --out_dir /path/to/pretrain/out

# 2) Run public transfer tasks as needed.
python scripts/downstream_tasks/airi_svh_joint_erosion_multiseed.py
python scripts/downstream_tasks/airi_svh_joint_jsn_multiseed.py
python scripts/downstream_tasks/rhpe_boneage_month.py

# 3) Run CATCH detector fine-tuning.
python scripts/catch_tasks/catch_joint_det_finetune_from_rsna.py

# 4) Run CATCH SvH binary classifiers before the gated regression scripts.
python scripts/catch_tasks/catch_svh_joint_binary_conformal.py
python scripts/catch_tasks/catch_svh_unilateral_binary_conformal.py
python scripts/catch_tasks/catch_svh_bilateral_binary_conformal.py

# 5) Run CATCH gated regression models.
python scripts/catch_tasks/catch_svh_joint_erosion_gate_regression_conformal.py
python scripts/catch_tasks/catch_svh_joint_jsn_gate_regression_conformal.py
python scripts/catch_tasks/catch_svh_unilateral_erosion_gate_regression_conformal.py
python scripts/catch_tasks/catch_svh_unilateral_jsn_gate_regression_conformal.py
python scripts/catch_tasks/catch_svh_bilateral_erosion_regression_conformal.py
python scripts/catch_tasks/catch_svh_bilateral_jsn_regression_conformal.py
```

For scripts with a `TASK` constant, edit the constant at the top of the file to run the corresponding erosion or JSN setting.
