# ra-pipeline

A modular research pipeline for automated analysis of rheumatoid arthritis (RA) radiographs.

This repository focuses on joint-centric and hand-level analysis of RA radiographs, with
current implementations covering joint detection, Sharp–van der Heijde (SvH) damage
classification and regression, and uncertainty estimation. The pipeline is designed for
methodological research rather than deployment, and individual components can be trained
and evaluated independently.

---

## Scope

The current scope of this repository includes:

- Automated joint detection from full radiographs (single-hand or bilateral)
- Joint-level SvH abnormality classification (0 vs >0)
- Joint-level SvH severity regression for erosion and joint space narrowing (JSN)
- Hand-level and bilateral-level SvH binary classification and regression
- Uncertainty quantification for regression outputs

The pipeline is modular and task-oriented. Joint detection, classification, regression,
and uncertainty estimation are implemented as separate components rather than a single
end-to-end system.

---

## Methodological Overview

- **Joint detection**  
  A detection model localizes a fixed set of anatomical joint landmarks directly from
  full RA radiographs. The detection scope is broader than SvH scoring and supports
  downstream joint-centric analysis.

- **SvH damage modeling**  
  SvH scoring is formulated at multiple task granularities (joint-level, single-hand,
  and bilateral). Binary abnormality detection and severity regression are trained as
  separate models, with regression using a soft-gated formulation initialized from the
  corresponding binary classifier.

- **Uncertainty estimation**  
  Predictive uncertainty is quantified for regression tasks using calibration-based
  methods, enabling reliability analysis alongside point estimates.

---

## Backbone and Initialization

Models in this repository leverage a pretrained hand X-ray foundation model for
initialization of visual encoders. The foundation model itself is **not** trained or
developed in this repository and is treated as a fixed or fine-tuned backbone depending
on the task.

---

## Repository Structure

```text
ra-pipeline/
├── detection/          # Joint detection models and evaluation
└── svh/
   ├── binary/         # SvH abnormality classification (joint / hand / bilateral)
   ├── regression/     # SvH severity regression (erosion / JSN)
   └── regression/    # SvH severity regression (erosion / JSN)
