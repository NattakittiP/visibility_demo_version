# Reading Visibility Classifier

A machine learning project for predicting **whether a person can read text** (`can_read`) based on visual and physical features such as viewing distance, font size, head orientation, and display medium.

---

## Overview

This script trains and evaluates two classification models on a reading visibility dataset, and produces a comprehensive set of exploratory and diagnostic figures.

**Target variable:** `can_read` (binary: 1 = can read, 0 = cannot read)

---

## Dataset

| File | Rows | Notes |
|------|------|-------|
| `reading_visibility_dataset_300rows.csv` | 300 | Default input file — rename in `CSV_PATH` if needed |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `distance_m` | Numeric | Viewing distance in metres |
| `font_size_pt` | Numeric | Font size in points |
| `text_height_mm` | Numeric | Physical text height in mm |
| `head_yaw_deg` | Numeric | Head yaw angle (degrees) |
| `head_pitch_deg` | Numeric | Head pitch angle (degrees) |
| `head_roll_deg` | Numeric | Head roll angle (degrees) |
| `angular_size_deg` | Numeric | Angular size of text (degrees) |
| `visibility_score` | Numeric | Continuous visibility score |
| `medium` | Categorical | Display medium type |
| `contrast` | Categorical | Contrast level |

---

## Models

| Model | Notes |
|-------|-------|
| **Logistic Regression** (L2) | `solver=liblinear`, `class_weight=balanced` |
| **Random Forest** | 400 trees, `class_weight=balanced_subsample` |

**Preprocessing pipeline (fold-safe):**
- Numeric: median imputation → StandardScaler
- Categorical: most-frequent imputation → OneHotEncoder
- Train/test split: 75% / 25%, stratified, `random_state=42`

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
python Project.py
```

> Make sure `reading_visibility_dataset_300rows.csv` is in the same directory, or update `CSV_PATH` in the script.

---

## Outputs

### Figures (300 DPI PNG)

| File | Description |
|------|-------------|
| `fig_corr_heatmap.png` | Correlation heatmap (numeric + dummy-encoded categoricals) |
| `fig_scatter_vs_visibility.png` | 2D scatter plots of each feature vs `visibility_score` |
| `fig_3d_visibility.png` | 3D scatter: distance × angular size × font size (color = visibility score) |
| `fig_3d_cheating.png` | 3D scatter: distance × font size × visibility (size = angular size, color = can_read) |
| `fig_roc.png` | ROC curves for both models |
| `fig_confusion_*.png` | Confusion matrices per model |
| `fig_calibration_rf.png` | Calibration curve (Random Forest) |
| `fig_feature_importance.png` | Tree-based feature importances (RF) |
| `fig_permutation_importance.png` | Permutation importances on test set (RF, 30 repeats) |
| `fig_pdp_1d.png` | 1D Partial Dependence Plots for key features |
| `fig_pdp_2d.png` | 2D Partial Dependence surfaces (feature interaction pairs) |

### CSV

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Accuracy, Precision, Recall, F1, AUC, AP, Brier score per model |

---

## Evaluation Metrics

| Metric | Direction |
|--------|-----------|
| Accuracy | ↑ |
| Precision / Recall / F1 | ↑ |
| AUROC | ↑ |
| Average Precision (AP) | ↑ |
| Brier Score | ↓ |

---
