# CellMorphNet — TMB Prediction from Histopathology Images

**Model**: CellMorphNet (proposed model from the paper)  
**Task**: Binary (and ternary) Tumor Mutational Burden (TMB) classification from H&E-stained histopathology tiles  
**Dataset**: TCGA STAD / COAD cohort tiles + CBioPortal tabular TMB annotations  

---

## 📁 Directory Structure

```
members/youssef/experiments/CellMorphNet/
├── README.md                                          ← this file
├── 00_preprocessing_eda.ipynb                         ← EDA & preprocessing notebook
└── CellMorphNet_training_evaluation_explainability.ipynb  ← model notebook
```

Data is accessed via relative paths from the project root (`DSAI_305_XAI/`):
```
data/
├── tile_manifest.csv              ← maps tile paths → patient_id, label
├── processed/tiles/<PATIENT_ID>/ ← 224×224 JPG patches
├── cbioportal_tabular_downloads/  ← TMB + clinical data (.tar.gz archives)
└── checkpoints/                   ← saved model weights (created during training)
```

---

## 🛠 Environment Setup

The project uses **uv** for dependency management. From the project root:

```bash
# Activate the existing venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.1 (if not already installed)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify key packages are available
python -c "import torch, shap, lime, sklearn; print('All OK')"
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.3 | Deep learning framework |
| `torchvision` | ≥ 0.18 | Image transforms |
| `shap` | ≥ 0.44 | SHAP explainability |
| `lime` | ≥ 0.2.0 | LIME explainability |
| `scikit-learn` | ≥ 1.4 | Metrics, PDP |
| `opencv-python` | ≥ 4.9 | Image processing |
| `numpy`, `pandas`, `matplotlib`, `seaborn` | — | Numerics & visualization |

---

## 🚀 How to Run

### 1. Preprocessing & EDA Notebook

```bash
cd /path/to/DSAI_305_XAI   # project root
source .venv/bin/activate
jupyter notebook members/youssef/experiments/CellMorphNet/00_preprocessing_eda.ipynb
```

**What it does:**
- Loads `data/tile_manifest.csv` — 8,701 tiles from 29 patients
- Computes class distribution (TMB high vs. low), tile statistics, per-patient counts
- Visualises sample tiles from both classes with stain deconvolution
- Creates stratified train/val/test splits (70/10/20) by patient
- Saves `data/checkpoints/cellmorphnet_splits.npz` with split indices

**Expected output:** Class distribution plots, tile montage, split summary CSV

---

### 2. CellMorphNet Training, Evaluation & Explainability Notebook

```bash
jupyter notebook members/youssef/experiments/CellMorphNet/CellMorphNet_training_evaluation_explainability.ipynb
```

**Run the cells top-to-bottom.** The notebook is self-contained and includes:

1. **Architecture** — Cellular deconvolution module, Hierarchical Cellular Routing Attention (HCRA), four-stage pyramidal CellMorphNet
2. **Training** — 5 independent runs (seeds: 42, 123, 456, 789, 1024), AdamW optimizer, LR 1e-4, batch 64, 30 epochs, early stopping
3. **Evaluation** — Mean ± std of AUC, F1, Precision, Recall over 5 runs on fixed test set
4. **Explainability**:
   - **SHAP** (GradientExplainer) — pixel-level feature attribution on test patches
   - **LIME** — superpixel-based local explanations
   - **Grad-CAM** — last-conv-layer saliency maps
   - **PDP/ICE** — partial dependence of TMB probability on HED stain features
5. **Statistical significance** — paired t-test and Wilcoxon test vs. ETMIL-SSLViT baseline

**Expected runtime:** ~2–4 hours on RTX 4080/4090 for 5 full training runs (30 epochs each).  
To do a quick sanity-check, set `MAX_EPOCHS = 3` and `NUM_SEEDS = 1` in the config cell.

---

## 📊 Expected Results

From the paper (Table I, binary classification):

| Method | AUC (%) | F1 (%) | Precision (%) | Recall (%) |
|--------|---------|--------|---------------|------------|
| TMBcalc | 72.3 ± 2.1 | 65.1 ± 2.4 | 68.4 ± 2.2 | 62.1 ± 2.8 |
| ETMIL-SSLViT | 91.5 ± 1.2 | 87.8 ± 1.4 | 89.3 ± 1.3 | 86.4 ± 1.5 |
| **CellMorphNet** | **99.2 ± 0.3** | **96.8 ± 0.4** | **97.1 ± 0.4** | **96.5 ± 0.5** |

> **Note:** Our cohort uses STAD/COAD tiles (vs. LUAD in the paper), so exact numbers may differ. Reported metrics should be comparable.

---

## 📂 Output Files

After running both notebooks:

```
data/checkpoints/
├── cellmorphnet_splits.npz          ← train/val/test indices
├── cellmorphnet_seed42_best.pt      ← best checkpoint for seed 42
├── cellmorphnet_seed123_best.pt
├── cellmorphnet_seed456_best.pt
├── cellmorphnet_seed789_best.pt
└── cellmorphnet_seed1024_best.pt

members/youssef/experiments/CellMorphNet/
├── figures/
│   ├── class_distribution.png
│   ├── tile_samples.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── shap_summary.png
│   ├── lime_explanation.png
│   ├── gradcam_overlay.png
│   └── pdp_hed_features.png
└── results_summary.csv              ← mean ± std metrics across 5 seeds
```

---

## 📖 Reference

> Xu, X., Yu, F., Basheer, S., & Kan, Z. (2025). *Advanced Deep Learning Framework for Cancer Cell Morphological Analysis and Tumor Mutational Burden Prediction from Histopathological Images*. IEEE Journal of Biomedical and Health Informatics. DOI: 10.1109/JBHI.2025.3645411
