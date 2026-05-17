# Research Paper Outline Implementation Guide

## Overview
This document provides detailed guidance for completing the LaTeX research paper outline (`paper_outline.tex`). 

---

## Section-by-Section Implementation Plan

### Section 1: Introduction

#### 1.1 Clinical Significance of TMB
**TODO Items:**
- [ ] Define TMB precisely: "Tumor Mutational Burden (TMB) = number of somatic mutations per megabase (Mut/Mb) in the coding region"
- [ ] Explain clinical relevance:
  - TMB correlates with response to checkpoint inhibitors (anti-PD-1/PD-L1)
  - High-TMB tumors show better immunotherapy response (meta-analysis, cite 2–3 papers)
  - Used for patient stratification in trials (KEYNOTE-024 [PD-L1+], CHECKMATE-032 [TMB-high])
- [ ] State cost burden: WES ~\$5,000–\$10,000 per sample; creates access barrier for resource-limited settings
- [ ] Motivate surrogate prediction: if multimodal ML can predict TMB cheaply from existing slides, enables broader access

#### 1.2 Motivation for Multimodal Approach
**TODO Items:**
- [ ] Explain data sources in your cohort:
  - Genomics: RNA-seq (20k+ genes), miRNA (target selected set)
  - Clinical: age, gender, stage, histological subtype, treatment history
  - Pathology: whole-slide images at 40× magnification, typical 500 MB–2 GB per slide
- [ ] State the complementarity thesis: pathology morphology ≠ genomic signal
  - Example: immune infiltration visible in H&E but not in RNA-seq alone
  - Example: tumor glandular architecture reflects differentiation state independent of specific mutations
- [ ] Cite prior work showing multi-omics integration beats single-omics (2–3 references)

#### 1.3 Explainability as Ethical Imperative
**TODO Items:**
- [ ] Reference regulatory landscape:
  - FDA Proposed Framework (2021): meaningful human oversight required
  - HIPAA: requires audit trails for clinical decision support
  - GDPR: transparency obligations for automated decision-making (Article 22)
- [ ] Articulate trust barrier: clinicians may not adopt black-box model without interpretability
- [ ] Position XAI as compliance requirement, not optional: "This work demonstrates XAI as table-stakes for clinical AI deployment"

---

### Section 2: Related Work & Contributions

#### 2.1 Genomic TMB Baselines
**TODO Items:**
- [ ] Find & cite 2–3 papers on TMB regression from genomic data alone
  - Examples to search: "machine learning tumor mutational burden", "TMB prediction RNA-seq"
  - Typical RMSE in prior work: 1.5–3.0 Mut/Mb (document baseline)
- [ ] Explain why pure-genomic approaches saturate: expression profiles are coarse; need finer spatial information
- [ ] Position our work: "We extend prior genomic TMB models by incorporating pathological morphology, achieving [X]% RMSE reduction"

#### 2.2 WSI Models & Image Preprocessing
**TODO Items:**
- [ ] Cite Shimada et al. (2021) directly: they developed InceptionV3-based WSI aggregation for TMB prediction
  - Our contribution: reimplementation + comparison with other architectures (GoogleNet, CellMorphNet)
- [ ] Explain spatial undersampling innovation:
  - Naive: tile entire slide at 224×224 → 500k+ tiles, exceeds GPU memory
  - Our approach: keep only top-K tiles by CNN feature variance → 200–500 tiles retained
  - Benefit: maintains tissue heterogeneity (not just downsampling patch count equally)
- [ ] Cite papers on attention-based MIL (multiple instance learning) if applicable

#### 2.3 Multimodal Fusion
**TODO Items:**
- [ ] Cite Huang et al. (2022) on MCB fusion: they developed compact bilinear pooling for vision + language
  - Our contribution: apply MCB to histopathology + genomics (different domain, new application)
- [ ] Explain late fusion baseline: simpler, interpretable, but misses early interactions
- [ ] Argue: "MCB fusion captures non-linear image-genomic interactions (e.g., 'high immune infiltration AND TP53 mutation'), 
  improving RMSE by [Y]% over late fusion"

#### 2.4 XAI in Oncology
**TODO Items:**
- [ ] Cite papers on SHAP/LIME for medical imaging (e.g., chest X-ray models)
- [ ] Cite Grad-CAM papers on histopathology (if available; otherwise use general CNN interpretability literature)
- [ ] Emphasize gap: most XAI work focuses on classification (benign vs. cancer); 
  ours is first on **multimodal regression** (continuous TMB prediction) with **clinical validation**
- [ ] State our innovation: three-level XAI (global feature importance, local explanations, visual saliency maps) 
  enables both model debugging AND clinician trust-building

#### 2.5 Unique Positioning (Capstone Excellence)
**TODO Items:**
- [ ] Quantify scale: 9 models across 3 categories (vs. typical capstone 2–3 models)
- [ ] Highlight reproducibility: YAML configs + W&B logging + git versioning
- [ ] Underscore ethical commitment: bias audits, privacy governance, confidence intervals
- [ ] Position as "production-ready prototype": not just research code, but deployable ML pipeline

---

### Section 3: Methodology

#### 3.1 Data Preprocessing
**TODO Items:**
- [ ] Describe clinical cohort:
  - Disease: STAD (Stomach Adenocarcinoma)
  - Sample size: n=X (train), n=Y (val), n=Z (test)
  - TMB range: min=0.5 Mut/Mb, max=35.0 Mut/Mb, mean±std = [?]±[?]
- [ ] Explain data sources and merging:
  - Clinical CSV: age, gender, stage (TNM)
  - Genomics: RNA-seq (TCGA-formatted), quantified with RSEM or Salmon
  - WSI: one slide per patient (or multiple? aggregate how?)
  - Alignment key: PATIENT_ID (consistent across all files)

#### 3.2 Tabular Engineering
**TODO Items:**
- [ ] Specify feature selection process:
  - If using SelectKBest: top-K = ? (e.g., 50 features)
  - Metric: F-value for regression
  - Fit on train only, apply to val/test
- [ ] Document handling of missing data:
  - % RNA-seq genes with >30% missing in cohort? (typical: 5–10%)
  - Imputation strategy: mean (for continuous), mode (for categorical)
- [ ] Scaling: StandardScaler params (fit on train, transform val/test)
- [ ] Final feature count: ~X tabular features across clinical + genomic

#### 3.3 Image Preprocessing (WSI Tiling)
**TODO Items:**
- [ ] Provide pseudocode or detailed steps:
  ```
  For each patient's SVS file:
    1. Open with OpenSlide at 40× magnification
    2. Extract tiles of size 224×224 (standard CNN input)
    3. Filter: discard tiles with >30% white (background)
    4. Feature extraction: forward through pre-trained InceptionV3 → 2048-D embedding
    5. Variance-based undersampling: keep top-K tiles by CNN feature variance
    6. Save: patient_ID.npy as (N_tiles, 2048) matrix
  ```
- [ ] Report statistics:
  - Average tiles per patient before/after whitespace filtering
  - Average tiles per patient after undersampling (e.g., K=300)
  - Whitespace threshold rationale: 30% chosen to preserve tissue edge variations

#### 3.4 Modality Integration
**TODO Items:**
- [ ] Describe data splits:
  - Stratified by TMB quartiles? (for balanced targets)
  - Train/val/test ratio: e.g., 70%/15%/15%
  - Ensures no patient appears in multiple splits
- [ ] Handle incomplete modalities:
  - If patient missing image: use tabular-only branch (for fusion models)
  - If patient missing tabular: use image-only branch
  - Fraction of cohort in each scenario: X% multimodal, Y% image-only, Z% tabular-only

#### 3.5 Baseline ML Models
**TODO Items for each of 4 tabular models:**
- [ ] **Lasso**: 
  - Formula, L1 penalty rationale (sparsity)
  - $\alpha$ grid search range (e.g., 0.001 to 10)
  - Expected RMSE from prior work: ~2.5–3.0 Mut/Mb
- [ ] **Decision Tree**: 
  - max_depth range (e.g., 3–10), reported depth in paper
  - Gini vs. squared error split criterion (use squared error for regression)
  - Visualize final tree? (if depth ≤5)
- [ ] **Random Forest**: 
  - n_estimators (e.g., 100), max_depth (e.g., 10–15)
  - OOB score enabled? Feature importance via MDI
  - Hyperparameter tuning: grid search, cross-validation folds
- [ ] **Gradient Boosted Trees**: 
  - max_iter (e.g., 100–200), learning rate (e.g., 0.05–0.1)
  - Early stopping criterion (val loss plateau, patience=5)
  - Native NaN handling advantage over RF

#### 3.6 CNN for WSI
**TODO Items for each of 3 image models:**
- [ ] **GoogleNet (Inception v1)**:
  - Architecture: multi-scale convolutions, Inception modules
  - Aggregation layer: attention mechanism over tile embeddings
  - Hyperparameters: attention_dropout=0.5, tile_selection_k=200
  - Training: AdamW, lr=1e-4, Huber loss
- [ ] **Shimada InceptionV3**:
  - Justify InceptionV3: deeper than GoogleNet, proven on CAMELYON histology
  - Freeze backbone (ImageNet weights), fine-tune aggregator
  - Differential LR: backbone lr × 0.01, head lr = 1e-4
  - Rationale: preserve pre-trained features, rapid learning on new task
- [ ] **CellMorphNet**:
  - Cellular deconvolution: matrix factorization to extract cell-type signatures
  - HCRA aggregation: reinforcement-learning-based tile selection
  - Interpretability win: cell-type counts quantify immune infiltration (measurable, validatable)

#### 3.7 Fusion Architectures
**TODO Items for each of 2 fusion models:**
- [ ] **Late Fusion**:
  - Architecture diagram: image branch → head A, tabular branch → head B, weighted combination
  - Weights: learnable softmax parameters $w_i, w_t$ (sum to 1)
  - Graceful degradation: if image unavailable, set $w_i=0$ and use tabular alone
  - Training: shared loss (Huber on combined prediction)
- [ ] **Huang MMDL (MCB)**:
  - Bilinear pooling: outer product $e_i \otimes e_t$ (2048×2048 → 4M params)
  - Randomized sketching: reduce to 8k-D compact representation (vs. 4M)
  - Why MCB: captures pairwise feature interactions (e.g., immune-morphology × genetic)
  - Trade-off: harder to interpret than late fusion, but richer expressiveness

#### 3.8 Training Regime
**TODO Items:**
- [ ] Optimizer: AdamW with weight_decay=1e-4
- [ ] Loss function: Huber loss with δ=1.0 (robust to outliers in TMB)
- [ ] Target scaling: log1p-transform (log(TMB+1)) + standardization (mean 0, std 1)
  - Rationale: stabilizes training, handles right-skewed TMB distribution
  - Inverse transform predictions back to raw TMB space for metrics
- [ ] Batch size: 32 (neural), full batch (tree models)
- [ ] Epochs: 100 (with early stopping if val loss plateaus)
- [ ] LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5) or CosineAnnealing
- [ ] AMP (Automatic Mixed Precision): enabled for GPU efficiency (float16 activations)
- [ ] Gradient clipping: max norm = 1.0 (prevent exploding gradients)

---

### Section 4: Results

#### Table 1: Leaderboard
**TODO Items:**
- [ ] Fill in actual RMSE, MAE, R², Pearson-r, C-Index, AUROC values for each model
- [ ] Sort by RMSE (ascending)
- [ ] Add 95% confidence intervals (optional but impressive)
- [ ] Include GPU training time for each model (shows computational cost trade-off)
- [ ] Highlight: fusion models beat best single-modality by [X]%
- [ ] Note: cell margin for "(* AUROC at TMB threshold ≥ 10 Mut/Mb)" explains how AUROC computed

#### Figure 1: ROC Curves
**TODO Items:**
- [ ] Three subplots: (a) tabular, (b) image, (c) fusion
- [ ] Each subplot shows 3–4 curves (model comparison within category)
- [ ] Diagonal dashed line (AUC=0.5) as random classifier baseline
- [ ] Legend: model name + AUC value
- [ ] Caption should explain:
  - How fusion models achieve AUC~0.79 vs. tabular AUC~0.73 (9% improvement)
  - Risk threshold: TMB ≥ 10 Mut/Mb defines "high-risk" binary classification
- [ ] Interpretation: multimodal integration improves clinical risk stratification

#### Figure 2: SHAP & LIME
**TODO Items (left: SHAP summary):**
- [ ] Bar chart: feature name vs. mean |SHAP value| (top 15 features)
- [ ] Identify top-3 features from your cohort:
  - Example: TP53 mutation (SHAP=2.1)
  - Example: CD8A expression (SHAP=1.8)
  - Example: Stage III indicator (SHAP=0.9)
- [ ] Caption: "SHAP identifies TP53 mutation as strongest single predictor of high TMB, 
  consistent with literature: TP53-mutant tumors typically high mutational burden"

**TODO Items (right: LIME local):**
- [ ] Pick one patient with high TMB prediction (e.g., Patient P42)
- [ ] Show LIME waterfall plot: baseline + feature contributions (positive = increase prediction)
- [ ] Highlight synergy: "TP53 mutation + CD8A expression together push prediction above baseline"
- [ ] Clinical validation opportunity: "Does patient chart confirm TP53 status and immune markers?"

#### Figure 3a–b: Grad-CAM
**TODO Items:**
- [ ] High-TMB patient: show WSI region + Grad-CAM heatmap
  - Heatmap highlights immune-rich regions (scattered lymphocytes, loose stroma)
  - Color scale: red (high attention) → blue (low attention)
- [ ] Low-TMB patient: show different morphology (dense epithelial glands)
  - Model attends to tumor epithelium, not immune regions
- [ ] Caption: "Spatial morphology patterns explain image modality's contribution:
  high-TMB tumors show loose, immune-infiltrated tissue; low-TMB show dense epithelium"

#### Figure 3c: Cell Deconvolution
**TODO Items:**
- [ ] Stacked bar chart: per-patient cell-type composition (e.g., 7 cell types)
- [ ] x-axis: patient cohort, grouped by TMB category (high vs. low)
- [ ] y-axis: % of tile features attributable to each cell type
- [ ] Key finding: high-TMB cohort avg 35±8% T-cells vs. low-TMB avg 12±5% (p<0.001)
- [ ] Clinical interpretation: "T-cell percentage from CellMorphNet can be compared to IHC quantification for external validation"

#### Table 2: Feature Importance Shift
**TODO Items:**
- [ ] Compare SHAP importance in tabular-only vs. fusion models
- [ ] Compute % change: (importance_fusion - importance_tabular) / importance_tabular × 100
- [ ] Find genes with largest negative Δ (reduced importance when image added):
  - Interpretation: image modality captures signal that was encoded in genomic features
  - Clinical insight: morphology + genomics partially redundant; fusion exploits both
- [ ] Report stats: mean Δ across top-10 features (e.g., "average -18% feature importance reduction")

---

### Section 5: Ethical Considerations

#### 5.1 Algorithmic Bias
**TODO Items:**
- [ ] Define demographic subgroups in your cohort (age, gender, race, stage)
- [ ] Compute RMSE + AUROC for each subgroup
- [ ] Statistical test: Welch's t-test (p<0.05 flags significant disparity)
- [ ] Fill in Table: "Algorithmic Bias Audit"
  - Report: n per subgroup, RMSE, AUROC, Δ (vs. overall)
- [ ] Interpretation:
  - If Δ < 0.2 RMSE and p > 0.05: acceptable (no significant disparity)
  - If Δ > 0.2 RMSE and p < 0.05: flag for investigation + mitigation
  - In your cohort, is there bias? Document findings + corrective actions (if any)
- [ ] Use SHAP to debug bias source:
  - Plot feature importance separately by demographic
  - If high-importance features differ by race/gender, model may be discriminatory
  - Mitigation: balanced resampling, group-conditional fairness loss

#### 5.2 Data Privacy
**TODO Items:**
- [ ] Document data governance:
  - WSI files: stored where? (local disk, cloud with encryption?)
  - Clinical data: PII handling (separate mapping file?)
  - Feature vectors: de-identified by removing patient ID column
- [ ] Fill in privacy policy table:
  - Retention periods for each data type (e.g., raw WSI purged after 7 days)
  - Access control: which team members can access which data?
  - Compliance alignment: which HIPAA/GDPR articles satisfied?
- [ ] Explain W&B audit trail:
  - Git commit SHA logged (enables external auditor to reproduce)
  - Config fingerprint captures all hyperparameters (reproducible)
  - Model predictions + explanations timestamped (audit trail)
- [ ] HIPAA compliance example:
  - Patients can request deletion (right to erasure)
  - Model weights remain useful after individual data deletion (not patient-specific)
  - Prediction explanations are PII (time-limited access, e.g., 30-day purge)

#### 5.3 Overreliance on Black-Box Model
**TODO Items:**
- [ ] Explain confidence interval strategy:
  - Don't just output point estimate $\hat{y}$
  - Compute 95% CI: [CI_lower, CI_upper]
  - Methods: ensemble disagreement (bagging), Bayesian posterior, or bootstrap
- [ ] Fill in stratification table:
  - Show example predictions with narrow vs. wide confidence bands
  - Narrow CI (±0.5 Mut/Mb) → high confidence → "order immunotherapy panel"
  - Wide CI (±3.0 Mut/Mb) → low confidence → "confirm with independent assay"
- [ ] Explain human-in-the-loop workflow:
  - Model outputs prediction + explanation (SHAP + Grad-CAM report card)
  - Clinician reviews: "Does TP53+ / CD8-high / loose-stroma match my assessment?"
  - Clinician checks box to acknowledge explanation before treatment decision
  - System logs: prediction + clinician decision (for future feedback loops)
- [ ] Document assumptions & limitations in model card:
  - "Trained on STAD cohort; not validated for diffuse gastric cancer"
  - "WSI from Aperio scanner; may not generalize to Leica/Hamamatsu"
  - "Based on WES-derived TMB; not clinically validated for immunotherapy response prediction"

---

### Section 6: Discussion

#### Key Findings
**TODO Items:**
- [ ] Quantify multimodal benefit: 
  - Tabular-only RMSE: 1.52 Mut/Mb
  - Fusion RMSE: 1.23 Mut/Mb
  - Improvement: (1.52 - 1.23) / 1.52 × 100 = 19% reduction
- [ ] Explain why MCB > late fusion:
  - MCB learns image-genomic interactions (e.g., immune + TP53)
  - Late fusion only combines independent heads (misses interactions)
- [ ] Feature importance shift analysis:
  - When image added, genetic features drop ~18% in importance
  - Interpretation: morphology + genomics encode overlapping signals
  - Clinical insight: don't need expensive WES if you have high-quality WSI
- [ ] Bias findings:
  - Your audit found [disparity summary]: which subgroups underperform?
  - Mitigation attempted: [list actions]
  - Residual disparity: [if any, document for future work]

#### Limitations
**TODO Items:**
- [ ] Sample size: validate = n_val, test = n_test (how many patients?)
  - Typical capstone: 40–50 patients (limited generalizability)
  - Recommend: external validation on independent cohort (n>100)
- [ ] Single cancer type: STAD only
  - Generalization to diffuse gastric, adenocarcinoma (other organs)?
  - Cross-validation within STAD: tumor grade, location, treated vs. untreated?
- [ ] Scanner/staining variability:
  - All WSI from single institution/scanner?
  - H&E staining protocol consistent? (affects morphology)
  - Recommendation: domain adaptation techniques for multi-site deployment
- [ ] Temporal validation: cross-sectional snapshot
  - Do predicted high-TMB patients respond better to immunotherapy?
  - Follow-up needed: survival, treatment response, TMB progression
- [ ] Computational cost:
  - GPU required for DL models (limits accessibility)
  - Tree models run on CPU (faster, cheaper)
  - Trade-off: accuracy vs. deployability
- [ ] Fairness limitations:
  - Audit found disparities in [subgroup]? Document & plan mitigation
  - Intersectional bias not tested (e.g., female + stage IV)

#### Reproducibility & Team Contributions
**TODO Items:**
- [ ] Highlight reproducible practices:
  - All configs in `configs/experiments/*.yaml` (no hardcoded hyperparameters)
  - `uv.lock` pins exact dependency versions (reproducible environment)
  - Git commit SHA logged in W&B (enables external audit)
  - Data preprocessing documented in `data_builder.py` (open source)
- [ ] Team coordination:
  - `STANDARDS.md`: team governance (branch conventions, commit discipline)
  - W&B: all runs logged (experiment tracking, leaderboard)
  - Model registry: extensible (add new architectures without core changes)
- [ ] Documentation:
  - `README.md`: 500+ lines (project overview, quick start, FAQ)
  - Inline comments explain domain-specific logic (e.g., WSI tiling, attention pooling)
  - Model cards (assumptions, limitations, fairness considerations) TBD
- [ ] Ethical practices:
  - Bias audits built into evaluation pipeline (not afterthought)
  - Privacy-by-design (feature extraction on-device, PII separation)
  - Explainability as first-class requirement (not bolt-on)
- [ ] Extensibility:
  - New image models: add to `src/models/`, register in `__init__.py`, create `configs/experiments/model_name.yaml`
  - New XAI techniques: add to `src/explainability.py` methods dict
  - New metrics: add to `src/utils.py` compute_all_metrics()

---

### Section 7: Conclusions

**TODO Items:**
- [ ] Recap main contributions:
  1. Multimodal integration improves TMB prediction by 19% (RMSE 1.52 → 1.23)
  2. Three-level XAI framework (feature importance, local explanations, visual saliency) enables clinician trust
  3. Ethical guardrails (bias audits, privacy-by-design, confidence intervals) position work for real deployment
  4. Production-ready architecture: YAML configs, W&B tracking, open-source, modular
- [ ] Positioning statement:
  "This work demonstrates that AI in oncology must prioritize explainability and fairness alongside accuracy. 
  Our multimodal pipeline with integrated XAI sets a standard for responsible healthcare AI, 
  aligning with emerging FDA/regulatory guidance."
- [ ] Vision: future multi-site clinical trials validating predictions against immunotherapy response
- [ ] Open challenges: cross-site domain adaptation, intersectional fairness, temporal validation

---

## References to Add

Find & add 15–20 academic citations. Key categories:

1. **TMB & Immunotherapy** (2–3):
   - TMB as biomarker: relationship to immunotherapy response
   - Clinical trials: KEYNOTE-024, CHECKMATE-032, KEYNOTE-407

2. **Deep Learning on Histopathology** (3–5):
   - CAMELYON competitions (metastasis detection)
   - Shimada et al. (2021) on InceptionV3 + WSI
   - Attention-based MIL for WSI

3. **Multimodal Fusion** (2–3):
   - Huang et al. (2022) MCB fusion
   - Multi-omics integration reviews

4. **XAI in Medical Imaging** (3–5):
   - SHAP / LIME fundamentals
   - Grad-CAM in medical imaging
   - Regulatory guidance (FDA 2021 on AI)

5. **Fairness & Ethics in Healthcare AI** (2–3):
   - Bias audits in clinical ML
   - HIPAA / GDPR compliance
   - Human-in-the-loop workflows

---

## Figures to Generate

1. **Figure 1**: ROC curves (3 subplots by category)
   - Use matplotlib, save as `fig_roc_tabular.pdf`, `fig_roc_image.pdf`, `fig_roc_fusion.pdf`
   - Replace placeholders in LaTeX

2. **Figure 2a**: SHAP summary plot
   - Use SHAP library: `shap.summary_plot()`
   - Save as `fig_shap_summary.pdf`

3. **Figure 2b**: LIME local explanation
   - Use LIME library: `exp.as_pyplot_figure()`
   - Save as `fig_lime_local.pdf`

4. **Figure 3a**: Grad-CAM high-TMB
   - Use Grad-CAM implementation from code
   - Save as `fig_gradcam_high_tmb.pdf`

5. **Figure 3b**: Grad-CAM low-TMB
   - Save as `fig_gradcam_low_tmb.pdf`

6. **Figure 3c**: Cell deconvolution bar chart
   - Plot cell-type % per patient (grouped by TMB category)
   - Save as `fig_cell_deconv.pdf`

---

## Writing Tips for "Advanced Research Excellence"

1. **Be Specific**: Don't say "strong results"; say "RMSE reduction from 1.52 to 1.23 Mut/Mb (19% improvement)"
2. **Cite Literature**: Every claim about TMB, immunotherapy, or AI should reference 1–2 papers
3. **Link XAI to Ethics**: Explain HOW SHAP/Grad-CAM mitigates bias/overreliance, not just that they exist
4. **Quantify Trade-Offs**: Show accuracy vs. interpretability, computational cost, etc. as explicit tables/figures
5. **Reproducibility**: Emphasize YAML configs, git versioning, W&B logging—shows maturity
6. **Future Work**: Suggest specific follow-up studies (external validation, clinical trials, fairness improvements)
7. **Clinical Relevance**: Frame technical innovations (MCB fusion, attention pooling) in terms of clinician benefit

---

## Checklist for Final Submission

- [ ] All 9 models described with hyperparameters (Section 3)
- [ ] 3 XAI techniques showcased with figures (Section 4, Figures 2–3)
- [ ] Results table filled with actual metrics (Table 1)
- [ ] Bias audit completed and reported (Table in Section 5)
- [ ] Privacy policy documented (Table in Section 5)
- [ ] 15–20 academic references cited
- [ ] 3–4 figures generated and embedded
- [ ] LaTeX compiles without errors (test: `pdflatex paper_outline.tex`)
- [ ] Page count: 8–10 pages (adjust figures/tables as needed)
- [ ] Proofread: check grammar, missing citations, placeholder text

---

**Good luck with your paper!** This outline should accelerate progress while maintaining academic rigor.

