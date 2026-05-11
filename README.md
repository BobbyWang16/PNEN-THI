<div align="center">

# 🔬 PNET-ITH

### Pancreatic Neuroendocrine Tumor — Intra-tumoral Heterogeneity Analysis & Ordinal Grading

<p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-≥3.9-blue.svg" alt="Python"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-≥1.0-orange.svg" alt="scikit-learn"></a>
  <a href="https://pyradiomics.readthedocs.io/"><img src="https://img.shields.io/badge/PyRadiomics-≥3.0-red.svg" alt="PyRadiomics"></a>
  <img src="https://img.shields.io/badge/patients-546-blueviolet.svg" alt="Cohort">
  <img src="https://img.shields.io/badge/centers-5-yellow.svg" alt="Centers">
  <img src="https://img.shields.io/badge/license-Academic-green.svg" alt="License">
</p>

<p><b>A multi-center, multi-modal radiomics framework that integrates dual-phase CT intra-tumoral heterogeneity (ITH) analysis with ordinal machine-learning classifiers to non-invasively predict the WHO grade (G1 / G2 / G3) of Pancreatic Neuroendocrine Tumors (P-NEN).</b></p>

<p>
  <a href="#-highlights">Highlights</a> ·
  <a href="#-pipeline-overview">Pipeline</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-stage-a--tumor-heterogeneity-clustering">Algorithm</a> ·
  <a href="#-citation">Citation</a> ·
  <a href="说明文档.md">📖 中文文档</a>
</p>

</div>

---

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [📐 Pipeline Overview](#-pipeline-overview)
- [📊 Dataset](#-dataset)
- [🗂️ Repository Layout](#️-repository-layout)
- [🔬 Stage A — Tumor Heterogeneity Clustering](#-stage-a--tumor-heterogeneity-clustering)
- [📈 Stage B — Heterogeneity Feature Engineering](#-stage-b--heterogeneity-feature-engineering)
- [🎯 Stage C — Ordinal Grade Prediction](#-stage-c--ordinal-grade-prediction)
- [🔍 Stage D — Interpretation & Sub-analyses](#-stage-d--interpretation--sub-analyses)
- [🚀 Quick Start](#-quick-start)
- [✅ Quality Guarantees](#-quality-guarantees)
- [💡 Key Design Choices](#-key-design-choices)
- [🔁 Reproducibility](#-reproducibility)
- [📚 Documentation Map](#-documentation-map)
- [📖 Citation](#-citation)
- [📝 License](#-license)

---

## ✨ Highlights

- 🧠 **Prior-guided ConstrainedKMeans** — Learns radiologist-annotation priors and enforces hard cluster-size constraints (≥ 5%) to overcome empty/tiny-cluster failure modes of vanilla GMM/KMeans on small lesions.
- 🔒 **Three guaranteed invariants** — Every patient ends up with **exactly 3 HU-ordered, biologically interpretable sub-regions** (Necrosis → Stroma → Hyper-vascular).
- 📊 **Three complementary heterogeneity indices** — DDI (distribution) + DEHI (dynamic enhancement) + SHI (spatial) = **120-D THI** feature block.
- 🎯 **Ordinal-aware classification** — 6 models × 7 feature combinations × 3 selectors, scored with **Quadratic Weighted Kappa** that matches the clinical cost of mis-grading.
- 🏥 **Multi-center validation** — 546 patients across 5 centers (TJ / ZS / HN / XY / SX), with inter-/intra-observer ICC reproducibility study.

---

## 📐 Pipeline Overview

```text
  ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
  │ Dual-phase CT│───►│   Stage A      │───►│   Stage B       │───►│   Stage C      │
  │  + Tumor ROI │    │ ITH Clustering │    │ Heterogeneity   │    │ Ordinal Grading│
  │  + DOT (opt) │    │  (3 sub-regs)  │    │ Feature Engin.  │    │  G1 < G2 < G3  │
  └──────────────┘    └────────────────┘    └─────────────────┘    └────────────────┘
                             │                      │                      │
                             ▼                      ▼                      ▼
                      *_clusters.nii.gz      feature_thi.xlsx        results/best/
                                             feature_rad.xlsx             + SHAP
                                                                          + ICC
                                                                          + survival
```

---

## 📊 Dataset

| Item | Value |
|---|---|
| **Total patients** | **546** |
| **Modalities** | Arterial-phase CT · Venous-phase CT · Tumor ROI mask (NIfTI) |
| **Outcome** | WHO Grade ∈ {G1, G2, G3} (ordinal; *Cancer treated as G3 by treatment equivalence*) + (time, event) |
| **Cohort split** | TJ = **425** (train = 297 / inter-test = 128) — primary site<br>External = **121** (ZS = 56, HN = 29, XY = 26, SX = 10) |
| **Grade distribution** | G1 = 226, G2 = 228, G3 = 92 — encoded 0 / 1 / 2 |
| **Feature blocks** | Clinical (24) · Radiomics (200) · Heterogeneity / THI (120) |

> Stratified by the **(Grade × Event)** joint label so that grade imbalance and survival imbalance are jointly preserved across folds. See [group.py](group.py).

---

## 🗂️ Repository Layout

<details>
<summary><b>Click to expand full tree</b></summary>

```text
PNET-ITH/
├── data/                                  # Raw and processed inputs
│   ├── crop_A/   crop_V/   crop_ROI/      # NIfTI (mask voxel = 2 = tumor)
│   ├── crop_DOT/                          # Expert point annotations (optional)
│   ├── modeling_data.xlsx                 # name, hospital, grade, event, time
│   └── inter.csv / exter.csv              # Train / inter-test / external splits
│
├── clinical/                              # Univariate + multivariate clinical analysis
├── cluster/
│   ├── origin/                            # Legacy GMM / KMeans baseline (no prior)
│   ├── without_point/                     # ConstrainedKMeans without prior (ablation)
│   ├── with_point/                        # Prior-guided ConstrainedKMeans  ★ main
│   └── point_prior_knowledge.pkl
├── ICC/                                   # Inter- / intra-observer reproducibility
├── rad/                                   # PyRadiomics tables + analysis notebooks
├── score/                                 # Heterogeneity feature tables (modeling inputs)
│   ├── feature_thi.xlsx   (546 × 120)
│   ├── feature_rad.xlsx   (546 × 200)
│   ├── feature_ddi.xlsx   (546 × 68)
│   ├── feature_dehi.xlsx  (546 × 30)
│   └── feature_shi.xlsx   (546 × 10)
├── results/                               # Final figures, SHAP, subgroup, survival
│
├── ───── Core scripts ─────
├── point_cluster.py        # ★ Prior-guided ConstrainedKMeans (main pipeline)
├── nonpoint_cluster.py     #   No-prior ablation (same path minus prior init)
├── ith_analysis.py         #   Legacy GMM/KMeans + necrosis spatial prior
├── DDI.py / DEHI.py / SHI.py   # Three heterogeneity indices
├── cal_rad_feature.py      # PyRadiomics feature extraction
├── group.py                # Stratified train / inter-test / external split
├── hyper_tune.py           # 7 × 3 × 6 GridSearchCV hyper-parameter search
│
├── ───── Modeling notebooks ─────
├── classifier.ipynb        # ★ Final ordinal classifier (6 models × 7 combos)
├── clinical.ipynb          # Clinical statistics
├── icc.ipynb               # ICC study
├── figure.ipynb            # Publication-grade figures
│
└── README.md / 说明文档.md / ith_analysis.md
```

</details>

---

## 🔬 Stage A — Tumor Heterogeneity Clustering

Three biologically interpretable sub-regions per tumor:

<div align="center">

| Label | Region | HU |
|:-:|:-:|:-:|
| **1** | Necrosis / low-activity | 🟦 lowest  |
| **2** | Stroma / mid-activity   | 🟩 middle  |
| **3** | Hyper-vascular / rich   | 🟥 highest |

</div>

### Per-voxel 9-D feature

```python
[ weighted_a_hu,    # 0  HU × arterial_weight (2.0)
  a_texture,        # 1  3×3 std filter
  a_gradient,       # 2  3-axis Sobel magnitude
  V − A,            # 3  ★ core enhancement difference
  V / A,            # 4  enhancement ratio
  |V − A|,          # 5
  V_HU,             # 6
  (weighted_a + V) / 2,   # 7  fused mean
  sign(V − A) ]     # 8  ∈ {-1, 0, 1}
```

### ConstrainedKMeans — two-stage greedy assignment

```python
def _constrained_assign(X, centers, min_size):
    # Stage 1: each cluster receives its min_size nearest samples first
    # Stage 2: remaining samples are assigned to their nearest cluster
```

**Hard guarantee**: `min(cluster_size) ≥ 0.05 × n_voxels`.

### Triple post-hoc validation

1. **Force exactly 3 clusters** — split the largest cluster (KMeans k=2) or merge the smallest into its nearest neighbour.
2. **Min-fraction ≥ 5 %** — iteratively merge under-sized clusters into the nearest cluster (max 5 iterations).
3. **HU-ascending re-ordering** — enforce `μ₁ < μ₂ < μ₃` (Necrosis < Stroma < Hyper-vascular).

> Detailed walkthrough in [说明文档.md §2](说明文档.md) and [ith_analysis.md](ith_analysis.md).

---

## 📈 Stage B — Heterogeneity Feature Engineering

| Index | Script | Features | What it captures |
|---|---|---|---|
| **DDI** — Distribution | [DDI.py](DDI.py) | 68 / phase | Between-/within-variance, centroid separation, histogram overlap, shape diversity (skew / kurt / entropy / IQR) |
| **DEHI** — Dynamic Enhancement | [DEHI.py](DEHI.py) | 30 | Per-cluster strong / mild / no / wash-out ratios, A–V Pearson correlation, uniformity |
| **SHI** — Spatial | [SHI.py](SHI.py) | 10 | Spherical-coordinate descriptors: radial concentration, azimuthal Gini, polar hemisphere bias, pattern diversity |
| **THI** (composite) | merged | **120** | DDI(A+V) + DEHI + SHI + cluster fractions → `feature_thi.xlsx` |
| **PyRadiomics** | [cal_rad_feature.py](cal_rad_feature.py) | **200** | Shape + first-order + GLCM / GLRLM / GLSZM / GLDM (both phases) |

DDI composite (illustrative):

```text
DDI = 0.30·Imbalance + 0.25·Separation + 0.25·Overlap_sep + 0.20·Shape_div
```

---

## 🎯 Stage C — Ordinal Grade Prediction

### Feature selection — fit on training fold only

`OrdinalFeatureSelector` (in [classifier.ipynb](classifier.ipynb)):

```text
Missing/Variance ──► Kruskal-Wallis ──► Mutual Information ──► Spearman ρ ──► Top-K rank
 (>10% / <0.01)         (p > 0.05)          (< median MI)         (|ρ| ≥ 0.90)   (0.6·KW + 0.4·MI)
```

### Models × Combos

<table>
<tr>
<th>Models (6)</th>
<th>Feature combos (7)</th>
</tr>
<tr>
<td>

- `LR` — L2 / saga / C=0.005
- `SVM` — RBF / C=1.0 / γ=0.01
- `DT` — depth 3
- `RF` — n=200 / depth 7
- `GBDT` — n=200 / lr=0.01
- `KNN` — k=15 / Manhattan

</td>
<td>

- Clinical
- Radiomics
- Heterogeneity (THI)
- Clinical + Radiomics
- Clinical + Heterogeneity
- Radiomics + Heterogeneity
- **All** (C + R + H)

</td>
</tr>
</table>

All models use `class_weight='balanced'` to handle the G3 minority (~17 %) without SMOTE leakage.

### Hyper-parameter search — [hyper_tune.py](hyper_tune.py)

```text
7 feature combos × 3 selector profiles (loose / standard / strict) × 6 models = 126 configurations
StratifiedKFold(5, shuffle=True, random_state=42), scoring = roc_auc_ovr
```

### Ordinal-aware metrics

| Metric | Why we use it |
|---|---|
| **QWK** (Quadratic Weighted Kappa) | **Primary** — penalises cross-grade errors more than adjacent ones |
| **RPS** (Ranked Probability Score) | Calibration on cumulative class probabilities |
| **Macro / Micro / Weighted AUC (OvR)** | Discrimination ability |
| **Ordinal MAE** | Expected ordinal distance `mean(|y - ŷ|)` |
| **Accuracy / F1 / PPV / NPV / Sensitivity / Specificity** | Per-class diagnostics |

---

## 🔍 Stage D — Interpretation & Sub-analyses

| Analysis | Location | Purpose |
|---|---|---|
| **SHAP** | `results/shap*` | Global + local feature attribution for the final ordinal model |
| **Ki-67 correlation** | `results/figures_THI_Ki67/` ([figure.ipynb](figure.ipynb)) | Validate THI vs. Ki-67 proliferation index |
| **Inter-/intra-rater ICC** | `ICC/` ([icc.ipynb](icc.ipynb)) | Reproducibility under 2 radiologists × 2 rounds |
| **Tumor-size sub-group** | `results/subgroup/` | ≤ 2 cm vs. > 2 cm performance |
| **Survival** | `results/survival0/` | Kaplan–Meier / Cox on (time, event) |
| **Prior-vs-no-prior ablation** | [figure.ipynb](figure.ipynb) cell 5 | Quantify gain from expert prior |

---

## 🚀 Quick Start

### 1. Install

```bash
# Python ≥ 3.9 recommended
pip install numpy pandas nibabel SimpleITK pyradiomics \
            scikit-learn scipy seaborn matplotlib shap \
            statsmodels openpyxl tqdm
```

### 2. Data layout

```text
data/
├── crop_A/<id>.nii.gz       # Arterial-phase CT
├── crop_V/<id>.nii.gz       # Venous-phase CT
├── crop_ROI/<id>.nii.gz     # Mask (voxel value = 2 inside tumor)
└── crop_DOT/<id>.nii.gz     # (Optional) expert annotations: voxel 1/2/3 → 3 classes
```

### 3. End-to-end run

```bash
# (1) Prior-guided clustering — main pipeline
python point_cluster.py
#   → cluster/point_prior_knowledge.pkl
#   → cluster/with_point/<id>_clusters.nii.gz
#   → cluster/with_point/results_summary.csv

# (2) No-prior ablation (optional)
python nonpoint_cluster.py
#   → cluster/without_point/<id>_clusters.nii.gz

# (3) Heterogeneity features
python DDI.py    # → score/DDI/{A,V}_features.csv
python DEHI.py   # → score/DEHI/DEHI_features.csv
python SHI.py    # → score/SHI/spatial_heterogeneity_features.csv

# (4) PyRadiomics features
python cal_rad_feature.py  # → data/V_radiomics.csv (and similarly for A)

# (5) Train / inter / external split
python group.py            # → data/inter.csv, data/exter.csv

# (6) Hyper-parameter search (optional)
python hyper_tune.py       # → results/hyper_tune/

# (7) Final classifier
jupyter lab classifier.ipynb
#   → results/best/{results, selected_features_summary.txt, results_proba.xlsx}
```

### 4. Key parameters

| Script | Parameter | Default | Notes |
|---|---|---|---|
| `point_cluster.py` | `ARTERIAL_WEIGHT` | `2.0` | Arterial-HU weight in feature vector |
| `point_cluster.py` | `MIN_CLUSTER_FRACTION` | `0.05` | Hard cluster-size constraint |
| `point_cluster.py` | `n_clusters` | `3` | Fixed (biological tri-partition) |
| `DDI.py` | `bins / hu_range` | `200 / (-160, 240)` | Histogram config |
| `DEHI.py` | `hu_range` | `(-200, 200)` | Voxel HU filter |
| `SHI.py` | `voxel_size` | `(1.0, 1.0, 1.0) mm` | Spacing for spherical geometry |
| `group.py` | `test_size` | `0.3` | Inter-test fraction of TJ cohort |

---

## ✅ Quality Guarantees

For **every** patient processed by the prior-guided pipeline, the following invariants are **hard-enforced** by post-processing:

```diff
+ ✓ Exactly 3 clusters present (Label ∈ {1, 2, 3})
+ ✓ Each cluster contains ≥ 5 % of tumor voxels
+ ✓ Mean HU is monotonically increasing: μ₁ < μ₂ < μ₃
+ ✓ Labels are biologically interpretable: Necrosis → Stroma → Hyper-vascular
```

These invariants ensure that downstream heterogeneity descriptors and the final ordinal classifier operate on a **consistent, semantically aligned representation** across the entire 546-patient cohort and any new patient processed with the same pipeline.

---

## 💡 Key Design Choices

| Component | Choice | Why |
|---|---|---|
| **Clustering** | `ConstrainedKMeans` + learned prior | Hard guarantee against empty / < 5 % clusters that vanilla GMM / KMeans produce on small lesions |
| **n_clusters** | **Fixed at 3** | Aligns with histopathological tri-partition (necrosis / stroma / hyper-vascular) |
| **Label semantics** | **HU-ascending re-ordering** | Removes the permutation ambiguity inherent in clustering — `Class 1` is always lowest density |
| **Phase fusion** | `0.6 × A + 0.4 × (A − V)` | Arterial dominance with venous-enhancement complement |
| **Feature scaling** | `RobustScaler` (q5–q95) | CT HU contains outliers from beam-hardening / partial volume; IQR-based scaling is more stable |
| **Class imbalance** | `class_weight='balanced'` (no SMOTE) | Avoids over-sampling leakage; G3 ~ 17 % minority handled at model level |
| **Model selection** | Six diverse classifiers | Covers linear (LR), kernel (SVM-RBF), and tree-ensemble (RF / GBDT) decision boundaries |
| **Evaluation metric** | **Quadratic Weighted Kappa** | Penalises larger-rank errors more than adjacent — matches clinical cost of mis-grading |
| **External validation** | 4 hospitals | Tests transportability across scanner vendors / acquisition protocols |

---

## 🔁 Reproducibility

- `random_state = 42` is fixed throughout splitting, KMeans, SMOTE-not-used, and model fitting.
- The prior file `cluster/point_prior_knowledge.pkl` is a pickled `dict`; **regenerate** it via `learn_prior_knowledge()` in [point_cluster.py](point_cluster.py) whenever the annotation set changes.
- The `ICC/` study uses a held-out subset annotated by **2 radiologists at 2 timepoints**, reporting ICC for cluster fractions, DDI, DEHI, SHI and downstream model probabilities.
- Notebook outputs are heavy; strip them before commit:
  ```bash
  jupyter nbconvert --clear-output --inplace *.ipynb
  ```

---

## 📚 Documentation Map

| File | Language | Content |
|---|---|---|
| **README.md** | English | This file — GitHub overview |
| [说明文档.md](说明文档.md) | 中文 | Full framework specification (recommended for in-depth reading) |
| [ith_analysis.md](ith_analysis.md) | 中文 | Stage-A clustering algorithm in depth |

---

## 📖 Citation

If you use this code or any of its derived feature tables, please cite the accompanying manuscript (under review) and acknowledge the **PNET-ITH** framework:

```bibtex
@misc{pnet-ith,
  title  = {PNET-ITH: Pancreatic Neuroendocrine Tumor Intra-tumoral
            Heterogeneity Analysis and Ordinal Grading Prediction},
  note   = {Manuscript under review},
  year   = {2026}
}
```

---

## 📝 License

For **academic and non-commercial research** use. Please contact the authors for clinical or commercial licensing.

---

<div align="center">

If you find this work useful, please ⭐ the repo — it helps others discover it.

</div>
