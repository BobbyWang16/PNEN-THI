<div align="center">

# PNEN-THI

**Dual-phase CT habitats for preoperative three-tier grading**

Expert-initialized habitats · Interpretable heterogeneity · Frozen inference

[English](README.md) &nbsp; · &nbsp; [简体中文](README_zh.md)

[Quick start](#quick-start) &nbsp; / &nbsp; [Input data](#input-data) &nbsp; / &nbsp; [Core method](#core-method) &nbsp; / &nbsp; [Image scores](#image-scores) &nbsp; / &nbsp; [Grading](#grading) &nbsp; / &nbsp; [Training](#training)

</div>

<a href="assets/study_workflow.png"><img src="assets/study_workflow.png" alt="Study workflow: dual-phase CT, expert-initialized tumor habitats, heterogeneity and radiomics scores, and three-tier grading of pancreatic neuroendocrine neoplasms" width="100%"></a>

**From expert points to tumor grade.** PNET-ITH connects arterial- and venous-phase CT, three tumor habitats, interpretable image scores, and G1 / G2 / G3/NEC grading of pancreatic neuroendocrine neoplasms.

## Overview

**`predict.py` handles all inference; `train.py` refits the selected model specifications.** Image preparation, point-prior learning, constrained clustering, and feature calculation live in `preprocessing/`. The release includes a learned aggregate prior, selected fitted models, and one real deidentified image case. The case demonstrates the imaging branch and contains no clinical measurements, outcome, or expert-point annotations.

**Inference scope.** The frozen Model 8 and Model 9 components, clinical transformations, and THI/RAD score parameters are provided for independent prediction. Each model requires complete values for its listed clinical predictors. The original training-fitted Bayesian-Ridge chained-equation clinical imputers are not included; the release therefore does not reproduce the study's handling of missing clinical inputs. The supplied image case demonstrates habitat and image-score calculation; full grading additionally requires the selected model's clinical measurements.

## Setup

Use **Python 3.9** for the full workflow, including PyRadiomics:

```bash
python -m pip install -r preprocessing/requirements-radiomics.txt
```

For THI, inference from existing features, or training without radiomics extraction, install the base dependencies; this branch also supports **Python 3.12**:

```bash
python -m pip install -r requirements.txt
```

Full extraction was verified with Python 3.9.25, PyRadiomics 3.1.0, NumPy 1.26.4, SciPy 1.13.1, and SimpleITK 2.5.5. Model loading and training use scikit-learn 1.6.1. Run the commands below from the repository root.

## Input data

Use the same deidentified filename across folders:

```text
data/
├── A/case_001.nii.gz       # arterial CT in source HU; fixed reference
├── V/                     # original venous CT, before registration
├── V2A/case_001.nii.gz     # V registered and resampled to A
├── ROI/case_001.nii.gz     # segmentation on A: 0 background, 2 tumor
└── DOT/                   # sparse expert labels on A: 0 unmarked, 1/2/3 classes
```

`A`, `V2A`, and `ROI` must share a **1-mm isotropic grid**, shape, and affine; check anatomical registration as well as geometry. ROI may contain non-target label 1. `DOT` is needed only to learn a new prior. The supplied case includes `A`, `V2A`, and `ROI`; other patient images remain ignored by Git by default.

See the [data and preparation guide](data/README.md) for DICOM conversion, segmentation, registration, point annotation, and the complete calculation workflow.

## Quick start

Validate and organize the real case, then calculate habitats, all 18 heterogeneity features, and THI:

```bash
python -m preprocessing.prepare --data-root data --case-id case_001 --output-dir outputs/cases/case_001
python predict.py --thi-only --case outputs/cases/case_001 --output-dir outputs/demo
```

Preparation copies the images to `arterial.nii.gz`, `venous.nii.gz`, and `tumor_roi.nii.gz`, preserving values and geometry; optional `DOT` becomes `points.nii.gz`. Existing targets are not overwritten. Inference automatically loads **[`point_prior_n100.pkl`](point_prior_n100.pkl)** and writes `habitats.nii.gz`, `heterogeneity_features.json`, and `result.json`.

With the full radiomics environment, calculate both THI and RAD, also saving `radiomics_features.json`:

```bash
python predict.py --score-only --case outputs/cases/case_001 --output-dir outputs/demo_scores
```

## Core method

### 1 · Point-driven prior learning

Prior learning uses **A, ROI, and DOT only**. Each prepared training case needs `arterial.nii.gz`, `tumor_roi.nii.gz`, and `points.nii.gz` on the same grid. Expert labels **1/2/3** must be represented in the pooled training annotations; 0 is unmarked, and only points inside the tumor contribute.

At each point, arterial HU is transformed as `clip((HU + 160) / 400, 0, 1)`. The three features are **arterial intensity, local standard deviation in a 3 × 3 × 3 neighborhood, and 3D Sobel gradient magnitude**. Points are pooled by expert class; coordinatewise medians define three centroids, ordered by arterial intensity.

```bash
python -m preprocessing.learn_prior --cases /path/to/training_case_001 /path/to/training_case_002 --output outputs/point_prior.pkl
```

The released **`point_prior_n100.pkl`** preserves the original schema-2 pickle dictionary: `cluster_centroids` maps labels 1/2/3 to three NumPy feature arrays, and `provenance` contains aggregate counts. It summarizes **3,390 expert points from 100 training cases**, without patient identifiers. Image inference uses it by default; no DOT or relearning is required. Load pickle files from trusted sources.

### 2 · Prior-driven constrained clustering

Each tumor voxel has the five-dimensional vector **[A, local SD(A), Sobel(A), V, V − A]**, with A and V mapped from the fixed −160 to 240 HU window to 0–1. Within each tumor, `RobustScaler` median/IQR scaling is applied separately to the arterial block (first three dimensions) and venous/difference block (last two). Constant columns are removed; each block is divided by the square root of its retained dimension count.

The prior initializes **only the arterial dimensions**; other initial coordinates are zero in scaled space. Constrained K-means uses **K = 3**, at least **ceil(5% of tumor voxels)** per habitat, seed 42, up to 100 iterations, and relative tolerance 1e−4. Final labels **1 → 3** follow increasing mean arterial intensity; background is 0.

```bash
python -m preprocessing.cluster --case outputs/cases/case_001 --prior point_prior_n100.pkl --output outputs/habitats.nii.gz
```

Use **`--prior outputs/point_prior.pkl`** with `preprocessing.cluster`, or with `predict.py --case` when computing THI, to select your own learned prior. Omitting the option uses the released prior.

### 3 · Heterogeneity calculation

```bash
python -m preprocessing.heterogeneity --case outputs/cases/case_001 --habitats outputs/habitats.nii.gz --output outputs/thi_features.json
```

| Family | Count | Measurements |
| :--- | ---: | :--- |
| Distributional heterogeneity, DDI | 9 | Per-phase between/within log ratio, centroid separation, Jensen–Shannon divergence, dispersion diversity; habitat-balance entropy |
| Spatial heterogeneity, SHI | 4 | Centroid spread, radial Wasserstein distance, interface fraction, disconnected fraction |
| Dynamic-enhancement heterogeneity, DEHI | 5 | Enhancement-mean spread, Jensen–Shannon divergence, dispersion diversity, phase-correlation diversity, washout-fraction diversity |

These **18 features use source HU images**. Spatial measurements use physical coordinates from the affine, voxel volume, and equivalent tumor radius. `--include-reference` additionally returns `dehi_global_mean`, outside the core feature set.

## Image scores

**Dual-phase radiomics:** **200 features = 14 shared shape + 93 arterial + 93 venous**. PyRadiomics uses `Original` images, whole-image normalization scaled to 255, bin width 25, 1 × 1 × 1 mm resampling, and B-spline image interpolation. Feature prefixes are `original_shape_`, `A_original_`, and `V_original_`. Retain the intended source field of view because it affects whole-image normalization.

```bash
python -m preprocessing.radiomics --case outputs/cases/case_001 --output outputs/radiomics_features.json
```

**THI and RAD:** fixed linear formulas in `predict.SCORES` use **7 heterogeneity features** and **16 radiomics features**, respectively. Development used ElasticNet with `(alpha, l1_ratio)` of `(0.3, 0.2)` for THI and `(0.1, 0.5)` for RAD. Raw scores are mapped to 0–1 as `clip((raw − training_P1) / (training_P99 − training_P1), 0, 1)`. P1/P99 come from five-fold training OOF scores; future cases use the all-training formula and those fixed constants.

<sub>The original workflow graphic retains schematic quantile/min–max labels; numerical inference follows the training P1/P99 transformation in `SCORES`.</sub>

## Grading

All nine models use **one `predict.py`**. `MODELS` defines their inputs and grade rules; `CLINICAL` defines clinical transformations. **Models 1–7 are unpenalized proportional-odds logistic models**, learning coefficients and two ordered cutpoints, then selecting the largest of three class probabilities.

| Model | Clinical inputs | Image scores | Structure / decision |
| :--- | :--- | :--- | :--- |
| Model 1 | size, SFI, TFI, LDH, AST, IFI, RIF% | None | Proportional odds / largest class probability |
| Model 2 | None | RAD | Proportional odds / largest class probability |
| Model 3 | None | THI | Proportional odds / largest class probability |
| Model 4 | size, SFI, TFI, LDH, AST, IFI, RIF% | RAD | Proportional odds / largest class probability |
| Model 5 | size, SFI, TFI, LDH, AST, IFI, RIF% | THI | Proportional odds / largest class probability |
| Model 6 | None | RAD, THI | Proportional odds / largest class probability |
| Model 7 | SFI, TFI, IFI | RAD, THI | Proportional odds / largest class probability |
| Model 8 | LDH, AST, SFI | RAD, THI | RBF-SVC / MLP / polynomial SVM, weights 0.4 / 0.2 / 0.4; largest averaged probability |
| Model 9 | LDH, AST, SFI, RIF%, size | RAD, THI | Two cumulative binary ensembles / fixed risk thresholds |

**Models 1–6** have embedded fitted parameters. **Model 7** requires a bundle generated by `train.py`. **Models 8 and 9** use the supplied `weights.joblib`. A new bundle passed with `--weights` supplies its fitted parameters, clinical transforms, and score formulas.

```bash
python predict.py --model model1 --input /path/to/clinical_only.json
python predict.py --model model3 --case outputs/cases/case_001 --output-dir outputs/model3
python predict.py --model model7 --input /path/to/measured_features_or_scores.json --clinical /path/to/clinical.json --weights models1_7.joblib
python predict.py --model model8 --input /path/to/measured_features_or_scores.json --clinical /path/to/clinical.json --weights weights.joblib
```

Only the selected model's inputs are required. Image models accept `--case` instead of `--input`, adding `--clinical` when clinical variables are required. Models 3 and 5 compute THI without radiomics. Inspect model specifications with `--list-models` or `--describe model7`.

<details>
<summary><b>Input JSON, clinical units, and grade rules</b></summary>

- Input JSON contains `features` mapping exact feature names to measurements, or `scores` mapping required `THI`/`RAD` to **numeric 0–1 values**, not nested result objects.
- Clinical JSON maps required variable names to **complete raw observed values**; alternatively place them in the input JSON's `clinical` object. Model 1 needs only that object.
- SFI, TFI, and IFI are volume/height³ indices (cm³/m³); `RIF% = 100 × IFI / TFI`; size is in mm. Preserve the study measurement scale. The model applies its prescribed `log1p` and training mean/SD transforms.
- Model 9 averages crossed cumulative probabilities before constructing class probabilities. Its risk thresholds are **0.6772685594841208** and **0.9673987735976841**; equality selects the higher category.

</details>

## Training

Refit **Models 1–7 together**, including their required image-score formulas:

```bash
python train.py --input /path/to/training_features.csv --models model1 model2 model3 model4 model5 model6 model7 --fit-scores --output models1_7.joblib
```

For one model, select its name and provide only its required inputs:

```bash
python train.py --input /path/to/training_features.csv --models model7 --fit-scores --output model7.joblib
```

| Option | Use |
| :--- | :--- |
| `--models` | One name, several names, or `all` for Models 1–9; default `model8` |
| `--fit-scores` | Refit only the required THI/RAD formulas from their locked 7/16 raw feature sets |
| `--input` / `--output` | Training CSV / one new shared `.joblib` bundle; existing files are not overwritten |
| `--cohort-column` / `--training-label` | Defaults `Cohort` / `training`; only matching rows are fitted |
| `--grade-column` | Default `Grade_ordinal`: 0 = G1, 1 = G2, 2 = G3/NEC |

Training rows must contain **all three grades** and **complete required clinical measurements**. Score fitting needs at least **five training cases per grade** and exact raw feature columns listed in `SCORES[name]['terms']`. Model 1 needs only clinical columns and no `--fit-scores`; Models 2 and 3 fit RAD and THI, respectively. Further options are listed in `python train.py --help`.

**What is fitted.** Models 1–7 maximize the proportional-odds likelihood; Model 7 uses **[SFI, TFI, IFI, RAD, THI]**. Models 8 and 9 refit their fixed components while retaining ensemble weights and grade rules. Clinical statistics are learned only from training rows. All selected models, clinical transforms, and score formulas are saved together for `predict.py --weights`.

<details>
<summary><b>Score fitting, CSV provenance, and development details</b></summary>

With `--fit-scores`, five-fold `StratifiedKFold` (`shuffle=True`, seed 20260720) refits the locked feature sets. Each training fold learns median imputation, variance filtering, population-SD standardization, and ElasticNet. Entirely unobserved or constant features are discarded. Score orientation follows training Spearman correlation to grade; centering/scaling is learned within the same fold. OOF scores define P1/P99, followed by an all-training inference formula.

Without `--fit-scores`, supply required **numeric 0–1 OOF scores** as `THI`/`RAD` (or `THI_score_01`/`RAD_score_01`) and `THI_score_source`/`RAD_score_source`. Accepted source labels are `out_of_fold`, `nested_oof`, or `training_locked_7_feature_5fold_OOF_refit` / `training_locked_16_feature_5fold_OOF_refit`. These scores must match the published inference formulas and scaling; use `--fit-scores` to keep new-dataset formulas and scores together.

Models 1–7 use SciPy L-BFGS-B (`maxiter=2000`, `ftol=1e−11`, `gtol=1e−7`). The original clinical development excluded variables with at least 25% training missingness, used `log1p` for skewed nonnegative variables, screened proportional-odds likelihood-ratio tests at BH-FDR q&lt;0.10, and pruned redundancy at |Spearman rho|≥0.85. After feature selection, clinical variables were transformed, standardized using the transformed nonmissing training mean and population SD, and then imputed in standardized space. Five training-fitted Bayesian-Ridge chained-equation imputations with posterior sampling were averaged while preserving observed values; validation outcomes were excluded. The original fitted clinical imputers are not distributed. Complete required clinical inputs are needed for all clinical-containing inference models, including Models 8 and 9; image-feature median imputation is a separate preprocessing step.

Score regularization was compared within training cross-validation; Model 8 families, subsets, hyperparameters, and weights were selected using internal tuning. `train.py` refits the selected specifications; it does not repeat those selection searches or establish a new performance evaluation.

</details>

## Repository layout

```text
PNEN-THI/
├── predict.py / train.py       # shared inference / fixed-specification training
├── weights.joblib             # shared fitted model bundle
├── point_prior_n100.pkl       # original-format learned arterial prior
├── requirements.txt
├── preprocessing/             # prepare · learn_prior · cluster · heterogeneity · radiomics
├── data/                      # real case and bilingual preparation guides
└── assets/study_workflow.png
```

The real example uses neutral filenames and cleaned NIfTI headers while preserving voxel values and spatial geometry.

## Use and citation

For **academic and non-commercial research** use. Contact the authors for clinical or commercial licensing.

Until formal manuscript citation metadata are available, reference the [PNEN-THI repository](https://github.com/BobbyWang16/PNEN-THI). The manuscript and full research cohort are not included in this release.
