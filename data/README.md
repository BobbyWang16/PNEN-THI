# Input data and imaging workflow

[English](README.md) · [简体中文](README_zh.md) · [Project overview](../README.md)

Organize images by **phase and annotation type**, with one anonymous case identifier shared across folders. The released `case_001` contains real, preprocessed arterial CT, venous CT registered to the arterial reference, and a tumor ROI. It is ready for habitat and feature calculation.

## Folder structure

```text
data/
├── A/
│   └── case_001.nii.gz       # arterial reference, source HU, 1-mm grid
├── V/
│   └── .gitkeep              # place your original venous images here
├── V2A/
│   └── case_001.nii.gz       # venous CT registered and resampled to A
├── ROI/
│   └── case_001.nii.gz       # consensus tumor mask on the A grid
├── DOT/
│   └── .gitkeep              # optional expert-point maps for prior learning
├── case_001.json             # geometry and label description of the example
├── README.md
└── README_zh.md
```

For a new case, use the same basename in every applicable folder, for example `case_002.nii.gz`.

| Folder | Contents | Spatial and value requirements | Used for |
| :--- | :--- | :--- | :--- |
| `A` | Preprocessed arterial-phase CT | Reference space; 1 × 1 × 1 mm; retain HU | Prior learning, clustering, features |
| `V` | Original venous-phase CT before alignment to A | Native geometry may differ from A; retain HU | Upstream registration |
| `V2A` | Venous-phase CT transformed into arterial space | Same shape, spacing and affine as A; retain HU | Clustering and dual-phase features |
| `ROI` | Reviewed whole-tumor label map | Same grid as A; **2 = tumor**, 0 = background | Defines the analytic tumor |
| `DOT` | Sparse expert-point label map | Same grid as A; **0 = unannotated, 1/2/3 = low/intermediate/high perfusion** | Optional training-only prior learning |

`V2A` means **venous registered to arterial**, not the enhancement difference `V − A`. In the numerical method, “V” refers to the registered image from `V2A`. ROI and DOT label numbers belong to separate maps: a DOT point with label 1 remains a valid low-perfusion point inside ROI label 2. Point phenotypes are not pathological grades.

The example ROI also contains a non-target label 1, which the default calculation excludes. Convert your tumor mask to label 2 for the shared preparation and prediction entry points. Do not relabel every positive voxel as tumor if the map also includes pancreas or other structures. The individual method CLIs additionally support explicit `--tumor-label 1` for genuinely binary masks.

`V` and `DOT` are intentionally empty in the released example: its original unregistered venous series and expert points are not supplied. They are not required to use the supplied prior and calculate image features. No point labels are generated automatically.

## 1 Prepare CT and segment the tumor

Keep original scans and intermediate working images separately. The analysis inputs in `A`, `V2A`, `ROI` and optional `DOT` must ultimately share one **anatomically aligned 3D grid**, not merely matching array dimensions.

Starting from DICOM, use the [Slicer DICOM module](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html) to load the arterial and venous series, then save each volume as a 3D `.nii.gz` file with a deidentified case identifier. Retain HU and spatial geometry; use the volume data rather than screenshots or color-rendered images.

The study's upstream preprocessing specification is:

1. Generate preliminary pancreas and tumor masks separately for each phase using an nnU-Net trained on PANORAMA, then review the masks.
2. Resample CT to **1-mm isotropic spacing with B-spline interpolation**; resample masks with **nearest-neighbor interpolation**.
3. Define the patient-specific field of view from the three-dimensional union of both phases' pancreas and tumor masks, expanded by **20 mm in every direction**. Apply the corresponding physical field of view to both phases. Preserve source HU.
4. After venous-to-arterial registration, review and correct the arterial tumor mask, checking its boundary against both phases. The final consensus ROI is in arterial space.

The preliminary segmentation model is not bundled. You may supply reviewed masks from your own segmentation workflow or delineate them manually. With [3D Slicer Segmentations](https://slicer.readthedocs.io/en/5.4/user_guide/modules/segmentations.html), use A as reference geometry, review the tumor in three planes, and export the label map onto A's grid. ITK-SNAP can also be used for manual annotation and refinement.

**A may already be cropped.** The supplied case is an analysis-ready local field of view. Do not crop it again or replace its HU with windowed display values. The preparation command below validates and packages existing inputs; it performs no segmentation, registration or additional cropping.

## 2 Register V to A and save V2A

Use arterial CT as the **fixed reference** and venous CT as the **moving image**. The study specifies Euler 3D rigid registration followed by affine registration:

| Setting | Value |
| :--- | :--- |
| Metric | Mattes mutual information, 32 bins, full-voxel sampling |
| Multiresolution shrink factors | 4, 2, 1 |
| Smoothing sigmas | 2, 1, 0 mm |
| Optimizer | Gradient descent, learning rate 1.0 |
| Maximum iterations | 200 per stage |
| Convergence | Minimum value 1e−6; window 10 iterations |
| Final moving-image resampling | **Linear interpolation onto the A reference grid** |

The initial isotropic CT resampling and final registration resampling are separate operations with different interpolation settings. Use nearest-neighbor interpolation for any label maps being transformed.

An interactive route is [Slicer BRAINSFit](https://slicer.readthedocs.io/en/latest/user_guide/modules/brainsfit.html): select Fixed = A and Moving = V, perform rigid/affine alignment, and save the resampled output in `V2A`. Its settings must be chosen in that tool; this is an alternative manual route rather than an implementation of the study settings above. Export the output on the **fixed A grid**. Merely attaching or hardening a transform does not establish that the stored grids match.

Check axial, coronal and sagittal overlays at the tumor and surrounding vessels. Correct residual respiratory or bowel-motion mismatch where possible. A case with persistent gross mismatch is unsuitable for voxelwise dual-phase calculation. Finalize the ROI after this review.

## 3 Package an analysis case

Run commands from the repository root. The supplied real example can be prepared directly:

```bash
python -m preprocessing.prepare --data-root data --case-id case_001 --output-dir outputs/cases/case_001
```

The command checks 3D geometry, 1-mm spacing, matching affines, finite CT values and tumor label 2, then writes:

```text
outputs/cases/case_001/
├── arterial.nii.gz           # from data/A/case_001.nii.gz
├── venous.nii.gz             # from data/V2A/case_001.nii.gz
├── tumor_roi.nii.gz          # from data/ROI/case_001.nii.gz
└── points.nii.gz             # only when a valid matching DOT file exists
```

The packed `venous.nii.gz` always comes from **V2A**, never from unregistered `V`. Existing DOT maps are validated and copied automatically; absent DOT maps remain absent. Keep new patient scans local; only the curated example is allowed into version control by default.

Existing target files are not overwritten. Choose a new output directory when repeating preparation or adding DOT annotations to a previously prepared case.

## 4 Mark points and learn a prior

This step is needed only to learn a new prior from **your own training cases**. The already learned prior is **[`point_prior_n100.pkl`](../point_prior_n100.pkl) in the repository root**, preserving the original schema-2 pickle dictionary. It aggregates **3,390 expert points from 100 training cases**; `cluster_centroids` maps labels 1/2/3 to three NumPy arrays of three arterial features each, and `provenance` holds aggregate counts. `predict.PRIOR` loads this file for default image inference, which needs no DOT file or relearning. Load pickle files only from trusted sources.

For each training case, create a sparse integer label image with the exact A geometry. In ITK-SNAP or your annotation tool, mark representative intratumoral **single-voxel points** using 1 = low, 2 = intermediate and 3 = high perfusion; leave all other voxels at 0. Inspect the point map over A and the consensus ROI, and save it as `data/DOT/<case_id>.nii.gz`. Keep phenotype meanings consistent across cases. A centroid or predicted habitat map is not a substitute for expert point annotations.

Package each annotated case with the same preparation command and its actual identifier. Then pass its prepared directory to the prior learner:

```bash
python -m preprocessing.learn_prior --cases outputs/cases/your_training_case_001 outputs/cases/your_training_case_002 --output outputs/point_prior.pkl
```

Replace these training case names with your own prepared directories; they are not supplied examples. Prior learning uses only **A, ROI, and DOT**. Only points inside the selected tumor contribute, and the training set must collectively contain all three point classes. The learner pools normalized arterial intensity, 3 × 3 × 3 local SD and 3D Sobel magnitude at valid points, takes the coordinatewise median for each class, and orders the three centroids by arterial intensity. `outputs/point_prior.pkl` stores your new aggregate centroids and annotation counts in a compatible pickle dictionary. No validation cases or pathological grades enter this calculation.

## 5 Cluster three habitats

From the repository root, explicitly select the supplied prior (omitting `--prior` uses the same file):

```bash
python -m preprocessing.cluster --case outputs/cases/case_001 --prior point_prior_n100.pkl --output outputs/case_001/habitats.nii.gz
```

Or supply your training-derived prior:

```bash
python -m preprocessing.cluster --case outputs/cases/case_001 --prior outputs/point_prior.pkl --output outputs/case_001/habitats.nii.gz
```

The method uses `[A, local SD(A), Sobel(A), V, V − A]` with fixed HU normalization and per-case block scaling. The prior initializes the arterial feature dimensions. Constrained K-means returns **three habitats**, each at least **5% of the tumor**. Output 0 is background; labels 1, 2 and 3 follow increasing mean arterial intensity. These derived habitat labels are separate from both ROI and DOT annotations.

## 6 Calculate features and scores

Extract the **18 core heterogeneity features**: 9 DDI, 4 SHI and 5 DEHI measurements.

```bash
python -m preprocessing.heterogeneity --case outputs/cases/case_001 --habitats outputs/case_001/habitats.nii.gz --output outputs/case_001/heterogeneity_features.json
```

With the full radiomics environment, extract **200 dual-phase whole-tumor radiomics features**: 14 shared shape, 93 arterial and 93 registered venous.

```bash
python -m preprocessing.radiomics --case outputs/cases/case_001 --output outputs/case_001/radiomics_features.json
```

For the complete image workflow in one call, use the unified prediction script. It performs clustering and feature extraction internally:

```bash
python predict.py --thi-only --case outputs/cases/case_001 --prior point_prior_n100.pkl --output-dir outputs/demo_thi
python predict.py --score-only --case outputs/cases/case_001 --prior point_prior_n100.pkl --output-dir outputs/demo_scores
```

The same `--prior` option also works with `--model model3 --case ...`. Replace `point_prior_n100.pkl` with `outputs/point_prior.pkl` to use your own learned prior; default commands without this option remain valid.

The first command requires the base dependencies; the second also requires `preprocessing/requirements-radiomics.txt`. Both write derived habitat maps, measured features and `result.json`. Neither needs clinical measurements. For three-tier grading, Models 2, 3 and 6 use image scores alone; other models additionally need their specified clinical measurements. See the model-input table in the [main README](../README.md#grading), and its [Training section](../README.md#training) for feature tables and fixed-specification fitting.
