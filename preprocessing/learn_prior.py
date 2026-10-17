"""Step 1: learn aggregate expert-point centroids from training cases.

Each case needs arterial.nii.gz, tumor_roi.nii.gz, points.nii.gz. Points
are sparse expert labels 1, 2, 3 within the tumour. Only aggregate medians
and annotation counts are published. No patient identifiers are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import generic_filter, sobel

from .io import label_mask, load_matched, normalize_hu, save_json

FEATURE_NAMES = ("arterial_intensity", "arterial_local_sd", "arterial_gradient")


def point_features(case_dir, tumor_label=2):
    """Extract the source arterial vector at valid expert points within the ROI.

    Expert points outside the selected tumour are excluded, matching the
    study implementation. An empty valid point set is rejected.
    """
    case_dir = Path(case_dir)
    a_img, roi_img, p_img = load_matched(
        case_dir / "arterial.nii.gz", case_dir / "tumor_roi.nii.gz", case_dir / "points.nii.gz"
    )
    arterial = a_img.get_fdata(dtype=np.float32)
    if not np.isfinite(arterial).all():
        raise ValueError("Arterial CT contains non-finite values")
    roi = label_mask(roi_img, tumor_label)
    points = p_img.get_fdata()
    if not np.isfinite(points).all() or not np.isin(points, (0, 1, 2, 3)).all():
        raise ValueError("Point map must contain integer labels 0, 1, 2, 3")
    valid = roi & np.isin(points, (1, 2, 3))
    if not valid.any():
        raise ValueError("No expert points inside the tumour")
    arterial = normalize_hu(arterial)
    local_sd = generic_filter(arterial, np.std, size=3)
    gradient = np.sqrt(sum(sobel(arterial, axis=axis) ** 2 for axis in range(3)))
    features = np.column_stack((arterial[valid], local_sd[valid], gradient[valid])).astype(
        np.float64
    )
    return points[valid].astype(int), features


def learn_prior(case_dirs, tumor_label=2):
    """Pool expert points, take class medians, order three centers by intensity.

    Supply training cases only. Training/validation membership cannot be
    inferred from image files and remains the caller's responsibility.
    Class labels are pooled before taking medians, as in the study code.
    """
    case_dirs = [Path(path).resolve() for path in case_dirs]
    if not case_dirs or len(set(case_dirs)) != len(case_dirs):
        raise ValueError("Supply at least one distinct training case directory")
    by_label = {label: [] for label in (1, 2, 3)}
    for index, path in enumerate(case_dirs, 1):
        try:
            labels, features = point_features(path, tumor_label)
        except Exception:
            # Do not expose source paths or patient identifiers in console logs.
            raise ValueError(f"Training case {index} failed input/point validation") from None
        for label in by_label:
            by_label[label].append(features[labels == label])
    by_label = {label: np.concatenate(values, axis=0) for label, values in by_label.items()}
    if any(len(values) == 0 for values in by_label.values()):
        raise ValueError(
            "The training cases must collectively contain all three expert-point labels"
        )
    centers = {label: np.median(values, axis=0) for label, values in by_label.items()}
    ordered = sorted(centers, key=lambda label: centers[label][0])
    return {
        "schema_version": 2,
        "prior_learned": True,
        "feature_dimension": 3,
        "feature_names": list(FEATURE_NAMES),
        "cluster_centroids": {
            str(rank): centers[label].tolist() for rank, label in enumerate(ordered, 1)
        },
        "min_cluster_fraction": 0.05,
        "n_clusters": 3,
        "provenance": {
            "n_patients": len(case_dirs),
            "annotation_points": sum(len(x) for x in by_label.values()),
            "annotation_counts": {str(label): len(values) for label, values in by_label.items()},
        },
        "algorithm": {
            "window_hu": [-160, 240],
            "local_sd_size": [3, 3, 3],
            "gradient": "3D_Sobel",
            "centroid": "pooled_pointwise_median",
            "expert_prior_role": "arterial_centroid_initialization_only",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", type=Path, required=True, help="Training case directories only"
    )
    parser.add_argument("--output", type=Path, required=True, help="Aggregate JSON prior")
    parser.add_argument("--tumor-label", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    prior = learn_prior(args.cases, args.tumor_label)
    save_json(prior, args.output)
    print(f"Saved aggregate prior: {prior['provenance']['n_patients']} training cases, 3 centroids")


if __name__ == "__main__":
    main()
