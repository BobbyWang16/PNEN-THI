"""Step 3: 18 habitat heterogeneity measures in raw HU and physical space.

DDI, SHI, and DEHI numerical definitions are preserved from the study source.
dehi_global_mean is an optional reference, not a heterogeneity measure.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from nibabel.affines import apply_affine
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from .io import load_case, load_matched, save_json

EPS = np.finfo(np.float64).eps
CLUSTER_IDS = (1, 2, 3)
CORE_FEATURES = (
    "ddi_a_between_within_log",
    "ddi_a_centroid_separation",
    "ddi_a_js_divergence",
    "ddi_a_dispersion_diversity",
    "ddi_v_between_within_log",
    "ddi_v_centroid_separation",
    "ddi_v_js_divergence",
    "ddi_v_dispersion_diversity",
    "ddi_cluster_balance_entropy",
    "shi_centroid_spread",
    "shi_radial_wasserstein",
    "shi_interface_fraction",
    "shi_disconnected_fraction",
    "dehi_mean_spread",
    "dehi_js_divergence",
    "dehi_dispersion_diversity",
    "dehi_phase_correlation_diversity",
    "dehi_washout_fraction_diversity",
)


def cluster_values(image: np.ndarray, clusters: np.ndarray) -> dict[int, np.ndarray]:
    """Return finite image values for each ordered habitat."""
    return {
        cluster_id: np.asarray(image[clusters == cluster_id], dtype=np.float64)
        for cluster_id in CLUSTER_IDS
    }


def cluster_proportions(clusters: np.ndarray) -> np.ndarray:
    """Return the three habitat volume fractions."""
    counts = np.asarray([(clusters == cluster_id).sum() for cluster_id in CLUSTER_IDS])
    return counts / counts.sum()


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Return Shannon entropy normalized to [0, 1]."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(CLUSTER_IDS)))


def robust_scale(values: np.ndarray) -> float:
    """Return an IQR scale with a standard-deviation fallback."""
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.percentile(values, 75) - np.percentile(values, 25))
    if scale <= EPS:
        scale = float(np.std(values))
    return max(scale, EPS)


def shared_probability_histograms(
    values_by_cluster: dict[int, np.ndarray], bins: int = 64
) -> dict[int, np.ndarray]:
    """Create comparable probability histograms using robust shared limits."""
    pooled = np.concatenate([values_by_cluster[c] for c in CLUSTER_IDS])
    low, high = np.percentile(pooled, (0.5, 99.5))
    if high <= low:
        low, high = (float(pooled.min()), float(pooled.max() + 1.0))
    histograms = {}
    for cluster_id in CLUSTER_IDS:
        hist, _ = np.histogram(values_by_cluster[cluster_id], bins=bins, range=(low, high))
        probability = hist.astype(np.float64) + EPS
        histograms[cluster_id] = probability / probability.sum()
    return histograms


def weighted_pairwise(pair_values: dict[tuple[int, int], float], proportions: np.ndarray) -> float:
    """Return a volume-weighted mean across the three habitat pairs."""
    numerator = 0.0
    denominator = 0.0
    for (first, second), value in pair_values.items():
        weight = proportions[first - 1] * proportions[second - 1]
        numerator += weight * value
        denominator += weight
    return float(numerator / max(denominator, EPS))


def pairwise_js_divergence(histograms: dict[int, np.ndarray], proportions: np.ndarray) -> float:
    """Return volume-weighted Jensen-Shannon divergence in [0, 1]."""
    values = {
        (first, second): float(jensenshannon(histograms[first], histograms[second], base=2.0) ** 2)
        for first in CLUSTER_IDS
        for second in CLUSTER_IDS
        if first < second
    }
    return weighted_pairwise(values, proportions)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(values * weights))
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2)))


def _phase_features(
    image: np.ndarray, clusters: np.ndarray, phase: str
) -> tuple[dict[str, float], dict[int, np.ndarray]]:
    values = cluster_values(image, clusters)
    proportions = cluster_proportions(clusters)
    means = np.asarray([np.mean(values[c]) for c in CLUSTER_IDS])
    iqrs = np.asarray(
        [np.percentile(values[c], 75) - np.percentile(values[c], 25) for c in CLUSTER_IDS]
    )
    pooled = np.concatenate([values[c] for c in CLUSTER_IDS])
    global_mean = float(np.mean(pooled))
    between = float(np.sum(proportions * (means - global_mean) ** 2))
    within = float(np.sum(proportions * np.asarray([np.var(values[c]) for c in CLUSTER_IDS])))
    pair_distances = {
        (first, second): abs(means[first - 1] - means[second - 1])
        for first in CLUSTER_IDS
        for second in CLUSTER_IDS
        if first < second
    }
    histograms = shared_probability_histograms(values)
    prefix = f"ddi_{phase}"
    result: dict[str, float] = {
        f"{prefix}_between_within_log": float(np.log1p(between / max(within, EPS))),
        f"{prefix}_centroid_separation": weighted_pairwise(pair_distances, proportions)
        / robust_scale(pooled),
        f"{prefix}_js_divergence": pairwise_js_divergence(histograms, proportions),
        f"{prefix}_dispersion_diversity": _weighted_std(iqrs, proportions)
        / max(float(np.sum(proportions * iqrs)), EPS),
    }
    for cluster_id in CLUSTER_IDS:
        result[f"{prefix}_c{cluster_id}_mean"] = float(means[cluster_id - 1])
        result[f"{prefix}_c{cluster_id}_iqr"] = float(iqrs[cluster_id - 1])
    return (result, values)


def calculate_ddi_features(
    arterial: np.ndarray, venous: np.ndarray, clusters: np.ndarray
) -> tuple[dict[str, float], dict[str, dict[int, np.ndarray]]]:
    """Calculate arterial/venous distribution heterogeneity features."""
    a_features, a_values = _phase_features(arterial, clusters, "a")
    v_features, v_values = _phase_features(venous, clusters, "v")
    proportions = cluster_proportions(clusters)
    features = {**a_features, **v_features}
    features["ddi_cluster_balance_entropy"] = float(
        -np.sum(proportions * np.log(proportions)) / np.log(len(CLUSTER_IDS))
    )
    for cluster_id in CLUSTER_IDS:
        features[f"habitat_c{cluster_id}_fraction"] = float(proportions[cluster_id - 1])
    return (features, {"a": a_values, "v": v_values})


def _physical_coordinates(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return apply_affine(affine, np.argwhere(mask)).astype(np.float64, copy=False)


def _contact_statistics(clusters: np.ndarray) -> tuple[float, np.ndarray]:
    total_edges = 0
    different_edges = 0
    contacts = np.zeros((len(CLUSTER_IDS), len(CLUSTER_IDS)), dtype=np.int64)
    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(None, -1)
        second_slice[axis] = slice(1, None)
        first = clusters[tuple(first_slice)]
        second = clusters[tuple(second_slice)]
        inside = (first > 0) & (second > 0)
        total_edges += int(inside.sum())
        boundary = inside & (first != second)
        different_edges += int(boundary.sum())
        left = first[boundary].astype(np.int64) - 1
        right = second[boundary].astype(np.int64) - 1
        directed = np.bincount(left * len(CLUSTER_IDS) + right, minlength=9).reshape(3, 3)
        contacts += directed + directed.T
    return (different_edges / max(total_edges, 1), contacts)


def _disconnected_fraction(clusters: np.ndarray) -> float:
    proportions = cluster_proportions(clusters)
    structure = generate_binary_structure(3, 1)
    values = []
    for cluster_id in CLUSTER_IDS:
        components, count = label(clusters == cluster_id, structure=structure)
        sizes = np.bincount(components.ravel())[1:]
        fraction = 0.0 if count <= 1 else 1.0 - float(sizes.max() / sizes.sum())
        values.append(fraction)
    return float(np.sum(proportions * np.asarray(values)))


def calculate_shi_features(
    clusters: np.ndarray, affine: np.ndarray
) -> tuple[dict[str, float], dict[str, object]]:
    """Calculate physical-space habitat separation and fragmentation."""
    proportions = cluster_proportions(clusters)
    tumor_coords = _physical_coordinates(clusters > 0, affine)
    tumor_centroid = np.mean(tumor_coords, axis=0)
    voxel_volume = abs(float(np.linalg.det(np.asarray(affine)[:3, :3])))
    equivalent_radius = (3.0 * len(tumor_coords) * voxel_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    equivalent_radius = max(float(equivalent_radius), EPS)
    centroids: dict[int, np.ndarray] = {}
    radial: dict[int, np.ndarray] = {}
    for cluster_id in CLUSTER_IDS:
        coords = _physical_coordinates(clusters == cluster_id, affine)
        centroids[cluster_id] = np.mean(coords, axis=0)
        radial[cluster_id] = np.linalg.norm(coords - tumor_centroid, axis=1) / equivalent_radius
    centroid_distances = np.asarray(
        [np.linalg.norm(centroids[c] - tumor_centroid) for c in CLUSTER_IDS]
    )
    radial_pairs = {
        (first, second): float(wasserstein_distance(radial[first], radial[second]))
        for first in CLUSTER_IDS
        for second in CLUSTER_IDS
        if first < second
    }
    interface_fraction, contacts = _contact_statistics(clusters)
    features = {
        "shi_centroid_spread": float(
            np.sqrt(np.sum(proportions * centroid_distances**2)) / equivalent_radius
        ),
        "shi_radial_wasserstein": weighted_pairwise(radial_pairs, proportions),
        "shi_interface_fraction": float(interface_fraction),
        "shi_disconnected_fraction": _disconnected_fraction(clusters),
    }
    details: dict[str, object] = {"clusters": clusters, "radial": radial, "contacts": contacts}
    return (features, details)


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) <= EPS or np.std(second) <= EPS:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def calculate_dehi_features(
    arterial: np.ndarray, venous: np.ndarray, clusters: np.ndarray
) -> tuple[dict[str, float], dict[str, dict[int, np.ndarray] | np.ndarray]]:
    """Calculate scale-aware dynamic enhancement heterogeneity features."""
    enhancement = venous.astype(np.float64) - arterial.astype(np.float64)
    enhancement_values = cluster_values(enhancement, clusters)
    arterial_values = cluster_values(arterial, clusters)
    venous_values = cluster_values(venous, clusters)
    proportions = cluster_proportions(clusters)
    means = np.asarray([np.mean(enhancement_values[c]) for c in CLUSTER_IDS])
    iqrs = np.asarray(
        [
            np.percentile(enhancement_values[c], 75) - np.percentile(enhancement_values[c], 25)
            for c in CLUSTER_IDS
        ]
    )
    correlations = np.asarray(
        [_safe_correlation(arterial_values[c], venous_values[c]) for c in CLUSTER_IDS]
    )
    washout = np.asarray([np.mean(enhancement_values[c] < 0) for c in CLUSTER_IDS])
    pooled = np.concatenate([enhancement_values[c] for c in CLUSTER_IDS])
    mean_pairs = {
        (first, second): abs(means[first - 1] - means[second - 1])
        for first in CLUSTER_IDS
        for second in CLUSTER_IDS
        if first < second
    }
    histograms = shared_probability_histograms(enhancement_values)
    features: dict[str, float] = {
        "dehi_mean_spread": weighted_pairwise(mean_pairs, proportions) / robust_scale(pooled),
        "dehi_js_divergence": pairwise_js_divergence(histograms, proportions),
        "dehi_dispersion_diversity": _weighted_std(iqrs, proportions)
        / max(float(np.sum(proportions * iqrs)), EPS),
        "dehi_phase_correlation_diversity": _weighted_std(correlations, proportions),
        "dehi_washout_fraction_diversity": _weighted_std(washout, proportions),
        "dehi_global_mean": float(np.mean(pooled)),
    }
    for cluster_id in CLUSTER_IDS:
        index = cluster_id - 1
        features[f"dehi_c{cluster_id}_mean"] = float(means[index])
        features[f"dehi_c{cluster_id}_iqr"] = float(iqrs[index])
        features[f"dehi_c{cluster_id}_phase_correlation"] = float(correlations[index])
        features[f"dehi_c{cluster_id}_washout_fraction"] = float(washout[index])
    details: dict[str, dict[int, np.ndarray] | np.ndarray] = {
        "enhancement": enhancement_values,
        "arterial": arterial_values,
        "venous": venous_values,
        "washout": washout,
    }
    return (features, details)


def extract_features(
    arterial, venous, habitats, affine, *, include_reference=False, include_descriptors=False
):
    """Return 18 finite core features; optionally add 1 reference and 27 descriptors."""
    arterial = np.asarray(arterial)
    venous = np.asarray(venous)
    habitats = np.asarray(habitats)
    if arterial.ndim != 3 or arterial.shape != venous.shape or arterial.shape != habitats.shape:
        raise ValueError("CT and habitat arrays must share one three-dimensional shape")
    observed = set(np.unique(habitats).tolist())
    if not {1, 2, 3}.issubset(observed) or not observed.issubset({0, 1, 2, 3}):
        raise ValueError("Habitats must be 0/1/2/3 with all three habitats present")
    if not np.isfinite(arterial[habitats > 0]).all() or not np.isfinite(venous[habitats > 0]).all():
        raise ValueError("CT contains non-finite intensities within habitats")
    affine = np.asarray(affine, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:3, :3])) <= 0
    ):
        raise ValueError("Invalid physical-space affine")
    ddi, _ = calculate_ddi_features(arterial, venous, habitats)
    shi, _ = calculate_shi_features(habitats, affine)
    dehi, _ = calculate_dehi_features(arterial, venous, habitats)
    all_features = {**ddi, **shi, **dehi}
    features = {key: all_features[key] for key in CORE_FEATURES}
    if include_reference:
        features["dehi_global_mean"] = all_features["dehi_global_mean"]
    if include_descriptors:
        features.update(
            {
                key: value
                for key, value in all_features.items()
                if key not in CORE_FEATURES and key != "dehi_global_mean"
            }
        )
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Non-finite heterogeneity feature")
    return features


def extract_case(
    case_dir, habitat_path, *, tumor_label=2, include_reference=False, include_descriptors=False
):
    """Read a matched habitat map and require exact agreement with tumour support."""
    arterial, venous, mask, image = load_case(case_dir, tumor_label)
    _, habitat_image = load_matched(Path(case_dir) / "arterial.nii.gz", habitat_path)
    habitats = habitat_image.get_fdata()
    if not np.array_equal(habitats > 0, mask):
        raise ValueError("Habitat support differs from the selected tumour ROI")
    return extract_features(
        arterial,
        venous,
        habitats,
        image.affine,
        include_reference=include_reference,
        include_descriptors=include_descriptors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--habitats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Measured feature JSON")
    parser.add_argument("--tumor-label", type=int, choices=(1, 2), default=2)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--include-descriptors", action="store_true")
    args = parser.parse_args()
    result = extract_case(
        args.case,
        args.habitats,
        tumor_label=args.tumor_label,
        include_reference=args.include_reference,
        include_descriptors=args.include_descriptors,
    )
    save_json(result, args.output)
    print(f"Saved {len(result)} measured features")


if __name__ == "__main__":
    main()
