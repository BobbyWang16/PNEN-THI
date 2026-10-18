"""Step 2: expert-centroid-initialized constrained three-habitat clustering.

The numerical core is migrated from the study point_cluster.py. Intensity
windowing precedes feature construction; expert centroids only initialize
the arterial block. There is no phase weighting or probability penalty.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import generic_filter, sobel
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from .io import load_case, normalize_hu, save_habitats

N_HABITATS = 3
DEFAULT_MIN_CLUSTER_FRACTION = 0.05
DEFAULT_RANDOM_STATE = 42
MAX_CLUSTER_ITERATIONS = 100
CLUSTER_RELATIVE_TOLERANCE = 1e-4
ARTERIAL_FEATURE_COUNT = 3
VOXEL_FEATURE_NAMES = (
    "arterial_intensity",
    "arterial_local_sd",
    "arterial_gradient",
    "venous_intensity",
    "venous_minus_arterial",
)


class ConstrainedKMeans:
    """Deterministic K-means with a minimum cluster-size constraint."""

    def __init__(
        self,
        n_clusters=N_HABITATS,
        min_cluster_fraction=DEFAULT_MIN_CLUSTER_FRACTION,
        max_iter=MAX_CLUSTER_ITERATIONS,
        random_state=DEFAULT_RANDOM_STATE,
    ):
        self.n_clusters = n_clusters
        self.min_cluster_fraction = min_cluster_fraction
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit_predict(self, X, init_centers):
        """Assign and update with the source minimum-size constraint."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < self.n_clusters:
            raise ValueError("X must be a 2D matrix with at least n_clusters rows")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        if not 0 < self.min_cluster_fraction <= 1 / self.n_clusters:
            raise ValueError("min_cluster_fraction must lie in (0, 1/n_clusters]")
        np.random.seed(self.random_state)
        n_samples = len(X)
        min_size = int(np.ceil(n_samples * self.min_cluster_fraction))
        if min_size * self.n_clusters > n_samples:
            raise ValueError("Not enough tumour voxels for the minimum habitat sizes")
        if (
            init_centers is None
            or init_centers.shape != (self.n_clusters, X.shape[1])
            or not np.isfinite(init_centers).all()
        ):
            raise ValueError("Exactly three valid prior initialization centers are required")
        centers = init_centers.copy()
        labels = self._constrained_assign(X, centers, min_size)
        centers = self._update_centers(X, labels)
        for iteration in range(self.max_iter):
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) != self.n_clusters:
                labels = self._constrained_assign(X, centers, min_size)
                centers = self._update_centers(X, labels)
                continue
            new_labels = self._assign_labels(X, centers)
            new_unique, new_counts = np.unique(new_labels, return_counts=True)
            if len(new_unique) == self.n_clusters and np.all(new_counts >= min_size):
                new_centers = self._update_centers(X, new_labels)
                if np.allclose(centers, new_centers, rtol=CLUSTER_RELATIVE_TOLERANCE):
                    labels = new_labels
                    centers = new_centers
                    break
                labels = new_labels
                centers = new_centers
            else:
                labels = self._constrained_assign(X, centers, min_size)
                centers = self._update_centers(X, labels)
        self.cluster_centers_ = centers
        self.labels_ = labels
        return labels

    def _assign_labels(self, X, centers):
        """Nearest-center assignment."""
        distances = cdist(X, centers, metric="euclidean")
        return np.argmin(distances, axis=1)

    def _constrained_assign(self, X, centers, min_size):
        """Greedy distance ordering with at least min_size samples per cluster."""
        n_samples = len(X)
        distances = cdist(X, centers, metric="euclidean")
        labels = np.full(n_samples, -1, dtype=int)
        cluster_sizes = np.zeros(self.n_clusters, dtype=int)
        sample_indices, cluster_indices = np.meshgrid(
            np.arange(n_samples), np.arange(self.n_clusters), indexing="ij"
        )
        flat_samples = sample_indices.flatten()
        flat_clusters = cluster_indices.flatten()
        flat_distances = distances.flatten()
        sorted_idx = np.argsort(flat_distances)
        assigned = set()
        for idx in sorted_idx:
            sample = flat_samples[idx]
            cluster = flat_clusters[idx]
            if sample in assigned:
                continue
            if cluster_sizes[cluster] < min_size:
                labels[sample] = cluster
                cluster_sizes[cluster] += 1
                assigned.add(sample)
                if np.all(cluster_sizes >= min_size):
                    break
        unassigned = [i for i in range(n_samples) if i not in assigned]
        for sample in unassigned:
            cluster = np.argmin(distances[sample])
            labels[sample] = cluster
            cluster_sizes[cluster] += 1
        if np.any(labels == -1):
            unassigned_mask = labels == -1
            labels[unassigned_mask] = np.argmin(distances[unassigned_mask], axis=1)
        return labels

    def _update_centers(self, X, labels):
        """Update each center from its assigned voxels."""
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[np.random.randint(len(X))]
        return centers


class _Analyzer:

    def __init__(self, arterial, venous, mask, prior):
        self.arterial = arterial
        self.venous = venous
        self.mask = mask
        self.prior_knowledge = prior
        self.min_cluster_fraction = DEFAULT_MIN_CLUSTER_FRACTION
        self.random_state = DEFAULT_RANDOM_STATE
        self.n_clusters = N_HABITATS

    def extract_combined_features(self):
        """Return A, local SD(A), gradient(A), V, and V-A at tumour voxels."""
        tumor_voxels_a = self.arterial[self.mask]
        tumor_voxels_v = self.venous[self.mask]
        enhancement_diff = tumor_voxels_v - tumor_voxels_a
        return np.column_stack(
            [
                tumor_voxels_a,
                generic_filter(self.arterial, np.std, size=3)[self.mask],
                np.sqrt(sum((sobel(self.arterial, axis=i) ** 2 for i in range(3))))[self.mask],
                tumor_voxels_v,
                enhancement_diff,
            ]
        ).astype(np.float64)

    @staticmethod
    def _fit_nonconstant_robust_block(block):
        """Robust-scale nonconstant columns without a tunable variance cutoff."""
        block = np.asarray(block, dtype=np.float64)
        keep = np.ptp(block, axis=0) > 0
        if not np.any(keep):
            raise ValueError("clustering feature block contains no varying columns")
        scaler = RobustScaler().fit(block[:, keep])
        scaled = scaler.transform(block[:, keep])
        return (scaled, scaler, keep)

    def prepare_clustering_features(self, features=None):
        """Robust-scale each tumour block, then divide by sqrt(retained dimension)."""
        if features is None:
            features = self.extract_combined_features()
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(VOXEL_FEATURE_NAMES):
            raise ValueError(
                f"expected {len(VOXEL_FEATURE_NAMES)} voxel features {VOXEL_FEATURE_NAMES}, got shape {features.shape}"
            )
        arterial_scaled, arterial_scaler, arterial_keep = self._fit_nonconstant_robust_block(
            features[:, :ARTERIAL_FEATURE_COUNT]
        )
        av_scaled, av_scaler, av_keep = self._fit_nonconstant_robust_block(
            features[:, ARTERIAL_FEATURE_COUNT:]
        )
        arterial_scaled /= np.sqrt(arterial_scaled.shape[1])
        av_scaled /= np.sqrt(av_scaled.shape[1])
        combined = np.hstack([arterial_scaled, av_scaled])
        metadata = {
            "arterial_scaler": arterial_scaler,
            "arterial_keep": arterial_keep,
            "av_scaler": av_scaler,
            "av_keep": av_keep,
        }
        return (combined, metadata)

    def cluster_fixed_three_habitats(self, features=None):
        """Run the single prespecified three-habitat clustering path."""
        if features is None:
            features = self.extract_combined_features()
        clustering_features, metadata = self.prepare_clustering_features(features)
        init_centers = self._prior_initial_centers(metadata)
        model = ConstrainedKMeans(
            n_clusters=self.n_clusters,
            min_cluster_fraction=self.min_cluster_fraction,
            max_iter=MAX_CLUSTER_ITERATIONS,
            random_state=self.random_state,
        )
        labels = model.fit_predict(clustering_features, init_centers=init_centers)
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        min_size = int(np.ceil(n_samples * self.min_cluster_fraction))
        if len(unique_labels) != self.n_clusters or np.any(counts < min_size):
            raise RuntimeError("constrained K-means returned an invalid three-habitat solution")
        labels = self._reorder_labels_by_hu(labels, features[:, 0])
        return labels.astype(int) + 1

    def _prior_initial_centers(self, metadata):
        """Map the three training centroids into the current tumor feature space."""
        if len(self.prior_knowledge.cluster_centroids) != self.n_clusters:
            raise ValueError("expert prior must contain exactly three centroids")
        raw = np.asarray(
            [
                self.prior_knowledge.cluster_centroids[index + 1]
                for index in range(self.n_clusters)
            ],
            dtype=np.float64,
        )
        arterial_keep = metadata["arterial_keep"]
        prior_a = metadata["arterial_scaler"].transform(raw[:, arterial_keep])
        prior_a /= np.sqrt(prior_a.shape[1])
        av_dimension = int(np.sum(metadata["av_keep"]))
        prior_av = np.zeros((self.n_clusters, av_dimension), dtype=np.float64)
        prior_av /= np.sqrt(av_dimension)
        return np.hstack([prior_a, prior_av])

    def _reorder_labels_by_hu(self, labels, hu_values):
        """Relabel by ascending normalized arterial mean intensity."""
        unique_labels = np.unique(labels)
        if len(unique_labels) != self.n_clusters:
            raise ValueError("Expected exactly three habitat labels")
        hu_means = {lb: np.mean(hu_values[labels == lb]) for lb in unique_labels}
        sorted_labels = sorted(unique_labels, key=lambda x: hu_means[x])
        mapping = {sorted_labels[i]: i for i in range(len(sorted_labels))}
        new_labels = np.vectorize(mapping.get)(labels).astype(int)
        return new_labels


def load_prior(prior):
    """Validate safe JSON/dictionary schema 2 or the shipped aggregate schema."""
    if isinstance(prior, (str, Path)):
        prior = json.loads(Path(prior).read_text(encoding="utf-8-sig"))
    if not isinstance(prior, dict):
        raise ValueError("A prior JSON path or dictionary is required")
    schema = prior.get("schema_version", prior.get("source_prior_schema_version"))
    if schema != 2:
        raise ValueError("Only source prior schema 2 is supported")
    if not prior.get("prior_learned", True) or prior.get("n_clusters", 3) != 3:
        raise ValueError("A learned three-habitat prior is required")
    expected = list(VOXEL_FEATURE_NAMES[:3])
    if list(prior.get("feature_names", [])) != expected:
        raise ValueError("Prior arterial feature names/order do not match the locked method")
    if "centroids" in prior:
        raw = np.asarray(prior["centroids"], dtype=np.float64)
    else:
        values = prior.get("cluster_centroids", {})
        raw = np.asarray(
            [values.get(str(i), values.get(i, [])) for i in (1, 2, 3)], dtype=np.float64
        )
    if raw.shape != (3, 3) or not np.isfinite(raw).all():
        raise ValueError("Prior must contain three finite 3-D arterial centroids")
    if np.any(np.diff(raw[:, 0]) < 0):
        raise ValueError("Prior centroid labels must be ordered by arterial intensity")
    if np.any((raw[:, 0] < 0) | (raw[:, 0] > 1)) or np.any(raw[:, 1:] < 0):
        raise ValueError("Prior is not in the fixed normalized-image feature space")
    return SimpleNamespace(
        prior_learned=True, cluster_centroids={i + 1: row for i, row in enumerate(raw)}
    )


def cluster_arrays(arterial, venous, tumor_mask, prior):
    """Return labels 0/1/2/3, with habitats ordered by normalized arterial mean.

    Arrays must be matched 3-D grids in source HU. The minimum habitat size
    is ceil(5% of the tumour). Constant feature blocks are rejected as in
    the original source, rather than falling back to another algorithm.
    """
    arterial = np.asarray(arterial, dtype=np.float32)
    venous = np.asarray(venous, dtype=np.float32)
    mask = np.asarray(tumor_mask, dtype=bool)
    if arterial.ndim != 3 or arterial.shape != venous.shape or arterial.shape != mask.shape:
        raise ValueError("Arterial, venous, and tumour arrays must share one 3-D shape")
    if not np.isfinite(arterial).all() or not np.isfinite(venous).all():
        raise ValueError("Non-finite CT intensity")
    if mask.sum() < 3:
        raise ValueError("Three habitats require at least three tumour voxels")
    analyzer = _Analyzer(normalize_hu(arterial), normalize_hu(venous), mask, load_prior(prior))
    labels = analyzer.cluster_fixed_three_habitats()
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[mask] = labels
    return result


def cluster_case(case_dir, prior, *, tumor_label=2, output=None):
    """Validate a registered case and optionally save its habitat NIfTI."""
    arterial, venous, mask, reference = load_case(case_dir, tumor_label)
    result = cluster_arrays(arterial, venous, mask, prior)
    if output is not None:
        save_habitats(result, reference, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument(
        "--prior",
        type=Path,
        help="Learned aggregate JSON; defaults to the released root-level prior.json",
    )
    parser.add_argument("--output", required=True, type=Path, help="Habitat NIfTI (.nii.gz)")
    parser.add_argument("--tumor-label", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.prior is None:
        from predict import PRIOR

        prior = PRIOR
    else:
        prior = args.prior
    result = cluster_case(args.case, prior, tumor_label=args.tumor_label, output=args.output)
    print(
        "Saved three ordered habitats; minimum fraction = "
        + format(min(np.mean(result[result > 0] == i) for i in (1, 2, 3)), ".4f")
    )


if __name__ == "__main__":
    main()
