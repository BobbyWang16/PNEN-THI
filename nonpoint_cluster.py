"""
胰腺肿瘤异质性分析完整流程 - 无点先验对照版
- 不使用任何医生标注先验
- ConstrainedKMeans（内置5%约束）
- 强制3类 + HU梯度排序

与 point_cluster.py 一一对应，唯一差异：去掉先验知识引导，
其余特征/约束/后处理/输出格式完全一致，便于消融对比。
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import warnings
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import generic_filter, sobel
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')


# ============================================================================
#  ConstrainedKMeans - 带5%约束的KMeans（与 point_cluster.py 保持一致）
# ============================================================================

class ConstrainedKMeans:
    """带最小簇大小约束的KMeans"""

    def __init__(self, n_clusters=3, min_cluster_fraction=0.05, max_iter=100, random_state=42, init='k-means++'):
        self.n_clusters = n_clusters
        self.min_cluster_fraction = min_cluster_fraction
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init
        self.cluster_centers_ = None
        self.labels_ = None

    def fit_predict(self, X, init_centers=None):
        """带约束的KMeans聚类"""
        np.random.seed(self.random_state)
        n_samples = len(X)
        min_size = int(np.ceil(n_samples * self.min_cluster_fraction))

        # 初始化
        if init_centers is not None and init_centers.shape[0] == self.n_clusters:
            centers = init_centers.copy()
            print(f"    使用自定义中心初始化")
        elif self.init == 'k-means++':
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=1, init='k-means++')
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_

        # 初始约束分配
        labels = self._constrained_assign(X, centers, min_size)
        centers = self._update_centers(X, labels)

        # 迭代优化
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
                if np.allclose(centers, new_centers, rtol=1e-4):
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
        """标准最近邻分配"""
        distances = cdist(X, centers, metric='euclidean')
        return np.argmin(distances, axis=1)

    def _constrained_assign(self, X, centers, min_size):
        """带约束的分配：保证每个簇至少有min_size个样本"""
        n_samples = len(X)
        distances = cdist(X, centers, metric='euclidean')

        labels = np.full(n_samples, -1, dtype=int)
        cluster_sizes = np.zeros(self.n_clusters, dtype=int)

        sample_indices, cluster_indices = np.meshgrid(
            np.arange(n_samples), np.arange(self.n_clusters), indexing='ij'
        )
        flat_samples = sample_indices.flatten()
        flat_clusters = cluster_indices.flatten()
        flat_distances = distances.flatten()

        sorted_idx = np.argsort(flat_distances)

        # 第一阶段：每个簇至少分配min_size个样本
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

        # 第二阶段：分配剩余样本
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
        """更新簇中心"""
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[np.random.randint(len(X))]
        return centers


# ============================================================================
#  PancreaticTumorHeterogeneityAnalyzerNoPrior - 无先验肿瘤分析器
# ============================================================================

class PancreaticTumorHeterogeneityAnalyzerNoPrior:
    """胰腺癌异质性分析器（无先验对照版）"""

    def __init__(self, arterial_ct, venous_ct, tumor_mask,
                 arterial_weight=2.0, min_cluster_fraction=0.05):
        self.arterial = arterial_ct
        self.venous = venous_ct
        self.mask = tumor_mask.astype(bool)
        self.arterial_weight = arterial_weight
        self.min_cluster_fraction = min_cluster_fraction
        self.n_clusters = 3
        self.labels = None

    def extract_combined_features(self):
        """提取组合特征（与 point_cluster.py 完全一致的 9 维）"""
        tumor_voxels_a = self.arterial[self.mask]
        tumor_voxels_v = self.venous[self.mask]

        weighted_a = tumor_voxels_a * self.arterial_weight
        enhancement_diff = tumor_voxels_v - tumor_voxels_a

        return np.column_stack([
            weighted_a,
            generic_filter(self.arterial, np.std, size=3)[self.mask],
            np.sqrt(sum(sobel(self.arterial, axis=i)**2 for i in range(3)))[self.mask],
            enhancement_diff,
            tumor_voxels_v / (tumor_voxels_a + 1e-10),
            np.abs(enhancement_diff),
            tumor_voxels_v,
            (weighted_a + tumor_voxels_v) / 2,
            np.where(enhancement_diff > 0.05, 1, np.where(enhancement_diff < -0.05, -1, 0))
        ]).astype(np.float64)

    def adaptive_clustering(self, features=None):
        """自适应聚类（无先验，直接走基础路径）"""
        if features is None:
            features = self.extract_combined_features()

        print(f"\n🔬 聚类方法: basic (no-prior), 特征维度: {features.shape}")

        labels = self._basic_clustering(features)

        # 验证和修正
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        min_size = int(np.ceil(n_samples * self.min_cluster_fraction))

        print(f"\n  📊 聚类结果:")
        for label, count in zip(unique_labels, counts):
            status = "✓" if count >= min_size else "✗"
            print(f"    簇{label}: {count:5d} ({count/n_samples*100:5.1f}%) [{status}]")

        if len(unique_labels) != self.n_clusters or np.any(counts < min_size):
            print(f"  ⚠️  约束未满足，启动后处理...")
            labels = self._ensure_exactly_n_clusters(labels, features)
            labels = self._enforce_minimum_cluster_size_postprocess(labels, features)
            labels = self._ensure_exactly_n_clusters(labels, features)

        # HU梯度排序
        labels = self._ensure_hu_gradient(labels, features)

        return labels.astype(int) + 1

    def _basic_clustering(self, features):
        """基础聚类（RobustScaler + ConstrainedKMeans）"""
        X_scaled = RobustScaler().fit_transform(features)
        return self._basic_clustering_impl(X_scaled, init_centers=None)

    def _basic_clustering_impl(self, X_scaled, init_centers=None):
        """基础聚类实现"""
        try:
            constrained_kmeans = ConstrainedKMeans(
                n_clusters=self.n_clusters,
                min_cluster_fraction=self.min_cluster_fraction,
                max_iter=100,
                random_state=42
            )
            labels = constrained_kmeans.fit_predict(X_scaled, init_centers=init_centers)
            print(f"  ✓ ConstrainedKMeans(n={self.n_clusters}, min≥{self.min_cluster_fraction*100:.0f}%)")
            return self._reorder_labels_by_hu(labels, X_scaled[:, 0])

        except Exception as e:
            print(f"  ⚠️  ConstrainedKMeans失败: {e}, 回退到标准KMeans")
            if init_centers is not None:
                kmeans = KMeans(n_clusters=self.n_clusters, init=init_centers, n_init=1, random_state=42)
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            return self._reorder_labels_by_hu(labels, X_scaled[:, 0])

    def _reorder_labels_by_hu(self, labels, hu_values):
        """按HU值排序标签"""
        unique_labels = np.unique(labels)

        if len(unique_labels) != self.n_clusters:
            print(f"  ⚠️  标签数异常: {len(unique_labels)}, 期望{self.n_clusters}")

        hu_means = {lb: np.mean(hu_values[labels == lb]) for lb in unique_labels}
        sorted_labels = sorted(unique_labels, key=lambda x: hu_means[x])
        mapping = {sorted_labels[i]: i for i in range(len(sorted_labels))}
        new_labels = np.vectorize(mapping.get)(labels).astype(int)

        print(f"  🔄 按HU值排序: {[f'L{old}({hu_means[old]:.2f})' for old in sorted_labels]} -> [0,1,2]")

        return new_labels

    def _ensure_hu_gradient(self, labels, features):
        """确保HU值梯度"""
        unique_labels = np.unique(labels)
        if len(unique_labels) != 3:
            return labels

        hu_means = {lb: np.mean(features[labels == lb, 0]) for lb in unique_labels}
        sorted_labels = sorted(unique_labels, key=lambda x: hu_means[x])

        if sorted_labels == [0, 1, 2]:
            print(f"  ✅ HU梯度已满足")
            return labels

        mapping = {sorted_labels[i]: i for i in range(3)}
        new_labels = np.vectorize(mapping.get)(labels)

        print(f"  🔄 HU梯度重排")
        return new_labels

    def _ensure_exactly_n_clusters(self, labels, features):
        """强制保证n个类别"""
        unique_labels = np.unique(labels)
        n_current = len(unique_labels)

        if n_current == self.n_clusters:
            return labels

        print(f"  ⚠️  修正类别数: {n_current} -> {self.n_clusters}")

        if n_current < self.n_clusters:
            labels = self._split_clusters_to_target(labels, features, self.n_clusters)
        elif n_current > self.n_clusters:
            labels = self._merge_clusters_to_target(labels, features, self.n_clusters)

        return labels

    def _split_clusters_to_target(self, labels, features, target_n):
        """分裂簇"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_missing = target_n - len(unique_labels)

        new_labels = labels.copy()
        next_label = max(unique_labels) + 1

        for _ in range(n_missing):
            unique_labels, counts = np.unique(new_labels, return_counts=True)
            sorted_indices = np.argsort(-counts)
            largest_label = unique_labels[sorted_indices[0]]
            largest_mask = new_labels == largest_label
            largest_features = features[largest_mask]

            if len(largest_features) < 2:
                break

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(largest_features)

            largest_indices = np.where(largest_mask)[0]
            split_mask = sub_labels == 1
            new_labels[largest_indices[split_mask]] = next_label

            next_label += 1

        unique = np.unique(new_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        return np.vectorize(mapping.get)(new_labels).astype(int)

    def _merge_clusters_to_target(self, labels, features, target_n):
        """合并簇"""
        unique_labels = np.unique(labels)
        n_excess = len(unique_labels) - target_n

        new_labels = labels.copy()

        for _ in range(n_excess):
            unique_labels, counts = np.unique(new_labels, return_counts=True)
            sorted_indices = np.argsort(counts)
            smallest_label = unique_labels[sorted_indices[0]]

            centers = {
                label: np.mean(features[new_labels == label], axis=0)
                for label in unique_labels
            }

            smallest_center = centers[smallest_label]
            min_distance = np.inf
            merge_target = unique_labels[sorted_indices[1]]

            for label in unique_labels:
                if label != smallest_label:
                    distance = np.linalg.norm(centers[label] - smallest_center)
                    if distance < min_distance:
                        min_distance = distance
                        merge_target = label

            new_labels[new_labels == smallest_label] = merge_target

        unique = np.unique(new_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        return np.vectorize(mapping.get)(new_labels).astype(int)

    def _enforce_minimum_cluster_size_postprocess(self, labels, features, max_iterations=5):
        """后处理：最小簇约束"""
        n_samples = len(labels)
        min_size = int(np.ceil(n_samples * self.min_cluster_fraction))

        adjusted_labels = labels.copy()

        for iteration in range(max_iterations):
            unique_labels, counts = np.unique(adjusted_labels, return_counts=True)

            if len(unique_labels) != self.n_clusters:
                break

            small_clusters = unique_labels[counts < min_size]

            if len(small_clusters) == 0:
                break

            for small_label in small_clusters:
                small_mask = adjusted_labels == small_label
                other_labels = unique_labels[unique_labels != small_label]

                if len(other_labels) == 0:
                    break

                small_indices = np.where(small_mask)[0]
                centers = {
                    label: np.mean(features[adjusted_labels == label], axis=0)
                    for label in unique_labels
                }

                small_center = centers[small_label]
                distances = [np.linalg.norm(centers[other_label] - small_center) for other_label in other_labels]
                target_label = other_labels[np.argmin(distances)]

                adjusted_labels[small_indices] = target_label
                break

        return adjusted_labels

    def evaluate_clustering_quality(self, labels):
        """评估聚类质量"""
        unique, counts = np.unique(labels, return_counts=True)
        n_total = len(labels)

        features = self.extract_combined_features()
        hu_means = {lb: np.mean(features[labels == lb, 0]) for lb in unique}
        hu_gradient_correct = all(
            hu_means.get(i, 0) < hu_means.get(i+1, float('inf'))
            for i in range(len(unique)-1)
        )

        quality = {
            'n_clusters': len(unique),
            'cluster_sizes': dict(zip(unique.tolist(), counts.tolist())),
            'size_balance': counts.min() / counts.max() if len(counts) > 1 else 1.0,
            'min_cluster_fraction': counts.min() / n_total,
            'min_constraint_satisfied': (counts.min() / n_total) >= self.min_cluster_fraction,
            'exactly_3_clusters': len(unique) == 3,
            'hu_gradient_correct': hu_gradient_correct,
            'hu_means': hu_means,
        }

        return quality

    def compute_tumor_descriptors(self, nii_img=None):
        """计算肿瘤描述符"""
        if self.labels is None:
            raise ValueError("请先执行聚类!")

        voxel_vol = np.prod(nii_img.header.get_zooms()[:3]) if nii_img else 1.0
        features = self.extract_combined_features()

        descriptors = {
            'tumor_volume_mm3': np.sum(self.mask) * voxel_vol,
            'necrosis_fraction': np.sum(self.labels == 1) / len(self.labels),
            'medium_activity_fraction': np.sum(self.labels == 2) / len(self.labels),
            'high_activity_fraction': np.sum(self.labels == 3) / len(self.labels),
            'mean_arterial_hu': np.mean(features[:, 0]),
            'mean_enhancement_diff': np.mean(features[:, 3]),
            'enhancement_heterogeneity': np.std(features[:, 3]),
        }

        for label in [1, 2, 3]:
            mask = self.labels == label
            if np.sum(mask) > 0:
                descriptors[f'cluster_{label}_mean_hu'] = np.mean(features[mask, 0])
                descriptors[f'cluster_{label}_size'] = np.sum(mask)
                descriptors[f'cluster_{label}_mean_enhancement'] = np.mean(features[mask, 3])

        for i in [1, 2, 3]:
            for j in [1, 2, 3]:
                if i < j:
                    mask_i = self.labels == i
                    mask_j = self.labels == j
                    if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                        mean_i = np.mean(features[mask_i, 0])
                        mean_j = np.mean(features[mask_j, 0])
                        descriptors[f'intra_tumor_diff_class{i}_vs_{j}'] = abs(mean_j - mean_i)

        return descriptors


# ============================================================================
#  工具函数
# ============================================================================

def load_nifti_data(a_path, v_path, roi_path):
    """加载NIfTI数据（无先验版不需要标注路径）"""
    A_img = nib.load(str(a_path))
    V_img = nib.load(str(v_path))
    ROI_img = nib.load(str(roi_path))

    A = A_img.get_fdata(dtype=np.float32)
    V = V_img.get_fdata(dtype=np.float32)
    ROI = (ROI_img.get_fdata() == 2)

    A = np.clip((A + 160) / 400, 0, 1)
    V = np.clip((V + 160) / 400, 0, 1)

    return A_img, V_img, ROI_img, A, V, ROI


# ============================================================================
#  主要流程函数
# ============================================================================

def apply_clustering_to_patients(data_dir, output_dir, arterial_weight=2.0, min_cluster_fraction=0.05):
    """对所有患者执行无先验聚类（强制3类 + HU梯度排序）"""
    print("\n" + "="*70)
    print("无先验聚类（强制3类 + HU梯度排序）")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)
    results = []

    print(f"\n🔍 处理所有患者...")
    for a_file in sorted((data_path / "crop_A").glob("*.nii*")):
        name = a_file.name
        v_file = data_path / "crop_V" / name
        roi_file = data_path / "crop_ROI" / name

        output_path = Path(output_dir) / name.replace('.nii', '_clusters.nii')
        if output_path.exists():
            print(f"⏭️  跳过已处理: {name}")
            continue

        if v_file.exists() and roi_file.exists():
            print(f"\n{'='*70}\n🔍 {name}")

            try:
                A_img, V_img, ROI_img, A, V, ROI = load_nifti_data(a_file, v_file, roi_file)

                analyzer = PancreaticTumorHeterogeneityAnalyzerNoPrior(
                    A, V, ROI, arterial_weight, min_cluster_fraction
                )

                labels_1d = analyzer.adaptive_clustering()
                analyzer.labels = labels_1d

                # 保存3D标签
                labels_3d = np.zeros(ROI.shape, dtype=np.uint16)
                labels_3d[ROI] = labels_1d
                nib.save(nib.Nifti1Image(labels_3d, A_img.affine), str(output_path))

                # 评估
                quality = analyzer.evaluate_clustering_quality(labels_1d)
                descriptors = analyzer.compute_tumor_descriptors(A_img)

                print(f"\n📈 质量:")
                print(f"  簇数: {quality['n_clusters']} {'✓' if quality['exactly_3_clusters'] else '❌'}")
                print(f"  最小簇: {quality['min_cluster_fraction']*100:.1f}% | "
                      f"约束: {'✓' if quality['min_constraint_satisfied'] else '✗'}")
                print(f"  HU梯度: {'✓' if quality['hu_gradient_correct'] else '❌'}")
                print(f"  平衡度: {quality['size_balance']:.3f}")

                if 'hu_means' in quality:
                    print(f"  HU均值: ", end="")
                    for lb in sorted(quality['hu_means'].keys()):
                        print(f"L{lb}={quality['hu_means'][lb]:.2f} ", end="")
                    print()

                results.append({'case': name, **quality, **descriptors})
                print(f"\n✅ 已保存: {output_path.name}")

            except Exception as e:
                print(f"\n❌ 失败: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = Path(output_dir) / "results_summary.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n{'='*70}")
        print(f"📊 汇总已保存: {csv_path}")

        # 统计
        invalid_cases = df[df['n_clusters'] != 3]
        gradient_incorrect = df[df['hu_gradient_correct'] == False]
        constraint_failed = df[df['min_constraint_satisfied'] == False]

        print(f"\n📈 统计:")
        print(f"  总病例数: {len(df)}")
        print(f"  3类病例: {len(df[df['n_clusters'] == 3])}/{len(df)}")
        print(f"  HU梯度正确: {len(df[df['hu_gradient_correct'] == True])}/{len(df)}")
        print(f"  约束满足: {len(df[df['min_constraint_satisfied'] == True])}/{len(df)}")

        if len(invalid_cases) > 0:
            print(f"\n⚠️  {len(invalid_cases)}个病例类别数异常:")
            for _, row in invalid_cases.iterrows():
                print(f"    {row['case']}: {row['n_clusters']}类")

        if len(gradient_incorrect) > 0:
            print(f"\n⚠️  {len(gradient_incorrect)}个病例HU梯度异常")

        if len(constraint_failed) > 0:
            print(f"\n⚠️  {len(constraint_failed)}个病例未满足{min_cluster_fraction*100:.0f}%约束")

        if len(invalid_cases) == 0 and len(gradient_incorrect) == 0 and len(constraint_failed) == 0:
            print(f"\n✅ 所有{len(df)}个病例均满足：3类 + HU梯度正确 + {min_cluster_fraction*100:.0f}%约束")


# ============================================================================
#  主函数
# ============================================================================

if __name__ == "__main__":
    DATA_DIR = "./data"
    OUTPUT_DIR = "./cluster/without_point"

    ARTERIAL_WEIGHT = 2.0
    MIN_CLUSTER_FRACTION = 0.05

    print("\n" + "="*70)
    print("胰腺肿瘤异质性分析系统-无点标注先验聚类（对照版）")
    print("="*70)
    print("✨ 特性:")
    print("  1. 🚫 不使用任何医生标注先验")
    print("  2. 🎯 ConstrainedKMeans（内置5%约束）")
    print("  3. 🔒 强制保证3个类别（G1/G2/G3）")
    print("  4. 📈 HU值梯度排序（Label1<Label2<Label3）")
    print("  5. 🛡️  双重保障（约束+后处理）")
    print("  6. ⚖️  与 point_cluster.py 输出格式一致，便于消融对比")
    print("="*70)

    # 单阶段：直接对所有患者执行聚类
    apply_clustering_to_patients(DATA_DIR, OUTPUT_DIR, ARTERIAL_WEIGHT, MIN_CLUSTER_FRACTION)

    print("\n" + "="*70)
    print("✨ 完整流程完成!")
    print("="*70)
    print(f"\n📁 输出文件:")
    print(f"  - 聚类结果: {OUTPUT_DIR}/")
    print(f"  - 结果CSV: {OUTPUT_DIR}/results_summary.csv")
