"""
胰腺肿瘤异质性分析完整流程
- 先验知识学习（从标注数据）
- ConstrainedKMeans（内置5%约束）
- 强制3类 + HU梯度排序
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import warnings
import pickle
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import generic_filter, sobel
from scipy.stats import norm, gaussian_kde
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')


# ============================================================================
#  ConstrainedKMeans - 带5%约束的KMeans
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
#  TumorHeterogeneityPriorKnowledge - 先验知识学习器
# ============================================================================

class TumorHeterogeneityPriorKnowledge:
    """肿瘤异质性先验知识学习器"""
    
    def __init__(self, min_cluster_fraction=0.05):
        self.prior_learned = False
        self.inter_tumor_stats = {}
        self.intra_tumor_stats = {}
        self.feature_distributions = {}
        self.cluster_centroids = {}
        self.feature_importance = {}
        self.feature_dimension = 3
        self.min_cluster_fraction = min_cluster_fraction
        self.n_clusters = 3
        
    def learn_from_annotations(self, all_analyzers):
        """从标注数据学习先验知识"""
        print(f"\n🔬 从临床标注点学习先验知识...")
        
        all_annotation_data = {1: [], 2: [], 3: []}
        intra_tumor_differences = []
        
        for analyzer in all_analyzers:
            if not analyzer.has_annotations:
                continue
            
            features = analyzer.extract_arterial_features()
            self.feature_dimension = features.shape[1]
            
            tumor_class_hus = {1: [], 2: [], 3: []}
            for ann_idx, label in zip(analyzer.annotation_indices, analyzer.annotation_labels):
                all_annotation_data[label].append(features[ann_idx])
                tumor_class_hus[label].append(features[ann_idx, 0])
            
            intra_diffs = self._compute_intra_tumor_differences(tumor_class_hus)
            if intra_diffs:
                intra_tumor_differences.append(intra_diffs)
        
        total_annotations = sum(len(v) for v in all_annotation_data.values())
        if total_annotations < 10:
            print("⚠️  标注点不足，无法学习有效先验知识")
            return False
        
        print(f"  总标注点: {total_annotations}")
        for label in [1, 2, 3]:
            print(f"    Label{label}: {len(all_annotation_data[label])}")
        
        self._learn_intra_tumor_patterns(intra_tumor_differences)
        self._learn_inter_tumor_distribution(all_annotation_data)
        self._learn_feature_distributions(all_annotation_data)
        self._learn_cluster_centroids(all_annotation_data)
        self._compute_feature_importance(all_annotation_data)
        
        self.prior_learned = True
        print(f"✅ 先验知识学习完成")
        return True
    
    def _compute_intra_tumor_differences(self, tumor_class_hus):
        """计算单个肿瘤内三类点之间的HU差异"""
        diffs = {}
        valid_classes = [k for k, v in tumor_class_hus.items() if len(v) > 0]
        
        if len(valid_classes) < 2:
            return None
        
        for i, class_i in enumerate(valid_classes):
            for class_j in valid_classes[i+1:]:
                mean_i = np.mean(tumor_class_hus[class_i])
                mean_j = np.mean(tumor_class_hus[class_j])
                diff_key = f"class_{class_i}_vs_{class_j}"
                diffs[diff_key] = abs(mean_j - mean_i)
        
        for class_k in valid_classes:
            if len(tumor_class_hus[class_k]) > 1:
                diffs[f"class_{class_k}_intra_std"] = np.std(tumor_class_hus[class_k])
        
        return diffs
    
    def _learn_intra_tumor_patterns(self, intra_tumor_differences):
        """学习肿瘤内异质性模式"""
        if len(intra_tumor_differences) == 0:
            return
        
        all_diffs = {}
        for tumor_diffs in intra_tumor_differences:
            for key, value in tumor_diffs.items():
                if key not in all_diffs:
                    all_diffs[key] = []
                all_diffs[key].append(value)
        
        self.intra_tumor_stats = {}
        for key, values in all_diffs.items():
            self.intra_tumor_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        
        print(f"\n  📊 肿瘤内差异先验统计:")
        for key, stats in sorted(self.intra_tumor_stats.items()):
            if 'vs' in key:
                print(f"    {key}: {stats['median']:.2f} [{stats['q25']:.2f}, {stats['q75']:.2f}]")
    
    def _learn_inter_tumor_distribution(self, annotation_data):
        """学习肿瘤间分布规律"""
        for label in [1, 2, 3]:
            if len(annotation_data[label]) == 0:
                continue
            data = np.array(annotation_data[label])
            self.inter_tumor_stats[label] = {
                'mean_a_hu': np.mean(data[:, 0]),
                'std_a_hu': np.std(data[:, 0]),
                'median_a_hu': np.median(data[:, 0]),
            }
    
    def _learn_feature_distributions(self, annotation_data):
        """学习特征概率分布"""
        feature_names = ['weighted_a_hu', 'a_texture', 'a_gradient']
        
        for label in [1, 2, 3]:
            if len(annotation_data[label]) == 0:
                continue
            data = np.array(annotation_data[label])
            self.feature_distributions[label] = {}
            
            for i, name in enumerate(feature_names[:data.shape[1]]):
                if len(data[:, i]) > 1:
                    try:
                        self.feature_distributions[label][name] = gaussian_kde(data[:, i])
                    except:
                        mu, sigma = norm.fit(data[:, i])
                        self.feature_distributions[label][name] = (mu, sigma)
    
    def _learn_cluster_centroids(self, annotation_data):
        """学习类别中心（按HU值从低到高排序）"""
        centroids_by_label = {}
        for label in [1, 2, 3]:
            if len(annotation_data[label]) > 0:
                centroids_by_label[label] = np.median(np.array(annotation_data[label]), axis=0)
        
        # 按HU值排序
        if len(centroids_by_label) == 3:
            sorted_labels = sorted(centroids_by_label.keys(), 
                                 key=lambda x: centroids_by_label[x][0])
            self.cluster_centroids = {
                i+1: centroids_by_label[sorted_labels[i]] 
                for i in range(3)
            }
            print(f"\n  🎯 先验中心按HU排序: {sorted_labels} -> [1, 2, 3]")
            for i in range(3):
                print(f"    Label{i+1}: HU={self.cluster_centroids[i+1][0]:.2f}")
        else:
            self.cluster_centroids = centroids_by_label
    
    def _compute_feature_importance(self, annotation_data):
        """计算特征重要性"""
        all_features, all_labels = [], []
        for label in [1, 2, 3]:
            if len(annotation_data[label]) > 0:
                data = np.array(annotation_data[label])
                all_features.extend(data)
                all_labels.extend([label] * len(data))
        
        if len(all_features) == 0:
            return
            
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        n_features = all_features.shape[1]
        feature_importance = np.zeros(n_features)
        
        for i in range(n_features):
            feature_values = all_features[:, i]
            overall_mean = np.mean(feature_values)
            
            between_var = sum(
                np.sum(all_labels == label) * (np.mean(feature_values[all_labels == label]) - overall_mean) ** 2
                for label in [1, 2, 3] if np.sum(all_labels == label) > 0
            )
            
            within_var = sum(
                np.sum((feature_values[all_labels == label] - np.mean(feature_values[all_labels == label])) ** 2)
                for label in [1, 2, 3] if np.sum(all_labels == label) > 0
            )
            
            feature_importance[i] = between_var / within_var if within_var > 0 else 0
        
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        self.feature_importance = feature_importance
        print(f"\n  特征重要性: {dict(zip(['A_HU', 'A_texture', 'A_gradient'][:n_features], feature_importance))}")
    
    def get_prior_probabilities(self, features, current_labels=None, n_total=None):
        """获取特征向量的先验概率"""
        if not self.prior_learned:
            return np.array([0.33, 0.33, 0.34])
        
        probabilities = []
        
        for label in [1, 2, 3]:
            if label in self.cluster_centroids:
                centroid = self.cluster_centroids[label]
                weights = self.feature_importance if len(self.feature_importance) == len(features) else np.ones_like(features)
                distance = np.sqrt(np.sum(weights * (features[:len(centroid)] - centroid) ** 2))
                max_distance = np.sqrt(np.sum(weights * (np.ptp(features[:len(centroid)])) ** 2))
                similarity = 1.0 - min(distance / (max_distance + 1e-10), 1.0)
            else:
                similarity = 0.5
            
            prob_dist = 1.0
            if label in self.feature_distributions:
                for i, name in enumerate(['weighted_a_hu', 'a_texture', 'a_gradient'][:len(features)]):
                    if name in self.feature_distributions[label]:
                        dist = self.feature_distributions[label][name]
                        if isinstance(dist, gaussian_kde):
                            prob_dist *= max(dist(features[i])[0], 1e-10)
                        else:
                            mu, sigma = dist
                            prob_dist *= norm.pdf(features[i], mu, sigma) + 1e-10
            
            cluster_penalty = 1.0
            if current_labels is not None and n_total is not None:
                current_count = np.sum(current_labels == label)
                current_fraction = current_count / n_total
                if current_fraction < self.min_cluster_fraction:
                    cluster_penalty = 1.5
            
            probabilities.append((0.7 * prob_dist + 0.3 * similarity) * cluster_penalty)
        
        probabilities = np.array(probabilities)
        return probabilities / (np.sum(probabilities) + 1e-10)
    
    def get_intra_tumor_consistency_score(self, features, labels):
        """评估聚类结果与肿瘤内差异先验的一致性"""
        if not self.intra_tumor_stats:
            return 1.0
        
        score = 1.0
        unique_labels = np.unique(labels)
        
        for i, label_i in enumerate(unique_labels):
            for label_j in unique_labels[i+1:]:
                mask_i = labels == label_i
                mask_j = labels == label_j
                
                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    mean_i = np.mean(features[mask_i, 0])
                    mean_j = np.mean(features[mask_j, 0])
                    observed_diff = abs(mean_j - mean_i)
                    
                    diff_key = f"class_{min(label_i, label_j)}_vs_{max(label_i, label_j)}"
                    if diff_key in self.intra_tumor_stats:
                        expected_median = self.intra_tumor_stats[diff_key]['median']
                        expected_range = (
                            self.intra_tumor_stats[diff_key]['q25'],
                            self.intra_tumor_stats[diff_key]['q75']
                        )
                        
                        if expected_range[0] <= observed_diff <= expected_range[1]:
                            score *= 1.1
                        else:
                            deviation = min(
                                abs(observed_diff - expected_range[0]),
                                abs(observed_diff - expected_range[1])
                            ) / expected_median
                            score *= max(0.8, 1.0 - 0.1 * deviation)
        
        return np.clip(score, 0.5, 1.5)

    def save_prior_knowledge(self, filepath):
        """保存先验知识"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'inter_tumor_stats': self.inter_tumor_stats,
                'intra_tumor_stats': self.intra_tumor_stats,
                'feature_distributions': self.feature_distributions,
                'cluster_centroids': self.cluster_centroids,
                'feature_importance': self.feature_importance,
                'prior_learned': self.prior_learned,
                'feature_dimension': self.feature_dimension,
                'min_cluster_fraction': self.min_cluster_fraction,
                'n_clusters': self.n_clusters
            }, f)
        print(f"\n💾 先验知识已保存: {filepath}")
    
    def load_prior_knowledge(self, filepath):
        """加载先验知识"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.__dict__.update(data)
            print(f"\n📂 先验知识已加载: {filepath}")
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False


# ============================================================================
#  PancreaticTumorHeterogeneityAnalyzer - 肿瘤分析器
# ============================================================================

class PancreaticTumorHeterogeneityAnalyzer:
    """胰腺癌异质性分析器"""

    def __init__(self, arterial_ct, venous_ct, tumor_mask, doctor_annotations=None, 
                 arterial_weight=2.0, prior_knowledge=None, min_cluster_fraction=0.05):
        self.arterial = arterial_ct
        self.venous = venous_ct
        self.mask = tumor_mask.astype(bool)
        self.arterial_weight = arterial_weight
        self.prior_knowledge = prior_knowledge
        self.min_cluster_fraction = min_cluster_fraction
        self.n_clusters = 3
        
        self.annotation_points = None
        self.annotation_labels = None
        self.annotation_indices = None
        self.has_annotations = False
        self.labels = None
        
        if doctor_annotations is not None:
            self._parse_doctor_annotations(doctor_annotations)

    def _parse_doctor_annotations(self, doctor_annotations):
        """解析医生标注"""
        try:
            coords = np.where(doctor_annotations > 0)
            if len(coords[0]) == 0:
                return
            
            points = np.column_stack(coords)
            labels = doctor_annotations[coords]
            
            valid_points, valid_labels = [], []
            for point, label in zip(points, labels):
                if self.mask[tuple(point)] and label in [1, 2, 3]:
                    valid_points.append(point)
                    valid_labels.append(label)
            
            if len(valid_points) == 0:
                return
            
            self.annotation_points = np.array(valid_points)
            self.annotation_labels = np.array(valid_labels)
            
            mask_coords = np.array(np.where(self.mask)).T
            self.annotation_indices = [
                np.argmin(np.sum((mask_coords - point) ** 2, axis=1))
                for point in self.annotation_points
            ]
            self.has_annotations = True
            
        except Exception as e:
            pass

    def extract_arterial_features(self):
        """提取A期特征"""
        tumor_voxels_a = self.arterial[self.mask]
        
        return np.column_stack([
            tumor_voxels_a * self.arterial_weight,
            generic_filter(self.arterial, np.std, size=3)[self.mask],
            np.sqrt(sum(sobel(self.arterial, axis=i)**2 for i in range(3)))[self.mask]
        ]).astype(np.float64)
    
    def extract_combined_features(self):
        """提取组合特征"""
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

    def adaptive_clustering(self, features=None, method='prior_guided'):
        """自适应聚类"""
        if features is None:
            features = self.extract_combined_features()
        
        print(f"\n🔬 聚类方法: {method}, 特征维度: {features.shape}")
        
        if method == 'prior_guided' and self.prior_knowledge and self.prior_knowledge.prior_learned:
            labels = self._prior_guided_clustering(features)
        elif self.has_annotations:
            labels = self._annotation_guided_clustering(features)
        else:
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
        
        # 验证一致性
        if self.prior_knowledge and self.prior_knowledge.prior_learned:
            consistency_score = self.prior_knowledge.get_intra_tumor_consistency_score(features, labels)
            print(f"  🎯 肿瘤内差异一致性: {consistency_score:.3f}")
        
        return labels.astype(int) + 1

    def _prior_guided_clustering(self, features):
        """先验引导聚类"""
        a_features = features[:, :3]
        av_features = features[:, 3:]
        
        # 特征选择
        if hasattr(self.prior_knowledge, 'feature_importance') and len(self.prior_knowledge.feature_importance) > 0:
            importance = self.prior_knowledge.feature_importance
            selected_indices = np.where(importance > np.mean(importance) * 0.8)[0]
            if len(selected_indices) == 0:
                selected_indices = np.arange(a_features.shape[1])
        else:
            selected_indices = np.arange(a_features.shape[1])
        
        a_processed = RobustScaler().fit_transform(a_features[:, selected_indices])
        
        av_variances = np.var(av_features, axis=0)
        av_mask = av_variances > np.percentile(av_variances, 30)
        av_processed = RobustScaler().fit_transform(av_features[:, av_mask])
        
        combined = np.hstack([a_processed * 0.6, av_processed * 0.4])
        
        # 准备先验中心
        init_centers = None
        if len(self.prior_knowledge.cluster_centroids) == 3:
            try:
                prior_centers_raw = [
                    self.prior_knowledge.cluster_centroids[i+1][:len(selected_indices)] 
                    for i in range(3)
                ]
                scaler_a = RobustScaler()
                scaler_a.fit(a_features[:, selected_indices])
                prior_a_scaled = scaler_a.transform(prior_centers_raw)
                prior_av_scaled = np.zeros((3, av_processed.shape[1]))
                init_centers = np.hstack([prior_a_scaled * 0.6, prior_av_scaled * 0.4])
                print(f"  🎯 使用先验中心初始化")
            except:
                pass
        
        # ConstrainedKMeans
        try:
            constrained_kmeans = ConstrainedKMeans(
                n_clusters=self.n_clusters,
                min_cluster_fraction=self.min_cluster_fraction,
                max_iter=100,
                random_state=42
            )
            labels = constrained_kmeans.fit_predict(combined, init_centers=init_centers)
            print(f"  ✓ ConstrainedKMeans(n={self.n_clusters}, min≥{self.min_cluster_fraction*100:.0f}%)")
            
        except Exception as e:
            print(f"  ⚠️  ConstrainedKMeans失败: {e}, 回退到GMM")
            try:
                if init_centers is not None:
                    gmm = GaussianMixture(
                        n_components=self.n_clusters, 
                        random_state=42, 
                        max_iter=100, 
                        n_init=1,
                        means_init=init_centers
                    )
                else:
                    gmm = GaussianMixture(
                        n_components=self.n_clusters, 
                        random_state=42, 
                        max_iter=100, 
                        n_init=3
                    )
                
                gmm.fit(combined)
                posterior_probs = gmm.predict_proba(combined)
                
                prior_probs = np.array([
                    self.prior_knowledge.get_prior_probabilities(
                        a_features[i], 
                        current_labels=None,
                        n_total=len(a_features)
                    )
                    for i in range(len(a_features))
                ])
                
                combined_probs = 0.65 * posterior_probs + 0.35 * prior_probs
                labels = np.argmax(combined_probs, axis=1)
                print(f"  ✓ GMM(n={self.n_clusters}) + 先验概率")
                
            except Exception as e2:
                print(f"  ⚠️  GMM也失败: {e2}, 使用标准KMeans")
                labels = self._basic_clustering_impl(combined, init_centers)
        
        return self._reorder_labels_by_hu(labels, a_features[:, 0])
    
    def _annotation_guided_clustering(self, features):
        """标注引导聚类"""
        X_scaled = RobustScaler().fit_transform(features)
        
        init_centers = None
        if self.has_annotations and len(np.unique(self.annotation_labels)) == 3:
            try:
                centers = []
                for label in [1, 2, 3]:
                    mask = self.annotation_labels == label
                    if np.sum(mask) > 0:
                        label_indices = np.array(self.annotation_indices)[mask]
                        centers.append(np.mean(X_scaled[label_indices], axis=0))
                
                if len(centers) == 3:
                    init_centers = np.array(centers)
                    hu_means = [np.mean(features[self.annotation_labels == i, 0]) for i in [1, 2, 3]]
                    sorted_idx = np.argsort(hu_means)
                    init_centers = init_centers[sorted_idx]
                    print(f"  🎯 使用标注中心初始化")
            except:
                pass
        
        try:
            constrained_kmeans = ConstrainedKMeans(
                n_clusters=self.n_clusters,
                min_cluster_fraction=self.min_cluster_fraction,
                max_iter=100,
                random_state=42
            )
            labels = constrained_kmeans.fit_predict(X_scaled, init_centers=init_centers)
            print(f"  ✓ ConstrainedKMeans + 标注")
            
        except Exception as e:
            print(f"  ⚠️  失败: {e}, 回退")
            labels = self._basic_clustering_impl(X_scaled, init_centers)
        
        return self._reorder_labels_by_hu(labels, features[:, 0])
    
    def _basic_clustering(self, features):
        """基础聚类"""
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
            print(f"  ✓ ConstrainedKMeans")
            return labels
            
        except Exception as e:
            print(f"  ⚠️  失败: {e}, 使用标准KMeans")
            if init_centers is not None:
                kmeans = KMeans(n_clusters=self.n_clusters, init=init_centers, n_init=1, random_state=42)
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            return labels

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

def load_nifti_data(a_path, v_path, roi_path, ann_path=None):
    """加载NIfTI数据"""
    A_img = nib.load(str(a_path))
    V_img = nib.load(str(v_path))
    ROI_img = nib.load(str(roi_path))
    
    A = A_img.get_fdata(dtype=np.float32)
    V = V_img.get_fdata(dtype=np.float32)
    ROI = (ROI_img.get_fdata() == 2)
    
    A = np.clip((A + 160) / 400, 0, 1)
    V = np.clip((V + 160) / 400, 0, 1)
    
    ANN = None
    if ann_path and Path(ann_path).exists():
        try:
            ANN_img = nib.load(str(ann_path))
            ANN = ANN_img.get_fdata()
        except:
            pass
    
    return A_img, V_img, ROI_img, A, V, ROI, ANN


# ============================================================================
#  主要流程函数
# ============================================================================

def learn_prior_knowledge(data_dir, arterial_weight=2.0, min_cluster_fraction=0.05, save_path="./prior_knowledge.pkl"):
    """学习先验知识"""
    print("\n" + "="*70)
    print("阶段1: 学习先验知识（强制3类 + HU梯度排序）")
    print("="*70)
    
    prior = TumorHeterogeneityPriorKnowledge(min_cluster_fraction)
    analyzers = []
    
    data_path = Path(data_dir)
    a_files = sorted((data_path / "crop_A").glob("*.nii*"))
    
    print(f"\n🔍 扫描标注数据...")
    for a_file in a_files:
        name = a_file.name
        v_file = data_path / "crop_V" / name
        roi_file = data_path / "crop_ROI" / name
        ann_file = data_path / "crop_DOT" / name
        
        if all(f.exists() for f in [v_file, roi_file, ann_file]):
            try:
                _, _, _, A, V, ROI, ANN = load_nifti_data(a_file, v_file, roi_file, ann_file)
                analyzer = PancreaticTumorHeterogeneityAnalyzer(
                    A, V, ROI, ANN, arterial_weight, 
                    prior_knowledge=None, 
                    min_cluster_fraction=min_cluster_fraction
                )
                if analyzer.has_annotations:
                    analyzers.append(analyzer)
                    print(f"  ✓ {name}: {len(analyzer.annotation_labels)}个标注")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
    
    if len(analyzers) == 0:
        print("\n❌ 无有效标注数据")
        return None
    
    print(f"\n✅ 共找到 {len(analyzers)} 个带标注病例")
    
    success = prior.learn_from_annotations(analyzers)
    if success and save_path:
        prior.save_prior_knowledge(save_path)
    
    return prior


def apply_prior_to_patients(prior_path, data_dir, output_dir, arterial_weight=2.0, min_cluster_fraction=0.05):
    """应用先验知识到患者（强制3类 + HU梯度排序）"""
    print("\n" + "="*70)
    print("阶段2: 应用先验知识（强制3类 + HU梯度排序）")
    print("="*70)
    
    prior = TumorHeterogeneityPriorKnowledge(min_cluster_fraction)
    if not prior.load_prior_knowledge(prior_path):
        print("❌ 无法加载先验知识")
        return
    
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
                A_img, V_img, ROI_img, A, V, ROI, _ = load_nifti_data(a_file, v_file, roi_file)
                
                analyzer = PancreaticTumorHeterogeneityAnalyzer(
                    A, V, ROI, None, arterial_weight, prior, min_cluster_fraction
                )
                
                labels_1d = analyzer.adaptive_clustering(method='prior_guided')
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
    PRIOR_PATH = "./cluster/point_prior_knowledge.pkl"
    OUTPUT_DIR = "./cluster/with_point"

    ARTERIAL_WEIGHT = 2.0
    MIN_CLUSTER_FRACTION = 0.05
    
    print("\n" + "="*70)
    print("胰腺肿瘤异质性分析系统-带点标注先验聚类")
    print("="*70)
    print("✨ 特性:")
    print("  1. 🧠 先验知识学习（从标注数据）")
    print("  2. 🎯 ConstrainedKMeans（内置5%约束）")
    print("  3. 🔒 强制保证3个类别（G1/G2/G3）")
    print("  4. 📈 HU值梯度排序（Label1<Label2<Label3）")
    print("  5. 🛡️  三重保障（先验+约束+后处理）")
    print("  6. 📊 肿瘤内/间差异先验融合")
    print("="*70)
    
    # 阶段1: 学习或加载先验知识
    if not os.path.exists(PRIOR_PATH):
        learn_prior_knowledge(DATA_DIR, ARTERIAL_WEIGHT, MIN_CLUSTER_FRACTION, PRIOR_PATH)
    else:
        print("  ✓ 将使用现有先验知识")

    # 阶段2: 应用先验知识进行聚类
    apply_prior_to_patients(PRIOR_PATH, DATA_DIR, OUTPUT_DIR, ARTERIAL_WEIGHT, MIN_CLUSTER_FRACTION)
    
    print("\n" + "="*70)
    print("✨ 完整流程完成!")
    print("="*70)
    print(f"\n📁 输出文件:")
    print(f"  - 先验知识: {PRIOR_PATH}")
    print(f"  - 聚类结果: {OUTPUT_DIR}/")
    print(f"  - 结果CSV: {OUTPUT_DIR}/results_summary.csv")