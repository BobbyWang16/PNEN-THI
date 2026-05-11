import numpy as np  
import pandas as pd  
import nibabel as nib  
from pathlib import Path  
import warnings  
import os  

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering  
from sklearn.mixture import GaussianMixture  
from sklearn.preprocessing import RobustScaler  
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  
from sklearn.neighbors import KNeighborsClassifier  
from scipy.ndimage import generic_filter, sobel  
from scipy.spatial.distance import cdist  

warnings.filterwarnings('ignore')  


# ============================================================================  
#  ConstrainedKMeans - 带5%约束的KMeans  
# ============================================================================  

class ConstrainedKMeans:  
    """带最小簇大小约束的KMeans"""  
    
    def __init__(self, n_clusters=3, min_cluster_fraction=0.05, max_iter=100, random_state=42):  
        self.n_clusters = n_clusters  
        self.min_cluster_fraction = min_cluster_fraction  
        self.max_iter = max_iter  
        self.random_state = random_state  
        self.cluster_centers_ = None  
        self.labels_ = None  
        
    def fit_predict(self, X):  
        """带约束的KMeans聚类"""  
        np.random.seed(self.random_state)  
        n_samples = len(X)  
        min_size = int(np.ceil(n_samples * self.min_cluster_fraction))  
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=5)  
        labels = kmeans.fit_predict(X)  
        centers = kmeans.cluster_centers_  
        
        # 初始约束分配  
        labels = self._constrained_assign(X, centers, min_size)  
        centers = self._update_centers(X, labels)  
        
        # 迭代优化  
        for iteration in range(self.max_iter):  
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
        """最近邻分配"""  
        distances = cdist(X, centers, metric='euclidean')  
        return np.argmin(distances, axis=1)  
    
    def _constrained_assign(self, X, centers, min_size):  
        """带约束的分配"""  
        n_samples = len(X)  
        distances = cdist(X, centers, metric='euclidean')  
        
        labels = np.full(n_samples, -1, dtype=int)  
        cluster_sizes = np.zeros(self.n_clusters, dtype=int)  
        
        # 贪心分配：按距离排序，优先保证最小簇大小  
        sample_indices, cluster_indices = np.meshgrid(  
            np.arange(n_samples), np.arange(self.n_clusters), indexing='ij'  
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
        
        # 分配剩余样本  
        unassigned = [i for i in range(n_samples) if i not in assigned]  
        for sample in unassigned:  
            cluster = np.argmin(distances[sample])  
            labels[sample] = cluster  
            cluster_sizes[cluster] += 1  
        
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
#  高效肿瘤聚类分析器  
# ============================================================================  

class FastTumorClusterer:  
    """高效的肿瘤异质性聚类分析器 - 无先验约束"""  

    def __init__(self, arterial_ct, venous_ct, tumor_mask, arterial_weight=2.0, n_clusters=3, min_cluster_fraction=0.05):  
        self.arterial = arterial_ct  
        self.venous = venous_ct  
        self.mask = tumor_mask.astype(bool)  
        self.arterial_weight = arterial_weight  
        self.n_clusters = n_clusters  
        self.min_cluster_fraction = min_cluster_fraction  
        self.features = None  
        self.labels = None  
        self.algorithm = None  

    def extract_features(self):  
        """快速特征提取"""  
        tumor_a = self.arterial[self.mask].astype(np.float64)  
        tumor_v = self.venous[self.mask].astype(np.float64)  
        
        weighted_a = tumor_a * self.arterial_weight  
        enh_diff = tumor_v - tumor_a  
        enh_ratio = tumor_v / np.clip(tumor_a, 1e-10, None)  
        
        # 局部纹理特征  
        sobel_mag = np.sqrt(sum(sobel(self.arterial, axis=i)**2 for i in range(3)))[self.mask]  
        
        self.features = np.column_stack([  
            weighted_a,                    # 0  
            enh_diff,                      # 1  
            enh_ratio,                     # 2  
            np.abs(enh_diff),              # 3  
            tumor_v,                       # 4  
            (weighted_a + tumor_v) / 2,    # 5  
            sobel_mag,                     # 6  
        ]).astype(np.float64)  
        
        return self.features  

    def _cluster_constrained_kmeans(self, X_scaled):  
        """ConstrainedKMeans聚类"""  
        ckm = ConstrainedKMeans(  
            n_clusters=self.n_clusters,  
            min_cluster_fraction=self.min_cluster_fraction,  
            max_iter=100,  
            random_state=42  
        )  
        labels = ckm.fit_predict(X_scaled)  
        print(f"      ✓ ConstrainedKMeans (min≥{self.min_cluster_fraction*100:.0f}%)")  
        return labels  

    def _cluster_kmeans(self, X_scaled):  
        """标准KMeans聚类"""  
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)  
        labels = kmeans.fit_predict(X_scaled)  
        print(f"      ✓ KMeans")  
        return labels  

    def _cluster_gmm(self, X_scaled):  
        """高斯混合模型 - 健壮版本"""  
        X_float64 = X_scaled.astype(np.float64)  
        
        # 尝试多种配置  
        for reg_covar, covar_type in [  
            (1e-3, 'full'),  
            (1e-2, 'full'),  
            (1e-2, 'diag'),  
            (1e-1, 'tied'),  
        ]:  
            try:  
                gmm = GaussianMixture(  
                    n_components=self.n_clusters,  
                    covariance_type=covar_type,  
                    reg_covar=reg_covar,  
                    random_state=42,  
                    n_init=2,  
                    max_iter=200,  
                    warm_start=False,  
                    verbose=0  
                )  
                labels = gmm.fit_predict(X_float64)  
                print(f"      ✓ GMM (cov={covar_type}, reg={reg_covar})")  
                return labels  
            
            except Exception as e:  
                continue  
        
        # 回退到KMeans  
        print(f"      ⚠️  GMM失败，回退到KMeans")  
        return self._cluster_kmeans(X_scaled.astype(np.float32))  

    def _cluster_agglomerative_sampled(self, X_scaled):  
        """采样凝聚聚类（解决内存问题）"""  
        n_samples = len(X_scaled)  
        
        # 智能采样  
        if n_samples > 50000:  
            sample_size = 10000  
        elif n_samples > 10000:  
            sample_size = 5000  
        else:  
            sample_size = n_samples  
        
        if sample_size < n_samples:  
            indices = np.random.choice(n_samples, sample_size, replace=False)  
            X_sample = X_scaled[indices]  
            
            agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='complete')  
            labels_sample = agg.fit_predict(X_sample)  
            
            # KNN扩展到全集  
            knn = KNeighborsClassifier(n_neighbors=3)  
            knn.fit(X_sample, labels_sample)  
            labels = knn.predict(X_scaled)  
            print(f"      ✓ Agglomerative (采样{sample_size}/{n_samples})")  
        else:  
            agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='complete')  
            labels = agg.fit_predict(X_scaled)  
            print(f"      ✓ Agglomerative")  
        
        return labels  

    def _cluster_dbscan(self, X_scaled):  
        """DBSCAN聚类"""  
        dbscan = DBSCAN(eps=0.6, min_samples=5)  
        labels = dbscan.fit_predict(X_scaled)  
        
        unique_labels = np.unique(labels)  
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)  
        
        if n_clusters_found != self.n_clusters:  
            print(f"      ⚠️  DBSCAN产生{n_clusters_found}类，回退到KMeans")  
            return self._cluster_kmeans(X_scaled.astype(np.float32))  
        
        print(f"      ✓ DBSCAN")  
        return labels  

    def _cluster_spectral(self, X_scaled):  
        """谱聚类"""  
        try:  
            # 对大样本采样  
            n_samples = len(X_scaled)  
            if n_samples > 10000:  
                indices = np.random.choice(n_samples, 5000, replace=False)  
                X_sample = X_scaled[indices]  
                
                sc = SpectralClustering(  
                    n_clusters=self.n_clusters,  
                    affinity='nearest_neighbors',  
                    n_neighbors=10,  
                    random_state=42  
                )  
                labels_sample = sc.fit_predict(X_sample)  
                
                knn = KNeighborsClassifier(n_neighbors=5)  
                knn.fit(X_sample, labels_sample)  
                labels = knn.predict(X_scaled)  
                print(f"      ✓ Spectral (采样)")  
            else:  
                sc = SpectralClustering(  
                    n_clusters=self.n_clusters,  
                    affinity='nearest_neighbors',  
                    n_neighbors=10,  
                    random_state=42  
                )  
                labels = sc.fit_predict(X_scaled)  
                print(f"      ✓ Spectral")  
            
            return labels  
        
        except Exception as e:  
            print(f"      ⚠️  Spectral失败: {e}，回退到KMeans")  
            return self._cluster_kmeans(X_scaled.astype(np.float32))  

    def cluster(self, method='constrained_kmeans'):  
        """统一聚类接口"""  
        if self.features is None:  
            self.extract_features()  
        
        # 特征缩放  
        X_scaled = RobustScaler().fit_transform(self.features)  
        
        n_samples = len(self.features)  
        print(f"    🔬 {method.upper()} 聚类中 ({n_samples} 样本)...")  
        
        try:  
            if method == 'constrained_kmeans':  
                labels = self._cluster_constrained_kmeans(X_scaled)  
            elif method == 'kmeans':  
                labels = self._cluster_kmeans(X_scaled)  
            elif method == 'gmm':  
                labels = self._cluster_gmm(X_scaled)  
            elif method == 'agglomerative':  
                labels = self._cluster_agglomerative_sampled(X_scaled.astype(np.float32))  
            elif method == 'dbscan':  
                labels = self._cluster_dbscan(X_scaled.astype(np.float32))  
            elif method == 'spectral':  
                labels = self._cluster_spectral(X_scaled.astype(np.float32))  
            else:  
                raise ValueError(f"未知算法: {method}")  
            
            # 按HU值排序标签  
            labels = self._reorder_labels_by_hu(labels, self.features[:, 0])  
            
            # 强制3类 + 最小簇约束  
            labels = self._ensure_exactly_n_clusters(labels, self.features)  
            labels = self._enforce_minimum_cluster_size(labels, self.features)  
            labels = self._ensure_exactly_n_clusters(labels, self.features)  
            
            # 最终排序  
            labels = self._reorder_labels_by_hu(labels, self.features[:, 0])  
            
            self.labels = labels.astype(np.int32)  
            self.algorithm = method  
            
            return self.labels + 1  # 1-indexed  
        
        except Exception as e:  
            print(f"      ✗ {method} 失败: {e}")  
            import traceback  
            traceback.print_exc()  
            return None  

    def _reorder_labels_by_hu(self, labels, hu_values):  
        """按HU值排序标签"""  
        valid_mask = labels >= 0  
        if np.sum(valid_mask) == 0:  
            return labels  
        
        valid_labels = labels[valid_mask]  
        valid_hu = hu_values[valid_mask]  
        
        unique_labels = np.unique(valid_labels)  
        label_means = {lb: np.mean(valid_hu[valid_labels == lb]) for lb in unique_labels}  
        
        sorted_old = sorted(label_means.keys(), key=lambda x: label_means[x])  
        mapping = {sorted_old[i]: i for i in range(len(sorted_old))}  
        
        new_labels = np.full_like(labels, -1)  
        for old_label, new_label in mapping.items():  
            new_labels[labels == old_label] = new_label  
        
        return new_labels  

    def _ensure_exactly_n_clusters(self, labels, features):  
        """强制n个类别"""  
        unique_labels = np.unique(labels[labels >= 0])  
        n_current = len(unique_labels)  
        
        if n_current == self.n_clusters:  
            return labels  
        
        if n_current < self.n_clusters:  
            return self._split_clusters(labels, features, self.n_clusters)  
        else:  
            return self._merge_clusters(labels, features, self.n_clusters)  

    def _split_clusters(self, labels, features, target_n):  
        """分裂簇"""  
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)  
        n_missing = target_n - len(unique_labels)  
        
        new_labels = labels.copy()  
        next_label = max(unique_labels) + 1 if len(unique_labels) > 0 else 0  
        
        for _ in range(n_missing):  
            unique_labels, counts = np.unique(new_labels[new_labels >= 0], return_counts=True)  
            if len(counts) == 0:  
                break  
            
            largest_label = unique_labels[np.argmax(counts)]  
            largest_mask = new_labels == largest_label  
            largest_indices = np.where(largest_mask)[0]  
            
            if len(largest_indices) < 2:  
                break  
            
            largest_features = features[largest_indices]  
            
            try:  
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)  
                sub_labels = kmeans.fit_predict(largest_features)  
                
                split_mask = sub_labels == 1  
                new_labels[largest_indices[split_mask]] = next_label  
                next_label += 1  
            except:  
                break  
        
        return new_labels  

    def _merge_clusters(self, labels, features, target_n):  
        """合并簇"""  
        unique_labels = np.unique(labels[labels >= 0])  
        n_excess = len(unique_labels) - target_n  
        
        new_labels = labels.copy()  
        
        for _ in range(n_excess):  
            unique_labels = np.unique(new_labels[new_labels >= 0])  
            if len(unique_labels) <= target_n:  
                break  
            
            # 找最小簇  
            counts = [np.sum(new_labels == lb) for lb in unique_labels]  
            smallest_label = unique_labels[np.argmin(counts)]  
            
            # 找最近的簇合并  
            centers = {  
                lb: np.mean(features[new_labels == lb], axis=0)  
                for lb in unique_labels  
            }  
            
            smallest_center = centers[smallest_label]  
            merge_target = None  
            min_dist = np.inf  
            
            for lb in unique_labels:  
                if lb != smallest_label:  
                    dist = np.linalg.norm(centers[lb] - smallest_center)  
                    if dist < min_dist:  
                        min_dist = dist  
                        merge_target = lb  
            
            if merge_target is not None:  
                new_labels[new_labels == smallest_label] = merge_target  
        
        return new_labels  

    def _enforce_minimum_cluster_size(self, labels, features, max_iter=5):  
        """最小簇约束后处理"""  
        n_samples = len(labels)  
        min_size = int(np.ceil(n_samples * self.min_cluster_fraction))  
        
        for _ in range(max_iter):  
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)  
            
            small_clusters = unique_labels[counts < min_size]  
            if len(small_clusters) == 0:  
                break  
            
            for small_label in small_clusters:  
                small_indices = np.where(labels == small_label)[0]  
                other_labels = unique_labels[unique_labels != small_label]  
                
                if len(other_labels) == 0:  
                    break  
                
                centers = {  
                    lb: np.mean(features[labels == lb], axis=0)  
                    for lb in unique_labels  
                }  
                
                small_center = centers[small_label]  
                distances = [np.linalg.norm(centers[lb] - small_center) for lb in other_labels]  
                target_label = other_labels[np.argmin(distances)]  
                
                labels[small_indices] = target_label  
                break  
        
        return labels  

    def compute_metrics(self, labels, features):  
        """计算聚类指标"""  
        valid_mask = labels >= 0  
        if np.sum(valid_mask) < 2 or len(np.unique(labels[valid_mask])) < 2:  
            return None  
        
        X_scaled = RobustScaler().fit_transform(features[valid_mask])  
        labels_valid = labels[valid_mask]  
        
        metrics = {}  
        
        try:  
            metrics['silhouette'] = float(silhouette_score(X_scaled, labels_valid))  
        except:  
            pass  
        
        try:  
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_scaled, labels_valid))  
        except:  
            pass  
        
        try:  
            metrics['davies_bouldin'] = float(davies_bouldin_score(X_scaled, labels_valid))  
        except:  
            pass  
        
        return metrics if metrics else None  

    def get_descriptors(self, voxel_vol=1.0):  
        """获取肿瘤描述符"""  
        if self.labels is None:  
            return None  
        
        descriptors = {  
            'algorithm': self.algorithm,  
            'tumor_volume_mm3': float(np.sum(self.mask) * voxel_vol),  
            'n_clusters': len(np.unique(self.labels[self.labels >= 0])),  
        }  
        
        for label in range(self.n_clusters):  
            mask = self.labels == label  
            if np.sum(mask) > 0:  
                descriptors[f'cluster_{label+1}_size'] = int(np.sum(mask))  
                descriptors[f'cluster_{label+1}_fraction'] = float(np.sum(mask) / len(self.labels))  
                descriptors[f'cluster_{label+1}_mean_a'] = float(np.mean(self.features[mask, 0]))  
                descriptors[f'cluster_{label+1}_mean_v'] = float(np.mean(self.features[mask, 4]))  
                descriptors[f'cluster_{label+1}_mean_enh'] = float(np.mean(self.features[mask, 1]))  
        
        return descriptors  


# ============================================================================  
#  工具函数  
# ============================================================================  

def load_nifti(a_path, v_path, roi_path):  
    """加载NIfTI文件"""  
    A_img = nib.load(str(a_path))  
    V_img = nib.load(str(v_path))  
    ROI_img = nib.load(str(roi_path))  
    
    A = np.clip((A_img.get_fdata(dtype=np.float32) + 160) / 400, 0, 1)  
    V = np.clip((V_img.get_fdata(dtype=np.float32) + 160) / 400, 0, 1)  
    ROI = ROI_img.get_fdata() == 2  
    
    return A_img, V_img, ROI_img, A, V, ROI  


def get_basename(filepath):  
    """提取文件名"""  
    name = Path(filepath).name  
    for ext in ['.nii.gz', '.nii']:  
        if name.endswith(ext):  
            return name[:-len(ext)]  
    return name  


def find_case_files(data_dir):  
    """查找匹配的文件组"""  
    data_path = Path(data_dir)  
    cases = []  
    
    a_folder, v_folder, roi_folder = (  
        data_path / "crop_A",  
        data_path / "crop_V",  
        data_path / "crop_ROI"  
    )  
    
    if not all([f.exists() for f in [a_folder, v_folder, roi_folder]]):  
        print("❌ 缺少必需的文件夹")  
        return []  
    
    for a_file in sorted(a_folder.glob("*.nii*")):  
        base = get_basename(a_file)  
        
        v_file = next((f for f in v_folder.glob("*.nii*") if get_basename(f) == base), None)  
        roi_file = next((f for f in roi_folder.glob("*.nii*") if get_basename(f) == base), None)  
        
        if v_file and roi_file:  
            cases.append({  
                'name': base,  
                'a_path': a_file,  
                'v_path': v_file,  
                'roi_path': roi_file  
            })  
    
    return cases  


def process_all_cases(data_dir, output_dir, methods=None):  
    """处理所有病例 - 多算法对比"""  
    
    if methods is None:  
        methods = ['constrained_kmeans', 'kmeans', 'gmm', 'agglomerative', 'dbscan', 'spectral']  
    
    os.makedirs(output_dir, exist_ok=True)  
    cases = find_case_files(data_dir)  
    
    if not cases:  
        print("❌ 未找到匹配的文件")  
        return  
    
    print(f"\n{'='*80}")  
    print(f"🔬 胰腺肿瘤异质性分析 - 聚类算法对比")  
    print(f"{'='*80}")  
    print(f"对比算法: {', '.join([m.upper() for m in methods])}")  
    print(f"病例数: {len(cases)}")  
    print(f"特性:")  
    print(f"  ✓ 无先验知识约束")  
    print(f"  ✓ 强制3个聚类")  
    print(f"  ✓ HU值梯度排序")  
    print(f"  ✓ 5%最小簇约束")  
    print(f"{'='*80}\n")  
    
    all_results = []  
    comparison_data = []  
    
    for case_idx, case_info in enumerate(cases, 1):  
        case_name = case_info['name']  
        
        print(f"\n[{case_idx}/{len(cases)}] 📋 {case_name}")  
        print(f"{'-'*80}")  
        
        try:  
            A_img, V_img, ROI_img, A, V, ROI = load_nifti(  
                case_info['a_path'],  
                case_info['v_path'],  
                case_info['roi_path']  
            )  
            
            if not np.any(ROI):  
                print(f"  ⚠️  ROI为空，跳过\n")  
                continue  
            
            voxel_vol = float(np.prod(A_img.header.get_zooms()[:3]))  
            case_result = {'case': case_name}  
            
            for method_idx, method in enumerate(methods, 1):  
                print(f"  [{method_idx}/{len(methods)}] {method.upper()}")  
                
                try:  
                    clusterer = FastTumorClusterer(A, V, ROI, n_clusters=3)  
                    clusterer.extract_features()  
                    
                    labels_1d = clusterer.cluster(method)  
                    
                    if labels_1d is None:  
                        print(f"      ✗ 聚类失败")  
                        case_result[f'{method}_silhouette'] = None  
                        case_result[f'{method}_calinski'] = None  
                        case_result[f'{method}_davies_bouldin'] = None  
                        continue  
                    
                    # 保存聚类结果  
                    labels_3d = np.zeros(ROI.shape, dtype=np.uint16)  
                    labels_3d[ROI] = labels_1d  
                    
                    output_file = Path(output_dir) / f"{case_name}_{method}.nii.gz"  
                    nib.save(nib.Nifti1Image(labels_3d, A_img.affine), output_file)  
                    
                    # 计算指标  
                    metrics = clusterer.compute_metrics(clusterer.labels, clusterer.features)  
                    
                    # 获取描述符  
                    desc = clusterer.get_descriptors(voxel_vol)  
                    desc['case'] = case_name  
                    desc['method'] = method  
                    
                    if metrics:  
                        desc.update({f'metric_{k}': v for k, v in metrics.items()})  
                        case_result[f'{method}_silhouette'] = metrics.get('silhouette')  
                        case_result[f'{method}_calinski'] = metrics.get('calinski_harabasz')  
                        case_result[f'{method}_davies_bouldin'] = metrics.get('davies_bouldin')  
                    
                    all_results.append(desc)  
                    
                    # 打印指标  
                    if metrics:  
                        print(f"      ✓ Silhouette: {metrics.get('silhouette', 'N/A'):.4f}")  
                        print(f"        Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A'):.2f}")  
                        print(f"        Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.4f}")  
                        print(f"        Cluster sizes: {desc.get('cluster_1_size', 0)}/{desc.get('cluster_2_size', 0)}/{desc.get('cluster_3_size', 0)}")  
                
                except Exception as e:  
                    print(f"      ✗ {method} 失败: {e}")  
                    case_result[f'{method}_silhouette'] = None  
                    case_result[f'{method}_calinski'] = None  
                    case_result[f'{method}_davies_bouldin'] = None  
                    continue  
            
            comparison_data.append(case_result)  
        
        except Exception as e:  
            print(f"  ✗ 加载失败: {e}")  
            continue  
    
    # 保存结果  
    if all_results:  
        df_detailed = pd.DataFrame(all_results)  
        detailed_path = Path(output_dir) / "results_detailed.csv"  
        df_detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')  
        print(f"\n✓ 详细结果: {detailed_path}")  
                             
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        comparison_path = Path(output_dir) / "results_comparison.csv"
        df_comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"✓ 对比结果: {comparison_path}")
        
        print(f"\n{'='*80}")
        print("📊 算法对比汇总:")
        print(f"{'='*80}")
        
        # 统计每个算法的表现
        algorithm_stats = {}
        for method in methods:
            silhouette_col = f'{method}_silhouette'
            calinski_col = f'{method}_calinski'
            davies_col = f'{method}_davies_bouldin'
            
            valid_silhouette = [v for v in df_comparison[silhouette_col] if pd.notna(v)]
            valid_calinski = [v for v in df_comparison[calinski_col] if pd.notna(v)]
            valid_davies = [v for v in df_comparison[davies_col] if pd.notna(v)]
            
            algorithm_stats[method] = {
                'success_rate': len(valid_silhouette) / len(df_comparison) * 100 if len(df_comparison) > 0 else 0,
                'mean_silhouette': np.mean(valid_silhouette) if valid_silhouette else None,
                'std_silhouette': np.std(valid_silhouette) if valid_silhouette else None,
                'mean_calinski': np.mean(valid_calinski) if valid_calinski else None,
                'std_calinski': np.std(valid_calinski) if valid_calinski else None,
                'mean_davies': np.mean(valid_davies) if valid_davies else None,
                'std_davies': np.std(valid_davies) if valid_davies else None,
            }
        
        # 打印统计表
        print(f"\n{'算法':<20} {'成功率':<12} {'Silhouette':<20} {'Calinski-H':<20}")
        print(f"{'-'*72}")
        
        for method in methods:
            stats = algorithm_stats[method]
            silhouette_str = f"{stats['mean_silhouette']:.4f}±{stats['std_silhouette']:.4f}" if stats['mean_silhouette'] is not None else "N/A"
            calinski_str = f"{stats['mean_calinski']:.2f}±{stats['std_calinski']:.2f}" if stats['mean_calinski'] is not None else "N/A"
            
            print(f"{method:<20} {stats['success_rate']:>10.1f}% {silhouette_str:<20} {calinski_str:<20}")
        
        # 排序算法
        print(f"\n{'='*80}")
        print("🏆 算法排名 (按Silhouette评分):")
        print(f"{'='*80}")
        
        ranked = sorted(
            [(m, algorithm_stats[m]['mean_silhouette']) for m in methods if algorithm_stats[m]['mean_silhouette'] is not None],
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (method, score) in enumerate(ranked, 1):
            medal = ['🥇', '🥈', '🥉'][min(rank-1, 2)]
            print(f"{medal} {rank}. {method:<20} Silhouette: {score:.4f}")
        
        print(f"\n{'='*80}")
        print(f"✨ 完成! 已处理 {len(cases)} 个病例")
        print(f"{'='*80}\n")
        print(f"📁 输出文件:")
        print(f"  - 详细结果: {detailed_path}")
        print(f"  - 对比结果: {comparison_path}")
        print(f"  - 聚类文件: {output_dir}/")


# ============================================================================
#  主函数
# ============================================================================

if __name__ == "__main__":
    DATA_DIR = "./data"
    OUTPUT_DIR = "./cluster_results"
    
    # 可选的算法列表
    METHODS = [
        'constrained_kmeans',  # 带5%约束的KMeans
        'kmeans',              # 标准KMeans
        'gmm',                 # 高斯混合模型
        'agglomerative',       # 凝聚聚类
        'dbscan',              # 密度聚类
        'spectral',            # 谱聚类
    ]
    
    print("\n" + "="*80)
    print("🔬 胰腺肿瘤异质性分析系统 - 无先验约束版本")
    print("="*80)
    print("✨ 核心特性:")
    print("  1. ✓ 去掉点标注先验知识约束")
    print("  2. ✓ 6种聚类算法自动对比")
    print("  3. ✓ 强制3个聚类类别（G1/G2/G3）")
    print("  4. ✓ HU值梯度自动排序（Label1<Label2<Label3）")
    print("  5. ✓ 5%最小簇大小自动约束")
    print("  6. ✓ 自动回退和错误处理")
    print("  7. ✓ 多项聚类评估指标")
    print("\n🎯 聚类算法:")
    for i, method in enumerate(METHODS, 1):
        print(f"  {i}. {method.upper():<25} - ", end="")
        if method == 'constrained_kmeans':
            print("带最小簇约束的KMeans")
        elif method == 'kmeans':
            print("标准K均值聚类")
        elif method == 'gmm':
            print("高斯混合模型（多配置尝试）")
        elif method == 'agglomerative':
            print("层次凝聚聚类（采样优化）")
        elif method == 'dbscan':
            print("密度聚类（自适应eps）")
        elif method == 'spectral':
            print("谱聚类（图论方法）")
    
    print(f"\n📊 评估指标:")
    print(f"  - Silhouette Score (轮廓系数)")
    print(f"  - Calinski-Harabasz Index (CH指数)")
    print(f"  - Davies-Bouldin Index (DB指数)")
    print(f"  - 聚类大小分布")
    print(f"  - HU值梯度一致性")
    
    print(f"{'='*80}\n")
    
    # 运行主流程
    process_all_cases(DATA_DIR, OUTPUT_DIR, methods=METHODS)
    
    print("\n" + "="*80)
    print("✅ 所有处理完成!")
    print("="*80)
    print(f"\n📁 输出目录: {OUTPUT_DIR}/")
    print(f"\n📋 输出文件说明:")
    print(f"  • *_constrained_kmeans.nii.gz   - ConstrainedKMeans聚类结果")
    print(f"  • *_kmeans.nii.gz               - 标准KMeans聚类结果")
    print(f"  • *_gmm.nii.gz                  - GMM聚类结果")
    print(f"  • *_agglomerative.nii.gz        - 凝聚聚类结果")
    print(f"  • *_dbscan.nii.gz               - DBSCAN聚类结果")
    print(f"  • *_spectral.nii.gz             - 谱聚类结果")
    print(f"  • results_detailed.csv          - 详细聚类指标（所有算法）")
    print(f"  • results_comparison.csv        - 算法对比汇总")
    print("\n💡 建议:")
    print(f"  1. 查看 results_comparison.csv 了解各算法在全数据集上的表现")
    print(f"  2. Silhouette > 0.5 表示聚类效果良好")
    print(f"  3. Davies-Bouldin 越小越好（< 1.5优秀）")
    print(f"  4. 选择成功率高、评分稳定的算法用于生产")