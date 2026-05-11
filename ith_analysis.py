import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import ConvergenceWarning
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")

class TumorHeterogeneityAnalyzer:
    """
    胰腺肿瘤异质性分析
    聚类：1=坏死(低活跃), 2=间质(中等), 3=富血供(高活跃)
    """

    def __init__(self, arterial_ct, venous_ct, tumor_mask):
        self.arterial = arterial_ct
        self.venous = venous_ct
        self.mask = tumor_mask.astype(bool)
        self.labels = None

    def extract_features(self):
        """提取双期特征: [A, V, V-A]"""
        a = self.arterial[self.mask > 0]
        v = self.venous[self.mask > 0]
        return np.column_stack([a, v, v - a]).astype(np.float64)

    def cluster(self, features, method='gmm'):
        """三类聚类算法"""
        if method == 'gmm':
            labels = self._gmm_clustering(features)
        else:
            labels = self._kmeans_clustering(features)
        return (labels.astype(int) + 1)

    def _gmm_clustering(self, features, q_low=30, q_high=70, q_enh=70, min_frac=0.1):
        """稳健GMM聚类"""
        N = len(features)
        if N < 30:
            return self._kmeans_clustering(features)

        mean_hu = (features[:, 0] + features[:, 1]) / 2
        enh = features[:, 2]
        X = np.column_stack([mean_hu, enh])

        low_q = np.percentile(mean_hu, q_low)
        high_q = np.percentile(mean_hu, q_high)
        enh_q = np.percentile(enh, q_enh)

        low_mask = mean_hu <= low_q
        high_mask = (mean_hu >= high_q) | (enh >= enh_q)
        mid_mask = ~(low_mask | high_mask)

        init_means = np.vstack([
            X[low_mask].mean(axis=0) if low_mask.any() else X.mean(axis=0),
            X[mid_mask].mean(axis=0) if mid_mask.any() else X.mean(axis=0),
            X[high_mask].mean(axis=0) if high_mask.any() else X.mean(axis=0),
        ])

        scaler = RobustScaler(quantile_range=(5, 95))
        X_scaled = scaler.fit_transform(X)
        init_means_scaled = scaler.transform(init_means)

        for cov in ['diag', 'full']:
            for reg in [1e-4, 5e-4, 1e-3]:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        gmm = GaussianMixture(
                            n_components=3, covariance_type=cov, reg_covar=reg,
                            means_init=init_means_scaled, n_init=1, max_iter=400, random_state=42
                        )
                        labels = gmm.fit_predict(X_scaled)

                    unique, counts = np.unique(labels, return_counts=True)
                    if len(unique) == 3 and (counts.min() / N >= min_frac):
                        return self._reorder_labels(labels, features)
                except:
                    continue

        return self._kmeans_clustering(features)

    def _kmeans_clustering(self, features):
        """种子初始化KMeans"""
        mean_hu = (features[:, 0] + features[:, 1]) / 2
        seeds_idx = [np.argmin(np.abs(mean_hu - np.percentile(mean_hu, p))) for p in [10, 50, 90]]
        km = KMeans(n_clusters=3, init=features[seeds_idx], n_init=1, random_state=42)
        labels = km.fit_predict(features)
        return self._reorder_labels(labels, features)

    @staticmethod
    def _reorder_labels(labels, features):
        """按mean_hu排序标签"""
        mean_hu = (features[:, 0] + features[:, 1]) / 2
        means = [mean_hu[labels == lb].mean() for lb in np.unique(labels)]
        mapping = {old: new for new, old in enumerate(np.argsort(means))}
        return np.vectorize(mapping.get)(labels).astype(int)

    def apply_necrosis_prior(self, labels_1d, interior_q=0.4, min_cc_vox=200, 
                            keep_top_k=1, kernel_size=3):
        """空间先验：坏死优先内部成团"""
        labels0 = labels_1d - 1
        labels3d = np.full(self.mask.shape, -1, dtype=np.int16)
        labels3d[self.mask] = labels0

        dist = ndi.distance_transform_edt(self.mask)
        d_thr = np.quantile(dist[self.mask], interior_q)

        k = max(3, int(kernel_size))
        kernel = np.ones((k, k, k))
        kernel[k // 2, k // 2, k // 2] = 0

        nec = (labels3d == 0)

        boundary_nec = nec & (dist < d_thr)
        if np.any(boundary_nec):
            n1 = ndi.convolve((labels3d == 1).astype(int), kernel, mode='constant')
            n2 = ndi.convolve((labels3d == 2).astype(int), kernel, mode='constant')
            labels3d[boundary_nec] = np.where(n1[boundary_nec] >= n2[boundary_nec], 1, 2)

        nec = (labels3d == 0)
        if np.any(nec):
            struct = ndi.generate_binary_structure(3, 1)
            cc, num = ndi.label(nec, structure=struct)
            sizes = np.bincount(cc.ravel()) if num > 0 else np.array([0])
            sizes[0] = 0

            keep = np.zeros_like(sizes, dtype=bool)
            if keep_top_k and keep_top_k > 0:
                keep[np.argsort(sizes)[-keep_top_k:]] = True
            keep[sizes >= max(1, int(min_cc_vox))] = True

            nec_clean = nec & keep[cc]
            nec_clean = ndi.binary_closing(nec_clean, structure=struct, iterations=1)

            drop = nec & (~nec_clean)
            if np.any(drop):
                n1 = ndi.convolve((labels3d == 1).astype(int), kernel, mode='constant')
                n2 = ndi.convolve((labels3d == 2).astype(int), kernel, mode='constant')
                labels3d[drop] = np.where(n1[drop] >= n2[drop], 1, 2)

        return (labels3d[self.mask] + 1).astype(np.uint16)

    def compute_descriptors(self, a_img=None):
        """计算肿瘤异质性特征"""
        if self.labels is None:
            raise ValueError("需先运行聚类!")

        try:
            dz, dy, dx = a_img.header.get_zooms()[:3]
            voxel_vol = dz * dy * dx
        except:
            voxel_vol = 1.0

        features = self.extract_features()
        mean_hu = (features[:, 0] + features[:, 1]) / 2
        enh = features[:, 2]
        dist = ndi.distance_transform_edt(self.mask)

        n_voxels = self.mask.sum()
        n1 = np.sum(self.labels == 1)
        n2 = np.sum(self.labels == 2)
        n3 = np.sum(self.labels == 3)
        total = len(self.labels)
        
        desc = {
            'tumor_volume_mm3': float(n_voxels * voxel_vol),
            'class1_necrosis_voxels': int(n1),
            'class1_necrosis_fraction': float(n1 / total),
            'class2_stroma_voxels': int(n2),
            'class2_stroma_fraction': float(n2 / total),
            'class3_rich_voxels': int(n3),
            'class3_rich_fraction': float(n3 / total),
            'necrosis_centrality': float(np.mean(dist[self.mask][self.labels == 1]) / 
                                        (np.max(dist) + 1e-6)) if n1 > 0 else 0.0,
            'cluster_diversity': float(-np.sum((np.bincount(self.labels) / total) * 
                                              np.log2(np.bincount(self.labels) / total + 1e-10))),
            'enhancement_heterogeneity': float(np.std(enh) / (np.mean(np.abs(enh)) + 1e-6)),
        }
        return desc


def batch_process_clustering(input_dir="./data", output_dir="./cluster/origin", hu_range=(-260, 340)):
    """
    批量处理肿瘤聚类
    
    输出结构:
        cluster/origin/
        ├─ clustering_table.csv              (汇总表格)
        ├─ summary_visualization.png         (汇总图表)
        ├─ clustering_labels/
        │  ├─ patient_001_clustering.nii.gz
        │  ├─ patient_002_clustering.nii.gz
        │  └─ ...
        └─ patient_analysis/
           ├─ patient_001_analysis.png
           ├─ patient_002_analysis.png
           └─ ...
    """
    # 创建子文件夹
    clustering_dir = os.path.join(output_dir, "clustering_labels")
    analysis_dir = os.path.join(output_dir, "patient_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    a_dir = f"{input_dir}/crop_A"
    v_dir = f"{input_dir}/crop_V"
    m_dir = f"{input_dir}/crop_ROI"
    
    for d in [a_dir, v_dir, m_dir]:
        if not os.path.exists(d):
            print(f"❌ 目录不存在: {d}")
            return
    
    files = sorted([f for f in os.listdir(a_dir) if f.endswith('.nii.gz')])
    
    if not files:
        print(f"❌ 未找到 .nii.gz 文件")
        return
    
    print(f"\n{'='*80}")
    print(f"肿瘤异质性聚类处理")
    print(f"{'='*80}\n")
    
    results_list = []
    successful = 0
    
    for idx, file in enumerate(files, 1):
        try:
            print(f"[{idx:3d}/{len(files)}] {file:<50}", end=" ", flush=True)
            
            A = nib.load(f"{a_dir}/{file}").get_fdata(dtype=np.float32)
            V = nib.load(f"{v_dir}/{file}").get_fdata(dtype=np.float32)
            M = (nib.load(f"{m_dir}/{file}").get_fdata() == 2)
            
            A = np.clip(A, hu_range[0], hu_range[1])
            V = np.clip(V, hu_range[0], hu_range[1])
            
            analyzer = TumorHeterogeneityAnalyzer(A, V, M)
            features = analyzer.extract_features()
            labels = analyzer.cluster(features, method='gmm')
            analyzer.labels = labels
            
            labels_refined = analyzer.apply_necrosis_prior(
                labels, interior_q=0.4, min_cc_vox=200, keep_top_k=1, kernel_size=3
            )
            analyzer.labels = labels_refined
            
            # 保存3D标签到子文件夹
            labels_3d = np.zeros(M.shape, dtype=np.uint16)
            labels_3d[M] = labels_refined
            
            A_img = nib.load(f"{a_dir}/{file}")
            hdr = nib.Nifti1Header()
            hdr.set_data_dtype(np.uint16)
            hdr.set_qform(A_img.affine, code=1)
            labels_img = nib.Nifti1Image(labels_3d, affine=A_img.affine, header=hdr)
            
            patient_id = file.replace('.nii.gz', '')
            nib.save(labels_img, os.path.join(clustering_dir, f"{patient_id}_clustering.nii.gz"))
            
            # 计算特征
            desc = analyzer.compute_descriptors(a_img=A_img)
            results_list.append({'patient_id': patient_id, **desc})
            
            # 绘制单个患者的分析图到子文件夹
            plot_single_patient(analyzer, patient_id, analysis_dir)
            
            print(f"✓")
            successful += 1
            
        except Exception as e:
            print(f"✗ {str(e)[:40]}")
            continue
    
    # 生成汇总表格
    if results_list:
        df_all = pd.DataFrame(results_list)
        
        # 保存汇总表格到主文件夹
        table_path = os.path.join(output_dir, "clustering_table.csv")
        df_all.to_csv(table_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"✓ 处理完成: {successful}/{len(files)} 个样本")
        print(f"{'='*80}\n")
        
        # 显示表格
        print("聚类结果汇总表格:")
        print("-" * 130)
        print(df_all.to_string(index=False))
        print("-" * 130 + "\n")
        
        # 生成汇总可视化图表到主文件夹
        plot_summary_visualization(df_all, output_dir)
        
        print(f"\n{'='*80}")
        print(f"输出目录: {output_dir}")
        print(f"├─ clustering_table.csv")
        print(f"├─ summary_visualization.png")
        print(f"├─ clustering_labels/")
        print(f"│  ├─ patient_001_clustering.nii.gz")
        print(f"│  ├─ patient_002_clustering.nii.gz")
        print(f"│  └─ ...")
        print(f"└─ patient_analysis/")
        print(f"   ├─ patient_001_analysis.png")
        print(f"   ├─ patient_002_analysis.png")
        print(f"   └─ ...")
        print(f"{'='*80}\n")
        
    else:
        print(f"\n❌ 没有样本处理成功")


def plot_single_patient(analyzer, patient_id, output_dir):
    """绘制单个患者的聚类分析图"""
    # 禁用中文警告
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Patient {patient_id} - Clustering Analysis', 
                 fontsize=14, fontweight='bold')
    
    features = analyzer.extract_features()
    labels = analyzer.labels
    
    # 1. 双期CT的散点图
    ax1 = axes[0, 0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    class_names = ['Necrosis', 'Stroma', 'Rich']
    
    for k in [1, 2, 3]:
        mask = labels == k
        ax1.scatter(features[mask, 0], features[mask, 1], 
                   c=colors[k-1], label=f'Class {k} ({class_names[k-1]})', 
                   alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Arterial Phase (HU)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Venous Phase (HU)', fontsize=11, fontweight='bold')
    ax1.set_title('Dual-phase CT Features', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. 增强差的分布
    ax2 = axes[0, 1]
    enh = features[:, 2]
    
    for k in [1, 2, 3]:
        mask = labels == k
        ax2.hist(enh[mask], bins=20, alpha=0.6, label=f'Class {k}', 
                color=colors[k-1], edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Enhancement (V-A, HU)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Voxel Count', fontsize=11, fontweight='bold')
    ax2.set_title('Enhancement Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. 类别比例
    ax3 = axes[1, 0]
    class_counts = [np.sum(labels == k) for k in [1, 2, 3]]
    class_fracs = [c / len(labels) * 100 for c in class_counts]
    
    bars = ax3.bar(class_names, class_fracs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, frac, count in zip(bars, class_fracs, class_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{frac:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Fraction (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Cluster Composition', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, max(class_fracs) * 1.2])
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. 特征统计
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计量
    mean_hu_all = (features[:, 0] + features[:, 1]) / 2
    mean_hu_by_class = [mean_hu_all[labels == k].mean() for k in [1, 2, 3]]
    std_hu_by_class = [mean_hu_all[labels == k].std() for k in [1, 2, 3]]
    enh_by_class = [enh[labels == k].mean() for k in [1, 2, 3]]
    
    stats_text = "Feature Statistics\n" + "-"*35 + "\n"
    for k in [1, 2, 3]:
        stats_text += f"Class {k} ({class_names[k-1]}):\n"
        stats_text += f"  Mean HU: {mean_hu_by_class[k-1]:.1f}+/-{std_hu_by_class[k-1]:.1f}\n"
        stats_text += f"  Mean Enh: {enh_by_class[k-1]:.1f}\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    # 不捕获警告，而是使用英文绘图
    plt.savefig(os.path.join(output_dir, f"{patient_id}_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_visualization(df, output_dir):
    """绘制汇总的可视化图表"""
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tumor Heterogeneity Clustering Analysis - Summary', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    n_patients = len(df)
    x = np.arange(n_patients)
    width = 0.6
    
    # 1. 三类体素数量堆叠柱状图
    ax1 = axes[0, 0]
    class_counts = df[['class1_necrosis_voxels', 'class2_stroma_voxels', 'class3_rich_voxels']].values
    
    ax1.bar(x, class_counts[:, 0], width, label='Class 1 (Necrosis)', color='#FF6B6B', alpha=0.8)
    ax1.bar(x, class_counts[:, 1], width, bottom=class_counts[:, 0], 
            label='Class 2 (Stroma)', color='#4ECDC4', alpha=0.8)
    ax1.bar(x, class_counts[:, 2], width, bottom=class_counts[:, 0] + class_counts[:, 1],
            label='Class 3 (Rich)', color='#45B7D1', alpha=0.8)
    
    ax1.set_xlabel('Patient', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Voxel Count', fontsize=11, fontweight='bold')
    ax1.set_title('Voxel Distribution (Stacked)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    if n_patients <= 20:
        ax1.set_xticks(x)
        ax1.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=9)
    
    # 2. 三类比例堆叠柱状图
    ax2 = axes[0, 1]
    fractions = df[['class1_necrosis_fraction', 'class2_stroma_fraction', 'class3_rich_fraction']].values
    
    ax2.bar(x, fractions[:, 0], width, label='Class 1 (Necrosis)', color='#FF6B6B', alpha=0.8)
    ax2.bar(x, fractions[:, 1], width, bottom=fractions[:, 0],
            label='Class 2 (Stroma)', color='#4ECDC4', alpha=0.8)
    ax2.bar(x, fractions[:, 2], width, bottom=fractions[:, 0] + fractions[:, 1],
            label='Class 3 (Rich)', color='#45B7D1', alpha=0.8)
    
    ax2.set_xlabel('Patient', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Fraction', fontsize=11, fontweight='bold')
    ax2.set_title('Cluster Proportion Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    if n_patients <= 20:
        ax2.set_xticks(x)
        ax2.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=9)
    
    # 3. 肿瘤体积分布
    ax3 = axes[1, 0]
    volumes = df['tumor_volume_mm3'].values
    ax3.bar(x, volumes, width=0.6, color='#98D8C8', alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_xlabel('Patient', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Volume (mm^3)', fontsize=11, fontweight='bold')
    ax3.set_title('Tumor Volume Distribution', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    if n_patients <= 20:
        ax3.set_xticks(x)
        ax3.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=9)
    
    # 4. 异质性指标
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(x, df['cluster_diversity'].values, 'o-', linewidth=2.5, 
                     markersize=8, label='Cluster Diversity', color='#FF6B6B')
    line2 = ax4_twin.plot(x, df['enhancement_heterogeneity'].values, 's-', linewidth=2.5,
                          markersize=8, label='Enhancement Heterogeneity', color='#45B7D1')
    
    ax4.set_xlabel('Patient', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cluster Diversity', fontsize=11, fontweight='bold', color='#FF6B6B')
    ax4_twin.set_ylabel('Enhancement Heterogeneity', fontsize=11, fontweight='bold', color='#45B7D1')
    ax4.set_title('Heterogeneity Metrics', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#FF6B6B')
    ax4_twin.tick_params(axis='y', labelcolor='#45B7D1')
    ax4.grid(alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=10, loc='upper left')
    
    if n_patients <= 20:
        ax4.set_xticks(x)
        ax4.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary visualization saved: {os.path.join(output_dir, 'summary_visualization.png')}")


# ============ 主流程 ============
if __name__ == "__main__":
    
    batch_process_clustering(
        input_dir="./data",
        output_dir="./cluster/origin",
        hu_range=(-260, 340)
    )