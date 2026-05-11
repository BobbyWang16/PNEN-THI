import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

def calculate_tumor_heterogeneity_score(image_path, cluster_path, bins=200, hu_range=(-160, 240)):
    """
    计算肿瘤异质性综合评分 (DDI - Dose Distribution Heterogeneity Index)
    
    特征说明：
    ├─ 1. 组间-组内方差比 (Imbalance Index)
    │   └─ 衡量三个聚类的HU值分布离散度：高值→聚类差异大→异质性强
    │
    ├─ 2. 分布中心分离度 (Separation Index)
    │   └─ 聚类中心在HU值空间的距离：高值→聚类中心相隔远→异质性强
    │
    ├─ 3. 分布重叠度 (Overlap Separation)
    │   └─ 直方图重叠程度：高值→重叠少→聚类独立性强→异质性强
    │
    └─ 4. 分布形状异质性 (Shape Diversity)
        └─ 结合偏度、峰度、熵、IQR变异系数：衡量聚类分布形状差异
    """
    
    # 读取并预处理数据
    image_data = np.squeeze(nib.load(image_path).get_fdata())
    cluster_data = np.squeeze(nib.load(cluster_path).get_fdata())
    
    if len(image_data.shape) == 4:
        image_data = image_data[:, :, :, 0]
    if len(cluster_data.shape) == 4:
        cluster_data = cluster_data[:, :, :, 0]
    
    # 提取三个聚类的统计特征
    cluster_hu_values = {}
    cluster_stats = {}
    histograms = {}
    
    for cluster_id in [1, 2, 3]:
        hu_values = image_data[cluster_data == cluster_id]
        hu_values = hu_values[np.isfinite(hu_values)]
        hu_values = hu_values[(hu_values >= hu_range[0]) & (hu_values <= hu_range[1])]
        
        if len(hu_values) == 0:
            raise ValueError(f"Cluster {cluster_id} 为空")
        
        cluster_hu_values[cluster_id] = hu_values
        
        # 计算单聚类统计量
        cluster_stats[cluster_id] = {
            'count': len(hu_values),
            'mean': np.mean(hu_values),
            'std': np.std(hu_values),
            'median': np.median(hu_values),
            'min': np.min(hu_values),
            'max': np.max(hu_values),
            'range': np.max(hu_values) - np.min(hu_values),
            'iqr': np.percentile(hu_values, 75) - np.percentile(hu_values, 25),
            'cv': np.std(hu_values) / (abs(np.mean(hu_values)) + 1e-8),
            'skewness': skew(hu_values),
            'kurtosis': kurtosis(hu_values),
            'entropy': entropy(np.histogram(hu_values, bins=bins, range=hu_range, density=True)[0] + 1e-8)
        }
        
        # 计算直方图
        hist, bin_edges = np.histogram(hu_values, bins=bins, range=hu_range, density=True)
        histograms[cluster_id] = {'hist': hist, 'bins': bin_edges}
    
    # 提取基础数据
    n1, n2, n3 = [cluster_stats[i]['count'] for i in [1, 2, 3]]
    mean1, mean2, mean3 = [cluster_stats[i]['mean'] for i in [1, 2, 3]]
    std1, std2, std3 = [cluster_stats[i]['std'] for i in [1, 2, 3]]
    proportions = np.array([n1, n2, n3]) / (n1 + n2 + n3)
    
    print(f"  聚类分布: C1={n1}({proportions[0]*100:.1f}%), C2={n2}({proportions[1]*100:.1f}%), C3={n3}({proportions[2]*100:.1f}%)")
    
    # ========== 特征计算 ==========
    
    # 特征1: 组间-组内方差比（衡量聚类紧密性）
    all_hu = np.concatenate([cluster_hu_values[i] for i in [1, 2, 3]])
    global_mean = np.mean(all_hu)
    between_ss = sum([cluster_stats[i]['count'] * (cluster_stats[i]['mean'] - global_mean) ** 2 for i in [1, 2, 3]])
    within_ss = sum([cluster_stats[i]['count'] * (cluster_stats[i]['std'] ** 2) for i in [1, 2, 3]])
    imbalance_index = np.tanh(between_ss / (within_ss + 1e-8) / 10)
    
    # 特征2: 聚类中心分离度（衡量聚类空间分布）
    centers = np.array([mean1, mean2, mean3])
    center_dists = [abs(centers[i] - centers[j]) * proportions[i] * proportions[j] 
                    for i in range(3) for j in range(i+1, 3)]
    normalized_separation = np.mean(center_dists) / (hu_range[1] - hu_range[0])
    
    # 特征3: 直方图重叠度（衡量聚类独立性）
    hist1, hist2, hist3 = [histograms[i]['hist'] for i in [1, 2, 3]]
    overlap = (np.sum(np.minimum(hist1, hist2)) * proportions[0] * proportions[1] +
               np.sum(np.minimum(hist1, hist3)) * proportions[0] * proportions[2] +
               np.sum(np.minimum(hist2, hist3)) * proportions[1] * proportions[2])
    separation_score = 1 - overlap / 3
    
    # 特征4: 分布形状异质性（衡量聚类形状差异）
    skew_vals = np.array([cluster_stats[i]['skewness'] for i in [1, 2, 3]])
    kurt_vals = np.array([cluster_stats[i]['kurtosis'] for i in [1, 2, 3]])
    ent_vals = np.array([cluster_stats[i]['entropy'] for i in [1, 2, 3]])
    iqr_vals = np.array([cluster_stats[i]['iqr'] for i in [1, 2, 3]])
    
    shape_diversity = (
        0.25 * np.tanh(np.std(skew_vals)) +
        0.25 * np.tanh(np.std(kurt_vals) / 2) +
        0.30 * np.std(ent_vals) +
        0.20 * np.tanh(np.std(iqr_vals) / (np.mean(iqr_vals) + 1e-8) * 2)
    )
    
    # ========== DDI综合评分 ==========
    heterogeneity_score = (
        0.30 * imbalance_index +
        0.25 * normalized_separation +
        0.25 * separation_score +
        0.20 * shape_diversity
    )
    
    return {
        'DDI_score': heterogeneity_score,
        'components': {
            'imbalance_index': imbalance_index,
            'separation_index': normalized_separation,
            'overlap_separation': separation_score,
            'shape_diversity': shape_diversity
        },
        'cluster_stats': cluster_stats,
        'cluster_proportions': proportions.tolist(),
        'histograms': histograms,
        'detailed_metrics': {
            'bw_variance_ratio': between_ss / (within_ss + 1e-8),
            'between_ss': between_ss,
            'within_ss': within_ss,
            'skewness_values': skew_vals.tolist(),
            'kurtosis_values': kurt_vals.tolist(),
            'entropy_values': ent_vals.tolist(),
            'centers': centers.tolist(),
            'global_mean': float(global_mean)
        }
    }

def plot_cluster_distributions(results, save_path, phase_label=''):
    """绘制聚类HU值分布"""
    
    plt.ioff()
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, cid in enumerate([1, 2, 3]):
        hist = results['histograms'][cid]['hist']
        bins = results['histograms'][cid]['bins']
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        stats = results['cluster_stats'][cid]
        prop = results['cluster_proportions'][i]
        
        ax.fill_between(bin_centers, hist, alpha=0.6, color=colors[i],
                       label=f'C{i+1} (n={stats["count"]:,}, {prop*100:.1f}%)',
                       edgecolor=colors[i], linewidth=2.5)
        ax.plot(bin_centers, hist, color=colors[i], linewidth=2, alpha=0.9)
        ax.axvline(stats['mean'], color=colors[i], linestyle='--', linewidth=2.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.set_xlabel('HU Value', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title(f'HU Distribution {phase_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    
    ax.text(0.98, 0.02, f'DDI: {results["DDI_score"]:.4f}',
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#E8F8F5', edgecolor='#333', linewidth=1.5),
           ha='right', va='bottom')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def extract_ddi_features_to_dataframe(results, patient_id, phase_label):
    """提取特征到DataFrame"""
    row_data = {'patient_id': patient_id, 'phase': phase_label, 'DDI': results['DDI_score']}
    
    # 主要组件
    row_data.update(results['components'])
    
    # 聚类统计
    for cid in [1, 2, 3]:
        stats = results['cluster_stats'][cid]
        for key in ['count', 'mean', 'std', 'median', 'min', 'max', 'iqr', 'skewness', 'kurtosis', 'entropy']:
            row_data[f'c{cid}_{key}'] = stats[key]
    
    # 详细指标
    row_data.update(results['detailed_metrics'])
    
    return pd.DataFrame([row_data])

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DDI (肿瘤异质性指数) 分析")
    print("="*60 + "\n")
    
    for dataset in ["A", "V"]:
        phase = "Arterial" if dataset == "A" else "Venous"
        print(f"\n处理{phase}期...\n")
        
        img_dir = f"./data/crop_{dataset}"
        cluster_dir = f"./cluster/with_point"
        save_dir = f"./score/DDI/{dataset}_figures"
        csv_path = f"./score/DDI/{dataset}_features.csv"
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        result_df = pd.DataFrame()
        namelist = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
        
        for idx, name in enumerate(namelist, 1):
            pid = name.split(".nii.gz")[0]
            print(f"[{idx:3d}/{len(namelist)}] {pid}...", end=' ')
            
            try:
                cf = f"{cluster_dir}/{pid}_clusters.nii.gz"
                if not os.path.exists(cf):
                    print("❌ 无聚类文件")
                    continue
                
                results = calculate_tumor_heterogeneity_score(f"{img_dir}/{name}", cf)
                print(f"✓ DDI={results['DDI_score']:.4f}")
                
                result_df = pd.concat([result_df, extract_ddi_features_to_dataframe(results, pid, dataset)], ignore_index=True)
                plot_cluster_distributions(results, f"{save_dir}/{pid}_DDI.png", phase_label=phase)
                
            except Exception as e:
                print(f"❌ {str(e)[:40]}")
        
        if len(result_df) > 0:
            result_df.to_csv(csv_path, index=False)
            print(f"\n✓ 已保存 {len(result_df)} 个样本")