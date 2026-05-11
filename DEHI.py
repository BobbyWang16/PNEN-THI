import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

def calculate_enhancement_heterogeneity(arterial_path, venous_path, cluster_path, 
                                      hu_range=(-200, 200)):
    """计算多时相强化异质性指数"""
    
    # 读取数据
    arterial_data = nib.load(arterial_path).get_fdata()
    venous_data = nib.load(venous_path).get_fdata()
    cluster_data = nib.load(cluster_path).get_fdata()
    
    # 处理维度
    arterial_data = np.squeeze(arterial_data)
    venous_data = np.squeeze(venous_data)
    cluster_data = np.squeeze(cluster_data)
    
    # 形状匹配
    min_shape = np.minimum.reduce([arterial_data.shape, venous_data.shape, cluster_data.shape])
    arterial_data = arterial_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    venous_data = venous_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    cluster_data = cluster_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # 计算强化
    enhancement_data = venous_data - arterial_data
    
    # 获取聚类
    valid_clusters = [int(label) for label in np.unique(cluster_data) if label != 0]
    if len(valid_clusters) < 2:
        raise ValueError(f"Need at least 2 clusters, found: {valid_clusters}")
    
    cluster_features = {}
    
    for cluster_id in valid_clusters:
        mask = cluster_data == cluster_id
        
        arterial_vals = arterial_data[mask]
        venous_vals = venous_data[mask]
        enhancement_vals = enhancement_data[mask]
        
        # 数据过滤
        valid_idx = (
            np.isfinite(arterial_vals) & np.isfinite(venous_vals) &
            (arterial_vals >= hu_range[0]) & (arterial_vals <= hu_range[1]) &
            (venous_vals >= hu_range[0]) & (venous_vals <= hu_range[1])
        )
        
        if np.sum(valid_idx) == 0:
            continue
        
        arterial_vals = arterial_vals[valid_idx]
        venous_vals = venous_vals[valid_idx]
        enhancement_vals = enhancement_vals[valid_idx]
        
        # 强化特征
        mean_enh = np.mean(enhancement_vals)
        std_enh = np.std(enhancement_vals)
        
        # 强化模式
        strong = np.sum(enhancement_vals > 20) / len(enhancement_vals)
        mild = np.sum((enhancement_vals > 5) & (enhancement_vals <= 20)) / len(enhancement_vals)
        no_enh = np.sum((enhancement_vals >= -5) & (enhancement_vals <= 5)) / len(enhancement_vals)
        washout = np.sum(enhancement_vals < -5) / len(enhancement_vals)
        
        dominant = ['washout', 'no_enhancement', 'mild_enhancement', 'strong_enhancement'][
            np.argmax([washout, no_enh, mild, strong])
        ]
        
        # 时相相关性
        if len(arterial_vals) > 1:
            phase_corr = stats.pearsonr(arterial_vals, venous_vals)[0]
        else:
            phase_corr = 0
        
        # 均匀性
        uniformity = 1 / (1 + std_enh / (abs(mean_enh) + 1e-8))
        
        cluster_features[cluster_id] = {
            'mean_enhancement': mean_enh,
            'std_enhancement': std_enh,
            'enhancement_range': np.max(enhancement_vals) - np.min(enhancement_vals),
            'strong_ratio': strong,
            'mild_ratio': mild,
            'no_enhancement_ratio': no_enh,
            'washout_ratio': washout,
            'dominant_pattern': dominant,
            'phase_correlation': phase_corr,
            'uniformity': uniformity,
            'arterial_mean': np.mean(arterial_vals),
            'venous_mean': np.mean(venous_vals),
            'n_voxels': len(enhancement_vals),
            'arterial_vals': arterial_vals,
            'venous_vals': venous_vals,
            'enhancement_vals': enhancement_vals
        }
    
    # 计算异质性指数
    means = [cluster_features[c]['mean_enhancement'] for c in valid_clusters]
    
    mag_diversity = np.std(means) / (abs(np.mean(means)) + 1e-8)
    
    # 模式多样性
    patterns = np.array([[cluster_features[c]['strong_ratio'],
                         cluster_features[c]['mild_ratio'],
                         cluster_features[c]['no_enhancement_ratio'],
                         cluster_features[c]['washout_ratio']] 
                        for c in valid_clusters])
    
    pattern_div = 0
    n_pairs = 0
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            cos_sim = np.dot(patterns[i], patterns[j]) / (
                np.linalg.norm(patterns[i]) * np.linalg.norm(patterns[j]) + 1e-8
            )
            pattern_div += 1 - cos_sim
            n_pairs += 1
    pattern_div = pattern_div / (n_pairs + 1e-8)
    
    # 稳定性多样性
    corrs = [cluster_features[c]['phase_correlation'] for c in valid_clusters]
    stab_div = np.std(corrs)
    
    # 综合指数
    het_index = 0.4 * mag_diversity + 0.4 * pattern_div + 0.2 * stab_div
    
    return {
        'heterogeneity_index': het_index,
        'cluster_features': cluster_features,
        'valid_clusters': valid_clusters,
        'mean_enhancement': np.mean(means),
        'enhancement_range': np.max(means) - np.min(means)
    }


def plot_enhancement_analysis(results, save_path):
    """绘制单张强化异质性图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    cluster_features = results['cluster_features']
    valid_clusters = sorted(results['valid_clusters'])
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    x_pos = np.arange(len(valid_clusters))
    width = 0.25
    
    # 左图：强化特征对比
    means = [cluster_features[c]['mean_enhancement'] for c in valid_clusters]
    stds = [cluster_features[c]['std_enhancement'] for c in valid_clusters]
    uniforms = [cluster_features[c]['uniformity'] for c in valid_clusters]
    
    ax1.bar(x_pos - width, means, width, label='Mean Enhancement', 
            alpha=0.85, color=colors[0], edgecolor='black', linewidth=1)
    ax1.bar(x_pos, [u*50-25 for u in uniforms], width, label='Uniformity (scaled)', 
            alpha=0.85, color=colors[1], edgecolor='black', linewidth=1)
    ax1.bar(x_pos + width, stds, width, label='Std Enhancement', 
            alpha=0.85, color=colors[2], edgecolor='black', linewidth=1)
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
    ax1.set_ylabel('HU Value', fontsize=11, fontweight='bold')
    ax1.set_title('Enhancement Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'C{c}' for c in valid_clusters], fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # 右图：A期vs V期
    arterial_means = [cluster_features[c]['arterial_mean'] for c in valid_clusters]
    venous_means = [cluster_features[c]['venous_mean'] for c in valid_clusters]
    
    ax2.plot(x_pos, arterial_means, 'o-', linewidth=2.5, markersize=10,
             color=colors[0], label='Arterial (A)', alpha=0.9, markeredgecolor='black', markeredgewidth=1)
    ax2.plot(x_pos, venous_means, 's-', linewidth=2.5, markersize=10,
             color=colors[1], label='Venous (V)', alpha=0.9, markeredgecolor='black', markeredgewidth=1)
    
    # 添加增强标签
    for i in range(len(valid_clusters)):
        enh = venous_means[i] - arterial_means[i]
        mid_y = (arterial_means[i] + venous_means[i]) / 2
        ax2.text(i + 0.2, mid_y, f'+{enh:.0f}', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', alpha=0.9))
    
    ax2.set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('HU Value', fontsize=11, fontweight='bold')
    ax2.set_title('Multi-Phase Response', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'C{c}' for c in valid_clusters], fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    
    fig.suptitle(f'Patient {patient_id} - Dynamic Enhancement Heterogeneity Analysis',
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def extract_features(results, patient_id):
    """提取特征到DataFrame"""
    
    cf = results['cluster_features']
    vc = results['valid_clusters']
    
    row = {
        'patient_id': patient_id,
        'DEHI': results['heterogeneity_index'],
        'n_clusters': len(vc),
        'mean_enhancement_avg': np.mean([cf[c]['mean_enhancement'] for c in vc]),
        'enhancement_range': results['enhancement_range']
    }
    
    # 强化模式统计
    strong_ratios = [cf[c]['strong_ratio'] for c in vc]
    mild_ratios = [cf[c]['mild_ratio'] for c in vc]
    washout_ratios = [cf[c]['washout_ratio'] for c in vc]
    
    row['strong_enhancement_avg'] = np.mean(strong_ratios)
    row['mild_enhancement_avg'] = np.mean(mild_ratios)
    row['washout_avg'] = np.mean(washout_ratios)
    
    # 时相特征
    phase_corrs = [cf[c]['phase_correlation'] for c in vc]
    uniformities = [cf[c]['uniformity'] for c in vc]
    
    row['phase_correlation_avg'] = np.mean(phase_corrs)
    row['uniformity_avg'] = np.mean(uniformities)
    
    # 基线和峰值
    arterial_means = [cf[c]['arterial_mean'] for c in vc]
    venous_means = [cf[c]['venous_mean'] for c in vc]
    
    row['arterial_mean_avg'] = np.mean(arterial_means)
    row['venous_mean_avg'] = np.mean(venous_means)
    
    # 每个聚类的详细信息
    for i, cid in enumerate(vc[:3]):
        prefix = f'C{cid}_'
        row[f'{prefix}mean_enh'] = cf[cid]['mean_enhancement']
        row[f'{prefix}std_enh'] = cf[cid]['std_enhancement']
        row[f'{prefix}strong'] = cf[cid]['strong_ratio']
        row[f'{prefix}mild'] = cf[cid]['mild_ratio']
        row[f'{prefix}washout'] = cf[cid]['washout_ratio']
        row[f'{prefix}phase_corr'] = cf[cid]['phase_correlation']
        row[f'{prefix}uniformity'] = cf[cid]['uniformity']
        row[f'{prefix}n_voxels'] = cf[cid]['n_voxels']
    
    return pd.DataFrame([row])


if __name__ == "__main__":
    
    arterial_dir = './data/crop_A'
    venous_dir = './data/crop_V'
    cluster_dir = './cluster/with_point'
    output_dir = './score/DEHI'
    plots_dir = os.path.join(output_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    result_df = pd.DataFrame()
    
    cluster_files = [f for f in os.listdir(cluster_dir) if f.endswith('.nii.gz')]
    print(f"\nProcessing {len(cluster_files)} files...\n")
    
    for i, cluster_file in enumerate(cluster_files):
        try:
            patient_id = cluster_file.split('_clusters')[0]
            
            arterial_path = os.path.join(arterial_dir, f"{patient_id}.nii.gz")
            venous_path = os.path.join(venous_dir, f"{patient_id}.nii.gz")
            cluster_path = os.path.join(cluster_dir, cluster_file)
            
            if not (os.path.exists(arterial_path) and os.path.exists(venous_path)):
                continue
            
            results = calculate_enhancement_heterogeneity(arterial_path, venous_path, cluster_path)
            
            plot_path = os.path.join(plots_dir, f'{patient_id}_DEHI.png')
            plot_enhancement_analysis(results, plot_path)
            
            df_row = extract_features(results, patient_id)
            result_df = pd.concat([result_df, df_row], ignore_index=True)
            
            print(f"✓ {i+1}/{len(cluster_files)} - {patient_id}: DEHI={results['heterogeneity_index']:.4f}")
            
        except Exception as e:
            print(f"✗ Error processing {cluster_file}: {str(e)}")
            continue
    
    # 保存结果
    if len(result_df) > 0:
        csv_path = os.path.join(output_dir, 'DEHI_features.csv')
        result_df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved: {csv_path}")
        print(f"✓ Total patients: {len(result_df)}")
        print(f"\nFeature summary:")
        print(result_df[['patient_id', 'DEHI', 'n_clusters', 'mean_enhancement_avg', 
                        'phase_correlation_avg', 'uniformity_avg']].head(10))