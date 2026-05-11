import os
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class SpatialHeterogeneityAnalyzer:
    """肿瘤聚类空间异质性分析"""
    
    def __init__(self, voxel_size=(1.0, 1.0, 1.0)):
        self.voxel_size = np.array(voxel_size)
    
    @staticmethod
    def gini_coefficient(x):
        """基尼系数：衡量数据分布不均匀性[0,1]"""
        x = np.asarray(x) + 1e-8
        sorted_x = np.sort(x)
        n = len(x)
        return (2 * np.sum((n - np.arange(1, n+1) + 1) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n
    
    @staticmethod
    def calculate_entropy(data, bins=10):
        """分布熵：衡量分布的复杂性和不确定性"""
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]
        probs = hist / np.sum(hist)
        return -np.sum(probs * np.log2(probs + 1e-8))
    
    @staticmethod
    def _calculate_skewness(data):
        """偏度：衡量分布的对称性"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std)**3) if std > 0 else 0
    
    def extract_cluster_spherical_features(self, cluster_data, cluster_id, tumor_centroid):
        """
        提取单个聚类的球坐标特征
        
        特征说明：
        ├─ 径向 (Radial): 到肿瘤中心的距离分布
        │  └─ mean_r/std_r: 平均/标准差距离
        │  └─ r_concentration: 距离集中度[0,1]，值高→分布紧密
        │  └─ r_entropy: 径向分布的复杂性
        │
        ├─ 方位角 (Azimuth): 水平面上的角度分布(0-2π)
        │  └─ theta_uniformity: 方位均匀性[0,1]，值高→均匀分布
        │  └─ theta_concentration: 方位集中度，值高→集中分布
        │  └─ theta_circular_concentration: 圆形浓度[0,1]
        │  └─ gini_coefficient: 基尼系数，衡量不均匀性
        │
        ├─ 极角 (Polar): 竖直方向的角度分布(0-π)
        │  └─ hemisphere_bias: 上下半球偏差，>1→上半球偏移
        │  └─ upper_hemisphere_ratio: 上半球占比
        │
        └─ 空间模式分类: 中心-球形/中心-集中/方向性/半球性/复杂不规则
        """
        coords = np.where(cluster_data == cluster_id)
        coords_3d = np.array([coords[0], coords[1], coords[2]]).T * self.voxel_size
        relative_coords = coords_3d - tumor_centroid
        
        x, y, z = relative_coords[:, 0], relative_coords[:, 1], relative_coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2 * np.pi
        phi = np.arccos(np.clip(z / (r + 1e-8), -1, 1))
        
        # 径向特征
        radial_features = {
            'mean_r': float(np.mean(r)),
            'std_r': float(np.std(r)),
            'max_r': float(np.max(r)),
            'r_concentration': float(np.exp(-np.std(r)/(np.mean(r) + 1e-8))),
            'r_entropy': float(self.calculate_entropy(r, bins=8)),
        }
        
        # 方位角特征
        theta_sectors = np.array([
            np.sum((theta >= i * 2*np.pi/8) & (theta < (i+1) * 2*np.pi/8))
            for i in range(8)
        ])
        
        expected_theta = len(theta) / 8
        theta_uniformity = 1 - np.std(theta_sectors) / (expected_theta + 1e-8)
        theta_concentration = np.max(theta_sectors) / (np.mean(theta_sectors) + 1e-8)
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        circular_conc = np.sqrt(np.mean(sin_theta)**2 + np.mean(cos_theta)**2)
        
        azimuth_features = {
            'theta_uniformity': float(max(0, theta_uniformity)),
            'theta_concentration': float(min(theta_concentration, 3.0)),
            'theta_circular_concentration': float(circular_conc),
            'theta_entropy': float(self.calculate_entropy(theta, bins=8)),
            'gini_coefficient': float(self.gini_coefficient(theta_sectors)),
        }
        
        # 极角特征
        phi_bands = np.array([
            np.sum((phi >= i * np.pi/6) & (phi < (i+1) * np.pi/6))
            for i in range(6)
        ])
        
        upper_hemisphere = np.sum(phi < np.pi/2)
        lower_hemisphere = np.sum(phi >= np.pi/2)
        
        polar_features = {
            'phi_mean': float(np.mean(phi)),
            'phi_std': float(np.std(phi)),
            'hemisphere_bias': float((upper_hemisphere + 1) / (lower_hemisphere + 1)),
            'upper_hemisphere_ratio': float(upper_hemisphere / len(phi)),
        }
        
        # 空间模式分类
        spatial_pattern = self._classify_spatial_pattern(radial_features, azimuth_features, polar_features)
        
        return {
            'radial_features': radial_features,
            'azimuth_features': azimuth_features,
            'polar_features': polar_features,
            'spatial_pattern': spatial_pattern,
            'volume': len(coords_3d),
            'raw_coords': {'r': r, 'theta': theta, 'phi': phi, 'x': x, 'y': y, 'z': z}
        }
    
    @staticmethod
    def _classify_spatial_pattern(radial_feat, azimuth_feat, polar_feat):
        """空间模式分类：5种模式"""
        r_conc = radial_feat['r_concentration']
        theta_unif = azimuth_feat['theta_uniformity']
        theta_conc = azimuth_feat['theta_concentration']
        hemisphere_bias = polar_feat['hemisphere_bias']
        
        if r_conc > 0.7:
            return 'central_spherical' if theta_unif > 0.8 else 'central_concentrated'
        elif theta_conc > 2.0:
            return 'directional'
        elif hemisphere_bias > 2.0 or hemisphere_bias < 0.5:
            return 'hemispherical'
        else:
            return 'complex_irregular'
    
    def calculate_global_heterogeneity(self, cluster_spherical_features, valid_clusters):
        """
        计算全局空间异质性指标
        
        特征说明：
        ├─ radial_position_diversity: 聚类径向位置差异(std/mean)
        ├─ radial_shape_diversity: 聚类径向集中度差异
        ├─ azimuth_uniformity_diversity: 聚类方位均匀性差异
        ├─ azimuth_concentration_diversity: 聚类方位浓度差异
        ├─ polar_position_diversity: 聚类极角位置差异
        ├─ hemisphere_diversity: 聚类上下半球偏差
        ├─ gini_diversity: 聚类基尼系数差异(不均匀性)
        ├─ pattern_diversity: 聚类空间模式多样性[0,1]
        ├─ volume_diversity: 聚类体积差异
        ├─ density_diversity: 聚类密度差异
        └─ spatial_heterogeneity_index: 综合空间异质性指数[0,1]
        """
        cluster_list = list(valid_clusters)
        
        # 径向多样性
        radial_means = np.array([cluster_spherical_features[c]['radial_features']['mean_r'] for c in cluster_list])
        radial_concentrations = np.array([cluster_spherical_features[c]['radial_features']['r_concentration'] for c in cluster_list])
        
        radial_position_diversity = float(np.std(radial_means) / (np.mean(radial_means) + 1e-8))
        radial_shape_diversity = float(np.std(radial_concentrations))
        
        # 方位角多样性
        theta_uniformities = np.array([cluster_spherical_features[c]['azimuth_features']['theta_uniformity'] for c in cluster_list])
        theta_circular_conc = np.array([cluster_spherical_features[c]['azimuth_features']['theta_circular_concentration'] for c in cluster_list])
        
        azimuth_uniformity_diversity = float(np.std(theta_uniformities))
        azimuth_concentration_diversity = float(np.std(theta_circular_conc))
        
        # 极角多样性
        phi_means = np.array([cluster_spherical_features[c]['polar_features']['phi_mean'] for c in cluster_list])
        polar_position_diversity = float(np.std(phi_means) / (np.pi / 2 + 1e-8))
        
        upper_ratios = np.array([cluster_spherical_features[c]['polar_features']['upper_hemisphere_ratio'] for c in cluster_list])
        hemisphere_diversity = float(np.std(upper_ratios))
        
        # 基尼系数多样性
        gini_coefficients = np.array([cluster_spherical_features[c]['azimuth_features']['gini_coefficient'] for c in cluster_list])
        gini_diversity = float(np.std(gini_coefficients))
        
        # 空间模式多样性
        patterns = [cluster_spherical_features[c]['spatial_pattern'] for c in cluster_list]
        pattern_counts = Counter(patterns)
        probs = np.array([count/len(patterns) for count in pattern_counts.values()])
        shannon_entropy = entropy(probs, base=2)
        max_entropy = np.log2(len(pattern_counts)) if len(pattern_counts) > 1 else 0
        pattern_diversity = float(shannon_entropy / (max_entropy + 1e-8))
        
        # 体积和密度多样性
        volumes = np.array([cluster_spherical_features[c]['volume'] for c in cluster_list])
        volume_diversity = float(np.std(volumes) / (np.mean(volumes) + 1e-8))
        
        density = volumes / (np.array([cluster_spherical_features[c]['radial_features']['max_r']**3 for c in cluster_list]) + 1e-8)
        density_diversity = float(np.std(density) / (np.mean(density) + 1e-8))
        
        # 综合异质性指数
        spatial_heterogeneity_index = float(
            (radial_position_diversity + azimuth_concentration_diversity + polar_position_diversity + pattern_diversity) / 4.0
        )
        
        return {
            'n_clusters': len(cluster_list),
            'radial_position_diversity': radial_position_diversity,
            'radial_shape_diversity': radial_shape_diversity,
            'azimuth_uniformity_diversity': azimuth_uniformity_diversity,
            'azimuth_concentration_diversity': azimuth_concentration_diversity,
            'polar_position_diversity': polar_position_diversity,
            'hemisphere_diversity': hemisphere_diversity,
            'gini_diversity': gini_diversity,
            'pattern_diversity': pattern_diversity,
            'volume_diversity': volume_diversity,
            'density_diversity': density_diversity,
            'spatial_heterogeneity_index': spatial_heterogeneity_index,
        }
    
    def process_single_case(self, cluster_path):
        """处理单个病例"""
        cluster_data = np.squeeze(nib.load(cluster_path).get_fdata())
        
        while len(cluster_data.shape) > 3:
            cluster_data = cluster_data[:, :, :, 0]
        
        valid_clusters = [int(l) for l in np.unique(cluster_data) if l != 0]
        
        if len(valid_clusters) < 2:
            raise ValueError(f"需要至少2个聚类，实际{len(valid_clusters)}个")
        
        tumor_mask = cluster_data > 0
        tumor_coords = np.where(tumor_mask)
        tumor_coords_3d = np.array([tumor_coords[0], tumor_coords[1], tumor_coords[2]]).T * self.voxel_size
        tumor_centroid = np.mean(tumor_coords_3d, axis=0)
        
        cluster_spherical_features = {
            cid: self.extract_cluster_spherical_features(cluster_data, cid, tumor_centroid)
            for cid in valid_clusters
        }
        
        global_features = self.calculate_global_heterogeneity(cluster_spherical_features, valid_clusters)
        
        return global_features, cluster_spherical_features


class SpatialVisualization:
    """空间分布可视化"""
    
    CLUSTER_COLORS = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#FFA07A', 
                      5: '#98D8C8', 6: '#F7DC6F', 7: '#BB8FCE', 8: '#85C1E2'}
    
    @staticmethod
    def get_cluster_color(cid):
        if cid in SpatialVisualization.CLUSTER_COLORS:
            return SpatialVisualization.CLUSTER_COLORS[cid]
        np.random.seed(cid)
        return f'#{np.random.randint(0, 0xFFFFFF):06x}'
    
    @staticmethod
    def create_2d_heterogeneity_figure(patient_id, cluster_spherical_features, output_path):
        """1x3分布图：方位角、径向距离、Z轴"""
        valid_clusters = sorted(cluster_spherical_features.keys())
        n_clusters = len(valid_clusters)
        
        fig = plt.figure(figsize=(18, 5))
        fig.suptitle(f'Patient {patient_id} - Spatial Heterogeneity', fontsize=14, fontweight='bold')
        
        # 方位角分布（极坐标）
        ax1 = fig.add_subplot(1, 3, 1, projection='polar')
        theta_bins = np.linspace(0, 2*np.pi, 9)
        bar_width = 2*np.pi / 8
        
        for i, cid in enumerate(valid_clusters):
            theta = cluster_spherical_features[cid]['raw_coords']['theta']
            theta_counts, _ = np.histogram(theta, bins=theta_bins)
            offset = (i - n_clusters/2 + 0.5) * bar_width / n_clusters
            ax1.bar(theta_bins[:-1] + offset, theta_counts, width=bar_width/n_clusters,
                   label=f'C{cid}', color=SpatialVisualization.get_cluster_color(cid), 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax1.set_theta_offset(np.pi / 2)
        ax1.set_theta_direction(-1)
        ax1.set_title('Azimuth Distribution\n(Top View)', fontsize=12, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=9)
        ax1.grid(True, alpha=0.15, linestyle='--')
        
        # 径向距离（箱线图）
        ax2 = fig.add_subplot(1, 3, 2)
        radii_data = [cluster_spherical_features[cid]['raw_coords']['r'] for cid in valid_clusters]
        colors_list = [SpatialVisualization.get_cluster_color(cid) for cid in valid_clusters]
        
        bp = ax2.boxplot(radii_data, labels=[f'C{c}' for c in valid_clusters],
                        patch_artist=True, showmeans=True, medianprops=dict(color='red', linewidth=2))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Radial Distance (mm)', fontsize=11, fontweight='bold')
        ax2.set_title('Radial Distribution\n(Distance to Centroid)', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Z轴分布（直方图）
        ax3 = fig.add_subplot(1, 3, 3)
        for cid in valid_clusters:
            z = cluster_spherical_features[cid]['raw_coords']['z']
            ax3.hist(z, bins=12, alpha=0.6, label=f'C{cid}', 
                    color=SpatialVisualization.get_cluster_color(cid), edgecolor='black', linewidth=0.5)
        
        ax3.set_xlabel('Z-axis Coordinate (mm)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Voxel Count', fontsize=11, fontweight='bold')
        ax3.set_title('Z-axis Distribution\n(Coronal View)', fontsize=12, fontweight='bold', pad=10)
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def batch_process_clusters(cluster_dir, output_dir, voxel_size=(1.0, 1.0, 1.0)):
    """批量处理聚类文件"""
    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, 'spatial_heterogeneity_features.csv')
    cluster_files = sorted([f for f in os.listdir(cluster_dir) if f.endswith('.nii.gz')])
    
    if not cluster_files:
        print(f"错误: {cluster_dir} 中未找到 .nii.gz 文件")
        return
    
    print(f"\n{'='*80}")
    print(f"肿瘤聚类空间异质性分析")
    print(f"{'='*80}")
    print(f"聚类目录: {cluster_dir}")
    print(f"找到 {len(cluster_files)} 个聚类文件")
    print(f"体素尺寸: {voxel_size}")
    print(f"{'='*80}\n")
    
    analyzer = SpatialHeterogeneityAnalyzer(voxel_size=voxel_size)
    results_list = []
    successful = 0
    
    for i, file in enumerate(cluster_files, 1):
        try:
            print(f"[{i:3d}/{len(cluster_files)}] {file:<50}", end=" ", flush=True)
            
            patient_id = file.split('_clusters')[0]
            global_features, cluster_details = analyzer.process_single_case(os.path.join(cluster_dir, file))
            
            global_features['patient_id'] = patient_id
            results_list.append(global_features)
            
            figure_path = os.path.join(figure_dir, f'{patient_id}_heterogeneity.png')
            SpatialVisualization.create_2d_heterogeneity_figure(patient_id, cluster_details, figure_path)
            
            successful += 1
            print(f"✓ (n_clusters: {global_features['n_clusters']})")
            
        except Exception as e:
            print(f"✗ {str(e)[:45]}")
    
    if results_list:
        df = pd.DataFrame(results_list)
        cols = ['patient_id'] + [c for c in df.columns if c != 'patient_id']
        df[cols].to_csv(output_csv, index=False)
        
        print(f"\n{'='*80}")
        print(f"处理完成: {successful}/{len(cluster_files)} 个文件")
        print(f"特征保存: {output_csv}")
        print(f"图形保存: {figure_dir}")
        print(f"特征维度: {len(df.columns)} 列")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    batch_process_clusters(
        cluster_dir='./cluster/with_point',
        output_dir='./score/SHI',
        voxel_size=(1.0, 1.0, 1.0)
    )