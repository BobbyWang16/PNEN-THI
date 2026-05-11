"""
三分级有序分类 - 超参数调优脚本
支持 7 种特征组合 × 3 种筛选配置 × 6 种模型的 GridSearchCV 调参。
G1 < G2 < G3/Cancer (有序关系)
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


# ============================================================================
# 特征筛选器
# ============================================================================
class OrdinalFeatureSelector:
    """
    有序多分类特征筛选器
    流程：缺失/方差 → Kruskal-Wallis → 互信息 → 斯皮尔曼相关 → 综合排序
    """

    def __init__(self,
                 variance_threshold=0.01,
                 missing_threshold=0.1,
                 correlation_threshold=0.95,
                 pvalue_threshold=0.05,
                 mi_threshold=None,
                 n_top_features=None,
                 verbose=True,
                 random_state=42):
        self.variance_threshold = variance_threshold
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.pvalue_threshold = pvalue_threshold
        self.mi_threshold = mi_threshold
        self.n_top_features = n_top_features
        self.verbose = verbose
        self.random_state = random_state

        self.removal_log = {}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def fit_transform(self, X, y, feature_list=None):
        """执行特征筛选，返回 (X_selected, kept_features)"""
        feature_list = feature_list or X.columns.tolist()
        X_work = X[feature_list].copy()
        y_work = np.asarray(y, dtype=int)

        self._log(f"\n{'='*70}\n🔍 有序三分类特征筛选\n{'='*70}")
        self._log(f"输入: {X_work.shape[0]} 样本 × {len(feature_list)} 特征")
        self._log(f"类别分布: {dict(enumerate(np.bincount(y_work)))}\n")

        kept = set(feature_list)

        kept = self._filter_missing_variance(X_work, kept)
        kept, kw_scores = self._filter_kruskal_wallis(X_work, y_work, kept)
        kept = self._filter_mutual_info(X_work, y_work, kept)
        kept = self._filter_correlation(X_work, y_work, kept)
        kept_list = self._rank_and_topk(X_work, y_work, kept, kw_scores)

        X_selected = X_work[kept_list].copy()

        self._log(f"{'='*70}")
        self._log(f"原始: {len(feature_list)} → 保留: {len(kept_list)} "
                  f"({len(kept_list)/len(feature_list)*100:.1f}%)")
        self._log(f"{'='*70}\n")
        return X_selected, kept_list

    # ------------------------------------------------------------------ #
    def _filter_missing_variance(self, X_work, kept):
        """筛选1: 缺失值 & 方差"""
        self._log("[1/5] 缺失值与方差筛选")
        removed = []
        for col in list(kept):
            missing_ratio = X_work[col].isna().sum() / len(X_work)
            if missing_ratio > self.missing_threshold:
                removed.append(col)
                self.removal_log[col] = f"缺失值{missing_ratio:.2%}"
                self._log(f"  ✗ {col:30s} - 缺失值 {missing_ratio:.2%}")
                continue
            var = X_work[col].var()
            if var < self.variance_threshold:
                removed.append(col)
                self.removal_log[col] = f"方差{var:.6f}"
                self._log(f"  ✗ {col:30s} - 方差 {var:.6f}")
        self._log(f"  ✓ 移除 {len(removed)} 个\n")
        return kept - set(removed)

    def _filter_kruskal_wallis(self, X_work, y_work, kept):
        """筛选2: Kruskal-Wallis 检验"""
        self._log("[2/5] Kruskal-Wallis秩和检验 (有序类间差异)")
        removed = []
        kw_scores = {}
        groups = [X_work.loc[y_work == c, :].values for c in np.unique(y_work)]
        for col in list(kept):
            col_idx = list(X_work.columns).index(col)
            group_data = [g[:, col_idx] for g in groups]
            try:
                h_stat, pval = kruskal(*group_data)
                kw_scores[col] = h_stat
                if pval > self.pvalue_threshold:
                    removed.append(col)
                    self.removal_log[col] = f"KW检验p={pval:.4f}"
                    self._log(f"  ✗ {col:30s} - p值 {pval:.4f} (不显著)")
            except Exception:
                removed.append(col)
                self.removal_log[col] = "KW检验失败"
        self._log(f"  ✓ 移除 {len(removed)} 个\n")
        return kept - set(removed), kw_scores

    def _filter_mutual_info(self, X_work, y_work, kept):
        """筛选3: 互信息"""
        self._log("[3/5] 互信息筛选 (特征-标签关联)")
        try:
            X_clean = X_work[list(kept)].fillna(0)
            X_scaled = StandardScaler().fit_transform(X_clean)
            mi_scores = mutual_info_classif(X_scaled, y_work, random_state=self.random_state)
            mi_dict = dict(zip(X_clean.columns, mi_scores))

            if self.mi_threshold is None:
                self.mi_threshold = np.median(list(mi_dict.values()))

            removed = [c for c, s in mi_dict.items() if s < self.mi_threshold]
            for col in removed:
                self.removal_log[col] = f"MI{mi_dict[col]:.4f}"
                self._log(f"  ✗ {col:30s} - MI {mi_dict[col]:.4f}")
            self._log(f"  ✓ 移除 {len(removed)} 个 (阈值={self.mi_threshold:.4f})\n")
            return kept - set(removed)
        except Exception as e:
            self._log(f"  ⚠ MI计算失败: {str(e)[:30]}\n")
            return kept

    def _filter_correlation(self, X_work, y_work, kept):
        """筛选4: 斯皮尔曼相关"""
        self._log("[4/5] 斯皮尔曼相关性筛选 (移除冗余特征)")
        kept_list = list(kept)
        if len(kept_list) <= 1:
            return kept

        X_subset = X_work[kept_list].fillna(0)
        corr = X_subset.corr(method='spearman').abs()
        removed = set()

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] < self.correlation_threshold:
                    continue
                col1, col2 = corr.columns[i], corr.columns[j]
                corr1_y, _ = spearmanr(X_work[col1].fillna(0), y_work)
                corr2_y, _ = spearmanr(X_work[col2].fillna(0), y_work)
                remove_col = col2 if abs(corr1_y) >= abs(corr2_y) else col1
                removed.add(remove_col)
                kept_col = col1 if remove_col == col2 else col2
                self.removal_log[remove_col] = f"与{kept_col}相关ρ={corr.iloc[i, j]:.3f}"
                self._log(f"  ✗ {remove_col:30s} - ρ {corr.iloc[i, j]:.3f}")

        self._log(f"  ✓ 移除 {len(removed)} 个\n")
        return kept - removed

    def _rank_and_topk(self, X_work, y_work, kept, kw_scores):
        """筛选5: 综合评分 + Top-K"""
        self._log("[5/5] 特征重要性评分")
        kept_list = list(kept)
        if not kept_list:
            return kept_list

        X_final = X_work[kept_list].fillna(0)
        kw_max = max(kw_scores.values()) if kw_scores else 1

        # 全体MI一次性计算
        try:
            X_scaled_all = StandardScaler().fit_transform(X_final)
            mi_all = mutual_info_classif(X_scaled_all, y_work, random_state=self.random_state)
            mi_max = max(mi_all) if max(mi_all) > 0 else 1
            mi_dict = dict(zip(kept_list, mi_all))
        except Exception:
            mi_dict, mi_max = {c: 0 for c in kept_list}, 1

        importance = {
            col: (kw_scores.get(col, 0) / kw_max) * 0.6 + (mi_dict[col] / mi_max) * 0.4
            for col in kept_list
        }
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        self._log("  特征排序 (Top-10):")
        for i, (col, score) in enumerate(sorted_feats[:10], 1):
            self._log(f"    {i:2d}. {col:30s} │ 分数 {score:.4f}")

        if self.n_top_features and self.n_top_features < len(kept_list):
            kept_list = [c for c, _ in sorted_feats[:self.n_top_features]]
            self._log(f"  ✓ 选择 Top-{self.n_top_features} 特征\n")

        return kept_list


# ============================================================================
# 配置区
# ============================================================================
# 6 个模型类映射
MODEL_CLASSES = {
    'LR':   LogisticRegression,
    'SVM':  SVC,
    'DT':   DecisionTreeClassifier,
    'RF':   RandomForestClassifier,
    'GBDT': GradientBoostingClassifier,
    'KNN':  KNeighborsClassifier,
}

# 模型基础参数（不参与调参）
BASE_CONFIGS = {
    'LR':   dict(multi_class='multinomial', max_iter=1000, tol=1e-4,
                 random_state=42, class_weight='balanced'),
    'SVM':  dict(decision_function_shape='ovr', probability=True,
                 random_state=42, class_weight='balanced'),
    'DT':   dict(criterion='gini', class_weight='balanced', random_state=42),
    'RF':   dict(max_samples=0.8, bootstrap=True,
                 class_weight='balanced_subsample', random_state=42, n_jobs=-1),
    'GBDT': dict(validation_fraction=0.15, n_iter_no_change=20,
                 tol=1e-4, random_state=42),
    'KNN':  dict(n_jobs=-1),
}

# 超参数搜索空间
PARAM_GRIDS = {
    'LR': {
        'C':       [0.01, 0.05, 0.1, 0.5, 1.0],
        'penalty': ['l1', 'l2'],
        'solver':  ['saga'],
    },
    'SVM': {
        'C':      [0.1, 0.5, 1.0, 5.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma':  ['scale', 'auto', 0.01, 0.1],
    },
    'DT': {
        'max_depth':         [2, 3, 4, 5],
        'min_samples_split': [10, 20, 30, 50],
        'min_samples_leaf':  [5, 10, 15, 20],
        'ccp_alpha':         [0.0, 0.005, 0.01, 0.02],
    },
    'RF': {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [3, 4, 5, 7, None],
        'min_samples_leaf': [5, 10, 15, 20],
        'max_features':     ['sqrt', 'log2', 0.5],
    },
    'GBDT': {
        'n_estimators':  [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth':     [2, 3, 4],
        'subsample':     [0.6, 0.8, 1.0],
        'max_features':  ['sqrt', 0.5, 0.8],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan', 'minkowski'],
    },
}

# 特征筛选器配置
SELECTOR_CONFIGS = {
    'loose':    dict(missing_threshold=0.3,  variance_threshold=0.001,
                     pvalue_threshold=0.10,  mi_threshold=None,
                     correlation_threshold=0.95, n_top_features=None),
    'standard': dict(missing_threshold=0.1,  variance_threshold=0.01,
                     pvalue_threshold=0.05,  mi_threshold=None,
                     correlation_threshold=0.90, n_top_features=None),
    'strict':   dict(missing_threshold=0.05, variance_threshold=0.05,
                     pvalue_threshold=0.01,  mi_threshold=None,
                     correlation_threshold=0.80, n_top_features=15),
}


# ============================================================================
# 数据加载与拆分
# ============================================================================
def load_and_split_data():
    """
    加载临床/影像/异质性特征 + 内部&外部分组标签
    返回 (feature_combinations, summary_dict)
    """
    print("\n[Step 1] 加载和合并数据...")

    # 特征
    clinical = pd.read_excel('./clinical/clinical_data.xlsx')
    clinical['name_hospital'] = clinical['name'] + '_' + clinical['hospital']
    clinical.set_index('name_hospital', inplace=True)
    X_clinical = clinical.iloc[:, 2:-3].sort_index()
    X_rad = pd.read_excel('./score/feature_rad.xlsx', index_col=0).sort_index()
    X_ith = pd.read_excel('./score/feature_thi.xlsx', index_col=0).sort_index()

    print(f"  临床: {X_clinical.shape}, 放射组学: {X_rad.shape}, 异质性: {X_ith.shape}")

    # 标签
    inter = pd.read_csv('./data/inter.csv')
    inter['name_hospital'] = inter['name'] + '_' + inter['hospital']
    inter.set_index('name_hospital', inplace=True)

    exter = pd.read_csv('./data/exter.csv')
    exter['name_hospital'] = exter['name'] + '_' + exter['hospital']
    exter.set_index('name_hospital', inplace=True)

    # 索引对齐
    print("\n[Step 2] 数据对齐...")
    inter_idx = inter.index.intersection(X_clinical.index)
    exter_idx = exter.index.intersection(X_clinical.index)
    inter_aligned = inter.loc[inter_idx].copy()
    exter_aligned = exter.loc[exter_idx].copy()

    # 训练 / 内部验证 拆分
    train_mask = inter_aligned['split'] == 'train'
    val_mask   = inter_aligned['split'] == 'inter'

    def _slice(df, idx, mask=None):
        out = df.loc[idx]
        return out.loc[mask] if mask is not None else out

    splits = {
        'X_clinical_train': _slice(X_clinical, inter_idx, train_mask),
        'X_rad_train':      _slice(X_rad,      inter_idx, train_mask),
        'X_ith_train':      _slice(X_ith,      inter_idx, train_mask),
        'X_clinical_val':   _slice(X_clinical, inter_idx, val_mask),
        'X_rad_val':        _slice(X_rad,      inter_idx, val_mask),
        'X_ith_val':        _slice(X_ith,      inter_idx, val_mask),
        'X_clinical_exter': _slice(X_clinical, exter_idx),
        'X_rad_exter':      _slice(X_rad,      exter_idx),
        'X_ith_exter':      _slice(X_ith,      exter_idx),
        'y_train':          inter_aligned.loc[train_mask, 'grade'],
        'y_val':            inter_aligned.loc[val_mask, 'grade'],
        'y_exter':          exter_aligned['grade'],
    }

    print(f"  训练: {splits['X_clinical_train'].shape[0]}, "
          f"内部验证: {splits['X_clinical_val'].shape[0]}, "
          f"外部验证: {splits['X_clinical_exter'].shape[0]}")
    print(f"  训练分布: {dict(splits['y_train'].value_counts().sort_index())}")
    print(f"  内部分布: {dict(splits['y_val'].value_counts().sort_index())}")
    print(f"  外部分布: {dict(splits['y_exter'].value_counts().sort_index())}")

    return splits


def build_feature_combinations(splits):
    """7种特征组合"""
    print("\n[Step 3] 定义特征组合...")

    def _make(train_blocks, val_blocks, exter_blocks):
        return {
            'X_train': pd.concat(train_blocks, axis=1) if len(train_blocks) > 1 else train_blocks[0],
            'X_val':   pd.concat(val_blocks,   axis=1) if len(val_blocks)   > 1 else val_blocks[0],
            'X_exter': pd.concat(exter_blocks, axis=1) if len(exter_blocks) > 1 else exter_blocks[0],
            'y_train': splits['y_train'],
            'y_val':   splits['y_val'],
            'y_exter': splits['y_exter'],
        }

    blocks = {
        'C': ('X_clinical_train', 'X_clinical_val', 'X_clinical_exter'),
        'R': ('X_rad_train',      'X_rad_val',      'X_rad_exter'),
        'H': ('X_ith_train',      'X_ith_val',      'X_ith_exter'),
    }

    def pick(keys):
        return (
            [splits[blocks[k][0]] for k in keys],
            [splits[blocks[k][1]] for k in keys],
            [splits[blocks[k][2]] for k in keys],
        )

    combos = {
        'Clinical':                _make(*pick('C')),
        'Radiomics':               _make(*pick('R')),
        'Heterogeneity':           _make(*pick('H')),
        'Clinical+Radiomics':      _make(*pick('CR')),
        'Clinical+Heterogeneity':  _make(*pick('CH')),
        'Radiomics+Heterogeneity': _make(*pick('RH')),
        'All':                     _make(*pick('CRH')),
    }
    print(f"  特征组合: {list(combos.keys())}")
    return combos


# ============================================================================
# 调参与构建
# ============================================================================
def tune_one(model_name, X_train, y_train, cv_strategy, scoring):
    """对单个模型做 Pipeline + GridSearchCV 调参"""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    MODEL_CLASSES[model_name](**BASE_CONFIGS[model_name])),
    ])
    grid_params = {f'clf__{k}': v for k, v in PARAM_GRIDS[model_name].items()}

    grid = GridSearchCV(
        estimator=pipe, param_grid=grid_params,
        cv=cv_strategy, scoring=scoring, n_jobs=-1, refit=True, verbose=0,
    )
    grid.fit(X_train, y_train)
    clean_params = {k.replace('clf__', ''): v for k, v in grid.best_params_.items()}
    return clean_params, round(grid.best_score_, 4)


def run_tuning(feat_combos, cv_folds=5, scoring='roc_auc_ovr'):
    """主调参循环：7特征组合 × 3筛选配置 × 6模型"""
    print("\n[Step 5] 超参数调优开始...\n")
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    best_params = {}
    kept_features = {}
    records = []

    for feat_name, feat_data in feat_combos.items():
        best_params[feat_name] = {}
        kept_features[feat_name] = {}
        X_tr, y_tr = feat_data['X_train'], feat_data['y_train']

        for sel_name, sel_kwargs in SELECTOR_CONFIGS.items():
            print(f"\n{'='*65}")
            print(f"  特征组合: {feat_name}  |  筛选配置: {sel_name}")
            print(f"{'='*65}")

            selector = OrdinalFeatureSelector(**sel_kwargs, verbose=False, random_state=42)
            X_tr_sel, kept = selector.fit_transform(X_tr, y_tr)
            kept_features[feat_name][sel_name] = kept
            print(f"  保留特征: {len(kept)} / {X_tr.shape[1]}")

            best_params[feat_name][sel_name] = {}
            for model_name in MODEL_CLASSES:
                params, score = tune_one(model_name, X_tr_sel, y_tr, cv_strategy, scoring)
                best_params[feat_name][sel_name][model_name] = params
                print(f"  [{model_name}] CV {scoring}={score:.4f}  "
                      f"n_feat={len(kept)}  params={params}")

                records.append({
                    'feature_combo':   feat_name,
                    'selector':        sel_name,
                    'model':           model_name,
                    'n_features':      len(kept),
                    f'cv_{scoring}':   score,
                    'best_params':     str(params),
                    'kept_features':   str(kept),
                })

    return best_params, kept_features, records


def save_results(records, scoring, save_dir):
    """保存调参结果与透视表"""
    print("\n[Step 6] 保存调参结果...")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(records)
    df_path = os.path.join(save_dir, 'tuning_results.csv')
    df.to_csv(df_path, index=False)
    print(f"  ✓ 调参结果: {df_path}")

    pivot = df.pivot_table(
        index=['feature_combo', 'selector'], columns='model',
        values=f'cv_{scoring}', aggfunc='first',
    )
    pivot_path = os.path.join(save_dir, 'tuning_pivot.csv')
    pivot.to_csv(pivot_path)
    print(f"\n── CV {scoring} 汇总 ──")
    print(pivot.to_string())
    print(f"\n  ✓ 透视表: {pivot_path}")
    return df, pivot


def build_best_models(best_params, kept_features, feat_name, sel_name):
    """构建指定 (feat_name, sel_name) 下的最优模型字典"""
    kept = kept_features[feat_name][sel_name]
    models = {}
    for model_name, model_cls in MODEL_CLASSES.items():
        merged = {**BASE_CONFIGS[model_name],
                  **best_params[feat_name][sel_name][model_name]}
        models[model_name] = model_cls(**merged)
    return models, kept


# ============================================================================
# 主流程
# ============================================================================
def main():
    result_root = './results/hyper_tune'
    dir_results = os.path.join(result_root, 'results')
    os.makedirs(dir_results, exist_ok=True)

    splits = load_and_split_data()
    feat_combos = build_feature_combinations(splits)
    best_params, kept_features, records = run_tuning(
        feat_combos, cv_folds=5, scoring='roc_auc_ovr',
    )
    save_results(records, scoring='roc_auc_ovr', save_dir=dir_results)

    print("\n✓ 调参完成。可用 build_best_models(best_params, kept_features, "
          "feat_name, sel_name) 构建最优模型。")
    return best_params, kept_features


if __name__ == '__main__':
    main()
