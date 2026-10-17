<div align="center">

# PNEN-THI

**基于双期 CT 生境的术前三分类分级**

专家先验初始化生境 · 可解释异质性 · 冻结模型推理

[English](README.md) &nbsp; · &nbsp; [简体中文](README_zh.md)

[快速开始](#快速开始) &nbsp; / &nbsp; [核心方法](#核心方法) &nbsp; / &nbsp; [影像评分](#影像评分) &nbsp; / &nbsp; [分级推理](#分级推理) &nbsp; / &nbsp; [训练](#训练)

</div>

<a href="assets/study_workflow.png"><img src="assets/study_workflow.png" alt="研究流程：双期 CT、专家先验初始化的肿瘤生境、异质性与影像组学评分，以及胰腺神经内分泌肿瘤三分类分级" width="100%"></a>

**研究流程。** PNET-ITH 方法连接动脉期与静脉期 CT、三类肿瘤生境、异质性特征及胰腺神经内分泌肿瘤分级模型。本仓库提供核心数值方法、选定的冻结模型，以及一例去标识的真实影像病例。

## 概览

PNEN-THI 将 PNET-ITH 实现收敛为两个入口：**`predict.py` 负责推理**，**`train.py` 负责固定规格的重新拟合**。图像读写、专家先验学习、生境聚类和特征提取统一放在 `preprocessing/` 中。

| 阶段 | 输入 | 输出 |
| :--- | :--- | :--- |
| 专家先验 | 训练 CT、肿瘤 ROI、稀疏专家点 | 三类汇总动脉特征中心 |
| 生境 | 已配准双期 CT、ROI、汇总先验 | 三类有序的肿瘤内生境 |
| 影像评分 | 生境异质性与双期影像组学特征 | 固定 0–1 标度的 THI 和 RAD |
| 分级 | 影像评分与完整临床输入 | G1 / G2 / G3/NEC 概率、风险值与分级 |

所附病例演示影像分支，不包含临床测量、结局或专家点标注。仓库不包含生成的患者输入或实验套件。

## 环境配置

完整影像流程使用 **Python 3.9**：

```bash
python -m pip install -r preprocessing/requirements-radiomics.txt
```

该依赖文件包含基础依赖和源影像组学环境。完整提取已在 Python **3.9.25**、PyRadiomics **3.1.0**、NumPy **1.26.4**、SciPy **1.13.1**、SimpleITK **2.5.5** 下验证。源模型加载与训练使用 scikit-learn **1.6.1**。

如仅计算 THI、从已有特征进行模型推理，或运行无需提取影像组学的固定规格训练：

```bash
python -m pip install -r requirements.txt
```

基础流程也支持 Python 3.12。以下命令均从仓库根目录运行。

## 快速开始

从真实示例计算生境、全部 18 项核心异质性特征和 THI：

```bash
python predict.py --thi-only --case examples/case_001 --output-dir outputs/demo
```

输出目录包含 `habitats.nii.gz`、`heterogeneity_features.json` 和 `result.json`。安装完整影像组学环境后，可提取 200 项影像组学特征并计算两个评分：

```bash
python predict.py --score-only --case examples/case_001 --output-dir outputs/demo_scores
```

此命令还输出 `radiomics_features.json`，记录提取设置与来源说明。同一入口可查看两个分级模型的规格：

```bash
python predict.py --list-models
python predict.py --describe model8
python predict.py --describe model9
```

## 核心方法

以下三个步骤公开完整的生境构建与异质性计算。已有配准 CT 和 ROI 的输入可跳过上游分割与配准。仓库不附自动分割模型或配准权重。

### 1 · 学习汇总专家先验

为新的训练数据集准备逐例标注：

```text
your_training_case/
├── arterial.nii.gz
├── tumor_roi.nii.gz
└── points.nii.gz
```

`points.nii.gz` 与 CT、ROI 使用相同网格，保存稀疏专家标签 **1、2、3**；0 表示未标注体素。仅统计所选肿瘤 ROI 内的点。输入须为**训练病例**：

```bash
python -m preprocessing.learn_prior --cases /path/to/training_case_001 /path/to/training_case_002 --output outputs/prior.json
```

动脉期 HU 按 `clip((HU + 160) / 400, 0, 1)` 变换。每个有效标注点提取**动脉强度**、**3 × 3 × 3 邻域局部标准差**和 **3D Sobel 梯度幅值**。汇集训练病例中的点，按专家标签分别计算各坐标的中位数，形成三个中心，再按动脉强度排序。输出只记录汇总中心和标注计数。

发布的 100 例汇总先验已嵌入 `predict.PRIOR`，无需重新学习即可使用。所附演示病例不含专家点图，不参与先验学习。

### 2 · 构建三类受约束生境

使用内置汇总先验：

```bash
python -m preprocessing.cluster --case examples/case_001 --output outputs/habitats.nii.gz
```

或使用自己的训练病例学习所得先验：

```bash
python -m preprocessing.cluster --case examples/case_001 --prior outputs/prior.json --output outputs/habitats_custom_prior.nii.gz
```

每个肿瘤体素的五维向量为 **[A、局部 SD(A)、Sobel(A)、V、V − A]**，其中 A、V 均采用固定 −160 至 240 HU 窗并映射至 0–1。对每例肿瘤，动脉特征块（前三维）与静脉／差值特征块（后两维）分别通过 `RobustScaler` 进行中位数／IQR 缩放，移除常量列，再将每块除以其保留维数的平方根。

专家中心**仅初始化三个动脉维度**；其余维度在缩放空间中初始化为 0。受约束 K-means 固定 **K = 3**，每类至少包含 **ceil(肿瘤体素数 × 5%)** 个体素，随机种子 **42**，最多 **100** 次迭代，相对收敛容差 **1e−4**。最终标签 **1 → 3** 按动脉平均强度递增排列，背景为 0。

### 3 · 测量肿瘤内异质性

```bash
python -m preprocessing.heterogeneity --case examples/case_001 --habitats outputs/habitats.nii.gz --output outputs/thi_features.json
```

输出包含 **18 项核心特征**：

| 特征组 | 数量 | 测量内容 |
| :--- | ---: | :--- |
| 分布异质性 DDI | 9 | 两期各自的类间／类内对数比、中心分离度、Jensen–Shannon 散度、离散程度多样性，以及生境体积平衡熵 |
| 空间异质性 SHI | 4 | 中心离散度、径向 Wasserstein 距离、界面比例、非连通比例 |
| 动态强化异质性 DEHI | 5 | 强化均值分离度、Jensen–Shannon 散度、离散程度多样性、双期相关性多样性、廓清比例多样性 |

异质性计算使用**原始 HU 影像**。SHI 通过 NIfTI affine 使用物理坐标，并考虑体素体积和肿瘤等效半径；用于聚类的归一化影像不替代原始 HU。`--include-reference` 可额外输出 `dehi_global_mean`，它属于参考测量，不计入 18 项核心特征。

## 影像评分

**双期影像组学。** 提取器输出 **200 项特征：14 项共享形状 + 93 项动脉期 + 93 项静脉期**。采用 PyRadiomics `Original` 影像、整幅影像归一化并缩放至 **255**、灰度箱宽 **25**，以及 **1 × 1 × 1 mm** 重采样和 B-spline 影像插值。形状特征以 `original_shape_` 开头，双期特征分别以 `A_original_`、`V_original_` 开头。

```bash
python -m preprocessing.radiomics --case examples/case_001 --output outputs/radiomics_features.json
```

整幅影像归一化受输入视野影响，使用时应保持预定的源预处理方式。历史汇总 200 维表中的九个 `SurfaceVolumeRatio` 值与对应源表存在差异。因此，新提取结果记录来源信息，**不宣称逐值重现历史 RAD**。

**冻结公式。** 历史开发通过 ElasticNet（`alpha=0.3`、`l1_ratio=0.2`）将 **18 项 THI 候选特征收敛为 7 项活动特征**，通过 `alpha=0.1`、`l1_ratio=0.5` 将 **200 项影像组学候选特征收敛为 16 项活动特征**。推理系数、特征统计量和缩放常数统一保存在 `predict.SCORES`。

先按固定线性公式计算原始评分，再应用 `clip((raw − training_P1) / (training_P99 − training_P1), 0, 1)`。训练 P1/P99 来自固定特征集的五折折外评分构建；新病例使用全训练集拟合的推理公式与这些固定常数。

<sub>图示说明：原研究流程图保留分位数／min–max 示意标签。实际数值推理以 `SCORES` 中固定的训练集 P1/P99 变换为准。</sub>

## 分级推理

两个模型共用**一个 `predict.py` 和一个 `weights.joblib`**。`MODELS` 保存组件、权重、输入顺序、超参数和分级规则，`CLINICAL` 保存固定的临床变量变换。输入实际测得的特征或已归一化评分，以及真实临床测量：

```bash
python predict.py --model model8 --input /path/to/measured_features_or_scores.json --clinical /path/to/clinical.json --weights weights.joblib
python predict.py --model model9 --input /path/to/measured_features_or_scores.json --clinical /path/to/clinical.json --weights weights.joblib
```

输入 JSON 使用 `features` 对象保存“精确特征名 → 测量值”，或使用 `scores` 对象保存已截断至 0–1 的 `THI` 和 `RAD`。临床 JSON 保存“临床变量名 → 原始实测值”；这些匹配值也可直接放在输入 JSON 的 `clinical` 对象中。

| 模型 | 结构 | 所需完整临床输入 | 分级规则 |
| :--- | :--- | :--- | :--- |
| Model 8 | RBF-SVC / MLP / 多项式 SVM，权重 0.4 / 0.2 / 0.4 | LDH、AST、SFI | 平均类别概率最大者 |
| Model 9 · 探索性 | 分别对应 ≥G2 与 G3/NEC 的两个累积二分类集成 | LDH、AST、SFI、RIF%、size | 对 `p(G2) + 2p(G3/NEC)` 使用固定阈值 |

Model 9 在累积概率交叉时先取二者平均，再构建三类概率。风险下、上阈值分别为 **0.6772685594841208**、**0.9673987735976841**，恰好等于阈值时归入更高类别。

SFI 为体积／身高³ 指数（cm³/m³）；`RIF% = 100 × IFI / TFI`，size 单位为 mm。临床测量须保持原研究标度，按规定执行 `log1p` 及训练均值／标准差变换。历史拟合 MICE 对象不可用，因此发布版要求**完整临床输入**。所附影像病例不提供分级所需的临床值。

Model 8 使用内部调优队列完成选择；Model 9 的选择还使用了外部性能约束，因此保留明确的**探索性**定位。发布规格保留这两种选择角色。

## 训练

使用自己的特征／临床表重新拟合选定的固定规格：

```bash
python train.py --input /path/to/training_features.csv --fit-scores --output new_weights.joblib
```

默认拟合 Model 8。如需同时重新拟合探索性 Model 9：

```bash
python train.py --input /path/to/training_features.csv --fit-scores --models model8 model9 --output new_weights.joblib
```

| CSV 列 | 要求 |
| :--- | :--- |
| `Cohort` | 仅取值为 `training` 的行参与拟合 |
| `Grade_ordinal` | 0 = G1，1 = G2，2 = G3/NEC |
| `LDH`、`AST`、`SFI` | 完整的原始临床测量 |
| `RIF%`、`size` | Model 9 额外需要 |
| 七项 THI 与十六项 RAD 特征 | 使用 `SCORES[name]['terms']` 列出的精确原始特征名 |

五折分层要求**每个等级至少五例训练病例**。已有特征列中的缺失值采用训练折中位数，全缺失或常量特征被移除；临床测量必须完整。可通过 `python train.py --help` 中的选项修改队列列、等级列及训练标签。

**评分拟合。** 脚本用五折 `StratifiedKFold`（`shuffle=True`，种子 **20260720**）重新拟合锁定的 7／16 项特征集。各折仅在对应训练分区学习中位数插补、方差过滤、总体标准差标准化和 ElasticNet；评分方向根据该训练分区中与有序等级的 Spearman 相关确定，中心化／缩放也在该分区学习。训练折外评分定义 P1/P99，随后在全部训练行拟合面向新病例的最终公式。

**分级拟合。** 临床均值／标准差仅从提供的训练行学习，变换选择、模型组件、超参数、权重与分级阈值保持固定。新权重包保存自己的临床变换及评分公式，`predict.py` 优先使用这些值。输出标记为 **`locked_spec_refit_not_historical_frozen`**：锁定活动特征集的新拟合，与历史全部候选特征的开发拟合存在数值差异。脚本不进行模型搜索、阈值重新调优或性能实验。

不使用 `--fit-scores` 时，须提供 `THI`／`RAD`（或 `THI_score_01`／`RAD_score_01`），并提供 `THI_score_source`／`RAD_score_source`。接受 `out_of_fold`、`nested_oof`，或原有 `training_locked_7_feature_5fold_OOF_refit`／`training_locked_16_feature_5fold_OOF_refit` 来源标签。该路径要求所供评分与发布的推理公式及缩放一致；新数据集使用 `--fit-scores` 可保持新公式与评分配套。

公开训练入口重新拟合已选定规格，不重建历史临床筛选或后验 MICE 插补。固定活动特征与共享折外缩放不构成嵌套特征选择评估，也不产生新的性能声明。

<details>
<summary><b>论文原始模型开发流程</b></summary>

临床筛选仅使用训练队列：排除缺失率至少 25% 的变量，对偏态非负变量作 log1p 变换，按单变量比例优势模型的似然比检验与 BH-FDR q&lt;0.10 筛选，再以 |Spearman rho|≥0.85 剔除冗余。五个训练集拟合、采用后验采样的 Bayesian-Ridge 链式插补结果取均值，保留已观察值，排除验证结局信息。原拟合插补器未保存，因此公开入口要求完整临床输入。评分正则化在训练交叉验证中比较；Model 8 的非线性算法、变量组合、超参数及融合权重通过内部调优队列选择，Model 9 还使用外部性能约束。公开 `train.py` 重新拟合这些已锁定规格，不重新执行上述选择搜索。

</details>

## 仓库结构

```text
PNEN-THI/
├── README.md
├── README_zh.md
├── predict.py                 # MODELS · SCORES · CLINICAL · PRIOR；全部推理
├── train.py                   # 固定规格重新拟合的统一入口
├── weights.joblib             # 选定拟合状态，共用一个文件
├── requirements.txt
├── preprocessing/
│   ├── io.py                  # NIfTI 几何与原始 HU 读写
│   ├── learn_prior.py         # 步骤 1：汇总专家点中心
│   ├── cluster.py             # 步骤 2：三类受约束生境聚类
│   ├── heterogeneity.py       # 步骤 3：18 项 DDI / SHI / DEHI 特征
│   ├── radiomics.py           # 200 项双期 Original 特征
│   └── requirements-radiomics.txt
├── assets/
│   └── study_workflow.png
└── examples/
    └── case_001/
        ├── arterial.nii.gz
        ├── venous.nii.gz
        ├── tumor_roi.nii.gz
        └── case.json
```

示例使用中性文件名及清理后的 NIfTI 头，保留体素值与空间几何信息。**肿瘤标签为 2**，ROI 还包含非目标标签 1。输入应处于相同的 **1-mm 各向同性网格**，具有相同尺寸与 affine，并在推理前核对解剖配准；网格一致本身不代表配准完成。

共享权重文件不含队列表或研究运行输出。拟合 SVC 仍需保留支持向量，kNN 仍需保留参考向量／类别，均属于模型状态。与源模型实现进行的私有迁移核对，其绝对误差 **≤1e−12**；该核对验证推理迁移，不使用发布示例估计性能。

## 使用与引用

仅供**学术及非商业研究**使用。临床或商业授权请联系作者。

正式稿件引用信息提供前，请引用 [PNEN-THI 仓库](https://github.com/BobbyWang16/PNEN-THI)。本发布包不包含稿件或完整研究队列。
