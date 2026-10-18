<div align="center">

# PNEN-THI

**基于双期 CT 生境的术前三分类分级**

专家先验初始化生境 · 可解释异质性 · 冻结模型推理

[English](README.md) &nbsp; · &nbsp; [简体中文](README_zh.md)

[快速开始](#快速开始) &nbsp; / &nbsp; [输入数据](#输入数据) &nbsp; / &nbsp; [核心方法](#核心方法) &nbsp; / &nbsp; [影像评分](#影像评分) &nbsp; / &nbsp; [分级推理](#分级推理) &nbsp; / &nbsp; [训练](#训练)

</div>

<a href="assets/study_workflow.png"><img src="assets/study_workflow.png" alt="研究流程：双期 CT、专家先验初始化的肿瘤生境、异质性与影像组学评分，以及胰腺神经内分泌肿瘤三分类分级" width="100%"></a>

**从专家点到肿瘤分级。** PNET-ITH 将动脉期与静脉期 CT、三类肿瘤生境及可解释影像评分连接起来，实现胰腺神经内分泌肿瘤的 G1 / G2 / G3/NEC 分级。

## 概览

**`predict.py` 统一推理，`train.py` 重新拟合已选定的模型规格。** 图像准备、点先验学习、受约束聚类和特征计算集中在 `preprocessing/`。发布包包含已学习的汇总先验、选定的拟合模型和一例去标识真实影像。该病例用于演示影像分支，不附临床测量、结局或专家点标注。

## 环境配置

包含 PyRadiomics 的完整流程使用 **Python 3.9**：

```bash
python -m pip install -r preprocessing/requirements-radiomics.txt
```

仅计算 THI、从已有特征推理，或训练时无需提取影像组学，可安装基础依赖；该分支也支持 **Python 3.12**：

```bash
python -m pip install -r requirements.txt
```

完整提取已在 Python 3.9.25、PyRadiomics 3.1.0、NumPy 1.26.4、SciPy 1.13.1、SimpleITK 2.5.5 下验证。模型加载与训练使用 scikit-learn 1.6.1。以下命令均从仓库根目录运行。

## 输入数据

同一病例在各目录使用相同的去标识文件名：

```text
data/
├── A/case_001.nii.gz       # 保留原始 HU 的动脉期 CT，作为固定参考
├── V/                     # 配准前的原始静脉期 CT
├── V2A/case_001.nii.gz     # V 配准并重采样到 A
├── ROI/case_001.nii.gz     # A 网格上的分割：0 背景，2 肿瘤
└── DOT/                   # A 网格上的稀疏专家点：0 未标注，1/2/3 类别
```

`A、V2A、ROI` 须具有相同的 **1-mm 各向同性网格**、尺寸与 affine，并同时核对解剖配准。ROI 可含非目标标签 1。`DOT` 仅在学习新先验时需要。所附病例提供 `A、V2A、ROI`；其他患者影像默认不会被 Git 收录。

DICOM 转换、分割、配准、点标注与完整计算步骤见[数据准备与完整流程指南](data/README_zh.md)。

## 快速开始

校验并整理真实病例，然后计算生境、全部 18 项异质性特征和 THI：

```bash
python -m preprocessing.prepare --data-root data --case-id case_001 --output-dir outputs/cases/case_001
python predict.py --thi-only --case outputs/cases/case_001 --output-dir outputs/demo
```

准备脚本将影像复制为 `arterial.nii.gz`、`venous.nii.gz`、`tumor_roi.nii.gz`，保留数值与几何信息；可选的 `DOT` 复制为 `points.nii.gz`，已有目标不会被覆盖。推理默认加载 **[`point_prior_n100.pkl`](point_prior_n100.pkl)**，输出 `habitats.nii.gz`、`heterogeneity_features.json` 和 `result.json`。

安装完整影像组学环境后，可同时计算 THI 和 RAD，并保存 `radiomics_features.json`：

```bash
python predict.py --score-only --case outputs/cases/case_001 --output-dir outputs/demo_scores
```

## 核心方法

### 1 · 点驱动先验学习

先验学习仅使用 **A、ROI 和 DOT**。每例训练病例需在相同网格准备 `arterial.nii.gz`、`tumor_roi.nii.gz`、`points.nii.gz`。汇总训练标注须覆盖专家类别 **1/2/3**；0 表示未标注，仅统计肿瘤内部的点。

动脉期 HU 按 `clip((HU + 160) / 400, 0, 1)` 变换，在各点提取**动脉强度、3 × 3 × 3 邻域局部标准差、3D Sobel 梯度幅值**。按专家类别汇集训练点，各坐标取中位数形成三个中心，再按动脉强度排序。

```bash
python -m preprocessing.learn_prior --cases /path/to/training_case_001 /path/to/training_case_002 --output outputs/point_prior.pkl
```

发布的 **`point_prior_n100.pkl`** 保留原项目 schema 2 的 pickle 字典：`cluster_centroids` 将标签 1/2/3 映射到三个 NumPy 特征数组，`provenance` 记录汇总计数。它汇集 **100 例训练病例的 3,390 个专家点**，不含患者标识。影像推理默认调用该文件，无需 DOT 或重新学习。只加载来源可信的 pickle 文件。

### 2 · 先验驱动的受约束聚类

每个肿瘤体素的五维向量为 **[A、局部 SD(A)、Sobel(A)、V、V − A]**，A、V 均由固定 −160 至 240 HU 窗映射至 0–1。每例肿瘤内，动脉块（前三维）与静脉／差值块（后两维）分别通过 `RobustScaler` 进行中位数／IQR 缩放，移除常量列，再将每块除以其保留维数的平方根。

先验**仅初始化动脉维度**，其余坐标在缩放空间中初始化为 0。受约束 K-means 固定 **K = 3**，每类至少 **ceil(肿瘤体素数 × 5%)** 个体素，种子 42，最多 100 次迭代，相对容差 1e−4。最终标签 **1 → 3** 按动脉平均强度递增排列，背景为 0。

```bash
python -m preprocessing.cluster --case outputs/cases/case_001 --prior point_prior_n100.pkl --output outputs/habitats.nii.gz
```

在 `preprocessing.cluster` 或需要计算 THI 的 `predict.py --case` 命令中，使用 **`--prior outputs/point_prior.pkl`** 可指定自己学习的先验；省略该选项时使用发布文件。

### 3 · 异质性特征计算

```bash
python -m preprocessing.heterogeneity --case outputs/cases/case_001 --habitats outputs/habitats.nii.gz --output outputs/thi_features.json
```

| 特征组 | 数量 | 测量内容 |
| :--- | ---: | :--- |
| 分布异质性 DDI | 9 | 两期各自的类间／类内对数比、中心分离度、Jensen–Shannon 散度、离散程度多样性；生境体积平衡熵 |
| 空间异质性 SHI | 4 | 中心离散度、径向 Wasserstein 距离、界面比例、非连通比例 |
| 动态强化异质性 DEHI | 5 | 强化均值分离度、Jensen–Shannon 散度、离散程度多样性、双期相关性多样性、廓清比例多样性 |

这 **18 项特征使用原始 HU 影像**。空间测量使用 affine 所定义的物理坐标、体素体积和肿瘤等效半径。`--include-reference` 可额外输出核心特征集之外的 `dehi_global_mean`。

## 影像评分

**双期影像组学：200 项 = 14 项共享形状 + 93 项动脉期 + 93 项静脉期。** PyRadiomics 使用 `Original` 影像、整幅影像归一化并缩放至 255、灰度箱宽 25、1 × 1 × 1 mm 重采样和 B-spline 影像插值。特征前缀分别为 `original_shape_`、`A_original_`、`V_original_`。输入视野会影响整幅归一化，应保持预定的源预处理方式。

```bash
python -m preprocessing.radiomics --case outputs/cases/case_001 --output outputs/radiomics_features.json
```

**THI 与 RAD：** `predict.SCORES` 中的固定线性公式分别使用 **7 项异质性特征**和 **16 项组学特征**。开发时 ElasticNet 的 `(alpha, l1_ratio)` 分别为 THI `(0.3, 0.2)`、RAD `(0.1, 0.5)`。原始评分通过 `clip((raw − training_P1) / (training_P99 − training_P1), 0, 1)` 映射至 0–1；P1/P99 来自训练集五折折外评分，新病例使用全训练集公式与这些固定常数。

<sub>原流程图保留分位数／min–max 示意标签；实际数值推理以 `SCORES` 中的训练集 P1/P99 变换为准。</sub>

## 分级推理

九个模型共用**一个 `predict.py`**。`MODELS` 定义输入与分级规则，`CLINICAL` 定义临床变换。**Model 1–7 为无惩罚的比例优势 Logistic 模型**，学习系数及两个有序截点，取三类概率中的最大者作为分级。

| 模型 | 临床输入 | 影像评分 | 结构／分级规则 |
| :--- | :--- | :--- | :--- |
| Model 1 | size、SFI、TFI、LDH、AST、IFI、RIF% | 无 | 比例优势模型／最大类别概率 |
| Model 2 | 无 | RAD | 比例优势模型／最大类别概率 |
| Model 3 | 无 | THI | 比例优势模型／最大类别概率 |
| Model 4 | size、SFI、TFI、LDH、AST、IFI、RIF% | RAD | 比例优势模型／最大类别概率 |
| Model 5 | size、SFI、TFI、LDH、AST、IFI、RIF% | THI | 比例优势模型／最大类别概率 |
| Model 6 | 无 | RAD、THI | 比例优势模型／最大类别概率 |
| Model 7 | SFI、TFI、IFI | RAD、THI | 比例优势模型／最大类别概率 |
| Model 8 | LDH、AST、SFI | RAD、THI | RBF-SVC／MLP／多项式 SVM，权重 0.4／0.2／0.4；最大平均概率 |
| Model 9 | LDH、AST、SFI、RIF%、size | RAD、THI | 两个累积二分类集成／固定风险阈值 |

**Model 1–6** 使用内置拟合参数；**Model 7** 需要 `train.py` 生成的权重包；**Model 8、9** 使用所附 `weights.joblib`。通过 `--weights` 传入新权重包，即调用其中拟合的参数、临床变换及评分公式。

```bash
python predict.py --model model1 --input /path/to/clinical_only.json
python predict.py --model model3 --case outputs/cases/case_001 --output-dir outputs/model3
python predict.py --model model7 --input /path/to/measured_features_or_scores.json --clinical /path/to/clinical.json --weights models1_7.joblib
python predict.py --model model8 --input /path/to/measured_features_or_scores.json --clinical /path/to/clinical.json --weights weights.joblib
```

仅需提供所选模型要求的输入。影像模型可用 `--case` 替代 `--input`，需要临床变量时还须提供 `--clinical`。Model 3、5 计算 THI 时无需组学。使用 `--list-models` 或 `--describe model7` 查看模型规格。

<details>
<summary><b>输入 JSON、临床单位与分级规则</b></summary>

- 输入 JSON 使用 `features` 保存“精确特征名 → 测量值”，或使用 `scores` 将所需 `THI`／`RAD` 映射到 **0–1 数值**，不使用嵌套的结果对象。
- 临床 JSON 保存“所需变量名 → **完整原始实测值**”，也可放在输入 JSON 的 `clinical` 对象中。Model 1 只需该对象。
- SFI、TFI、IFI 为体积／身高³ 指数（cm³/m³）；`RIF% = 100 × IFI / TFI`；size 单位为 mm。保持研究测量标度，模型按既定规则执行 `log1p` 和训练均值／标准差变换。
- Model 9 在累积概率交叉时先取均值，再构建类别概率。风险阈值为 **0.6772685594841208**、**0.9673987735976841**；等于阈值时归入更高类别。

</details>

## 训练

一次重新拟合 **Model 1–7**，并拟合这些模型需要的影像评分公式：

```bash
python train.py --input /path/to/training_features.csv --models model1 model2 model3 model4 model5 model6 model7 --fit-scores --output models1_7.joblib
```

训练单个模型时，选择其名称，仅提供对应输入：

```bash
python train.py --input /path/to/training_features.csv --models model7 --fit-scores --output model7.joblib
```

| 参数 | 用法 |
| :--- | :--- |
| `--models` | 单个名称、多个名称，或 `all` 选择 Model 1–9；默认 `model8` |
| `--fit-scores` | 从锁定的 7／16 项原始特征重新拟合所需 THI／RAD 公式 |
| `--input` / `--output` | 训练 CSV／一个新共享 `.joblib` 权重包；不覆盖已有文件 |
| `--cohort-column` / `--training-label` | 默认 `Cohort`／`training`，仅拟合匹配行 |
| `--grade-column` | 默认 `Grade_ordinal`：0 = G1，1 = G2，2 = G3/NEC |

训练行须包含**全部三个等级**及**完整的所需临床测量**。拟合评分时**每个等级至少五例训练病例**，原始特征列名须与 `SCORES[name]['terms']` 完全对应。Model 1 仅需临床列，无需 `--fit-scores`；Model 2、3 分别拟合 RAD、THI。更多选项见 `python train.py --help`。

**拟合内容。** Model 1–7 最大化比例优势似然；Model 7 的输入为 **[SFI、TFI、IFI、RAD、THI]**。Model 8、9 重新拟合固定组件，保留集成权重与分级规则。临床统计量仅从训练行学习。所有选定模型、临床变换和评分公式统一保存，由 `predict.py --weights` 调用。

<details>
<summary><b>评分拟合、CSV 来源与开发细节</b></summary>

使用 `--fit-scores` 时，五折 `StratifiedKFold`（`shuffle=True`，种子 20260720）重新拟合锁定特征集。各训练折学习中位数插补、方差过滤、总体标准差标准化及 ElasticNet；全缺失或常量特征被移除。评分方向依据训练分区中与等级的 Spearman 相关确定，中心化／缩放在同折学习。折外评分定义 P1/P99，随后拟合全训练集推理公式。

不使用 `--fit-scores` 时，以 `THI`／`RAD`（或 `THI_score_01`／`RAD_score_01`）提供所需 **0–1 数值折外评分**，并提供 `THI_score_source`／`RAD_score_source`。接受 `out_of_fold`、`nested_oof`，或 `training_locked_7_feature_5fold_OOF_refit`／`training_locked_16_feature_5fold_OOF_refit`。这些评分须与发布公式和缩放一致；新数据集使用 `--fit-scores` 可保持新公式与评分配套。

Model 1–7 使用 SciPy L-BFGS-B（`maxiter=2000`、`ftol=1e−11`、`gtol=1e−7`）。原始临床开发排除缺失率至少 25% 的变量，对偏态非负变量作 `log1p`，按比例优势似然比检验 BH-FDR q&lt;0.10 筛选，以 |Spearman rho|≥0.85 剔除冗余。五个训练集拟合、采用后验采样的 Bayesian-Ridge 链式插补结果取均值，保留已观察值，排除验证结局。原拟合插补器不可用，因此此入口要求完整临床输入。

评分正则化在训练交叉验证中比较；Model 8 的算法、变量组合、超参数与权重通过内部调优选择。`train.py` 重新拟合已选定规格，不重复这些选择搜索，也不构成新的性能评估。

</details>

## 仓库结构

```text
PNEN-THI/
├── predict.py / train.py       # 统一推理／固定规格训练
├── weights.joblib             # 共享拟合模型包
├── point_prior_n100.pkl       # 原始格式的动脉期学习先验
├── requirements.txt
├── preprocessing/             # prepare · learn_prior · cluster · heterogeneity · radiomics
├── data/                      # 真实病例与中英文准备指南
└── assets/study_workflow.png
```

真实示例采用中性文件名及清理后的 NIfTI 头，保留体素值与空间几何信息。

## 使用与引用

仅供**学术及非商业研究**使用。临床或商业授权请联系作者。

正式稿件引用信息提供前，请引用 [PNEN-THI 仓库](https://github.com/BobbyWang16/PNEN-THI)。本发布包不包含稿件或完整研究队列。
