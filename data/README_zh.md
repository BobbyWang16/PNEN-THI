# 输入数据与影像处理流程

[English](README.md) · [简体中文](README_zh.md) · [项目说明](../README_zh.md)

输入按**影像时相和标注类型**分文件夹保存，同一病例在不同文件夹中使用相同的匿名编号。发布的 `case_001` 包含真实的预处理动脉期、已配准到动脉期的静脉期和肿瘤 ROI，可直接用于生境与特征计算。

## 文件夹结构

```text
data/
├── A/
│   └── case_001.nii.gz       # 动脉期参考图像，保留 HU，1 mm 网格
├── V/
│   └── .gitkeep              # 放置自己的原始静脉期图像
├── V2A/
│   └── case_001.nii.gz       # 配准并重采样到 A 的静脉期
├── ROI/
│   └── case_001.nii.gz       # A 网格上的共识肿瘤掩膜
├── DOT/
│   └── .gitkeep              # 可选，学习先验使用的专家点图
├── case_001.json             # 示例的几何和标签说明
├── README.md
└── README_zh.md
```

新病例也使用一致的文件名，例如分别放入各目录的 `case_002.nii.gz`。

| 文件夹 | 保存内容 | 空间与数值要求 | 用途 |
| :--- | :--- | :--- | :--- |
| `A` | 预处理后的动脉期 CT | 参考空间；1 × 1 × 1 mm；保留 HU | 学习先验、聚类、提取特征 |
| `V` | 尚未配准到 A 的原始静脉期 CT | 可以与 A 的原始网格不同；保留 HU | 上游配准 |
| `V2A` | 变换到动脉期空间的静脉期 CT | 与 A 的形状、间距和仿射矩阵一致；保留 HU | 聚类和双期特征计算 |
| `ROI` | 经复核的完整肿瘤标签图 | 与 A 同网格；**2 为肿瘤，0 为背景** | 限定分析的肿瘤范围 |
| `DOT` | 稀疏专家点标签图 | 与 A 同网格；**0 为未标注，1/2/3 为低/中/高灌注点** | 可选，仅用训练集学习先验 |

`V2A` 表示**静脉期配准到动脉期**，不是增强差值 `V − A`。计算方法中写的 V，指的是 `V2A` 中已配准的静脉期。ROI 和 DOT 的数字标签相互独立：ROI 标签 2 内的 DOT 标签 1，表示肿瘤内一个有效的低灌注点。点的表型类别不代表病理分级。

示例 ROI 还包含非目标标签 1，默认计算会排除它。使用统一整理和预测入口时，请将自己的肿瘤掩膜保存为标签 2；如果标签图还含胰腺等其他结构，不要把所有正值都改成肿瘤。独立的方法模块另外支持用 `--tumor-label 1` 显式处理真正的二值掩膜。

发布示例的 `V` 和 `DOT` 目录有意留空：没有提供它的原始未配准静脉期和专家点标注。直接使用内置先验计算影像特征时，不需要这两类文件；程序不会自动编造点标注。

## 1 预处理与肿瘤分割

原始扫描和中间图像单独保留。最终放入 `A`、`V2A`、`ROI` 和可选 `DOT` 的文件应处于**解剖配准一致的三维网格**，不能只看数组大小相同。

从 DICOM 开始时，可用 [Slicer DICOM 模块](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html) 分别加载动脉期和静脉期序列，再将三维体数据保存为 `.nii.gz`，使用去标识病例编号命名。保留 HU 与空间几何信息，使用体数据，不使用截图或彩色渲染图。

稿件规定的上游处理流程如下：

1. 使用在 PANORAMA 上训练的 nnU-Net，分别获得双期的胰腺和肿瘤初始掩膜，并进行复核。
2. CT 用 **B-spline 插值**重采样到 **1 mm 各向同性**；掩膜用**最近邻插值**。
3. 用双期胰腺与肿瘤掩膜的三维并集确定个体化视野，各方向向外扩展 **20 mm**；双期按对应的物理视野裁剪。始终保留原始 HU。
4. 完成静脉期到动脉期的配准后，复核和修正动脉期肿瘤边界，并对照双期图像确认，形成动脉期空间的最终共识 ROI。

本仓库未打包初始分割模型。可以接入自己已有的分割流程，或手动勾画并复核。使用 [3D Slicer Segmentations](https://slicer.readthedocs.io/en/5.4/user_guide/modules/segmentations.html) 时，将 A 设为参考几何，在三个方向复核肿瘤，并导出到 A 网格的标签图；也可使用 ITK-SNAP 进行手动标注与修正。

**A 可以已经裁剪完成。** 发布示例是可直接分析的局部视野，无需再次裁剪，也不要把 HU 替换成窗宽窗位处理后的显示灰度。下面的整理命令只负责校验和打包已有输入，不会执行分割、配准或二次裁剪。

## 2 将 V 配准到 A 并保存为 V2A

将动脉期设为 **Fixed 参考图像**，静脉期设为 **Moving 待变换图像**。稿件使用三维 Euler 刚性配准，随后进行仿射配准：

| 设置 | 数值 |
| :--- | :--- |
| 相似性度量 | Mattes 互信息，32 个直方图箱，全体素采样 |
| 多分辨率降采样因子 | 4、2、1 |
| 平滑 sigma | 2、1、0 mm |
| 优化器 | 梯度下降，学习率 1.0 |
| 最大迭代次数 | 每阶段 200 次 |
| 收敛设置 | 最小变化 1e−6；窗口 10 次迭代 |
| 最终静脉期重采样 | **线性插值，输出到 A 的参考网格** |

初始的各向同性重采样与配准后的最终重采样是两步操作，插值设置不同。变换标签图时使用最近邻插值。

也可使用交互式 [Slicer BRAINSFit](https://slicer.readthedocs.io/en/latest/user_guide/modules/brainsfit.html)：选择 Fixed = A、Moving = V，完成刚性/仿射对齐，将重采样图像保存到 `V2A`。具体参数需在该工具中设置；这是一种手动处理途径，并非上述稿件参数的程序实现。输出必须落到 **A 的固定参考网格**；仅给图像附加或固化一个变换，并不代表存储网格已经一致。

在轴位、冠状位和矢状位叠加查看肿瘤及周围血管，必要时修正呼吸或肠蠕动导致的残余错位。仍存在明显错位的病例不适合逐体素双期计算。完成此项核查后再确定最终 ROI。

## 3 整理为分析病例

以下命令均在仓库根目录运行。发布的真实示例可直接整理：

```bash
python -m preprocessing.prepare --data-root data --case-id case_001 --output-dir outputs/cases/case_001
```

程序校验三维几何、1 mm 间距、仿射矩阵一致性、CT 数值有效性和肿瘤标签 2，然后生成：

```text
outputs/cases/case_001/
├── arterial.nii.gz           # 来自 data/A/case_001.nii.gz
├── venous.nii.gz             # 来自 data/V2A/case_001.nii.gz
├── tumor_roi.nii.gz          # 来自 data/ROI/case_001.nii.gz
└── points.nii.gz             # 仅在存在有效的同名 DOT 时生成副本
```

整理后的 `venous.nii.gz` 始终取自 **V2A**，不会读取未配准的 `V`。同名 DOT 存在时自动校验并复制；没有 DOT 就不生成点图。新增患者图像应留在本地，默认版本控制只允许提交已整理的公开示例。

脚本不覆盖已有目标文件。重复整理，或给已整理的病例补加 DOT 后，请选择新的输出目录。

## 4 标记专家点并生成先验文件

只有在使用**自己的训练病例**重新学习先验时才需要这一步。常规推理直接调用 `predict.PRIOR` 中发布的 100 例聚合先验，无需 DOT。

为每例训练图像建立与 A 几何完全一致的稀疏整数标签图。在 ITK-SNAP 或自己的标注工具中，选择肿瘤内部具有代表性的**单体素点**：1 为低灌注、2 为中灌注、3 为高灌注，其余位置保持 0。将点图叠加到 A 和共识 ROI 上复核，保存为 `data/DOT/<case_id>.nii.gz`。不同病例的类别含义必须保持一致。聚类中心或预测生境图不能代替专家点标注。

使用同一整理命令，为每例带 DOT 的训练病例生成分析目录；随后将实际目录传给先验学习模块：

```bash
python -m preprocessing.learn_prior --cases outputs/cases/your_training_case_001 outputs/cases/your_training_case_002 --output outputs/prior.json
```

将命令中的训练病例名替换为自己的已整理目录，这些训练病例不包含在发布示例中。仅位于选定肿瘤内部的点参与学习，全部训练病例合计须包含三类点。程序提取归一化动脉期强度、3 × 3 × 3 局部标准差和三维 Sobel 梯度幅值，汇总各类有效点后取逐坐标中位数，再按动脉期强度排列三个中心。`prior.json` 保存聚合中心和点数统计；不使用验证病例或病理分级。

## 5 聚类生成三类生境

使用发布的内置先验：

```bash
python -m preprocessing.cluster --case outputs/cases/case_001 --output outputs/case_001/habitats.nii.gz
```

或者调用自己的训练先验：

```bash
python -m preprocessing.cluster --case outputs/cases/case_001 --prior outputs/prior.json --output outputs/case_001/habitats.nii.gz
```

方法使用 `[A, local SD(A), Sobel(A), V, V − A]`，进行固定 HU 归一化和病例内分块缩放，先验用于初始化动脉期特征维度。约束 K-means 生成**三类生境**，每类至少占肿瘤的 **5%**。输出 0 为背景，1、2、3 按动脉期平均强度从低到高排列。这是计算得到的生境标签，与 ROI 和 DOT 标注分别保存。

## 6 计算特征与影像评分

提取 **18 个核心异质性特征**，包括 9 个 DDI、4 个 SHI 和 5 个 DEHI：

```bash
python -m preprocessing.heterogeneity --case outputs/cases/case_001 --habitats outputs/case_001/habitats.nii.gz --output outputs/case_001/heterogeneity_features.json
```

安装完整影像组学环境后，可提取 **200 个双期全肿瘤组学特征**：14 个共享形状特征、93 个动脉期特征和 93 个配准静脉期特征。

```bash
python -m preprocessing.radiomics --case outputs/cases/case_001 --output outputs/case_001/radiomics_features.json
```

也可用统一预测脚本一次完成影像流程，它会在内部进行聚类和特征提取：

```bash
python predict.py --thi-only --case outputs/cases/case_001 --output-dir outputs/demo_thi
python predict.py --score-only --case outputs/cases/case_001 --output-dir outputs/demo_scores
```

第一条只需基础依赖；第二条还需安装 `preprocessing/requirements-radiomics.txt`。两者均保存生境图、实测特征和 `result.json`，均无需临床值。三分类分级预测还需提供自己的真实、完整临床输入，详见[项目说明](../README_zh.md#分级推理)；特征表和固定方案训练见其中的[训练部分](../README_zh.md#训练)。
