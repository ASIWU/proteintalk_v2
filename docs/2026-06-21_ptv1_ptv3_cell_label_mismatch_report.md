# PTV1-flow 与 PTV3 cell_5fold 标签不一致原因报告

时间：2026-06-21 13:42 HKT

## 结论摘要

这次 PTV1-flow 与原始 PTV3 在 `cell_5fold` 上的 AUPRC 差异，主要不是单纯的模型结构问题，而是评估目标的标签命名空间不一致。两套流程虽然可以映射到同一个实验 key，即 `(Cell, pert_id1, pert_id2)`，但 PTV1-flow 使用的是 legacy benchmark 的 `pheno.csv` 数值标签，原始 PTV3 使用的是标准化 training-ready 表里的 `PRISM1st_label_total` 字符串标签。二者不是同一份标签。

在 5 个 `cell_5fold` test fold 的共同 key 上，PTV1 原始 0/1 标签与原始 PTV3 `PRISM1st_label_total` 标签共有 `3673` 个可比较样本，其中 `469` 个标签不一致，整体 mismatch rate 为 `12.77%`。这个比例足以显著改变 AUPRC，尤其是正类比例不同、且不一致方向不对称时。

标签对齐后，PTV3 的最终 label-aligned 实验 `20260620_ptv3_ptv1label_cell_v1_covdrop020_mse075` 在同一 PTV1 标签命名空间下达到 mean AP/AUPRC `0.862943`，超过 PTV1-flow 的 corrected mean AP/AUPRC `0.860341`。这说明之前的主要 gap 可以由标签不一致解释，而不是 PTV3 必然无法达到 PTV1-flow 水平。

## 涉及的数据与代码路径

PTV1-flow benchmark 标签来自：

- `baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified/cell_5fold/fold*/{train,valid,test}/cell_5fold_fold*_pheno.csv`
- `baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified/cell_5fold/fold*/{train,valid,test}/cell_5fold_fold*_loo_label.csv`
- 读取逻辑：`baseline/ptv2_benchmark_260514/PTV1_model_20260428/dataset.py`
- 评估逻辑：`baseline/ptv2_benchmark_260514/PTV1_model_20260428/main.py`

原始 PTV3 标签来自：

- `data/training_ready/ptv3/tasks/ptv3_main_singledrug/feature_table.parquet`
- 标签列：`PRISM1st_label_total`
- PTV3 response label 编码逻辑：`dataset/training_ready_fast_dataset.py`

本次对齐和验证用到的新证据文件：

- 原始 PTV1 vs 原始 PTV3 标签逐 key 对照：`outputs/2026-06/2026-06-21/20260621_ptv1_ptv3_original_label_mismatch_cell_5fold.csv`
- label-aligned PTV3 vs PTV1-flow 5-fold 结果：`outputs/2026-06/2026-06-20/20260620_ptv3_ptv1label_cell_v1_covdrop020_mse075_vs_ptv1_5fold.csv`
- PTV1 标签对齐任务构建脚本：`scripts/build_ptv3_ptv1_cellfold_task.py`
- 修正后的 PTV1/PTV3 key-level 对比脚本：`scripts/evaluate_ptv3_vs_ptv1_cellfold.py`

## PTV1-flow 标签是如何产生和使用的

PTV1-flow 的 `cell_5fold` 数据不是直接使用 PTV3 的 `PRISM1st_label_total`。它的 benchmark split 中，每个 fold 的 `loo_label.csv` 记录实验 key 和 time point，`pheno.csv` 记录 phenotype 标签。

关键行为如下：

1. `dataset.py` 读取 `loo_label.csv` 和 `pheno.csv`，并对 `pheno.csv` 执行 `fillna(-1)`。
2. PTV1 样本按 experiment key 聚合，key 形式是类似 `Cell#pert_id1#pert_id2`。
3. PTV1 要求同一个 key 同时有 `6` 和 `24` 两个 time point，才构成一个可用的 expression trajectory 样本。
4. 对每个 key，PTV1 使用该 key 第一条记录对应的 `pheno` 作为 phenotype 标签。
5. 标签语义是：
   - `1`：正类
   - `0`：负类
   - `-1`：缺失/未知/不可监督标签
6. 评估时，`main.py` 用 `valid_mask = (all_labels != -1)` 过滤掉 `-1` 标签。也就是说，`-1` 不应被当作负类。

这个行为直接影响评估样本数。5 个 test fold 中，PTV1 原始 test split 里有 `8756` 个具备 6h/24h 的 experiment key，但其中 `5083` 个是 `-1`，真正进入 AUPRC/AUROC 的 0/1 样本只有 `3673` 个。

| fold | 具备 6h/24h 的 key | `-1` masked | 有效 0/1 | 正类 1 | 负类 0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 671 | 166 | 505 | 187 | 318 |
| 1 | 2530 | 1522 | 1008 | 249 | 759 |
| 2 | 647 | 175 | 472 | 133 | 339 |
| 3 | 2569 | 1504 | 1065 | 225 | 840 |
| 4 | 2339 | 1716 | 623 | 133 | 490 |
| total | 8756 | 5083 | 3673 | 927 | 2746 |

## 原始 PTV3 标签是如何定义的

原始 PTV3 `ptv3_main_singledrug` 使用 training-ready 标准化表中的 `PRISM1st_label_total` 作为 single-drug response head 的标签。该列是字符串标签，主要取值是：

- `sensitive`
- `non-responsive`

PTV3 的 response encoder 会把这些字符串转成二分类训练目标。与 PTV1-flow 不同，原始 PTV3 任务的 stage-2 过滤规则会保留非 control 且有非空 `PRISM1st_label_total` 的 single-drug 行，因此它的标签来源和缺失处理都不同于 PTV1 的 `pheno.csv`。

这意味着：即使 PTV1 和 PTV3 指向同一个 `(Cell, pert_id1, pert_id2)`，二者也可能分别给出不同的 response label。这个差异不是模型训练时自然能消除的噪声，而是 benchmark target 本身不一样。

## 实测标签不一致情况

我用同一个 experiment key 对齐 PTV1 test fold 的有效 0/1 标签和原始 PTV3 的 `PRISM1st_label_total` 标签。对齐口径为：

- key：`(Cell, pert_id1, pert_id2)`
- PTV1：只统计 `pheno != -1` 的 key
- PTV3：把 `sensitive` 编码为 `1`，`non-responsive` 编码为 `0`
- 对 PTV3 多 time point 行，按 key 检查标签一致性；本次共同 key 上没有发现 PTV3 内部重复 key 标签冲突

逐 fold 结果如下：

| fold | common key 数 | PTV1 正类数 | PTV3 正类数 | mismatch 数 | mismatch rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 505 | 187 | 171 | 70 | 13.86% |
| 1 | 1008 | 249 | 208 | 129 | 12.80% |
| 2 | 472 | 133 | 96 | 79 | 16.74% |
| 3 | 1065 | 225 | 194 | 123 | 11.55% |
| 4 | 623 | 133 | 99 | 68 | 10.91% |
| total | 3673 | 927 | 768 | 469 | 12.77% |

整体 confusion table 如下，行是 PTV1 标签，列是原始 PTV3 标签：

| PTV1 label | PTV3 label 0 | PTV3 label 1 |
|---:|---:|---:|
| 0 | 2591 | 155 |
| 1 | 314 | 613 |

这个表说明两点：

1. 两套标签不是随机少量差错，而是有系统性差异。共同 key 上 `469/3673` 个标签不一致。
2. 不一致方向不对称。PTV1 正类但 PTV3 负类有 `314` 个，PTV1 负类但 PTV3 正类有 `155` 个。因此在这批 key 上，原始 PTV3 的正类定义更少，正类数是 `768`，而 PTV1 是 `927`。

## 为什么这会显著影响 AUPRC

AUPRC 对正类定义和正类比例非常敏感。这里有三个直接影响：

1. 正类集合不同。模型如果按 PTV3 `PRISM1st_label_total` 学到一个排序，在 PTV1 `pheno.csv` 标签上评估时，会把一部分 PTV1 正类、PTV3 负类的样本排低；同时会把一部分 PTV1 负类、PTV3 正类的样本排高。这两类都会损害 PTV1-label AUPRC。
2. base rate 不同。共同 key 上 PTV1 正类率是 `927/3673 = 25.24%`，原始 PTV3 正类率是 `768/3673 = 20.91%`。AUPRC 的 baseline 和排序压力都会随正类率变化。
3. fold 间差异不同。fold2 mismatch rate 最高，为 `16.74%`。这类 fold 会更容易出现“模型在自己标签上表现合理，但在另一套标签上被严重惩罚”的现象。

因此，直接比较“PTV1-flow 在 PTV1 标签上的 AUPRC”和“原始 PTV3 在 PTV3 标签或混合 key 标签上的 AUPRC”并不公平。它们回答的是两个不同问题。

## 另一个独立问题：PTV1 `predictions.csv` 的 key 不能直接用于对齐

除了真实标签不一致，还有一个分析陷阱：PTV1-flow 输出的 `predictions.csv` 里虽然有 `experiment_type`，但这个列在 `-1` 标签过滤后并不是可靠的逐样本 key。

PTV1 `main.py` 的逻辑是先收集 `all_predictions` 和 `all_labels`，然后用：

- `valid_mask = (all_labels != -1)`
- `all_labels = all_labels[valid_mask]`
- `all_predictions = all_predictions[valid_mask]`

过滤掉 `-1`。但是后面生成输出 CSV 时，`experiment_type` 不是按同一个 `valid_mask` 同步过滤，而是用 `valid_experiment_types[:len(all_predictions)]` 这样的截断方式构造。由于前面被 mask 掉的 `-1` key 可能出现在任意位置，截断后的 key 列会和 prediction/label 行错位。

后果是：如果直接拿 PTV1 `predictions.csv` 的 `experiment_type` 去和 PTV3 做 key-level common comparison，会得到额外的假 mismatch。这不是数据本身的标签差异，而是输出 CSV 的 key 对齐错误。

本次最终比较使用 `scripts/evaluate_ptv3_vs_ptv1_cellfold.py` 修正了这个问题：它不信任 PTV1 输出 CSV 的 `experiment_type`，而是从 raw `loo_label.csv` 和 `pheno.csv` 重新构造 PTV1 评估时真正保留下来的有效 key 顺序，然后再和 PTV3 prediction 对齐。

## 标签对齐后的验证结果

为了验证 gap 是否主要来自标签不一致，我构建了一个 PTV3 派生任务：

- 任务名：`ptv3_main_singledrug_ptv1_cell_5fold`
- 构建脚本：`scripts/build_ptv3_ptv1_cellfold_task.py`
- 核心规则：
  - 保留 PTV3 的特征、control、set-info 和 fast dataset 合约
  - 对同一个 `(Cell, pert_id1, pert_id2)`，把 PTV1 `pheno.csv` 的 0/1 标签映射到 PTV3 的 label 列
  - PTV1 `1` 映射为 `sensitive`
  - PTV1 `0` 映射为 `non-responsive`
  - PTV1 `-1` 保持 masked，不转换成负类

最终 5-fold 对比结果如下：

| fold | PTV1 AP | PTV1 AUROC | label-aligned PTV3 AP | label-aligned PTV3 AUROC | common label mismatch |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.883729 | 0.912118 | 0.896764 | 0.922376 | 0 |
| 1 | 0.868826 | 0.911255 | 0.875036 | 0.924203 | 0 |
| 2 | 0.887203 | 0.943753 | 0.900752 | 0.951228 | 0 |
| 3 | 0.824668 | 0.932085 | 0.833545 | 0.940450 | 0 |
| 4 | 0.837276 | 0.920040 | 0.808621 | 0.932730 | 0 |
| mean | 0.860341 | 0.923850 | 0.862943 | 0.934197 | 0 |

这个结果很关键：当 PTV3 也在 PTV1 的标签命名空间下训练和评估时，common-label mismatch 变成 `0`，PTV3 的 mean AP/AUPRC 也超过了 PTV1-flow。它支持以下判断：

- 原始 PTV3 低于 PTV1-flow 的主要原因是 label/objective mismatch。
- 如果要比较模型能力，必须先固定同一套 label definition、split 和 key-level evaluation 口径。
- 如果要比较数据标准本身，则需要明确区分“PTV1 pheno 标签”和“PTV3 PRISM 标签”，不能混用。

## 根因拆解

### 1. 标签来源不同

PTV1-flow 的 benchmark 标签来自 legacy `pheno.csv`。原始 PTV3 的标签来自标准化后的 `PRISM1st_label_total`。这两个字段没有被证明是同一规则、同一版本、同一阈值下产生的标签。它们可以共用 key，但不等价。

### 2. 缺失标签处理不同

PTV1 显式保留大量 `-1`，并在训练/评估时 mask。原始 PTV3 的 `ptv3_main_singledrug` 则倾向于在任务构建阶段要求非 control 行有非空 `PRISM1st_label_total`。如果把 PTV1 的 `-1` 忽略、错当 0、或者用输出 CSV 里的错位 key 做对齐，都会改变样本集合和标签含义。

### 3. 时间粒度不同

PTV1 一个样本绑定同一 key 的 6h 和 24h 表达变化。PTV3 原始表是按 time point 存在多行，报告时需要选择 `avgtime`、`bylasttime` 或其他 key-level aggregation。若把 PTV3 row-level prediction 直接和 PTV1 key-level label 混比，也会引入额外误差。

### 4. 输出 CSV 的 key 生成逻辑有错位风险

PTV1 `predictions.csv` 的 `experiment_type` 不是按 `valid_mask` 同步过滤出来的。它可以用于粗略查看预测值和标签，但不能作为可靠的 key-level join 字段。正确做法是从 raw split 文件重建有效 key 顺序。

### 5. 正类定义存在系统性偏移

共同 key 上，PTV1 正类是 `927`，PTV3 正类是 `768`。其中 `314` 个 PTV1 正类在 PTV3 是负类，`155` 个 PTV1 负类在 PTV3 是正类。这说明两套标签的正类边界存在系统性偏移，而不是少量随机错误。

## 建议

1. 所有 PTV1-flow benchmark 对比都应明确写出 label namespace。建议写成 `PTV1 pheno label` 或 `PTV3 PRISM1st_label_total label`，不要只写 `cell_5fold label`。
2. 与 PTV1-flow 比较模型能力时，应使用 label-aligned PTV3 任务 `ptv3_main_singledrug_ptv1_cell_5fold`，并保留 `-1` mask 语义。
3. 不要直接用 PTV1 `predictions.csv` 的 `experiment_type` 做 key-level join。应使用 raw `loo_label.csv` 和 `pheno.csv` 重建有效 key。
4. 每次报告 AUPRC 时，应同时报告正类数、负类数、masked `-1` 数和 mismatch table，避免把标签分布变化误判为模型变化。
5. PTV3 row-level prediction 和 PTV1 key-level sample 的比较必须显式声明 aggregation 方法，例如 `avgtime` 或 `bylasttime`。
6. 建议在派生任务或预测输出中同时保留：
   - `ptv1_cell_5fold_label`
   - `ptv1_cell_5fold_experiment_type`
   - 原始 `PRISM1st_label_total`
   - label source / label namespace metadata
7. 后续如果要判断哪套标签更“生物学正确”，需要回到原始实验读数、阈值、重复实验合并规则和 PRISM/benchmark 版本差异。本报告只证明当前 benchmark 中两套标签不一致，以及该不一致足以解释模型指标 gap。

## 附：核心证据文件

本报告的关键统计来自：

- `outputs/2026-06/2026-06-21/20260621_ptv1_ptv3_original_label_mismatch_cell_5fold.csv`
- `outputs/2026-06/2026-06-20/20260620_ptv3_ptv1label_cell_v1_covdrop020_mse075_vs_ptv1_5fold.csv`

前者记录了每个共同 key 的 PTV1 标签、原始 PTV3 标签和 mismatch 情况；后者记录了 label-aligned 后 PTV3 与 PTV1-flow 的 5-fold 指标对比。
