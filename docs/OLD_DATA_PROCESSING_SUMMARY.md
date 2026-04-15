# Utils Data Processing Summary

## 1. 总览

这个项目的数据处理是离线完成的，不是在 `train.py` 里边训练边做预处理。

训练主流程依赖四类产物：

1. 训练样本表 `*.parquet`
2. 元信息 `meta_info*.json`
3. embedding / 图矩阵 `*.npy`、`*.pkl`
4. 划分索引与 set 映射 `preprocess_path/*.pkl`

`train.py` 运行时只是把这些离线产物读进来，然后交给 `DatasetCSV` 采样。

---

## 2. 主数据处理流程

### 2.1 从原始矩阵和样本信息构建训练 parquet

代表脚本：

- `utils/1_dataprocess_full.py`

核心流程：

1. 读取原始表达矩阵和样本信息表。
2. 将表达矩阵中的蛋白列名统一为下划线前的部分。
3. 用 `sample_id` / `samp_ID` 对齐表达矩阵和样本信息。
4. 把每个样本的表达向量写入 `expressions_hvg` 列。
5. 对表达值做 `log1p`。
6. 将 `PRISM1st_label_total` 的空值填成 `non-responsive`。
7. 同时生成 `sensitive_label_mask`：
   - 原始标签缺失记为 `1`
   - 原始标签存在记为 `0`
8. 将 `targetv2` 解析成 `target_protein_list`，内容是蛋白整数 ID 列表。
9. 补充 `machineID_new` 等协变量信息。
10. 生成 `meta_info*.json` 并保存 parquet。

典型输出：

- `data/info_df.parquet`
- `data/info_df_fixed.parquet`
- `data/info_df_full.parquet`
- `data/meta_info.json`
- `data/meta_info_full.json`

---

### 2.2 元信息 `meta_info` 的作用

`meta_info*.json` 主要保存三类映射：

- `proteinname2id`
- `pert_id2smiles`
- `pert_id2id`

这些映射是后续所有 embedding、图矩阵、target 索引对齐的基准。

也就是说：

- `target_protein_list` 里的数字必须对应 `proteinname2id`
- drug embedding 的行顺序必须对应 `pert_id2id`
- protein embedding / PPI / PDI 的索引顺序必须对应 `proteinname2id`

如果这些映射版本不一致，训练虽然可能还能跑，但输入语义会错位。

---

### 2.3 生成 drug / protein embedding

相关脚本：

- `utils/3_embedding.py`
- `utils/3_metadata2emb.py`
- `utils/0311_build_embeddings_from_meta_and_fasta.py`

#### protein embedding

逻辑：

1. 读取 `proteinname2id`
2. 按蛋白顺序取序列
3. 用 ESM 模型编码
4. 对 token hidden state 做 mean pooling
5. 输出 `protein_embedding.npy`

#### drug embedding

逻辑：

1. 读取 `pert_id2smiles`
2. 用 RDKit 从 SMILES 生成 Morgan fingerprint
3. 输出：
   - `pertid_to_embedding*.pkl`
   - `pert_embedding*.npy`

新版脚本 `0311_build_embeddings_from_meta_and_fasta.py` 更严格，强调：

- embedding 必须按 index 对齐输出
- 缺序列或无效 SMILES 不能跳过
- 要用零向量占位
- 要保留缺失 mask

这是更安全的做法。

---

### 2.4 生成图矩阵

相关脚本：

- `utils/ppi_pdi/ppi/get_ppi_string.py`
- `utils/ppi_pdi/pdi/normalize_pdi.py`

用途：

- `ppi_string*.npy`
- `pdi_normalized*.npy`
- `ddi_normalized*.npy`

其中：

- PPI 矩阵按 `proteinname2id` 顺序构建
- PDI 会先 clip，再做 min-max normalize
- DDI 也是训练图结构的一部分

这些矩阵的行列顺序必须与 embedding 和 metadata 保持同一个索引体系。

---

## 3. 数据切分流程

代表脚本：

- `utils/2_datasplit.py`
- `utils/2_datasplit_saveeval.py`
- `utils/2_datasplit_ptv1.py`
- `utils/2_datasplit_ptv1_test.py`

### 3.1 set 的定义

切分前先构建 `set`。

逻辑：

1. 如果某行的 `sample_id` 出现在 `control` 列中，则该行被视为 control 行。
2. 其余行为 perturb 行。
3. 按 `control` 分组，每个 control family 对应一个 `set`。
4. 为每个 `set` 保存：
   - `control` 行索引
   - `perturb` 行索引

生成的核心文件：

- `row_to_set_index.pkl`
- `set_info.pkl`
- `set_to_grouping.pkl`

这一步决定了训练时 anchor 样本如何找到自己的 control。

---

### 3.2 普通数据切分

`utils/2_datasplit.py` 会基于 parquet 生成多种 train/test 划分：

- `random`
- `set`
- `pert_id`
- `Cell`
- `cell_type`
- `pert_id_5fold_fold*`
- `Cell_5fold_fold*`
- `cell_type_5fold_fold*`
- `pert_id_stratified_5fold_fold*`
- `all_train_subset_test`

每种划分都会生成：

- `train_indices_*.pkl`
- `test_indices_*.pkl`
- `train_set_info_*.pkl`
- `test_set_info_*.pkl`

---

### 3.3 `sensitive_label_mask` 的切分规则

在 `utils/2_datasplit.py` 和 `utils/2_datasplit_saveeval.py` 里：

- `sensitive_label_mask == 1` 的 perturb 样本会被视为“敏感数据”
- 这些样本不会进入 test
- 它们会被强制加入 train

注意：

这里的 “sensitive” 在这套代码里并不是药敏正例本身，而更接近“标签缺失或不适合作为评估集”的标记。

因此不要只看变量名，要看它的生成来源。

---

### 3.4 saveeval 版本

`utils/2_datasplit_saveeval.py` 在 train/test 基础上再切一个 valid。

额外产物：

- `valid_indices_*.pkl`
- `valid_set_info_*.pkl`

规则：

1. 先从非敏感训练样本中抽取一部分 valid
2. 敏感样本仍然只放在 train
3. valid 和 test 都不包含敏感样本

---

### 3.5 `all_train_subset_test`

这是一个特殊策略。

规则：

1. train 使用所有样本
2. test 只从非敏感 perturb 中抽样
3. test 是 train 的子集
4. test 采用放回采样

这个策略更像 overfitting check，不是严格意义上的独立测试集。

---

### 3.6 PTV1 固定划分分支

相关脚本：

- `utils/2_datasplit_ptv1_test.py`

这个分支不按随机或 family 划分，而是直接使用 parquet 里的 `data_split` 列：

- `train`
- `test`
- 其他值默认不进划分

它生成的目录通常是：

- `data/preprocessed_ptv1_fixed_split_test_v3/`

脚本 `scripts/train_0310_2.sh`、`scripts/train_0304_1.sh`、`scripts/train_0303_1.sh` 都属于这条路线。

---

## 4. 多来源数据合并流程

相关脚本：

- `utils/5_merge_df.py`

用途：

- 合并主数据 `info_df.parquet`
- 合并 dd / synergy 数据 `info_df_dd.parquet`

它会做的事：

1. 统一列名，如 `samp_id -> sample_id`
2. 补齐缺失列
3. 缺失 `pert_id` 时填 `"control"`
4. 补 `target_protein_list`
5. 统一 `machineID_new`
6. 推断 `cell_type`
7. 处理 `pert_time`
8. 处理 `pert_dose`
9. 最终保存合并 parquet / csv

典型输出：

- `data/info_df_merged.parquet`
- `data/info_df_merged_with_target_protein_lists.parquet`

---

## 5. `train.py` 如何消费这些数据

关键代码：

- `train.py`
- `dataset/dataset_csv.py`

### 5.1 `train.py` 读取的核心输入

`train.py` 启动时会读取：

- `args.data_path` 对应的 parquet
- `args.meta_data_path`
- `args.preprocess_path/row_to_set_index.pkl`
- `args.preprocess_path/train_indices_*.pkl`
- `args.preprocess_path/test_indices_*.pkl`
- `args.preprocess_path/train_set_info_*.pkl`
- `args.preprocess_path/test_set_info_*.pkl`
- protein / drug embedding
- ppi / pdi / ddi 矩阵

默认特征列：

- `expressions_hvg`
- `machineID_new`
- `Cell_plate`
- `Cell`
- `cell_type`
- `pert_id`
- `batch`
- `pert_time`
- `effective_key`
- `sensitive_label_mask`
- `target_protein_list`

如果模型分支需要，还会追加别的列。

---

### 5.2 `DatasetCSV` 的采样方式

`dataset/dataset_csv.py` 中：

- `valid_anchor_indices` 是可被采样的 perturb anchor
- 每个 anchor 会先通过 `row_to_set_index` 找到所属 set
- 再从 `set_info[set_idx]` 中取 control / perturb

#### train 模式

1. 当前 anchor 必定保留
2. 额外随机采 `group_size - 1` 个同 set perturb
3. 随机采 `group_size` 个 control

#### test 模式

1. 只取 1 个 anchor
2. 只取 1 个 control
3. 再复制到 `group_size`

最终输出结构是：

- `{'control': ..., 'perturb': ...}`

每个分支内部都是按 feature name 组织的 numpy 数组。

---

### 5.3 特征编码规则

在 `DataSpliter` 和 `DatasetCSV` 里：

- `expressions_hvg` 直接转 `float32` 向量
- `pert_id`
  - 异构图模型下通常映射为整数 index
  - 非异构图模型下可映射为 fingerprint embedding
- `target_protein_list` 会被 padding / truncate 到固定长度
- 分类特征如果没有显式 embedding 方法，就走 label encoder
- 所有最终张量都会 `astype(np.float32)`

---

## 6. 训练前必须满足的数据规范

### 6.1 parquet 行顺序必须稳定

split 文件里保存的是“行位置索引”，不是 `sample_id`。

因此：

- 只要 parquet 重新排序
- 或者删行、增行、reset index

就必须重新生成：

- `row_to_set_index.pkl`
- `train/test/valid_indices_*.pkl`
- `*_set_info_*.pkl`

否则索引会错位。

这是最重要的约束之一。

---

### 6.2 `expressions_hvg` 必须是定长数值向量

要求：

- 每一行都必须是同长度向量
- 类型必须能转成 `np.float32`

当前仓库里我实际看到：

- `data/info_df.parquet` 中 `expressions_hvg` 长度是 `2000`
- `data/ptv1_test_v3.parquet` 中 `expressions_hvg` 长度是 `5532`

不同数据集可以不同，但同一个 parquet 内部必须一致。

---

### 6.3 `target_protein_list` 必须是 `list[int]`

要求：

- 正常情况是若干蛋白 ID 组成的 list
- 没有 target 时必须是 `[]`
- 不能是 `NaN`
- 最好也不要是字符串 `"no"`

因为 `train.py` 里会直接对它做 `list(key)` 和 padding。

---

### 6.4 metadata、embedding、图矩阵必须是同版本

下面这些文件必须联动：

- `meta_info*.json`
- `protein_embedding.npy`
- `drug_embedding.npy`
- `ppi_string*.npy`
- `pdi_normalized*.npy`
- `ddi_normalized*.npy`
- parquet 里的 `target_protein_list`

尤其不能把：

- `meta_info.json`
- `meta_info_full.json`
- `meta_info_ptv1.json`

混着用。

---

### 6.5 label 列的语义要和数据集匹配

当前存在两种主要 label 语义：

- PTV3 / 主数据里常见 `PRISM1st_label_total`：
  - `sensitive`
  - `non-responsive`
- PTV1 分支里常见：
  - `Y`
  - `N`

所以：

- 常规分支走 `Sensitive_label_encoder`
- PTV1 分支走 `NY_label_encoder`

如果 label 语义和 encoder 不匹配，训练目标会错。

---

### 6.6 `preprocess_path` 必须和 `data_path` 配套

例如：

- `data/info_df.parquet` 应该搭配 `data/preprocessed_5fold*`
- `data/ptv1_test_v3.parquet` 应该搭配 `data/preprocessed_ptv1_fixed_split_test_v3`

不能只换 parquet 不换 split 目录。

---

## 7. 与脚本的对应关系

和 `train.py` 直接相关的脚本主要有：

- `scripts/train_0305_2.sh`
- `scripts/train_0304_1.sh`
- `scripts/train_0303_1.sh`
- `scripts/train_0310_2.sh`
- `scripts/train_0313_1.sh`

其中：

- `train_0305_2.sh` 更偏主数据 / `preprocessed_5fold_v2`
- `train_0310_2.sh`、`train_0304_1.sh`、`train_0303_1.sh` 更偏 PTV1 fixed split

`scripts/train_0408_*.sh` 主要调用的是 `train_dd.py`，不是当前这条 `train.py` 主线。

---

## 8. 一个最实用的理解方式

如果只从运行角度理解，这套流程可以压缩成一句话：

1. 先把原始样本整理成 parquet
2. 再从 parquet 生成 metadata、embedding、graph、split
3. 最后 `train.py` 读取这些离线产物训练

换句话说，`train.py` 的前提不是“有原始 csv”，而是“有一整套已经对齐好的中间产物”。

---

## 9. 训练前检查清单

在真正运行 `train.py` 前，建议至少确认下面几点：

1. `data_path` 对应的 parquet 存在，且列完整
2. `preprocess_path` 存在，并且和该 parquet 同版本
3. `meta_data_path` 与 parquet 使用同一套 `proteinname2id` / `pert_id2id`
4. `protein_embedding_path`、`drug_embedding_path`、`ppi_matrix_path`、`pdi_matrix_path`、`ddi_matrix_path` 与 metadata 对齐
5. `target_protein_list` 不是 `NaN`，而是 `[]` 或 `list[int]`
6. `expressions_hvg` 在整个 parquet 中长度一致
7. label 编码方式与当前数据集语义一致

---

## 10. 额外观察

当前 `train.py` 只加载：

- `train_indices_*`
- `test_indices_*`

并且把 `test_dataloader` 直接作为 `trainer.fit(..., val_dataloader)` 的验证集输入。

因此即使 `preprocessed_5fold_saveeval/` 目录里已经有 `valid_*` 文件，`train.py` 目前也没有显式使用它们。

如果后续你要严格区分 valid / test，这里需要再调整训练入口。
