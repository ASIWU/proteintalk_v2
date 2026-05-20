# Step2数据处理代码需求说明

当前已经根据以下文档完成了第一步数据预处理：

- `./docs/Data_Process.md`
- `./docs/data_process_summary_01.md`

现在需要继续编写代码，完成进一步的数据处理，使数据能够被模型直接读取和使用。

---

## 1. 数据构成要求

### 1.1 每个数据集的基础产物

除了 `extra_baseline` 之外，每个数据集都需要生成以下两个文件：

1. 自己对应的 `processed_csv`
2. 自己对应的 `expression matrix`

### 1.1.1 `ptv3_main_doubledrug` 的 single-drug 合并要求

`ptv3_main_doubledrug` 不能只保留 double-drug 自己的数据。Stage-2 的
`feature_table` 必须同时包含：

1. `ptv3_main_doubledrug` 自己过滤后的 processed rows；
2. `ptv3_main_singledrug` 过滤后的全部 processed rows。

当前实现中，合并进 double-drug `feature_table` 的 single-drug rows 使用
`feature_membership = "merged_single_drug"` 标记；double-drug 自己的 rows
使用 `feature_membership = "primary"` 标记。后续 Step3 split 必须把
`merged_single_drug` rows 加入 double-drug 的 train split，但不能加入
double-drug valid/test split，因为这些 single-drug rows 没有 `synergy`
label。

合并进 double-drug `feature_table`
的 single-drug rows 必须保留 `PRISM1st_label_total` / `PRISM2nd_label_total`
作为原始 metadata / audit 信息，但 double-drug `loss2` 只能使用 native
double-drug rows 的 `synergy`。因此 merged single-drug rows 的 active
`synergy` 必须清空，并使用
`training_label_scope = "single_drug_auxiliary_synergy_masked"` 标记。原始
single-drug 标签需要保存在 `auxiliary_source_*` audit columns 中，便于检查
数据来源和 label masking 是否正确。

重新生成 Stage-2 artifact 和 Step3 split 时使用：

```bash
conda run -n flow_v2 python utils/02_build_training_ready_data.py
conda run -n flow_v2 python utils/03_validate_training_ready_outputs.py
conda run -n flow_v2 python utils/09_build_data_splits.py --dataset-group all
```

### 1.2 control 行的保留规则

所有 `processed_csv` 都必须包含完整的 `control row`。

因此，对于 `extra data`，需要把 `extra_baseline` 中的 control 数据合并进去。

对于 `ptv1_extra_singledrug`，这里的 control 不来自 `extra_baseline`，而是来自 `ptv1_aivc` 中按 `cell -> protein_plate` 匹配到的 control 行。

### 1.3 非 control 行的过滤规则

以下过滤规则只作用于 **非 control 行**，control 行必须完整保留，不能因为标签为空而被删除。

具体规则如下：

1. 对 `single_drug` 数据：
   - 删除 `PRISM1st_label_total` 为空的非 control 行

2. 对 `double_drug` 数据：
   - 删除 `synergy` 为空的非 control 行

3. 对 `ptv3_extra_singledrug_*` 数据：
   - 删除 `PRISM2nd_label_total` 为空的非 control 行

4. 对 `ptv3_extra_doubledrug_*` 数据：
   - 删除 `PRISM1st_label_total` 为空的非 control 行

5. 对 `ptv1_aivc`：
   - 当前不新增 label-based filter
   - 因为这个 task 同时包含主 `PTV1` 的 control / single / anchor-combo 结构，先保留全部行

6. 对 `ptv1_extra_singledrug`：
   - 不再套用 `ptv3_extra_singledrug_*` 的非 control label 过滤规则
   - stage-1 已先保证只保留 unique `(cell, E115_id)` 的样本行
   - stage-2 只追加 matched control，不再删除这些 perturb 样本

---

## 2. info 数据处理

### 2.1 `target_protein_list` 的重新处理

需要对所有数据中的 `target_protein_list` 进行重新处理。

当前不再使用 `uniprotid list`，而是改为使用 `protein index list`。

具体要求：

1. 使用 `global_meta.json` 中的 `protein_index`
2. 将 `target_protein_list` 中的 protein / uniprot 信息映射为对应的 protein index
3. 最终保存到 processed csv 中的应该是 protein index list，而不是 uniprotid list

### 2.2 为离散特征添加 value-to-index 映射

需要对以下所有字段构建 `value_to_index` 映射，并添加到 `global_meta.json` 中，方便后续构建 one-hot 编码：

- `machineID_new`
- `Cell_plate`
- `Cell`
- `cell_type`
- `batch`
- `pert_time`
- `pert_dose1`
- `pert_dose2`

要求：

1. 每个字段都需要独立构建自己的 `value_to_index`; 两个`pert_dose`使用一个映射
2. 每个字段的映射中都必须额外添加一个 `"no"` value，方便后续进行特殊值处理或者外推处理
3. `pert_dose1` 和 `pert_dose2` 的 index 规则需要特殊处理：
   - 不再使用普通的“按唯一值排序后顺序编号”方式
   - 对于数值 dose，index 直接使用 `ceil(dose)`，例如：
     - `0.2 -> "1"`
     - `1.1 -> "2"`
   - index value 需要保留为 `string`
   - 缺失 dose 统一映射为 `"no"`
   - `"no"` 对应的 index value 必须是 `string(max_numeric_index + 1)`
4. 对于以下 5 个字段，需要特别注意格式一致性：
   - `machineID_new`
   - `Cell_plate`
   - `Cell`
   - `cell_type`
   - `batch`

格式一致性包括但不限于：

- 大小写
- 中横杠 `-` 与下划线 `_`
- 空格
- 其他可能导致同一个值被识别为不同值的格式差异

### 2.2.1 single-drug 的第二药物槽位

所有 single-drug rows 在进入 Stage-2 index encoding 前都必须满足
`pert_id2 == pert_id1`。这里包括 `ptv3_main_singledrug`、
`ptv3_extra_singledrug_*`、`ptv1_extra_singledrug`，以及 `ptv1_aivc`
中 `pert_id2` 原本为空的 single-drug rows。不能把 single-drug 的
`pert_id2` 编码为 `"no"`，否则 two-slot model input 会和 double-drug
训练约定不一致。

### 2.3 `protein_index` 和 `pert_index` 的特殊值补充

在 `global_meta.json` 中，需要对以下两个 index 添加特殊值：

1. `protein_index`
   - 添加 `"control"`
   - 添加 `"no"`

2. `pert_index`
   - 添加 `"no"`

这些特殊值用于后续外推处理或特殊情况处理。

---

## 3. expression 数据处理

需要对每个 `expression matrix` 进行 `log1p` 处理。

注意：

1. expression matrix 中可能包含 `NaN`
2. 对矩阵进行 `log1p` 时，需要正确处理 `NaN`
3. `NaN` 不应该导致程序报错
4. `NaN` 位置应该在处理后仍然保持为 `NaN`
5. training-ready expression matrix 必须使用每个 task/source 的完整
   protein union 轴；严禁 top-k / top2000 protein 截断。

---

## 4. 额外数据特征构建

下面这些特征构建代码需要编写完成，但由于部分步骤耗时较长或需要 GPU 权限，可以不实际执行，留给我后续运行。

---

### 4.1 drug embedding

需要参考以下代码，编写一份新的 `pert index -> drug embedding` 构建代码：

```text
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/3_embedding.py
````

要求：

1. 使用 `global_meta.json` 中的 `pert_index`
2. 根据 `pert_index` 的顺序构建 embedding
3. 最终保存为 `.pkl` 文件
4. 因为当前环境没有 GPU 权限，可以只写代码，不需要执行

* * *

### 4.2 protein embedding

需要参考以下代码，编写一份新的 `protein index -> protein embedding` 构建代码：

```
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/3_embedding.py
```

要求：

1. 使用 `global_meta.json` 中的 `protein_index`
2. 根据 `protein_index` 的顺序构建 embedding
3. 最终保存为 `.pkl` 文件
4. 因为当前环境没有 GPU 权限，可以只写代码，不需要执行

* * *

### 4.3 PPI matrix 构建

需要基于以下代码，编写一份新的、基于 `global_meta.json` 构建 PPI matrix 的代码：

```
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/ppi_pdi/ppi/process_ppi_string.py
```

要求：

1. PTV1 和 PTV3 需要分别构建自己的完整 PPI matrix
2. 每个 PPI matrix 的大小必须和 `global_meta.json` 中的 `protein_index` 长度一致
3. matrix 必须包含 `"control"` 和 `"no"`
4. matrix 的行列顺序必须和 `protein_index` 的顺序完全一致
5. 这样后续可以直接使用 protein index 对 matrix 进行索引
6. 因为该步骤运行时间较长，可以只写代码，不需要执行
7. 代码中需要加入进度条

* * *

### 4.4 DDI matrix 构建

需要基于以下代码，编写一份新的、基于 `global_meta.json` 构建 DDI matrix 的代码：

```
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/ppi_pdi/ddi.py
```

要求：

1. PTV1 和 PTV3 需要分别构建自己的完整 DDI matrix
2. 每个 DDI matrix 的大小必须和 `global_meta.json` 中的 `pert_index` 长度一致
3. matrix 必须包含 `"no"`
4. matrix 的行列顺序必须和 `pert_index` 的顺序完全一致
5. 这样后续可以直接使用 pert index 对 matrix 进行索引
6. 因为该步骤运行时间较长，可以只写代码，不需要执行
7. 代码中需要加入进度条

* * *

### 4.5 PDI matrix 构建

需要基于以下代码，编写一份新的、基于 `global_meta.json` 构建 PDI matrix 的代码：

```
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0303_inchikey2chemical_id.py
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0303_newpdi.py
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0303_uniprotid2protein_experimental.py
/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0304_buildpdi.py
```

要求：

1. 使用 `global_meta.json` 中的 `pert_index` 和 `protein_index`
2. 构建完整的 PDI matrix
3. matrix 的大小必须和对应 index 的长度一致
4. matrix 的顺序必须和 index 的顺序完全一致
5. 这样后续可以直接通过 index 对 matrix 进行索引
6. PDI 流程比较复杂，需要仔细处理 ID / 名称转换问题
7. 文献中的 PDI 使用的名称、当前数据中的 `pertid`、以及 protein 的 `uniprotid` 并不一致，因此都需要进行转换
8. 因为该步骤运行时间较长，可以只写代码，不需要执行
9. 代码中需要加入进度条

* * *

## 5. 完整 feature 构建

为了让训练代码可以直接读取数据，每个任务都需要生成一个融合所有 feature 的文件。

初步设想是使用 `csv`，如果有更合适的格式，也可以使用其他格式，但需要保证训练代码能够方便读取。

* * *

### 5.1 所有任务都需要包含的统一内容

每个任务的 feature 文件都需要从对应的 `processed_csv` 出发，并将以下字段全部转换为 index：

* `machineID_new`
* `Cell_plate`
* `Cell`
* `cell_type`
* `pert_time`
* `pert_dose1`
* `pert_dose2`

这些字段需要使用 `global_meta.json` 中对应的 `value_to_index` 映射进行转换。

其中：

- `pert_dose1_index`
- `pert_dose2_index`

需要使用共享的 dose 映射，并保持为 string index；具体规则为 `ceil(dose)`，缺失值使用 `"no"` 对应的 `string(max_numeric_index + 1)`。

此外，还需要将：

* `pert_id`

转换为：

* `pert index`

该转换同样使用 `global_meta.json` 中的 `pert_index`。

* * *

### 5.2 每个任务需要包含的特殊数据范围

不同任务的 feature 文件需要包含的数据范围如下。

#### 5.2.1 single drug 和 PTV1

`single drug` 和 `PTV1` 只需要包含它们各自数据处理后的所有行。

要求：

1. 包含所有处理后的数据行
2. 包含 control 行

#### 5.2.2 double drug

`double drug` 的 feature 文件需要包含：

1. `double drug` 自己处理后的数据
2. 上述 `single drug` 的数据

也就是说，double drug 任务中需要合并 single drug 数据。

#### 5.2.3 extra data

`extra data` 的 feature 文件需要包含：

1. extra data 自己处理后的数据
2. 它所需的所有 control 行

这些 control 行可能来自：

* single drug
* double drug
* extra baseline

* * *

### 5.3 每个任务的 ordered protein index list 构建

由于每个任务的 feature csv 可能包含不同数据来源，而不同来源的 protein 数量可能不一致，因此需要为每个任务单独构建一个 `ordered protein index list`。

具体要求如下：

1. 对于每个任务，收集该任务所有数据来源中的 `protein_order.json`
2. 对这些 `protein_order.json` 中的 protein 取并集
3. 使用 `global_meta.json` 将 protein 转换为 protein index
4. 构建该任务自己的 `ordered protein index list`
5. 该 list 的顺序需要固定，保证后续 expression matrix 可以稳定对齐

* * *

### 5.4 expression matrix 扩增与对齐

对于每个任务，需要基于该任务的 `ordered protein index list`，对 expression matrix 进行扩增和对齐。

要求：

1. 每个来源的 expression matrix 都需要映射到该任务的 `ordered protein index list`
2. 如果某个 protein 在原始 expression matrix 中不存在，则需要在扩增后的 matrix 中补充对应位置
3. 扩增后的 expression matrix 长度必须和该任务的 `ordered protein index list` 长度一致
4. 扩增后的 expression matrix 需要添加到 feature csv 中
5. 最终训练代码应该能够直接从 feature csv 中读取 expression feature

* * *

## 6. 最终代码目标

请根据以上要求编写完整的数据处理代码。

代码需要完成以下主要目标：

1. 为每个数据集生成符合要求的 `processed_csv`
2. 为每个数据集生成经过 `log1p` 处理后的 expression matrix
3. 更新并保存 `global_meta.json`
4. 将 `target_protein_list` 从 uniprotid list 转换为 protein index list
5. 为指定离散字段构建 `value_to_index`
6. 在 `protein_index` 和 `pert_index` 中添加 `"control"` 和 `"no"`
7. 编写 drug embedding 构建代码，并保存为 `.pkl`
8. 编写 protein embedding 构建代码，并保存为 `.pkl`
9. 编写 PPI matrix 构建代码
10. 编写 DDI matrix 构建代码
11. 编写 PDI matrix 构建代码
12. 为每个任务构建完整 feature 文件
13. 为每个任务构建对应的 `ordered protein index list`
14. 将 expression matrix 按任务的 `ordered protein index list` 扩增、对齐，并写入 feature 文件

其中，drug embedding、protein embedding、PPI、DDI、PDI 相关代码可以先不执行，但需要保证代码逻辑完整，并且能够在后续环境中直接运行。
