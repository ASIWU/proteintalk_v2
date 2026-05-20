# Step3：数据划分代码需求说明

在完成数据处理后，需要对数据进行 train / valid / test split。

## 1. 数据划分总体要求

请参考以下脚本的实现方式进行数据划分：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/2_datasplit_saveeval.py`

### 1.1 与原始代码的区别

需要注意以下改动：

1. **不再考虑敏感数据**
   - 所有数据都需要有 label。
   - 唯一例外是 `control`，`control` 可以没有 label。

2. **需要支持以下划分策略**
   - `random`
   - `pert_stratified`
   - `cell`
   - `cell_type`

### 1.2 需要包含验证集

除了 train / test 之外，还需要划分出 valid set。

对正式训练用 split，valid 不能为空；如果某个 group-level split 的
non-test groups/rows 数量足够，至少保留 1 个 validation group/row，同时
保证 train 不被清空。`train.py` 的 empty-valid fallback 只用于
`test_only`/debug 类路径，不能作为 5-fold 实验的正常 validation 来源。

### 1.3 保留原有 5-fold 和 include_all_train_subset_test 逻辑

仍然需要支持以下两类划分：

1. `5-fold`

2. `include_all_train_subset_test`
   - 所有数据都作为训练集。
   - 同时抽取 20% 的数据作为 test。
   - 注意：这里的 test 是 train 的子集。

### 1.4 不同数据类型的划分要求

需要注意，并不是所有数据都需要做 5-fold。

1. 只有以下数据需要做 5-fold：
   - `single_drug`
   - `double_drug`

2. `double_drug` 只需要做纯 `pert_id` 的 5-fold 划分。

   当前实现中，double-drug `pert_id_5fold_fold*` 的 fold key 是 canonical
   unordered pair，即 `pert_id1+pert_id2` 与 `pert_id2+pert_id1` 会进入同
   一个 split，避免 reversed-pair leakage。它不是 individual drug
   cold-start holdout。如果后续需要按单个 drug 完全 cold split，必须新增
   单独的 strategy，不能复用当前 `pert_id_5fold_fold*` 名称。

   对 `ptv3_main_doubledrug`，这里的 fold 只由 native double-drug rows
   决定；从 `ptv3_main_singledrug` 合并进 double-drug feature table 的
   rows 必须作为 train-only auxiliary rows 加入每一个 double-drug train
   split，不能加入 valid/test split。

3. 以下数据仅作为测试集使用：
   - `extra_guomics`
   - `nc`
   - `nature`

参考脚本：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/9_3_split.py`

重新生成 split 时使用以下命令，保证 `split_build_manifest.json` 包含所有 task：

```bash
conda run -n flow_v2 python utils/09_build_data_splits.py --dataset-group all
```

---

## 2. PTV1 数据的额外处理

PTV1 数据需要单独处理，并写入不同脚本中。

### 2.1 PTV1 的 pert_id 5-fold 划分

请参考以下脚本，对 PTV1 进行 `pert_id` 的 5-fold 划分：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/2_datasplit_ptv1.py`

### 2.2 PTV1 基于 experiment_type_list 的 train / valid / test 划分

请参考以下文件，对 PTV1 使用 `rawdata/ptv1/experiment_type_list` 进行 train / valid / test 划分：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/2_datasplit_ptv1_have.py`
- `/mnt/shared-storage-user/beam/wuhao/H100/proteintalk/ProteinTalkv2/utils/1_dataprocess_ptv1.ipynb`

### 2.3 ptv1_extra_single_drug 仅作为 test set

请参考以下文件，对 `ptv1_extra_single_drug` 仅做 test set 处理：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/2_datasplit_ptv1_test.py`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/1_dataprocess_ptv1_test_v3.ipynb`
