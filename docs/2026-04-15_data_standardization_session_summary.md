# 2026-04-15 Data Standardization Session Summary

## 背景

本轮工作基于 [data/Data_Process.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/Data_Process.md:1) 的第一阶段要求，目标是对 `data/rawdata/` 下的原始数据做逐文件梳理、标准化，并产出可复现的 task-level 输入与全局 meta。

本轮已经完成第一版可运行实现，并产出了标准化结果到 `data/standardized/`。

## 本轮新增代码

### 1. 主标准化脚本

文件: [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)

作用:

- 按 task 处理主单药、主双药、extra baseline、extra 单药、extra 双药、PTV1
- 输出每个 task 的:
  - `info.csv`
  - `expression_matrix.npy`
  - `protein_order.json`
  - `sample_ids.json`
  - `sample_id_to_row_index.json`
  - 小任务时额外输出 `expression_dict.pkl`
- 生成:
  - `data/standardized/ptv3/global_meta.json`
  - `data/standardized/ptv1/global_meta.json`
  - `data/standardized/file_audit.json`

### 2. 校验脚本

文件: [utils/01_validate_standardized_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/01_validate_standardized_outputs.py:1)

作用:

- 校验每个 task 的 `info.csv`、矩阵形状、蛋白顺序、sample 索引是否一致
- 校验 materialized `expression_dict.pkl` 的样本数是否匹配

## 本轮产物

输出目录:

- [data/standardized](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized:1)
- [data/standardized/file_audit.json](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized/file_audit.json:1)
- [data/standardized/ptv3/global_meta.json](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized/ptv3/global_meta.json:1)
- [data/standardized/ptv1/global_meta.json](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized/ptv1/global_meta.json:1)

验证结果:

- `ptv3_main_singledrug`: `28602 x 10982`
- `ptv3_main_doubledrug`: `3549 x 9205`
- `ptv3_extra_baseline`: `78 x 10169`
- `ptv1_aivc`: `15002 x 5576`
- extra 单药 / extra 双药: 当前输出为 metadata + 空表达结构

校验脚本已通过。

## 当前实现的重要规则

### 1. 主单药 / 主双药 / extra baseline / PTV1

- 有真实蛋白表达矩阵
- 蛋白列统一为 `UniProt ID`
- 按 `sample_id` 与样本表对齐

### 2. extra 单药 / extra 双药

当前仓库中没有对应的原始 perturbation proteome 矩阵，因此第一版实现采用统一接口输出:

- `info.csv`
- 空的 `expression_matrix.npy`
- 空的 `protein_order.json`
- 空的 `expression_dict.pkl`

这里“空表达结构”的含义是:

- 矩阵 shape 为 `(样本数, 0)`
- 不是数值填 0
- 而是“只有样本，没有蛋白表达列”

### 3. control 匹配

extra 数据的 control 候选池来自:

- 主单药中 `control == sample_id` 的 control 样本
- extra baseline

匹配优先级:

1. `Cell` 硬匹配
2. `Machine Match`
3. `Type Match`
4. `Batch Match`
5. `Plate Match`

匹配结果会记录审计列:

- `control_match_level`
- `control_match_source_task`
- `control_match_pool_kind`
- `control_match_score`
- `control_match_machine`
- `control_match_type`
- `control_match_batch`
- `control_match_plate`

### 4. PTV1 隔离

- `ptv1` 独立输出到 `data/standardized/ptv1`
- 不与 `ptv3` 共用索引空间

## 本轮遇到的问题

### 1. 主单药 info 文件编码问题

文件:

- `data/rawdata/singledrug/20260403_ptv3_v2_bind_bio_sampleID_machineID_details.csv`

问题:

- 该文件并非稳定 `utf-8`
- 实际需要 `latin1` fallback 才能完整读取

处理:

- 在主脚本里对该文件启用了多编码 fallback

### 2. 挂载盘不支持直接 `numpy.memmap`

问题:

- 运行主标准化脚本时，`np.lib.format.open_memmap(...)` 在挂载目录下报错:
  - `OSError: [Errno 19] No such device`

处理:

- 改为先把大矩阵写到 `/tmp`
- 完成后再 `move` 回 `data/standardized/...`
- 校验脚本不再使用 `mmap_mode='r'`，改为只读 `.npy` header 获取 shape

### 3. 主单药 `control` 字段有脏值

问题:

- 有 112 行 `control` 不是合法 `sample_id`
- 这些脏值其实是 SMILES，不是 sample family 引用

唯一 4 个异常值:

- `C=CCn1c(=O)c2cnc(Nc3ccc(N4CCN(C)CC4)cc3)nc2n1-c1cccc(C(C)(C)O)n1`
- `Cc1cc(NC(=O)NCCN2CCC(O)(Cc3ccccc3)CC2)c2ccccc2n1.Cl`
- `Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=NC(CC(=O)Nc1ccc(O)cc1)c1nnc(C)n1-2`
- `Cn1cc(CC(N)C(=O)O)c2ccccc21`

这些值对应的典型药物:

- `pert_id=115`, `drugname=Adavosertib`
- `pert_id=L9200_2016`
- `pert_id=L9200_2703`
- `pert_id=L9200_2896`

第一版处理:

- 这类值不会强行当作 `sample_id`
- 标准化结果中:
  - `control` 被清空
  - `control_raw` 保留原值
  - `control_status=unresolved_non_sample_id`

### 4. extra baseline 中有 3 条表达样本缺失原始 info

缺失样本:

- `S1_B`
- `S1_CAC`
- `S1_O`

第一版处理:

- 保留表达矩阵中的这 3 行
- 用占位 metadata 保留下来
- `raw_record_issue=info_missing_in_raw_file`

后续核查结果:

- 这 3 个样本没有被其他 extra task 用作 control
- 只在 `ptv3_extra_baseline` 自己里面存在

### 5. `pert_id -> smiles` 冲突较多

问题:

- `ptv3/global_meta.json` 中记录到 `937` 条冲突
- 大多是不同手性写法、盐形式、去手性写法、或不同但相关的 SMILES 规范化结果

第一版处理:

- `global_meta.json` 中的 `pertid_to_smiles` 实际已经是 first-seen wins
- 同时把后续冲突额外记在 `pert_mapping_conflicts`

## 用户在讨论阶段提出的修改边界

用户原始回复要点:

1. 为什么叫做表达结构是空矩阵？
2. 请告诉我不合法的 `sample_id` 是什么样的，我好判断如何修改
3. 缺乏原始 info 的 `extra_baseline` 可以直接删除
4. 冲突的 `pert_id -> smiles` 以第一次出现的为准，用 `pert_id` 对齐，因为手性不重要

## 我当时给出的回答摘要

### 对第 1 点的回答

我解释了“空矩阵”的含义:

- shape 是 `(样本数, 0)`
- 表示没有蛋白表达列
- 不是值全为 0

同时我提出建议:

- 如果不喜欢这种接口，可以改成 `info-only` 模式
- 即 metadata-only 的 extra task 不再生成空矩阵 / 空 dict

这个点用户尚未最终拍板。

### 对第 2 点的回答

我说明:

- 不合法的不是 `sample_id` 列本身
- 而是主单药 `control` 字段里出现了 4 个 SMILES 字符串

并把这 4 个具体值列了出来，供后续判断如何修。

### 对第 3 点的回答

我核查后确认:

- `S1_B`
- `S1_CAC`
- `S1_O`

这 3 个 extra baseline 样本没有被其他 task 使用为 control，因此直接删除是安全的。

### 对第 4 点的回答

我同意该规则，并进一步建议:

- 不只是 `global_meta.json` 里用 first-seen wins
- 各 task 的 `info.csv` 里也要把同一个 `pert_id` 的 `smiles` 统一替换成 canonical first-seen value

这样才是彻底按 `pert_id` 对齐，而不是只有 meta 对齐、task 内部仍各写各的。

## 下次继续时应执行的修改

### 已明确可以改的

1. 从 `ptv3_extra_baseline` 中删除缺失 info 的 3 个样本:
   - `S1_B`
   - `S1_CAC`
   - `S1_O`

2. 对 `ptv3` 全局 `pert_id -> smiles`:
   - 以第一次出现为准
   - 按 `pert_id` 统一回写到各 task 的 `info.csv`
   - 手性冲突不再保留为阻断问题

### 尚未最终拍板的

1. metadata-only 的 extra tasks 是否继续保留“空表达结构”

备选方案:

- 方案 A: 保留当前 `(n, 0)` 空矩阵输出
- 方案 B: 改为 `info-only`，不再生成空矩阵 / 空 dict

用户暂时离开前，尚未对这个点做最终确认。

## 建议的下次操作顺序

1. 修改 `utils/00_standardize_rawdata.py`
   - 删除 extra baseline 中缺失 info 的 3 个样本
   - 增加 canonical `pert_id -> smiles` 回写逻辑

2. 根据用户最终确认，决定:
   - 保留空表达结构
   - 或切换为 `info-only`

3. 重新运行:

```bash
source ~/.bashrc
conda activate flow_v2
python utils/00_standardize_rawdata.py
python utils/01_validate_standardized_outputs.py
```

4. 更新:

- `data/standardized/file_audit.json`
- `data/standardized/ptv3/global_meta.json`
- 本文档

