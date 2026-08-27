# official_fixed_residual_20260605 转录组预测输出使用说明

本文档说明如何复用以下目录中的 CPA/BioLord 预测转录组矩阵，用于后续二分类模型训练和测试。

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/maoxinjie/AIVC/ptv3_single_nonablation/output/official_fixed_residual_20260605
```

## 结论

该目录可以支持“使用 train 的模型生成转录组特征 + exp01/02/03 二分类标签训练，再使用 test 特征测试”的流程。

需要注意：

- prediction h5ad 的 `.X` 是 9843 维 common HGNC gene-axis 预测矩阵，可作为下游特征。
- 文件内没有低维 latent embedding：`obsm=[]`，`layers=[]`。如果下游模型必须使用低维 embedding，需要自行对 `.X` 做 PCA/autoencoder/feature selection。
- exp01/02/03 建议使用 `PRISM1st_label_total` 作为二分类标签：`sensitive` vs `non-responsive`。
- exp01/02/03 中 `PRISM2nd_label_total` 为空，不适合用于这些 split 的二分类训练。
- 若脚本读取 `PREDICTIONS.tsv`、`DONE`、`EXP07_REFERENCE_EPOCH`，需要注意里面有旧的 `/mnt/shared-storage-user/maoxinjie/...` 绝对路径；当前实际可用文件在上面的 gpfs 路径下。

## 目录结构

顶层有两个模型分支：

```text
official_fixed_residual_20260605/
  cpa/
  biolord/
```

每个模型分支都包含 exp01/02/03 的 5-fold 输出：

```text
{model}/split_vcbench_exp01_pert_stratified_fold{0..4}/
{model}/split_vcbench_exp02_cell_type_fold{0..4}/
{model}/split_vcbench_exp03_cell_fold{0..4}/
```

每个 split 目录下有三类 prediction h5ad：

```text
{split}_train_predictions.h5ad
{split}_val_predictions.h5ad
{split}_test_predictions.h5ad
```

例如：

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/maoxinjie/AIVC/ptv3_single_nonablation/output/official_fixed_residual_20260605/cpa/split_vcbench_exp01_pert_stratified_fold0/split_vcbench_exp01_pert_stratified_fold0_train_predictions.h5ad
```

## 文件内容

每个 exp01/02/03 prediction h5ad 满足：

- `.X`: 模型生成的 9843 维转录组预测矩阵。
- `.var_names`: common HGNC gene symbols，顺序与 `data/transcriptome_anndata/ptv3_single_nonablation/common_gene_axis/ptv3_main_singledrug.common_hgnc.h5ad` 一致。
- `.obs`: 样本 metadata，包含标签、split 标记、药物、细胞系等信息。

常用 obs 列：

```text
PRISM1st_label_total     # 推荐二分类标签: sensitive / non-responsive
PRISM2nd_label_total     # exp01/02/03 为空
cell
cell_type
drug_id
drug_name
drug_smiles
drug_smiles_primary
perturbation
split_vcbench_*
```

## exp01/02/03 split 名称

```python
SPLIT_PREFIXES = {
    "exp01": "split_vcbench_exp01_pert_stratified_fold",
    "exp02": "split_vcbench_exp02_cell_type_fold",
    "exp03": "split_vcbench_exp03_cell_fold",
}
```

每个 experiment 都有 fold0 到 fold4。

## 读取 train/test 特征和标签

下面的示例直接读取 `.X` 作为下游特征，并把 `PRISM1st_label_total` 映射成二分类标签。

```python
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/maoxinjie/AIVC/"
    "ptv3_single_nonablation/output/official_fixed_residual_20260605"
)

LABEL_MAP = {
    "non-responsive": 0,
    "sensitive": 1,
}

def load_prediction_xy(path, label_col="PRISM1st_label_total"):
    adata = ad.read_h5ad(path)
    X = np.asarray(adata.X, dtype=np.float32)

    y_series = adata.obs[label_col].map(LABEL_MAP)
    keep = y_series.notna().to_numpy()

    X = X[keep]
    y = y_series[keep].astype(int).to_numpy()
    obs = adata.obs.loc[keep].copy()
    return X, y, obs

def prediction_path(model, split, part):
    return ROOT / model / split / f"{split}_{part}_predictions.h5ad"

model = "cpa"
split = "split_vcbench_exp01_pert_stratified_fold0"

X_train, y_train, obs_train = load_prediction_xy(
    prediction_path(model, split, "train")
)
X_test, y_test, obs_test = load_prediction_xy(
    prediction_path(model, split, "test")
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(np.bincount(y_train), np.bincount(y_test))
```

## 遍历全部 exp01/02/03 folds

```python
from pathlib import Path

ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/maoxinjie/AIVC/"
    "ptv3_single_nonablation/output/official_fixed_residual_20260605"
)

SPLIT_PREFIXES = {
    "exp01": "split_vcbench_exp01_pert_stratified_fold",
    "exp02": "split_vcbench_exp02_cell_type_fold",
    "exp03": "split_vcbench_exp03_cell_fold",
}

def iter_prediction_files(model):
    for exp_name, prefix in SPLIT_PREFIXES.items():
        for fold in range(5):
            split = f"{prefix}{fold}"
            split_dir = ROOT / model / split
            yield {
                "model": model,
                "exp": exp_name,
                "fold": fold,
                "split": split,
                "train": split_dir / f"{split}_train_predictions.h5ad",
                "val": split_dir / f"{split}_val_predictions.h5ad",
                "test": split_dir / f"{split}_test_predictions.h5ad",
            }

for item in iter_prediction_files("cpa"):
    print(item["exp"], item["fold"], item["train"], item["test"])
```

## CPA 和 BioLord 如何使用

`cpa` 和 `biolord` 两个分支有相同的样本顺序、相同的 gene axis、相同的标签列，但 `.X` 是不同模型生成的预测矩阵。

推荐三种用法：

1. 单独使用 CPA 特征训练一个下游模型。
2. 单独使用 BioLord 特征训练一个下游模型。
3. 将 CPA 和 BioLord 的 `.X` 按列拼接，形成 19686 维特征后训练一个下游模型。

拼接示例：

```python
X_cpa, y_cpa, obs_cpa = load_prediction_xy(
    prediction_path("cpa", split, "train")
)
X_biolord, y_biolord, obs_biolord = load_prediction_xy(
    prediction_path("biolord", split, "train")
)

assert np.array_equal(y_cpa, y_biolord)
assert list(obs_cpa.index) == list(obs_biolord.index)

X_train = np.concatenate([X_cpa, X_biolord], axis=1)
y_train = y_cpa
```

如果样本量不大，可以直接用 9843 维特征；如果下游模型容易过拟合，建议先在 train 上 fit PCA/标准化器，再应用到 test，避免数据泄漏。

## exp07 extra inference

exp07 是 all-train 模型对 extra single-drug 数据的推理结果，位置如下：

```text
{model}/split_vcbench_exp07_all_single_for_extra/extra_single_test_only_predictions/
```

文件包括：

```text
split_vcbench_exp07_all_single_for_extra_mat1_480_faims_test_predictions.h5ad
split_vcbench_exp07_all_single_for_extra_mat1_qe_test_predictions.h5ad
split_vcbench_exp07_all_single_for_extra_mat2_480_faims_test_predictions.h5ad
split_vcbench_exp07_all_single_for_extra_mat2_qe_test_predictions.h5ad
split_vcbench_exp07_all_single_for_extra_mat3_qe_test_predictions.h5ad
split_vcbench_exp07_all_single_for_extra_mat4_qe_test_predictions.h5ad
```

这些文件也是 9843 维 common HGNC gene-axis `.X`。它们适合做额外测试/外部推理，但不是 exp01/02/03 的 fold train/test 文件。

## 已验证的完整性

2026-06-26 检查结果：

- 90 个 exp01/02/03 train/val/test prediction h5ad 全部存在。
- 12 个 exp07 extra single-drug prediction h5ad 全部存在。
- 所有 prediction h5ad 的 gene axis 均为 9843 个 common HGNC genes。
- exp01/02/03 输出的 `obs_names` 顺序与源 AnnData split 的非 control rows 完全一致。
- exp07 extra 输出的 `obs_names` 顺序与源 extra AnnData 的非 control rows 完全一致。
- 对全部 102 个 prediction h5ad 的 `.X` 抽样检查没有发现 NaN/Inf。

更详细的审计记录见：

```text
data/review_summary/2026-06-26_1539_official_fixed_residual_transcriptome_output_review.md
```

