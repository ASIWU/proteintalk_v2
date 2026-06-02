# VCBench Benchmark Data Format

## 1. AnnData 总体要求

每个数据集必须是一个可由 `scanpy.read_h5ad()` 读取的 AnnData 文件。

必须包含：

- `adata.X`：细胞 x 基因表达矩阵。VCBench 会在 datamodule 中转为 dense `float32`。
- `adata.obs`：每个细胞的 metadata。训练、验证、测试划分和扰动信息都从这里读取。
- `adata.var_names`：基因名或 feature id。不同数据集之间不强制完全一致，但同一个运行内 train/val/test 使用同一个 h5ad 的同一套基因。
- `adata.obs_names`：细胞 id。若使用外部 split CSV，CSV index 必须与 `adata.obs_names` 对齐。

推荐包含：

- `adata.layers["counts"]`：原始 counts，便于保留未归一化表达。当前示例 sweep 不依赖它；启用 `data.raw_counts_key` 前请确认模型/数据集代码路径是否按预期读取 raw counts。
- `adata.obsm["stack-embed"]`：预计算细胞 embedding。只有在配置 `data.embedding_key=stack-embed` 或模型使用 cell embedding 时才会读取。
- `adata.var["highly_variable_rank"]`：HVG 排名。只有在配置 `data.inference_top_hvg` 时才会读取。

## 2. `model_related/` 文件说明

`model_related/` 存放 perturbation encoder 和 GEARS 需要的外部资源。当前示例目录包含：

| 文件 | 用途 | 何时需要 | 配置项 |
| --- | --- | --- | --- |
| `ESM2_pert_features.pt` | gene perturbation 的 ESM2 embedding 字典 | 数据中有 `gene_pt`，并希望使用预训练 gene embedding 时 | `data.transform.gene_map_path` |
| `SMILES_pert_features.pt` | drug perturbation 的 SMILES/molecule embedding 字典 | 数据中有 `drug_pt`，并希望使用预训练 drug embedding 时 | `data.transform.drug_map_path` |
| `gene2go.pkl` | gene 到 GO term 的映射，GEARS 构图使用 | 运行 GEARS 时 | `model.gene2go_path` |
| `essential_all_data_pert_genes.pkl` | perturbation gene set，GEARS 使用 | 运行 GEARS 时 | `model.gene_set_path` |

`MixPertTransform` 会用 `torch.load(..., weights_only=False)` 读取 `.pt` 文件。它期望读取结果是一个 dict，key 是 perturbation 名称，value 是对应的 embedding tensor。

### 2.1 Gene perturbation

如果数据集是基因扰动，例如 Norman、Replogle、Adamson：

- `adata.obs["gene_pt"]` 中的每个 gene token 必须能在 `ESM2_pert_features.pt` 中找到对应 key，或让代码 fallback 到 one-hot。
- 组合基因扰动使用 `+` 分隔，例如 `KLF1+BAK1`；代码会按 `+` 拆开后分别查 embedding。
- control / non-gene perturbation 的 `gene_pt` 应填空字符串 `""`，空字符串会作为 null token 使用。

推荐配置：

```yaml
data: mix_pert
data.transform.gene_map_path: ./tasks_data/model_related/ESM2_pert_features.pt
data.transform.drug_map_path: null
```

### 2.2 Drug perturbation

如果数据集是药物扰动，例如 Sciplex、McFarland：

- `adata.obs["drug_pt"]` 中的每个 drug token 必须能在 `SMILES_pert_features.pt` 中找到对应 key，或让代码 fallback 到 one-hot。
- 示例数据里 `drug_pt` 通常使用小写药名，例如 `trametinib`、`dabrafenib`、`gemcitabine`、`tak-901`。
- 组合药物扰动同样使用 `+` 分隔。
- control / non-drug perturbation 的 `drug_pt` 应填空字符串 `""`。

推荐配置：

```yaml
data: mix_pert
data.transform.gene_map_path: null
data.transform.drug_map_path: ./tasks_data/model_related/SMILES_pert_features.pt
```

### 2.3 Environment 和 CRISPR perturbation

`env_pt` 和 `CRISPR` 默认没有外部 embedding 文件：

- `env_pt` 默认由当前数据中的 unique value 自动生成 one-hot，例如 Kang2018 的 `IFNB`。
- `CRISPR` 默认由当前数据中的 unique value 自动生成 one-hot，例如 `CRISPRi`。
- 如果将来准备了环境或 CRISPR embedding 字典，可分别设置 `data.transform.env_map_path` 和 `data.transform.crispr_map_path`。

Kang2018 属于环境扰动，通常不需要 `ESM2_pert_features.pt` 或 `SMILES_pert_features.pt`：

```yaml
data: mix_pert
data.transform.gene_map_path: null
data.transform.drug_map_path: null
data.transform.env_map_path: null
```

### 2.4 同时包含多种 perturbation

如果一个数据集同时有 gene 和 drug perturbation，需要同时提供两个 map：

```yaml
data.transform.gene_map_path: ./tasks_data/model_related/ESM2_pert_features.pt
data.transform.drug_map_path: ./tasks_data/model_related/SMILES_pert_features.pt
```

未提供 map 的 perturbation 类型会按当前 h5ad 中出现过的 token 生成 one-hot。这样可以运行，但不同数据集/不同 split 之间的表示空间可能不如预训练 embedding 稳定。

### 2.5 GEARS 额外资源

运行 GEARS 除了 h5ad 本身，还需要：

```yaml
model.gene2go_path: ${paths.data_dir}/model_related/gene2go.pkl
model.gene_set_path: ${paths.data_dir}/model_related/essential_all_data_pert_genes.pkl
```

本地示例中：

- `gene2go.pkl` 是 `dict[str, set[str]]`，例如 gene symbol 到 GO ids 的映射。
- `essential_all_data_pert_genes.pkl` 是 perturbation gene 列表/数组。

GEARS 的 `model.data_path` 还需要指向包含 `go.csv` 和 `co-express.csv` 的预计算目录，例如 `./tasks_data/gears_norman` 或 `./tasks_data/gears_replogle`。

## 3. `mix_pert` 数据模式的必需列

当前 sweep 大多使用 `data=mix_pert`。此模式把 gene、drug、environment、CRISPR 四类 perturbation 分开编码，然后在内部合并成 `_merged_pert_col_`。

`adata.obs` 必须包含下列列：

| 列名 | 类型/取值 | 作用 |
| --- | --- | --- |
| `gene_pt` | string/category；无基因扰动时填空字符串 `""` | 基因扰动名称。组合扰动用 `+` 连接，例如 `KLF1+BAK1`。 |
| `drug_pt` | string/category；无药物扰动时填 `""` | 药物扰动名称。需要与 `SMILES_pert_features.pt` 的 key 对齐，或由 one-hot 自动生成。 |
| `env_pt` | string/category；无环境扰动时填 `""` | 环境/刺激类扰动，例如 Kang2018 的 `IFNB`。 |
| `CRISPR` | string/category；无 CRISPR 时填 `""` | CRISPR 类型，例如 `CRISPRi`。 |
| `control` | bool | 是否为 control 细胞。`True` 表示 control，`False` 表示 perturbed。 |
| `split` | string/category；`train`、`val`、`test` | 数据划分。若通过 `data.task` 指定外部 CSV，可由 CSV 注入。 |
| 配置中的 `data.cov_keys` | string/category | 用于匹配 perturbed cell 和 control cell 的协变量。 |
| 配置中的 `data.result_avg_keys` | string/category | 用于评价结果聚合；如果不填，默认等于 `cov_keys`。 |

注意：

- control 细胞建议同时满足 `control=True`，并且 `gene_pt/drug_pt/env_pt/CRISPR` 为空字符串。
- 非 control 细胞应至少有一个 perturbation 列非空。
- 如果 `control` 列存在且为 bool，VCBench 优先使用它判断 control；否则会根据四个 perturbation 列是否为空或等于 `control_val` 推断。
- 组合扰动默认分隔符是 `+`，由 `data.perturbation_combination_delimiter` 和 `data.transform.comb_delim` 控制。

## 5. 常见配置对应的必需协变量列

必需的协变量列取决于运行时 YAML/CLI 配置，而不是固定在文件名中。示例 sweep 中的常见设置如下：

| 数据集 | 推荐 `data.cov_keys` | 推荐 `data.result_avg_keys` | 额外说明 |
| --- | --- | --- | --- |
| `norman2019_comb_stack.h5ad` | `[split_category]` | `[split_category]` | 用于区分 single/combo unseen 任务类别。 |
| `ReplogleWeissman2022_K562_stack_hvg_split.h5ad` | `[cell_line]` | `[cell_line]` | 示例中只有 K562。 |
| `SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad` | `[cell_line]` | `[cell_line]` | 三个 cell line：MCF7、K562、A549。 |
| `Kang2018_CD4Tcells_stack_split.h5ad` | `[cell_cluster, dataset]` 或按任务改成 `[cell]`/`[cell_cluster]` | 通常 `[cell_cluster]` | 文件中 `cell_cluster` 是 patient id，`cell` 是 CD4 T cells。 |
| `McFarlandTsherniak2020_filtered_stack_dose01time24.h5ad` | `[cell_cluster, dataset]` 或 `[cell_line]` | 通常 `[cell_cluster]`/`[cell_line]` | 文件中 `cell_cluster` 和 `cell_line` 都表示 cell line。 |
| Adamson cross-dataset files | `[cell_cluster]` 或 `[cell_line]` | `[cell_cluster]` | 示例中均为 K562。 |

如果你改了 YAML 中的 `data.cov_keys` 或 `data.result_avg_keys`，对应列就必须存在于 `adata.obs`。

## 6. Kang2018 示例

文件：`tasks_data/unseen_cells/Kang2018_CD4Tcells_stack_split.h5ad`

实际结构：

- shape：`11238 cells x 2000 genes`
- `obsm`：`X_pca`、`X_umap`、`stack-embed`
- `layers`：`counts`
- `var` 中包含：`name`、`highly_variable`、`highly_variable_rank` 等 HVG 信息

对 VCBench 必需/常用的 `obs` 列：

| 列名 | 示例值 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| `gene_pt` | `""` | 是 | Kang2018 不是基因扰动，因此为空。 |
| `drug_pt` | `""` | 是 | Kang2018 不是药物扰动，因此为空。 |
| `env_pt` | `""`、`IFNB` | 是 | IFN-beta stimulation 作为环境扰动编码。 |
| `CRISPR` | `""` | 是 | 非 CRISPR 数据，因此为空。 |
| `control` | `True`、`False` | 是 | control vs stimulated。 |
| `split` | `train`、`val`、`test` | 是 | 内置划分。 |
| `cell_cluster` | `patient_1488` 等 | 取决于配置，常用 | patient/batch 级协变量；也常用于结果聚合。 |
| `dataset` | `Kang2018_CD4Tcells` | 取决于配置，常用 | 数据集标签。 |
| `cell` | `CD4 T cells` | 取决于配置 | cell type 标签。 |
| `perturbation` | `control`、`IFNB` | 推荐 | 原始 perturbation 名称，便于检查和兼容其他模式。 |
| `perturbation_type` | `IFN-beta` | 推荐 | 扰动类型 metadata。 |
| `batch` / `replicate` | `patient_...` | 推荐 | 原始 patient/batch metadata。 |

若使用默认 `mix_pert.yaml`：

```yaml
data: mix_pert
data.data_path: ./tasks_data/unseen_cells/Kang2018_CD4Tcells_stack_split.h5ad
data.gene_key: gene_pt
data.drug_key: drug_pt
data.env_key: env_pt
data.crispr_key: CRISPR
data.control_val: control
data.cov_keys: [cell_cluster, dataset]
data.result_avg_keys: [cell_cluster]
data.perturbation_combination_delimiter: +
```

则 Kang2018 至少需要这些 `obs` 列：

```text
gene_pt, drug_pt, env_pt, CRISPR, control, split, cell_cluster, dataset
```

如果改成 `data.cov_keys: [cell]`，则还必须有 `cell` 列。

Kang2018 的 perturbation 信息在 `env_pt` 中，因此一般不需要 `ESM2_pert_features.pt` 或 `SMILES_pert_features.pt`；`IFNB` 会通过 `env_pt` 的 one-hot map 编码。

## 7. `gears` / 单一 perturbation 模式的必需列

如果使用 `data=gears` 或非 `mix_pert` 模式，VCBench 不使用 `gene_pt/drug_pt/env_pt/CRISPR` 四列，而是使用一个统一的 perturbation 列。

默认配置：

```yaml
data: gears
data.pert_key: perturbation
data.control_val: control
data.cov_keys: [cell_cluster]
data.result_avg_keys: [cell_cluster]
```

此时 `adata.obs` 至少需要：

```text
perturbation, control, split, cell_cluster
```

其中 `perturbation` 对 control 细胞通常为 `control`，非 control 细胞为基因/药物/组合扰动名称。

## 8. 示例数据集字段概览

| 文件 | shape | 关键 perturbation 列 | 关键 covariate/result 列 | 可选资源 |
| --- | --- | --- | --- | --- |
| `Kang2018_CD4Tcells_stack_split.h5ad` | `11238 x 2000` | `env_pt` = `IFNB`；`gene_pt/drug_pt/CRISPR` 为空 | `cell_cluster`、`cell`、`dataset` | `obsm["stack-embed"]`、`layers["counts"]`、`var["highly_variable_rank"]` |
| `McFarlandTsherniak2020_filtered_stack_dose01time24.h5ad` | `48310 x 2000` | `drug_pt` = `trametinib/dabrafenib/gemcitabine` | `cell_cluster`、`cell_line`、`dataset` | `obsm["stack-embed"]`、`layers["counts"]`、HVG rank |
| `norman2019_comb_stack.h5ad` | `118856 x 2000` | `gene_pt`，含单基因和 `+` 组合 | `split_category`、`cell_cluster` | `obsm["stack-embed"]` |
| `ReplogleWeissman2022_K562_stack_hvg_split.h5ad` | `152915 x 2000` | `gene_pt`，CRISPRi | `cell_line`、`cell_cluster`、`dataset` | `obsm["stack-embed"]`、`layers["counts"]`、HVG rank |
| `SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad` | `164771 x 2000` | `drug_pt` | `cell_line`、`cell_cluster`、`dataset` | `obsm["stack-embed"]`、`layers["counts"]`、HVG rank |
| `AdamsonWeissman2016_GSM2406675_10X001_processed.h5ad` | `5752 x 35635` | `gene_pt`，CRISPRi | `cell_line`、`cell_cluster` | 无 embedding/layers |
| `AdamsonWeissman2016_GSM2406681_10X010_filter.h5ad` | `56380 x 32738` | `gene_pt`，CRISPRi | `cell_line`、`cell_cluster` | 无 embedding/layers |

对应的 feature map 选择：

| 数据类型 | 示例数据集 | 推荐 feature map |
| --- | --- | --- |
| Gene / CRISPR gene perturbation | Norman、Replogle、Adamson | `data.transform.gene_map_path=./tasks_data/model_related/ESM2_pert_features.pt` |
| Drug perturbation | Sciplex、McFarland | `data.transform.drug_map_path=./tasks_data/model_related/SMILES_pert_features.pt` |
| Environment perturbation | Kang2018 | 默认 one-hot；通常不需要外部 map |

## 9. 外部 split CSV 格式

如果 h5ad 内没有 `obs["split"]`，可以通过 `data.task` 和 `data.split_dir` 让 VCBench 从外部 CSV 读取划分。

要求：

- CSV 第一列作为 index，必须是 cell id，并能匹配 `adata.obs_names`。
- 必须包含 `split` 列。
- `split` 只能是 `train`、`val`、`test`；空值会被忽略。
- 可选 `sample_count` 列，用于按 cell 重复采样；小于 0 会按 0 处理。

示例：

```csv
cell_id,split,sample_count
AAACCTGAGAAACCAT-1,train,1
AAACCTGAGAAACCGC-1,val,1
AAACCTGAGAAACCTA-1,test,1
```

## 10. 快速校验脚本

下面的脚本可用于检查一个 h5ad 是否满足当前配置需要的列：

```python
import anndata as ad

path = "./tasks_data/unseen_cells/Kang2018_CD4Tcells_stack_split.h5ad"
cov_keys = ["cell_cluster", "dataset"]
result_avg_keys = ["cell_cluster"]

required_obs = [
    "gene_pt",
    "drug_pt",
    "env_pt",
    "CRISPR",
    "control",
    "split",
    *cov_keys,
    *result_avg_keys,
]

adata = ad.read_h5ad(path, backed="r")
missing = sorted(set(required_obs) - set(adata.obs.columns))
if missing:
    raise ValueError(f"Missing obs columns: {missing}")

bad_split = set(adata.obs["split"].astype(str).unique()) - {"train", "val", "test"}
if bad_split:
    raise ValueError(f"Invalid split values: {bad_split}")

if adata.obs["control"].dtype != bool:
    raise TypeError("obs['control'] should be boolean.")

print("OK", adata.shape)
```

## 11. 常见问题

- `KeyError: ... not found in obs`：检查 YAML 中的 `data.cov_keys`、`data.result_avg_keys`、`gene_key/drug_key/env_key/crispr_key` 是否和 h5ad 列名一致。
- 预训练 perturbation embedding 查不到 key：确认 `gene_pt` token 与 `ESM2_pert_features.pt` 的 gene key 一致，或 `drug_pt` token 与 `SMILES_pert_features.pt` 的 drug key 一致；注意大小写和 `+` 分隔后的单个 token。
- `split column not found`：在 h5ad 的 `obs` 中加入 `split`，或提供 `data.task` + 外部 split CSV。
- control 匹配不到：确认每个 `cov_keys` 组合下至少有 control 细胞，否则训练/推理时无法为 perturbed cell 采样对应 control。
- 使用 `data.embedding_key=stack-embed` 报错：确认 `adata.obsm["stack-embed"]` 存在。
- 使用 `data.raw_counts_key` 报错：当前主线 sweep 通常不设置该项；若要启用 raw counts，请先确认对应代码路径读取的是 `layers` 还是 `obs` 字段，并保持 h5ad 与代码一致。
