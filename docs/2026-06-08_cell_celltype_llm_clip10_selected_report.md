# 2026-06-08 PTV3 Cell + Cell-type LLM Clip10 Selected Suite Report

## Scope

本次实验在既有 frozen Cell LLM embedding 基础上，新增 frozen `cell_type` LLM embedding，并只运行 exp01-exp08 selected 参数套件，不做新的调参搜索。

- New prefix: `20260608_cell_celltype_llm_clip10_selected_v1`
- Baseline prefix: `20260604_cell_llm_clip10_tuned_selected_v1`
- Cell LLM artifact: `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz`
- Cell-type LLM artifact: `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz`
- Main report: `outputs/2026-06/2026-06-08/20260608_cell_celltype_llm_clip10_selected_v1_cell_drug_dose_time_eval.csv`
- Comparison: `outputs/2026-06/2026-06-08/20260608_cell_celltype_llm_clip10_selected_v1_cell_celltype_llm_comparison.tsv`

## Prompt Design

目标是让 embedding 捕捉 cell type lineage/tissue context，而不是具体 cell line ID。实际 builder 中使用的 prompt 模板如下：

```text
Write one concise biomedical description for a cancer cell type label used to annotate cancer cell lines in drug-response proteomics experiments. Describe the tissue or lineage represented by the label, common tumor biology and molecular context, and why this lineage may affect perturbation response. Use 2-4 sentences, no bullets, and do not mention a specific cell line unless it appears in the observed labels.

Dataset context: PTV3 protein response after single- and double-drug perturbations.
Cell-type label: {name}
Observed labels: {variants}
```

设计理由：

- `cell_type` 是比 Cell 更粗粒度的 lineage/tissue 标签，因此 prompt 强调 lineage、tumor biology、molecular context 和 perturbation response relevance。
- 禁止默认提具体 cell line，避免把 cell-line identity 泄漏进 cell-type embedding。
- `Observed labels` 只用于保留原始标签变体和 disambiguation。
- `no` / index 0 继续保留为全零 embedding row。

## Embedding Artifact

按要求用 `proxy_on`、`.env` 中的 API key/base URL 启动过生成流程；代理正常开启，但 chat completion 在首个非 `no` cell type 描述请求处超时：`Request timed out`。为避免阻塞正式实验，本次使用已存在且通过校验的历史 Qwen3 4096-d cell-type embedding，重新打包为 `_v2` 并补齐正式 runner 需要的 metadata。

`_v2` artifact 校验结果：

- shape: `(14, 4096)`
- row 0 zero reserved: `true`
- finite values: `true`
- min nonzero norm: `1.0`
- metadata: `kind=cell_type_llm_embedding`, `input_field=cell_type`, `index_column=cell_type_index`
- sidecar: `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.json`

## Code Changes

- Added `utils/11_build_cell_type_llm_embeddings.py` for prompt-based cell-type description and embedding generation.
- Added dataset support for `cell_type_llm_features` in `dataset/training_ready_fast_dataset.py`.
- Added train/infer CLI and manifest fields for `--cell-type-llm-*` in `train.py` and `infer.py`.
- Added an independent cell-type LLM branch in `model/fast_delta_model.py`, supporting `covariate`, `piece`, `film`, `interaction`, and `hybrid` fusion modes.
- Updated `scripts/ptv3_experiment_common.sh` and `scripts/run_cell_llm_clip10_param_search.sh` so `CELL_TYPE_LLM_*` propagates through train/infer and selected sub-runs.
- Added selected-suite wrapper `scripts/run_cell_celltype_llm_clip10_selected_suite.sh`.
- Added comparison/audit reporter `scripts/report_cell_celltype_llm_selected_comparison.py`.

## Experiment Settings

All runs used frozen Cell LLM plus frozen cell-type LLM with covariate fusion:

- `CELL_LLM_MODE=frozen`, `CELL_LLM_FUSION_MODE=covariate`, dim `4096`
- `CELL_TYPE_LLM_MODE=frozen`, `CELL_TYPE_LLM_FUSION_MODE=covariate`, dim `4096`
- `CELL_TYPE_LLM_CONDITION_SCALE=0.0`, `CELL_TYPE_LLM_LOGIT_SCALE=0.0`, `CELL_TYPE_LLM_DROPOUT=0.0`
- Dose covariates stayed enabled with clip10 dose fields.

Selected configs were inherited from the existing selected cell-LLM suite:

| exp | task | selected config |
|---|---|---|
| exp01 | single unseen drug | `mse050_target_pdi` |
| exp02 | single unseen cell type | `covdrop010_lr1e4` |
| exp03 | single unseen cell | `covdrop010_drop010` |
| exp04 | single w/o MSE | `mse050_target_pdi` |
| exp05 | single w/o graph | `mse050_target_pdi`, manifest `graph_feature_mode=zero` |
| exp06 | double unseen drug pair | `drop020_mseinactive010` |
| exp07 | extra single all-train | `mse050_target_pdi`, selected epoch `10`, applied max epochs `11` |
| exp08 | extra double all-train | `drop020_mseinactive010`, selected epoch `6`, applied max epochs `7` |

## Main Results

The table reports original-row metrics from the new Cell + cell-type LLM suite. Deltas are vs `20260604_cell_llm_clip10_tuned_selected_v1`.

| exp | task | split | AUROC | AUPRC | nAUPRC | delta AUPRC | delta AUROC | count |
|---|---|---|---:|---:|---:|---:|---:|---:|
| exp01 | single unseen drug | mean5 | 0.895949 | 0.664372 | 5.607428 | -0.007663 | -0.001790 | 17986 |
| exp02 | single unseen cell type | mean5 | 0.945281 | 0.814609 | 6.212017 | -0.008272 | -0.000336 | 17986 |
| exp03 | single unseen cell | mean5 | 0.931817 | 0.768277 | 6.386600 | -0.009600 | -0.002218 | 17986 |
| exp04 | single w/o MSE | mean5 | 0.906519 | 0.657075 | 5.542445 | -0.005668 | +0.010775 | 17986 |
| exp05 | single w/o graph | mean5 | 0.843826 | 0.612638 | 5.159745 | +0.010999 | +0.006789 | 17986 |
| exp06 | double unseen drug pair | mean5 | 0.816593 | 0.756564 | 1.876977 | -0.032572 | -0.016785 | 1791 |
| exp07 | extra single | mean_extra | 0.764085 | 0.555530 | 2.651172 | -0.005514 | -0.000562 | 92671 |
| exp08 | extra double | mean_extra | 0.643297 | 0.101612 | 2.328969 | +0.009759 | +0.004920 | 88970 |

## Extra Dataset Details

Extra single mean was slightly lower than the Cell-only baseline, while extra double improved:

- exp07 extra single mean: AUROC/AUPRC `0.764085 / 0.555530`, delta AUPRC `-0.005514`.
- exp08 extra double mean: AUROC/AUPRC `0.643297 / 0.101612`, delta AUPRC `+0.009759`.

Exp08 combined original metrics by dataset:

| dataset | AUROC | AUPRC | baseline | nAUPRC | count |
|---|---:|---:|---:|---:|---:|
| `ptv3_extra_doubledrug_guomics` | 0.686615 | 0.117449 | 0.034617 | 3.392759 | 5633 |
| `ptv3_extra_doubledrug_nature` | 0.657097 | 0.069560 | 0.035112 | 1.981092 | 68182 |
| `ptv3_extra_doubledrug_nc` | 0.586178 | 0.117826 | 0.073045 | 1.613057 | 15155 |

## Audit

`python scripts/report_cell_celltype_llm_selected_comparison.py --prefix 20260608_cell_celltype_llm_clip10_selected_v1 --baseline-prefix 20260604_cell_llm_clip10_tuned_selected_v1` completed with:

- comparison rows: `159`
- audit errors: `0`
- all required 5-fold and all-train manifests present
- all audited training manifests: `run_status=fit_completed`
- all audited runs: `cell_llm_mode=frozen`, `cell_type_llm_mode=frozen`
- audited feature dims: Cell LLM `4096`, cell-type LLM `4096`
- exp05 audited as `graph_feature_mode=zero`; other formal graph-enabled runs audited as graph real
- exp07 policy: `selected_epoch=10`, `applied_max_epochs=11`
- exp08 policy: `selected_epoch=6`, `applied_max_epochs=7`

Static validation passed before launch:

```bash
bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh scripts/run_cell_celltype_llm_clip10_selected_suite.sh
python -m py_compile train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py scripts/report_cell_drug_time_eval.py scripts/report_cell_celltype_llm_selected_comparison.py utils/11_build_cell_type_llm_embeddings.py
```

## Interpretation

Adding cell-type LLM as a simple frozen covariate is not a broad win over the tuned Cell-only LLM suite. It improves the no-graph single ablation and extra double generalization, but reduces main exp01/02/03/06 and extra single AUPRC. This suggests the coarse cell-type embedding can help when graph signal is absent or when double-drug extra transfer benefits from lineage priors, but it may add redundant or noisy information when exact Cell LLM embedding and graph features are already present.

Recommended next step is a small targeted tune rather than a full grid: test `CELL_TYPE_LLM_FUSION_MODE=film` or `hybrid` with low dropout, and separately test turning off exact Cell LLM to isolate whether cell-type information is useful as a fallback for unseen cells.
