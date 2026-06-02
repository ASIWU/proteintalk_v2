# 2026-06-01 22:07 HKT LLM Cell-Type Embedding Experiment Review

## Scope

- Reviewed the fast training/inference path needed to add frozen LLM cell-type embeddings.
- Reviewed report propagation so materialized fold predictions keep the same `cell_type_llm` settings as training.
- Ran exp01 tuning and the full exp01-exp08 suite with `CELL_TYPE_LLM_MODE=frozen`.

## Implementation Notes

- `utils/11_build_cell_type_llm_embeddings.py` creates the frozen artifact from `.env` credentials and the `ptv3` training-ready metadata.
- `train.py` and `infer.py` both resolve the artifact through the same default derived path.
- `FastProteinTalkDataset` attaches per-row `cell_type_features`.
- `FastDeltaDrugResponseModel` concatenates categorical covariate embeddings and frozen cell-type features before the existing covariate projection.
- `scripts/ptv3_experiment_common.sh` carries `CELL_TYPE_LLM_MODE` and optional `CELL_TYPE_LLM_EMBEDDING_PATH` through train/infer/report commands.
- `scripts/report_cell_drug_time_eval.py` forwards the LLM flags when it materializes fold predictions.

## API and Artifact Check

- `proxy_on2` was enabled for network access.
- `ALL_PROXY` was unset for OpenAI-compatible client calls because `flow_v2` does not have SOCKS support installed.
- `gpt-5.4` chat completion returned successfully.
- `Qwen/Qwen3-Embedding-8B` embedding returned successfully with dimension `4096`.
- Artifact created:
  - `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.npz`
  - `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.json`

## Experiments

- exp01 tuning report: `logs/20260601_llm_celltype_exp01_v1_param_search_report.md`
- Selected exp01 config for full run: `mse050`.
- Full run prefix: `20260601_llm_celltype_full_v1`.
- Final report:
  - `logs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.md`
  - `outputs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.csv`
  - `outputs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.json`
- Result document: `docs/2026-06-01_llm_celltype_embedding_experiment_results.md`

## Validation

- `python -m py_compile utils/11_build_cell_type_llm_embeddings.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py train.py infer.py scripts/report_cell_drug_time_eval.py` passed.
- `bash -n scripts/ptv3_experiment_common.sh scripts/run_dose_param_search.sh scripts/exp_01_single_pert_stratified_5fold.sh scripts/exp_07_extra_single_all_train_infer.sh scripts/exp_08_extra_double_all_train_infer.sh` passed.
- Full suite completed with final `[done] llm-celltype full suite completed for BASE_PREFIX=20260601_llm_celltype_full_v1`.

## Notes

- exp07 contains the available extra single-drug mat subsets.
- guomics, nature, and nc are reported under exp08 as extra double-drug subsets, each split into `unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`, and `combined`.
- Extra datasets have missing or partial time metadata, so `missing_time` is nonzero for the collapsed by-last-time rows; the final report keeps those counts visible.
