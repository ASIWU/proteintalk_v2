# 2026-06-09 21:55 HKT PTV1 Cell-Type API Embedding Tuning Completion Review

## Scope

Reviewed and completed the PTV1 graph + frozen Cell LLM + frozen `cell_type` LLM migration run after regenerating the PTV1 `cell_type` embedding through the API path.

## Files And Artifacts Reviewed

- `utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py`
- `utils/11_build_cell_type_llm_embeddings.py`
- `utils/ptv1/07_build_ptv1_cell_llm_embeddings.py`
- `train.py`
- `infer.py`
- `scripts/ptv1/ptv1_experiment_common.sh`
- `scripts/ptv1/run_ptv1_cell_celltype_llm_fine_tune_search.sh`
- `scripts/ptv1/0427_build_ptv1_derived.sh`
- `scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`
- `logs/20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype_cell_celltype_tune_results.md`
- `outputs/20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype_cell_celltype_tune_results.tsv`
- `docs/2026-06-09_ptv1_cell_celltype_llm_exp11_13_tuned_results.md`

## Completion Summary

- Generated `data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz` with `proxy_on2` and `.env` API credentials.
- Confirmed artifact shape `(2, 4096)`, row 0 zero vector, row 1 `BREAST` normalized finite vector.
- Confirmed PTV1 Cell LLM artifact `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz` has shape `(20, 4096)` and aligns to `Cell_index`.
- Completed full PTV1 tuning prefix `20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype`.
- exp11 best: `mse075_drop010`, AUPRC/AUROC `0.914806 / 0.955703`.
- exp12 best: `mse075_drop010`, mean5 AUPRC/AUROC `0.624837 / 0.760972`.
- exp13 valid best: `mse050_target_pdi`, AUPRC/AUROC `0.601089 / 0.627653`.
- exp13 all-checkpoint oracle best: `mse050_lr1e4`, AUPRC/AUROC `0.636390 / 0.652502`, epoch `16`.

## Audit Results

- Static shell check passed: `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`.
- Static Python check passed: `python -m py_compile train.py infer.py utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`.
- Embedding validation passed for both PTV1 Cell LLM v2 and PTV1 `cell_type` LLM v3.
- TSV audit passed: 13 candidates, 65 exp12 folds, exp13 valid/oracle rows 218 for every candidate.
- Manifest audit passed: 13 exp11 manifests and 65 exp12 manifests use graph `real`, Cell LLM `frozen`, `cell_type` LLM `frozen`, and PTV1 `cell_type` embedding v3.

## Notes

- The exp13 oracle is diagnostic only because it selects a checkpoint using exp13 labels.
- No official row uses `GRAPH_FEATURE_MODE=off`.
- The workspace already had many unrelated modified/untracked files; this review did not revert or normalize them.
