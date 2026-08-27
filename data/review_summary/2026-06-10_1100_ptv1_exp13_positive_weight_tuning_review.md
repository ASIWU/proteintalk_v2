# 2026-06-10 11:00 HKT PTV1 exp13 Positive-Weight Tuning Review

## Scope

Reviewed the PTV1 graph + frozen Cell LLM + frozen cell-type LLM tuning workflow and added a dedicated positive-weight sweep for the exp13 valid-best baseline `mse050_target_pdi`.

## Files Reviewed

- `scripts/ptv1/ptv1_experiment_common.sh`
- `scripts/ptv1/exp_11_ptv1_random_split.sh`
- `scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh`
- `scripts/ptv1/run_ptv1_cell_celltype_llm_fine_tune_search.sh`
- `scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`
- `scripts/ptv3_experiment_common.sh`
- `train.py`
- `infer.py`
- `data/training_ready/ptv1/splits/ptv1_aivc/train_indices_fixed_experiment_type.pkl`
- `data/training_ready/ptv1/tasks/ptv1_aivc/feature_table.parquet`

## Implementation Summary

- Added `scripts/ptv1/run_ptv1_exp13_positive_weight_tune.sh`.
- Added `scripts/ptv1/report_ptv1_exp13_positive_weight_tune.py`.
- Fixed all non-positive-weight hyperparameters to the exp13 valid-best `mse050_target_pdi` configuration.
- Added `neg/pos` resolution from the exp11 train split: negatives `4688`, positives `2353`, ratio `1.9923501912`.
- Added oracle checkpoint exp11 test inference for the per-candidate best exp13 oracle epoch.
- Added strict reporting/audit checks for candidate count, row counts, checkpoint existence, graph-on policy, Cell LLM frozen v2, cell-type LLM frozen v3, and `MSE_TARGET_MODE=pdi`.

## Verification

- Passed: `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`.
- Passed: `python -m py_compile train.py infer.py scripts/ptv1/report_ptv1_exp13_positive_weight_tune.py scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`.
- Passed: `python utils/ptv1/03_validate_ptv1_training_ready.py`.
- Passed: `python utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only --output data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz`.
- Passed: `python utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py --validate-only --output data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz`.
- Passed no-op runner initialization with all run phases disabled; it resolved `neg/pos` as `4688 / 2353 = 1.9923501912`.

## Notes

- The full positive-weight GPU sweep was not launched in this review session.
- The generated report path is `docs/2026-06-10_ptv1_exp13_positive_weight_tuning_report.md`; it currently records the launch contract and will be overwritten with metric rows by the runner after completion.
- The workspace already contained many unrelated modified/untracked files; this review did not revert them.
