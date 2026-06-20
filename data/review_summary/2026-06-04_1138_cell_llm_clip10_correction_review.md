# 2026-06-04 11:38 HKT Cell LLM Clip10 Correction Review

## Scope

- Reviewed the LLM cell embedding path after identifying that the prior artifact used tissue-level `cell_type`.
- Implemented the corrected Cell-line LLM embedding and selected-suite launcher for the 2026-06-04 clip10 rerun.

## Findings

- The previous `cell_type_llm` path could be pointed at `Cell_index`, but still exposed the old public interface and defaulted to tissue `cell_type`; this made accidental reuse likely.
- Correct behavior is now fixed to `Cell_index` with no index-column override.
- The new artifact has 74 rows, matching `global_meta["value_to_index"]["Cell"]`; row 0 is the `no` zero vector.

## Changes Reviewed

- `train.py` / `infer.py`: public CLI and manifest keys migrated to `cell_llm_*`; loader fixed to `Cell_index`.
- `scripts/ptv3_experiment_common.sh` and `scripts/report_cell_drug_time_eval.py`: switched to `CELL_LLM_*` and `--cell-llm-*`.
- `utils/11_build_cell_llm_embeddings.py`: builds Qwen3 embeddings from `Cell` records while preserving the original prompt text.
- `scripts/run_cell_llm_clip10_selected_suite.sh`: added corrected exp01-exp08 runner with clip10 and artifact fail-fast validation.

## Validation

- `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py scripts/report_cell_drug_time_eval.py utils/11_build_cell_llm_embeddings.py`
- `bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_selected_suite.sh scripts/run_unseen_cell_llm_condition_search.sh scripts/ptv1/ptv1_experiment_common.sh scripts/ptv1/run_ptv1_param_search.sh`
- Artifact checks: shape `(74, 4096)`, max `Cell_index=73`, row 0 zero, nonzero rows finite, sidecar has no old cell-type LLM keys.

## Remaining Work

- The full corrected exp01-exp08 training suite was not launched in this code-update pass.
- After the suite completes, append the corrected report metrics to `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`.
