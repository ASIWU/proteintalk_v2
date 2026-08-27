# 2026-06-08 21:42 HKT Cell + Cell-type LLM Clip10 Tuning Launch Review

## Scope

- Implemented the tuning plan for the new graph + Cell LLM + cell-type LLM architecture.
- Kept exp05 as the no-graph diagnostic ablation while requiring Cell and cell-type LLM embeddings for the new suite.
- Restricted this launch to legacy hyperparameter tuning with fixed covariate LLM fusion.

## Files Reviewed and Updated

- `scripts/run_cell_llm_clip10_param_search.sh`
- `scripts/run_cell_celltype_llm_clip10_param_search.sh`
- `scripts/report_cell_celltype_llm_clip10_param_search.py`
- `scripts/report_cell_llm_clip10_param_search.py`
- `train.py`
- `infer.py`
- `dataset/training_ready_fast_dataset.py`
- `model/fast_delta_model.py`

## Verification

- Static shell check passed for the modified runners.
- Python compile checks passed for the new reporter and existing base reporter.
- The wrapper validated the fixed Cell and cell-type embedding artifacts before launch.
- Detached training was launched with:
  - `RUN_MODE=screen`
  - `ALLOW_EXISTING_RUN=1`
  - `SCREEN_PREFIX=20260608_cell_celltype_llm_clip10_tune_v1`
  - `FULL_PREFIX=20260608_cell_celltype_llm_clip10_tune_v1_full`
  - `SELECTED_PREFIX=20260608_cell_celltype_llm_clip10_tuned_selected_v1`

## Current State

- Launch log: `logs/20260608_cell_celltype_llm_clip10_tune_v1_launch.log`
- Detached launcher PID observed: `211364`
- The first screen job completed fold0 test for stage1/base/exp01 with AUPRC `0.579581` and AUROC `0.860253`, then advanced to fold2.
- No old artifacts were deleted; `ALLOW_EXISTING_RUN=1` was used only because an initial foreground debug run created a partial checkpoint directory.
