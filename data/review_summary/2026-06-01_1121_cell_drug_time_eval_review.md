# Cell-Drug Time-Collapsed Evaluation Review

Review time: 2026-06-01 11:21 HKT

## Scope

- Reviewed the existing PTV3 training, inference, extra-result reporting, and selected dose-parameter search workflow.
- Implemented a reusable evaluation path for reporting exp01-exp08 under original, cell-drug-bylasttime, and cell-drug-avgtime metrics.

## Findings

- Exp07 and exp08 already save `predictions.parquet`, so they can be re-evaluated without new inference.
- Exp01-exp06 training manifests save aggregate test metrics and checkpoints, but not per-row test probabilities. Re-evaluation therefore needs checkpoint-backed test inference, not retraining.
- Existing prediction files before this change may not contain `pert_time`; joining predictions back to the task feature table by `feature_row_index` is required for robust time aggregation.
- Main single- and double-drug feature tables have consistent labels within cell-drug groups in the inspected data, but the new reporter records and drops conflicting groups if future data introduces label conflicts.

## Code Changes

- Added `scripts/report_cell_drug_time_eval.py`.
- Added `scripts/run_selected_param_full_suite.sh`.
- Extended `infer.py` prediction metadata to include perturbation time, dose, and batch columns when present.

## Validation

- Python compile checks passed for `infer.py`, `scripts/report_cell_drug_time_eval.py`, and `scripts/report_ptv3_exp_results.py`.
- Shell syntax check passed for `scripts/run_selected_param_full_suite.sh`.
- Existing extra single and extra double prediction smoke checks ran through the new aggregation logic successfully.
