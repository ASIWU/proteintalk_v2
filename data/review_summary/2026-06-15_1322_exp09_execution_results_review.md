# 2026-06-15 13:22 HKT exp09 Execution Results Review

## Scope

- Executed `scripts/exp_09_unified_all_train_valid_oracle.sh` for exp09 after exp08.
- Verified unified-head all-data training, valid reference-epoch evaluation, oracle all-epoch evaluation, and detailed report generation.
- Reviewed the generated valid/oracle task structure and metrics artifacts.

## Findings

- No failed training or inference rows were found in `logs/20260615_1101_exp09_selectedref_v1_runtime_summary.tsv`: 460 rows, 0 nonzero statuses.
- The run produced 50 oracle epoch roots and 450 oracle task manifests, plus 9 valid task manifests.
- `valid` uses exp01/exp06 selected reference epochs:
  - exp07 extra single: epoch 5;
  - exp08 extra double: epoch 2.
- `oracle` selected best epochs by mean-extra original AUPRC:
  - exp07: epoch 2, AUPRC 0.596656, AUROC 0.780482;
  - exp08: epoch 8, AUPRC 0.101437, AUROC 0.655564.
- All valid/oracle manifests use `task_head=unified` and `task_label_policy=unified_synergy_first_else_response`.

## Artifacts

- Result doc: `docs/2026-06-15_exp09_unified_head_valid_oracle_results.md`
- Detailed markdown report: `logs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.md`
- Detailed CSV: `outputs/2026-06/2026-06-15/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.csv`
- Detailed JSON: `outputs/2026-06/2026-06-15/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.json`

## Notes

- The initial report subprocess was stopped after model outputs were complete because the old collapsed metric implementation did not finish after about 50 CPU-minutes.
- `scripts/report_cell_drug_time_eval.py::collapsed_frame()` was vectorized and the report was regenerated from the same prediction files. Training and inference outputs were not rerun for this report regeneration.

## Validation

- `python -m py_compile scripts/report_cell_drug_time_eval.py scripts/report_exp09_valid_oracle.py`
- Small synthetic collapsed-metric equivalence check for conflict and missing-time behavior.
- Runtime summary status check: 460 rows, 0 bad rows.
- Report artifact check: markdown, CSV, and JSON generated successfully.
