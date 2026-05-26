# 2026-05-26 14:55 HKT extra_doubledrug test_label / nAUPRC review

Scope: reviewed the two recent extra double-drug changes: `test` / `test_label`
pipeline support, post-processing report, nAUPRC metrics, and the impact on
`scripts/exp_0[1-8].sh`.

Findings:
- No blocking bug found in the new `test_label` / nAUPRC implementation.
- `scripts/exp_08_extra_double_all_train_infer.sh` now runs the three extra
  double-drug inference tasks and then prints/writes a grouped report with
  AUPRC, baseline, and nAUPRC.
- The report prints 9 rows: three rows per dataset
  (`unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`, `combined`).
  If only one result per dataset is desired, the report should be filtered to
  the `combined` rows or extended with a combined-only option.
- `exp_01` through `exp_06` are affected only by extra metric keys being logged
  (`auprc_baseline`, `nauprc`); checkpoint selection remains on `val/task_auprc`.
- `exp_07` is unaffected except that `show_extra_results.py` can display the new
  fields when future metrics contain them.

Validation:
- `bash -n` passed for `scripts/exp_01_single_pert_stratified_5fold.sh` through
  `scripts/exp_08_extra_double_all_train_infer.sh` and
  `scripts/ptv3_experiment_common.sh`.
- `python -m py_compile` passed for the changed Python files.
- `scripts/report_extra_doubledrug_test_label_auprc.py` successfully reported
  existing full extra-double predictions with the expected evaluable counts.
