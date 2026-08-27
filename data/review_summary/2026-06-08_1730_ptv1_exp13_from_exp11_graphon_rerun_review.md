# PTV1 exp_13 From exp_11 Graph-On Rerun Review

Reviewed at: 2026-06-08 17:30 HKT

Prefix: `20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010`

## Summary

- Implemented the exp_13-only official rerun path pinned to exp_11 best `mse050_drop010`.
- Added `scripts/ptv1/run_ptv1_exp13_from_exp11_selected.sh`.
- Added `scripts/ptv1/report_ptv1_exp13_from_exp11_selected.py`.
- Wrote the official report to `docs/2026-06-08_ptv1_exp13_from_exp11_graphon_rerun_report.md`.
- Wrote the TSV report to `outputs/2026-06/2026-06-08/20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010_exp13_from_exp11_selected_report.tsv`.

## Results

- Source exp_11: AUROC/AUPRC/nAUPRC `0.963021 / 0.926324 / 3.162962`, selected epoch `5`.
- exp_13 direct from exp_11: AUROC/AUPRC/nAUPRC `0.580062 / 0.561480 / 1.092882`, rows `218`.
- exp_13 all_train from exp_11 parameters: AUROC/AUPRC/nAUPRC `0.603900 / 0.579478 / 1.127913`, rows `218`.
- all_train manifest recorded `exp13_from_exp11_policy.selected_epoch=5` and `applied_max_epochs=6`.

## Audit

- Source exp_11 manifest is `fit_completed`.
- Source checkpoint exists and is `epoch=5-step=168.ckpt`.
- Source and all_train graph config is enabled: `GRAPH_FEATURE_MODE=real`, `GRAPH_STRUCTURAL_RP=1`, `GRAPH_DRUG_CONCAT=1`, `GRAPH_LOGIT_SCALE=2.0`.
- all_train key params match source exp_11 for learning rate, batch size, dropout, weight decay, MSE weight, MSE target mode, graph settings, and Cell LLM mode.
- Direct inference manifest references the source exp_11 checkpoint and manifest.
- Direct and all_train prediction parquet row counts are both `218`.
- The official report has no cross-candidate leaderboard rows and does not select a graph-off candidate.

## Notes

- The first sandboxed run failed during CUDA initialization and left partial `_v1` artifacts.
- No partial artifacts were deleted; the completed official rerun used auto-incremented `_v2`.
- Static checks passed:
  - `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`;
  - `python -m py_compile scripts/ptv1/*.py train.py infer.py`.
