# PTV1 Flow V2 Cell/Cell-Type Epoch-Matched Retrain Report

Date: 2026-06-17 HKT

## Scope

- Code/data root: `baseline/ptv2_benchmark_260514/PTV1_model_20260428`
- Data root used: `baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified`
- Datasets run: `cell_5fold`, `cell_type_5fold`
- Environment: `conda activate flow_v2`
- GPU execution: tmux session `gpu2`

The current PTV1 benchmark copy only has runnable `cell_5fold` and `cell_type_5fold` data/checkpoints. No additional pert-stratified data directory was used.

## Method

The run used README-compatible model and optimizer settings:

- `batch_size=64`
- `hidden_size=64`
- `dropout_rate=0`
- `learning_rate=0.001`
- `optimizer=adamw`
- `use_swag`
- `swag_lr=0.0005`
- `swag_adaptive`
- `random_seed=42`

Because the provided checkpoints are named by early best epochs, each fold was retrained through the corresponding provided checkpoint epoch plus one epoch. This is an epoch-matched reproducibility check against the provided checkpoint metrics, not a full 8000-epoch wall-clock run.

The training stage skipped per-epoch test-set evaluation and skipped per-epoch checkpoint files. Early stopping and best-checkpoint selection still used validation loss; final metrics were computed by reloading each fold's `best_checkpoint.pt` and predicting on that fold's test split.

## Outputs

- Run root: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_epochmatched_20260617_1617`
- Summary CSV: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_epochmatched_20260617_1617/summary.csv`
- Generated report: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_epochmatched_20260617_1617/report.md`
- Metrics files: 10/10 present.

## Result Summary

Tolerance used by the runner: retrained metric >= reference metric - 0.01.

| Dataset | Mean AUROC | Ref AUROC | dAUROC | Mean AUPRC | Ref AUPRC | dAUPRC | Mean AP | Ref AP | dAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cell_5fold | 0.9172 | 0.9184 | -0.0012 | 0.8473 | 0.8513 | -0.0040 | 0.8478 | 0.8518 | -0.0040 |
| cell_type_5fold | 0.9209 | 0.9226 | -0.0017 | 0.8522 | 0.8563 | -0.0041 | 0.8537 | 0.8577 | -0.0040 |

## Fold-Level Non-Matches

Most folds matched within the 0.01 tolerance. The following fold/metric checks did not:

| Dataset | Fold | AUROC d | AUPRC d | AP d | Notes |
|---|---:|---:|---:|---:|---|
| cell_5fold | 3 | -0.0125 | -0.0146 | -0.0148 | All three metrics below tolerance. |
| cell_5fold | 4 | -0.0002 | -0.0168 | -0.0166 | AUROC matched; AUPRC/AP below tolerance. |
| cell_type_5fold | 3 | -0.0123 | -0.0081 | -0.0081 | AUPRC/AP matched; AUROC below tolerance. |

## Conclusion

`flow_v2` can run the available PTV1 `cell` and `cell_type` benchmark data end to end. The epoch-matched retrain nearly reproduces the provided checkpoint means, but it does not exactly reach every provided checkpoint fold under a strict per-fold 0.01 tolerance. At dataset-mean level both tasks are close: mean deltas are about -0.001 to -0.004 across AUROC/AUPRC/AP.

