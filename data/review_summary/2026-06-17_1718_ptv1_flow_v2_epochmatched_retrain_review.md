# 2026-06-17 17:18 HKT PTV1 Flow V2 Epoch-Matched Retrain Review

Reviewed and exercised the PTV1 benchmark retraining path under `flow_v2` using only data present under `PTV1_model_20260428`.

## Scope

- Data used: `cell_5fold` and `cell_type_5fold` only.
- GPU execution: tmux session `gpu2`.
- Run root: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_epochmatched_20260617_1617`.

## Code Changes Reviewed

- `dataset.py`: optimized `L1000Dataset` row lookup and label/control indexing to avoid repeated full-frame filters during dataset construction.
- `trainer.py`: added `--skip_epoch_checkpoints` handling and configurable `--log_interval`.
- `config.py`: added CLI options for skipped epoch checkpoints and training log interval.
- `run_cell_celltype_5fold_retrain_eval.py`: added cell/cell-type 5-fold retrain orchestration, epoch-matched training caps from provided checkpoint filenames, skipped train-time test eval by default, final prediction/plot/summary generation, and comparison against the provided checkpoint evaluation summary.

## Results

- Completed 10/10 available folds.
- Metrics files present: 10/10.
- Mean deltas versus provided checkpoints:
  - `cell_5fold`: AUROC -0.0012, AUPRC -0.0040, AP -0.0040.
  - `cell_type_5fold`: AUROC -0.0017, AUPRC -0.0041, AP -0.0040.
- Fold-level misses beyond 0.01 tolerance:
  - `cell_5fold` fold3: AUROC/AUPRC/AP below tolerance.
  - `cell_5fold` fold4: AUPRC/AP below tolerance.
  - `cell_type_5fold` fold3: AUROC below tolerance.

## Validation

- `python -m py_compile` passed for the edited PTV1 runner/config/trainer files.
- `git diff --check` passed for edited PTV1 files.
- The generated run report is `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_epochmatched_20260617_1617/report.md`.

