# PTV1 Flow V2 Cell/Cell-Type README Retrain Report

Date: 2026-06-19 HKT

## Scope

- Code/data root: `baseline/ptv2_benchmark_260514/PTV1_model_20260428`
- Data root used: `baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified`
- Datasets run: `cell_5fold`, `cell_type_5fold`
- Environment: `conda activate flow_v2`
- GPU execution: tmux session/window `gpu2:brainctl`
- Parallelism: `--max-parallel 3`

Only the `cell_5fold` and `cell_type_5fold` tasks were run, because this repository copy only contains runnable PTV1 benchmark data/checkpoints for those two datasets.

## Method

The PTV1 README in `PTV1_model_20260428/readme.md` points to the benchmark evaluation scripts under `ptv1_eval`. For the requested retrain comparison, the runner used the PTV1 benchmark script hyperparameters:

- `total_epoch=8000`
- `batch_size=64`
- `hidden_size=64`
- `dropout_rate=0.0`
- `learning_rate=0.001`
- `optimizer=adamw`
- `use_swag`
- `swag_lr=0.0005`
- `swag_adaptive`
- `random_seed=42`

The implementation changes used for this run were operational only: task-level parallel orchestration, less frequent logging, and skipped per-epoch checkpoint files via `--skip_epoch_checkpoints`. Final metrics were computed from each fold's selected best checkpoint and compared with the provided PTV1 checkpoint metrics.

Command shape:

```bash
python baseline/ptv2_benchmark_260514/PTV1_model_20260428/run_cell_celltype_5fold_retrain_eval.py \
  --output-root baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720 \
  --log-interval 50 \
  --skip-existing \
  --max-parallel 3 \
  --fail-fast
```

## Outputs

- Run root: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720`
- Summary CSV: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720/summary.csv`
- Generated report: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720/report.md`
- Task logs: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720/task_logs`

## Result Summary

Tolerance used by the runner: retrained metric >= reference metric - 0.01.

| Dataset | Folds | Mean AUROC | Ref AUROC | dAUROC | Mean AUPRC | Ref AUPRC | dAUPRC | Mean AP | Ref AP | dAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cell_5fold | 5 | 0.9239 | 0.9184 | +0.0054 | 0.8599 | 0.8513 | +0.0086 | 0.8603 | 0.8518 | +0.0085 |
| cell_type_5fold | 5 | 0.9228 | 0.9226 | +0.0002 | 0.8608 | 0.8563 | +0.0045 | 0.8622 | 0.8577 | +0.0045 |

## Fold-Level Non-Matches

Dataset-level means reached or exceeded the provided checkpoint means for both tasks. The following individual fold/metric checks did not meet the 0.01 per-metric tolerance:

| Dataset | Fold | AUROC d | AUPRC d | AP d | Notes |
|---|---:|---:|---:|---:|---|
| cell_5fold | 4 | -0.0002 | -0.0168 | -0.0166 | AUROC matched; AUPRC/AP were below tolerance. |
| cell_type_5fold | 3 | -0.0123 | -0.0081 | -0.0081 | AUPRC/AP matched; AUROC was below tolerance. |

## Validation

- `summary.csv` contains 10 result rows, covering 5 folds each for `cell_5fold` and `cell_type_5fold`.
- All result statuses were `ok`.
- A final scan of the current run's `task_logs` found no `Traceback`, `RuntimeError`, CUDA OOM, killed process, or failed-task markers.

## Conclusion

At dataset-mean level, the README/script-aligned full retrain reaches the provided checkpoint results for both available PTV1 tasks. It should be treated as reproduced at the task-mean level, but not as a strict per-fold/per-metric exact reproduction because `cell_5fold` fold4 AUPRC/AP and `cell_type_5fold` fold3 AUROC remain below the 0.01 tolerance.
