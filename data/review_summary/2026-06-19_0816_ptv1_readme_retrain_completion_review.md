# 2026-06-19 08:16 HKT PTV1 README Retrain Completion Review

Reviewed the completed README/script-aligned retrain for the available PTV1 benchmark tasks under `flow_v2`.

## Scope

- Data used: `cell_5fold` and `cell_type_5fold` only.
- GPU execution: tmux session/window `gpu2:brainctl`.
- Run root: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720`.
- Parallelism: `--max-parallel 3`.

## Findings

- No code or run-log failure was found in the final completed run artifacts.
- `summary.csv` has 10 rows, all with status `ok`.
- Dataset means exceeded the provided checkpoint references for both tasks:
  - `cell_5fold`: AUROC +0.0054, AUPRC +0.0086, AP +0.0085.
  - `cell_type_5fold`: AUROC +0.0002, AUPRC +0.0045, AP +0.0045.
- Residual per-fold metric exceptions remain under the runner's 0.01 tolerance:
  - `cell_5fold` fold4: AUPRC -0.0168 and AP -0.0166.
  - `cell_type_5fold` fold3: AUROC -0.0123.

## Validation

- Confirmed `summary.csv` has 10 result rows for the two 5-fold tasks.
- Confirmed all statuses are `ok`.
- Scanned current `task_logs` for `Traceback`, `RuntimeError`, CUDA OOM, killed process, failed-task markers, and related exception strings; no matches were found.
- Stable report written to `docs/2026-06-19_ptv1_flow_v2_cell_celltype_readme_retrain_report.md`.

## Conclusion

The retrain reaches the provided checkpoint results at dataset-mean level for both available PTV1 tasks. It does not fully reproduce every individual fold/metric under the strict 0.01 tolerance.
