# 2026-06-17 15:31 HKT PTV1 Flow V2 Benchmark Review

Reviewed `baseline/ptv2_benchmark_260514/scripts/redme.md`, `PTV1_model_20260428/readme.md`, the available data/checkpoint directories, and the PTV1 model import path needed for prediction.

Findings:

- The scripts README lists `pert_stratified_5fold`, `cell_type_5fold`, and `cell_5fold`, but the current PTV1 benchmark data/checkpoint tree only has runnable `cell_5fold` and `cell_type_5fold` directories.
- `flow_v2` initially failed before prediction because `metrics.py` imported `torcheval.metrics.functional`, which triggered incompatible `torchaudio` CUDA 12.8 against `torch` CUDA 12.6.
- The `torcheval` import was unnecessary because `metrics.py` already defines local `binary_auroc` and `binary_auprc` functions.

Outcome:

- Removed the unused `torcheval` import and completed 10/10 available folds in `gpu2` with `flow_v2`.
- Results were written to `baseline/ptv2_benchmark_260514/eval_runs/flow_v2_cell_celltype_5fold_eval_20260617_1535_fixed`.
- Stable report: `docs/2026-06-17_ptv1_flow_v2_cell_celltype_benchmark_report.md`.
