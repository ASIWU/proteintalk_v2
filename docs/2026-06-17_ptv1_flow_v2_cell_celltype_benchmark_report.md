# PTV1 Flow V2 Cell/Cell-Type Benchmark Report

Date: 2026-06-17 15:31 HKT

## Scope

This run evaluated the available PTV1 benchmark data under:

- Model code: `baseline/ptv2_benchmark_260514/PTV1_model_20260428`
- Data root: `baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified`
- Checkpoint root: `baseline/ptv2_benchmark_260514/checkpoints`
- Conda env: `flow_v2`
- GPU execution: `gpu2` tmux session on H200 CUDA

The current copy contains runnable data and checkpoints for:

- `cell_5fold`, folds 0-4
- `cell_type_5fold`, folds 0-4

The historical scripts README mentions `pert_stratified_5fold`, but this repository copy does not contain a runnable `pert_stratified_5fold` data directory or checkpoint directory under the PTV1 benchmark paths above.

## Run Command

```bash
cd /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
mkdir -p /tmp/matplotlib-flow-v2
PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib-flow-v2 \
  python baseline/ptv2_benchmark_260514/PTV1_model_20260428/run_cell_celltype_5fold_eval.py \
  --output-root baseline/ptv2_benchmark_260514/eval_runs/flow_v2_cell_celltype_5fold_eval_20260617_1535_fixed \
  --fail-fast
```

## Code Compatibility Notes

- `main.py` now loads the provided full checkpoint dictionaries with `torch.load(..., weights_only=False)`, which is required with current PyTorch defaults.
- `metrics.py` no longer imports `torcheval.metrics.functional` at module import time. The file already defines local `binary_auroc` and `binary_auprc` functions, and the top-level `torcheval` import caused `flow_v2` to fail because `torch` is CUDA 12.6 while `torchaudio` is CUDA 12.8.
- No shared conda packages were modified for this successful run.

## Output Artifacts

- Output root: `baseline/ptv2_benchmark_260514/eval_runs/flow_v2_cell_celltype_5fold_eval_20260617_1535_fixed`
- Summary CSV: `baseline/ptv2_benchmark_260514/eval_runs/flow_v2_cell_celltype_5fold_eval_20260617_1535_fixed/summary.csv`
- Generated report: `baseline/ptv2_benchmark_260514/eval_runs/flow_v2_cell_celltype_5fold_eval_20260617_1535_fixed/report.md`
- Verification: 10 `metrics.txt` files found, matching 2 datasets x 5 folds.

## Per-Fold Metrics

| Dataset | Fold | Samples | Positives | AUROC | AUPRC | AP |
|---|---:|---:|---:|---:|---:|---:|
| cell_5fold | 0 | 505 | 187 | 0.9074 | 0.8779 | 0.8782 |
| cell_5fold | 1 | 1008 | 249 | 0.9109 | 0.8702 | 0.8704 |
| cell_5fold | 2 | 472 | 133 | 0.9431 | 0.8829 | 0.8834 |
| cell_5fold | 3 | 1065 | 225 | 0.9104 | 0.7723 | 0.7733 |
| cell_5fold | 4 | 623 | 133 | 0.9203 | 0.8533 | 0.8539 |
| cell_type_5fold | 0 | 1479 | 422 | 0.9061 | 0.8440 | 0.8442 |
| cell_type_5fold | 1 | 1214 | 298 | 0.9057 | 0.8365 | 0.8367 |
| cell_type_5fold | 2 | 139 | 49 | 0.9519 | 0.9263 | 0.9270 |
| cell_type_5fold | 3 | 70 | 32 | 0.9120 | 0.9206 | 0.9217 |
| cell_type_5fold | 4 | 771 | 126 | 0.9374 | 0.7540 | 0.7588 |

## Dataset Means

| Dataset | Folds | Mean AUROC | Mean AUPRC | Mean AP |
|---|---:|---:|---:|---:|
| cell_5fold | 5 | 0.9184 | 0.8513 | 0.8518 |
| cell_type_5fold | 5 | 0.9226 | 0.8563 | 0.8577 |

## Conclusion

`flow_v2` can run the available PTV1 benchmark after removing the unused import-time dependency on `torcheval.metrics.functional`. The available benchmark scope completed successfully: `cell_5fold` and `cell_type_5fold`, 10/10 folds.
