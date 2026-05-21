# 2026-05-21 14:46 HKT Baseline4 Single-GPU Ablation Review

## Scope

Reviewed the new baseline4 single-GPU experiment launcher and the completed
5-fold results for:

- baseline4 (`baseline3 + PCEP`, real PPI/PDI/DDI graph features);
- baseline4 w/o graph feature;
- baseline4 w/o MSE loss;
- baseline4 MSE-weight sweep at 0.05, 0.10, 0.25, and 0.50.

## Files Reviewed

- `new_version/run_baseline4_1gpu_parallel.sh`
- `new_version/runtime_logs/20260521_baseline4_b256_v1/summary.tsv`
- `new_version/runtime_logs/20260521_baseline4_mse_sweep_b256/summary.tsv`
- `new_version/ALGORITHM_UPDATE.md`

## Checks

- The launcher uses `CUDA_VISIBLE_DEVICES=${gpu_id}` and `--devices 1`, so each
  task is a true single-GPU run.
- The task queue assigns work across `GPU_IDS=0,1`, allowing two independent
  single-GPU experiments to run concurrently.
- `baseline4` uses real compressed graph features with `--graph-feature-mode real`.
- `baseline4_zero` keeps the same architecture but uses
  `--graph-feature-mode zero`.
- `baseline4_no_mse` keeps real graph features and PCEP but disables the MSE
  auxiliary loss with `--no-mse-loss`.
- All reported rows have `fit_completed` and `test_completed` in the summary
  TSVs.

## Results

Baseline4 matched ablation at `gpu=1`, `batch_size=256`, 50 epochs:

| method | AUPRC | AUROC | ACC | fit seconds/fold |
| --- | ---: | ---: | ---: | ---: |
| baseline4 | 0.666491 | 0.903489 | 0.915167 | 41.8 |
| baseline4 w/o graph feature | 0.561250 | 0.835845 | 0.912883 | 42.0 |
| baseline4 w/o MSE loss | 0.644541 | 0.888019 | 0.913273 | 41.2 |

MSE-weight sweep with real graph features:

| mse_weight | AUPRC | AUROC | ACC | fit seconds/fold |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.649057 | 0.887431 | 0.915543 | 42.4 |
| 0.10 | 0.656212 | 0.898468 | 0.916110 | 42.4 |
| 0.25 | 0.666491 | 0.903489 | 0.915167 | 41.8 |
| 0.50 | 0.658686 | 0.893493 | 0.912557 | 41.4 |

## Conclusion

The graph-feature ablation and MSE-loss ablation both support retaining the full
baseline4 recipe. `mse_weight=0.25` remains the best tested value by 5-fold mean
AUPRC and AUROC.
