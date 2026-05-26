# 2026-05-26 12:13 HKT exp01-exp08 Runtime Estimate Review

## Scope

- Estimated 2-GPU wall-clock time for running the full `exp_01` through `exp_08` suite with one shared model size.
- Used completed runtime artifacts only; no training jobs were launched.

## Inputs

- Baseline full-suite runtime source:
  - `logs/20260521_final_task_specific_runtime_summary.tsv`
  - Stage1 (`exp_01` through `exp_06`) has 30 fold-level training tasks.
  - Stage2 runs `exp_07` and `exp_08` in parallel.
- Model-size scaling source:
  - `logs/20260526_model_size_lr2e4_v1_model_size_report.md`
  - h512/h768 ratios were estimated from measured unseen-drug (`exp_01`-like) and unseen-cell (`exp_03`-like) fold runtimes versus h384.

## Estimate

Using the same task order as `scripts/0521_baseline4_8gpu_parallel.sh`, but with `GPU_IDS=0,1`:

| Setting | Stage1 exp01-exp06 | Stage2 exp07-exp08 | Total wall-clock |
| --- | ---: | ---: | ---: |
| h512, LR `2e-4` | ~28.2 min | ~1.5 min | ~29.7 min |
| h768, LR `2e-4` | ~34.6 min | ~1.9 min | ~36.5 min |

## Notes

- The estimate includes extra-data inference.
- h512 is about 6.8 minutes faster than h768 for the full 2-GPU suite.
- Expect a small practical buffer from cache state, W&B latency, and filesystem contention; round to about 30-32 minutes for h512 and 37-40 minutes for h768.
