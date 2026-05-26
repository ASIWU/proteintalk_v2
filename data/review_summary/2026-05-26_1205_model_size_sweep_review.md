# 2026-05-26 12:05 HKT Model Size Sweep Review

## Scope

- Explored fast_delta model size on both unseen-drug and unseen-cell 5-fold splits.
- Did not modify data files or data-processing code.
- Added only experiment/reporting scripts; standard training defaults are unchanged.

## Code Changes Reviewed

- `scripts/run_model_size_sweep_2gpu.sh`
  - Two-GPU launcher for capacity sweeps.
  - Runs unseen drug through `scripts/exp_01_single_pert_stratified_5fold.sh`.
  - Runs unseen cell through `scripts/exp_03_single_cell_5fold.sh`.
  - Uses task-specific default settings:
    - unseen drug: baseline4 single-drug config, `MSE_WEIGHT=0.25`, covariate UNK off;
    - unseen cell: full covariate UNK dropout `0.15`, `MSE_WEIGHT=0.075`.
- `scripts/model_size_sweep_report.py`
  - Reads `run_manifest.json` files.
  - Aggregates AUROC/AUPRC/ACC, fold runtime, parameter count, LR, MSE weight, and covariate UNK status.
  - Can merge multiple sweep prefixes into one markdown/json report.

## Validation

- `python -m py_compile scripts/model_size_sweep_report.py` passed.
- `bash -n scripts/run_model_size_sweep_2gpu.sh` passed.
- A 1-epoch/1-batch smoke run passed:
  - prefix `debug_modelsize_sweep_smoke`;
  - report `logs/debug_modelsize_sweep_smoke_model_size_report.md`.

## Completed Experiments

All runs used one GPU per fold, batch size `256`, `fast_delta`, real graph features, structural RP, graph-drug concat, and PCEP.

| Prefix | Purpose |
| --- | --- |
| `20260526_model_size_h128_v1` | Small lower-bound capacity at LR `3e-4` |
| `20260526_model_size_sweep_v1` | Main LR `3e-4` sweep: h192/h256/h384/h512/h768 |
| `20260526_model_size_lr2e4_v1` | Large-model LR `2e-4` sweep: h384/h512/h768 |
| `20260526_model_size_h1024_lr2e4_v1` | Boundary h1024 test at LR `2e-4` |

Combined report:

- `logs/20260526_model_size_combined_report.md`
- `logs/20260526_model_size_combined_report.json`

## Key Results

Same-day reference (`h384`, LR `3e-4`):

| Task | AUROC | AUPRC | sec/fold | Params |
| --- | ---: | ---: | ---: | ---: |
| unseen drug | 0.896696 | 0.652478 | 68.0 | 17.2M |
| unseen cell | 0.930355 | 0.748441 | 68.2 | 17.2M |

Best AUPRC settings:

| Task | Setting | AUROC | AUPRC | Delta AUPRC vs h384 LR3e-4 | sec/fold | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| unseen drug | h512, LR `2e-4` | 0.903166 | 0.677846 | +0.025367 | 82.4 | 26.4M |
| unseen cell | h768, LR `2e-4` | 0.934922 | 0.770017 | +0.021576 | 100.6 | 40.8M |

Efficiency lower-bound:

| Task | Setting | AUROC | AUPRC | Delta AUPRC vs h384 LR3e-4 | sec/fold | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| unseen drug | h128, LR `3e-4` | 0.894454 | 0.641002 | -0.011476 | 54.8 | 5.6M |
| unseen cell | h128, LR `3e-4` | 0.931033 | 0.744463 | -0.003978 | 54.6 | 5.6M |

## Interpretation

- Model size is not monotonic.
- For unseen drug, h512 with lower LR is the best tested capacity setting; h768 and h1024 do not improve further.
- For unseen cell, h768 with lower LR is the best tested capacity setting, but its gain over the historical `MSE_WEIGHT=0.075 + covUNK` reference (`0.767008`) is only about `+0.003`.
- h1024 is slower and worse on both tasks, so the useful capacity range appears to stop before hidden dim 1024.
- h128/h192 are viable speed/parameter-count ablations, but they are not best-performance settings.

## Recommendation

- If changing the standard single-drug unseen-drug baseline is acceptable, test/adopt h512 with LR `2e-4` next.
- Do not make h768 the global default solely for unseen cell; the effect is small and runtime increases by about 48% versus the same-day h384 rerun.
- Keep h128/h192 as efficiency ablations for paper/runtime discussion.
