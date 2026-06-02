# 2026-05-27 00:31 HKT MSE Gap Experiment Review

## Scope

Tried to expand the single unseen-drug w/o-MSE ablation gap beyond 5 AUPRC points on the 2-GPU server. No data files or data-processing code were modified.

## Code Changes Reviewed

- Added default-off architecture/training switches in `model/fast_delta_model.py`, `model/fast_lightning.py`, `train.py`, `infer.py`, and `scripts/ptv3_experiment_common.sh`.
- Extended `scripts/run_mse_gap_delta_screen_2gpu.sh` with screening presets for:
  - expression-delta bridge;
  - MSE warmup/pretrain;
  - high-dimensional delta branch;
  - zero-initialized delta decoder;
  - learnable delta-logit scale;
  - control-expression dropout;
  - true-delta teacher branch.

## Validation

- Static checks passed:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py`
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_mse_gap_delta_screen_2gpu.sh`
- All reported experiments ran on `GPU_IDS=0,1`, one GPU per task, batch size `256`.

## Key Results

| candidate | folds | with-MSE AUPRC | w/o-MSE AUPRC | gap |
| --- | ---: | ---: | ---: | ---: |
| h512 reference | 5 | 0.677846 | 0.651609 | +0.026237 |
| `learn_delta_b075_g2_init01` | 5 | 0.652089 | 0.644282 | +0.007807 |
| `zdelta_b0_g0_dim32` | 5 | 0.244446 | 0.120776 | +0.123670 |

## Conclusion

The target gap can be forced only by an extreme expression-mediated model that destroys base performance. No conservative model preserving the current hidden/graph classification path reached a 5-point full 5-fold AUPRC gap. The current h512 baseline should remain the recommended baseline.
