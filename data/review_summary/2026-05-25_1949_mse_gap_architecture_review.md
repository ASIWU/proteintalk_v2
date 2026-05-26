# 2026-05-25 19:49 HKT MSE-Gap Architecture Review

## Scope

Implemented and tested default-off architecture/loss variants intended to make the expression MSE auxiliary task more important for unseen single-drug prediction.

## Code Changes

- `model/fast_delta_model.py`
  - Added `response_delta_mode={off,summary,gate}`.
  - Added a low-rank projected predicted-delta summary and optional delta auxiliary response/synergy logit heads.
  - Added `response_delta_detach` so BCE does not backpropagate into the decoder-side delta signal.
- `model/fast_lightning.py`
  - Added sparse drug-specific MSE target weights.
  - Added `mse_weight_schedule={constant,warmup_decay}`.
  - Kept baseline behavior unchanged when new switches are left at defaults.
- `train.py`, `infer.py`, `scripts/ptv3_experiment_common.sh`
  - Added CLI/env plumbing and checkpoint manifest compatibility.
- `scripts/run_mse_gap_delta_screen_2gpu.sh`
  - Added paired two-GPU with-MSE / w/o-MSE screening for MSE-gap variants.

## Validation

- Static checks passed:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py`
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_mse_gap_delta_screen_2gpu.sh`
- Forward dry-runs passed for with-MSE and `--no-mse-loss`.
- 1-epoch smoke with `response_delta_mode=gate`, `response_delta_detach=1`, `delta_logit_scale=0.5`, and `mse_target_mode=pdi` completed after using `loss2` as smoke monitor.

## Results

Fold0/fold2 screening:

| method | folds | with-MSE AUROC/AUPRC | w/o-MSE AUROC/AUPRC | AUPRC gap |
| --- | --- | ---: | ---: | ---: |
| `delta_summary_pdi` | 0,2 | `0.877584 / 0.585393` | `0.861756 / 0.536349` | `+0.049044` |
| `delta_gate_pdi` | 0,2 | `0.874138 / 0.568220` | `0.863635 / 0.543747` | `+0.024472` |
| `delta_summary_all05` | 0,2 | `0.874113 / 0.566628` | `0.861756 / 0.536349` | `+0.030279` |
| `mse_target_pdi_w05` | 0,2 | `0.860494 / 0.559130` | `0.851408 / 0.523139` | `+0.035991` |

Full 5-fold confirmation for `delta_summary_pdi`:

| variant | AUROC | AUPRC | ACC |
| --- | ---: | ---: | ---: |
| with MSE | `0.902626` | `0.654613` | `0.915458` |
| w/o MSE | `0.898656` | `0.650194` | `0.914729` |
| gap | `+0.003970` | `+0.004418` | `+0.000729` |

Baseline4 reference:

| variant | AUROC | AUPRC | ACC |
| --- | ---: | ---: | ---: |
| baseline4 | `0.903489` | `0.666491` | `0.915167` |
| baseline4 w/o MSE | `0.888019` | `0.644541` | `0.913273` |
| gap | `+0.015470` | `+0.021950` | `+0.001894` |

## Conclusion

The new modules are technically valid and reproducible, but none should replace baseline4. The best early-screening gap did not survive full 5-fold validation, and with-MSE AUPRC was lower than baseline4.
