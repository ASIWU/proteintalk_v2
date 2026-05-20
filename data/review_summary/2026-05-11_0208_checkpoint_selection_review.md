# 2026-05-11 02:08 HKT Checkpoint Selection Review

## Scope

- Reviewed `train.py`, `model/training_ready_lightning.py`, and `scripts/ptv3_experiment_common.sh` checkpoint-selection behavior.

## Findings

- Before this change, scripts selected the best checkpoint with raw `MONITOR=val/total_loss` and `MONITOR_MODE=min`.
- `train.py` already supported raw Lightning monitor metrics, but the script interface did not expose a clear named choice for common validation metrics.
- Active-task validation AUPRC is logged as `val/task_auprc`; for single-drug tasks it maps to response AUPRC, and for double-drug tasks it maps to synergy AUPRC.

## Changes

- Added `--best-ckpt-metric` to `train.py` with named choices for total loss, loss1, loss2, and active-task AUPRC.
- Updated `scripts/ptv3_experiment_common.sh` to expose `BEST_CKPT_METRIC`, defaulting to `valid_auprc`.
- Kept raw `MONITOR` / `MONITOR_MODE` overrides for advanced/manual checkpointing.
- Updated active-task metric logging so `val/task_auprc` is available to checkpoint selection under DDP.
