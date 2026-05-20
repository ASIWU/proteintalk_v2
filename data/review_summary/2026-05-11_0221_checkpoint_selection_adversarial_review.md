# 2026-05-11 02:21 HKT Checkpoint Selection Adversarial Review

## Scope

- Re-reviewed the new best-checkpoint metric strategy across `train.py`, `model/training_ready_lightning.py`, and all relevant scripts under `scripts/`.

## Loopholes Found

1. `scripts/run_ptv3_training_experiments.sh` still used raw `MONITOR=val/total_loss`, so the legacy all-in-one runner would bypass the new default `valid_auprc` behavior.
2. `train.py` direct CLI still defaulted `--log-every-n-steps` to `5`, while script docs and wrapper defaults say `1`.
3. `SCHEDULER_NAME=plateau` still monitored `val/total_loss` even when checkpoint selection monitored `val/task_auprc` or another selected metric.
4. AUPRC can be undefined if the evaluated validation batches contain only one class; without a guard, checkpoint selection could silently proceed with a non-finite monitor.

## Fixes

- Updated the legacy runner to use `BEST_CKPT_METRIC=valid_auprc` by default and to keep raw `MONITOR` / `MONITOR_MODE` only as explicit overrides.
- Changed `train.py --log-every-n-steps` default to `1`.
- Added scheduler monitor/mode propagation so `ReduceLROnPlateau` follows the selected checkpoint metric.
- Added a monitor guard that fails training when the selected checkpoint monitor is missing or non-finite, unless `--allow-nonfinite-monitor` is explicitly set.
- Exposed `ALLOW_NONFINITE_MONITOR=1` in the script wrappers for explicit debugging only.
- Updated `scripts/README_ptv3_experiments.md` with the non-finite monitor behavior and plateau-scheduler alignment.

## Data Check

- Verified the full validation splits for current PTV3 single-drug and double-drug experiment strategies contain both active-label classes, including `all_train_subset_test`.
- This supports the default `valid_auprc` monitor for normal full-validation runs.

## Verification

- `py_compile` passed for `train.py` and `model/training_ready_lightning.py`.
- `bash -n` passed for every top-level `scripts/*.sh` file.
- `train.py --help` exposes `--best-ckpt-metric` and `--allow-nonfinite-monitor`.
- Metric mapping check passed:
  - `valid_auprc -> val/task_auprc, max`
  - `total_loss -> val/total_loss, min`
  - `loss1 -> val/loss1, min`
  - `loss2 -> val/loss2, min`
- Existing validators passed:
  - `utils/01_validate_standardized_outputs.py`
  - `utils/03_validate_training_ready_outputs.py`
