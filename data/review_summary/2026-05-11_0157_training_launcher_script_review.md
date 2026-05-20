# 2026-05-11 01:57 HKT Training Launcher Script Review

## Scope

- Reviewed `scripts/0509_1.sh` formatting and the existing PTV3 experiment scripts for double-drug, extra single-drug, and extra double-drug runs.

## Summary

- `scripts/0509_1.sh` is a thin launcher that exports the local W&B settings and passes run-specific environment variables into one `scripts/exp_*.sh` script.
- The requested targets already exist as:
  - `scripts/exp_06_double_pert_pair_5fold.sh`
  - `scripts/exp_07_extra_single_all_train_infer.sh`
  - `scripts/exp_08_extra_double_all_train_infer.sh`
- Added matching launchers `scripts/0509_2.sh`, `scripts/0509_3.sh`, and `scripts/0509_4.sh`.

## Notes

- Double-drug launchers use `BATCH_SIZE=64`, matching the recent documented double-drug run setting.
- Single-drug all-train launcher uses `BATCH_SIZE=128`, matching `scripts/0509_1.sh`.
