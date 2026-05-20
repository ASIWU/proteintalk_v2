# 2026-05-11 12:32 HKT - 0509 Wrapper Reference-Epoch Review

Reviewed `scripts/0509_1.sh` through `scripts/0509_4.sh` after the reference-epoch hardening.

Findings:

- `0509_1.sh` and `0509_2.sh` do not need direct reference-epoch changes because they produce the single/double 5-fold reference runs.
- `0509_3.sh` and `0509_4.sh` did need updates. Without `REFERENCE_5FOLD_CKPT_PATH`, they would still use the extra all-data run's validation-based best checkpoint.
- All four wrappers previously hard-coded inline environment values, which made bounded smoke tests and alternate production prefixes harder to run safely.

Fixes:

- Added shebangs and `set -euo pipefail`.
- Made the wrapper defaults externally overridable.
- Bound `0509_3.sh` to `checkpoints/20260510_single_pert_stratified_5fold` by default.
- Bound `0509_4.sh` to `checkpoints/20260510_double_pert_pair_5fold` by default.
- Set reference defaults to median aggregation, five required folds, saved `last.ckpt`, and no scheduler for the extra wrappers.

Validation:

- Shell syntax check passed for all four wrappers.
- 8-GPU bounded wrapper smoke passed for:
  - `20260511_wrapper_0509_1_smoke`
  - `20260511_wrapper_0509_2_smoke`
  - `20260511_wrapper_0509_3_smoke`
  - `20260511_wrapper_0509_4_smoke`
- Extra single smoke wrote 6 bounded extra outputs; extra double smoke wrote 3 bounded extra outputs.
- Extra wrapper manifests recorded the fixed reference epoch policy and selected `last.ckpt` paths.
