# 2026-07-19 13:28 HKT patientVali Output Format Comparison Review

## Scope

- Compared `outputs/2026-07/2026-07-19/20260719_patientVali260605v3_exp09_all_epoch_ckpts_rerun_v1` with `outputs/2026-06/2026-06-12/0612v3_all_epoch_ckpts` after a concern that their output formats differed.
- Also compared each directory with its same-stage counterpart:
  - raw: `outputs/2026-07/2026-07-19/20260719_patientVali260605v3_exp09_all_epoch_ckpts_rerun_v1` versus `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_all_epoch_ckpts`;
  - readable: `outputs/2026-07/2026-07-19/0719v3_exp09_all_epoch_ckpts_rerun_v1` versus `outputs/2026-06/2026-06-12/0612v3_all_epoch_ckpts`.
- This was a read-only review; no inference, aggregation, code, or output artifact was changed.

## Finding

- The initially named directories represent different stages, so their visible layouts are intentionally different:
  - `outputs/2026-07/2026-07-19/20260719_patientVali260605v3_exp09_all_epoch_ckpts_rerun_v1` is the raw inference directory. It contains checkpoint/task subdirectories, raw 30-column combined CSV/Parquet files, and `inference_summary.json`.
  - `outputs/2026-06/2026-06-12/0612v3_all_epoch_ckpts` is a readable export directory. It contains a 32-column readable combined CSV, summaries, README, manifest, and flattened per-checkpoint/task CSV exports.
- The new readable directory corresponding to the old `0612v3` readable directory is `outputs/2026-07/2026-07-19/0719v3_exp09_all_epoch_ckpts_rerun_v1`.

## Same-stage Verification

- New raw versus old raw:
  - the combined raw column schema is identical;
  - the top-level raw file set is identical: combined CSV, combined Parquet, and inference summary;
  - directory counts differ as expected because the new exp09 run has 50 checkpoint directories while the old exp07/exp08 run has 9.
- New readable versus old readable:
  - the 32-column readable schema is identical;
  - the top-level readable file set is identical: README, manifest, combined readable CSV, and three summary CSV files;
  - the new run has 400 per-task/checkpoint CSV files while the old run has 33, reflecting checkpoint/task coverage rather than a format change.
- Manifest roles agree with this pairing:
  - the new raw inference summary points to `outputs/2026-07/2026-07-19/0719v3_exp09_all_epoch_ckpts_rerun_v1` as its readable directory;
  - the old readable manifest points back to `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_all_epoch_ckpts` as its raw directory.

## Conclusion

- No output-format regression was found.
- The apparent difference came from comparing the new raw directory with the old readable directory. For human-readable consumption, use `outputs/2026-07/2026-07-19/0719v3_exp09_all_epoch_ckpts_rerun_v1`; for a raw-format comparison, compare the new `20260719...` directory with `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_all_epoch_ckpts`.
