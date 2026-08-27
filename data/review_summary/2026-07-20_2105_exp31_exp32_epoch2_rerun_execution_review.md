# Exp31/Exp32 Epoch-2 Rerun Execution Review

- Review time: 2026-07-20 21:05 HKT
- Scope: checkpoint override tooling, CPU preflight, H200 smoke/formal execution, reporter outputs, schema/coverage validation, and preservation of historical results.

## Findings

- The selected checkpoint exists and has SHA-256 `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076`.
- Exp31 reporter validation succeeded with `errors=[]`. All four training manifests are `fit_completed`, their `args.checkpoint_path` values resolve to epoch 2, all best checkpoints exist, and all four zero-shot inference manifests resolve to epoch 2.
- Exp31 runtime contains exactly 4 train and 8 infer rows with status `0`; every formal inference contains 1,827 finite probabilities in `[0,1]`.
- Exp31 final CSV is 56 rows by 16 columns. Its column order exactly matches the 2026-07-09 CSV, and the Markdown section sequence exactly matches the historical report. Neither the new Exp31 JSON nor Markdown contains `last.ckpt` or `epoch=5.ckpt`.
- Exp32 raw outputs contain 41,821 rows per device. Both manifests point to epoch 2, report `checkpoint_architecture_matches=true`, and contain zero checkpoint-config mismatches.
- Exp32 combined CSV and Parquet are both 83,642 rows by 18 columns, with identical schema to the 2026-07-10 CSV. Coverage is exactly B/CAC, 13 sample pairs, and 3,217 drugs; probabilities are finite and bounded; top-20 output has 520 rows; summary status is `complete`.
- The Exp32 Markdown identifies epoch 2 and retains the historical section sequence. Its data preflight continues to validate the immutable build summary's recorded epoch-5 hash as build provenance, while the selected inference checkpoint and all run manifests are independently and exactly validated as epoch 2.
- Smoke and formal GPU work ran only through `gpu2:0` on NVIDIA H200. CPU preflight/reporting ran locally under `flow_v2`. Exp31 and Exp32 formal jobs did not overlap.
- Historical 2026-07-09/10 results remain at their original paths with original modification timestamps. All new outputs use `20260720_*_epoch2` prefixes and `ALLOW_EXISTING_RUN` was not enabled.

## Code Changes Reviewed

- `scripts/exp_31_rnaseq_pdx_ft_benchmark.sh`: reporter compilation now follows `RUN_PREFLIGHT`; full reporter invocation passes the selected init checkpoint.
- `scripts/exp_32_organoid_exp09_single_sensitivity_infer.sh`: `EXP32_CHECKPOINT` is environment-overridable with the historical epoch-5 default preserved.
- `scripts/report_exp31_rnaseq_pdx_ft_benchmark.py`: exact fine-tune/zero-shot checkpoint validation and explicit JSON/Markdown checkpoint reporting.
- `scripts/report_exp32_organoid_exp09_single_sensitivity.py`: selected sibling checkpoint support within the validated exp09 run and exact inference-manifest checkpoint enforcement.

## Verification

- Passed `bash -n` for the common helper and both runners.
- Passed Python compilation and argument-help checks for both reporters.
- Passed Exp31 and Exp32 CPU preflight, both smoke validations, both formal result validations, historical schema/Markdown comparisons, and probability/coverage assertions.
- Final repository-wide `git diff --check`, repeated runner syntax checks, reporter compilation/help checks, expected-artifact assertions, and formal-log error scans all passed after the documentation updates.
