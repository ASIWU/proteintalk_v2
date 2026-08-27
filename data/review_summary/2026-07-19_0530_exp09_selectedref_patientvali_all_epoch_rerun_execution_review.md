# 2026-07-19 05:30 HKT Exp09 Selectedref patientVali All-Epoch Rerun Execution Review

## Scope

- Implemented and exercised the staged patientVali260605v3 inference workflow for all numbered checkpoints in `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra`.
- Reused `data/training_ready_patientVali260605v3` without rebuilding raw data, standardized data, task tables, splits, expression matrices, or derived features.
- Preserved all historical output directories and did not use `--force`.
- Reviewed the target entry point `utils/17_infer_patient_validation_exp09_all_epoch_ckpts.py`, its shared inference/export helper `utils/16_infer_patient_validation_all_epoch_ckpts.py`, the selectedref run manifest, the eight patientVali tasks/splits, and the historical selectedref output structure used as the schema baseline.

## Implementation Review

- Added mutually exclusive `--infer-only` and `--skip-infer` modes to `utils/17_infer_patient_validation_exp09_all_epoch_ckpts.py`.
- `--infer-only` invokes per-checkpoint/task `infer.py` jobs and returns without reading prediction tables, creating the readable output directory, or writing combined/summary files.
- Existing no-`--force` behavior is retained, so a repeated `--infer-only` command skips already completed prediction files and safely resumes remaining work.
- `--skip-infer` remains the CPU-only aggregation path and writes both raw combined files and all readable exports from existing per-task predictions.
- The code change was appended to `docs/2026-04-15_data_standardization_session_summary.md` without replacing its pre-existing uncommitted content.

## Preflight and Interface Verification

- Activated `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2` through the required conda setup.
- Confirmed exactly 50 numbered checkpoints with continuous epochs `0..49`; `last.ckpt` exists but was not selected.
- Confirmed all eight expected task directories and test-only split directories, plus all seven derived inference artifacts used by the launcher.
- Confirmed expected per-checkpoint coverage of 876 rows: 16 single-drug and 860 double-drug rows, giving 43,800 rows across 50 checkpoints.
- Confirmed both new output paths were absent before execution and approximately 99 GB was free on the shared filesystem.
- `python -m py_compile utils/17_infer_patient_validation_exp09_all_epoch_ckpts.py` passed.
- Supplying `--infer-only --skip-infer` together was rejected by argparse with exit code 2.

## GPU Execution

- Confirmed tmux session `gpu2:0` was attached to the shared-path GPU worker.
- Worker preflight reported one NVIDIA H200, CUDA available, and the `flow_v2` Python executable.
- Ran epoch-0 smoke inference with `--infer-only --max-checkpoints 1`; all eight tasks completed with 876 finite bounded probabilities.
- Smoke verification confirmed every task had `predictions.parquet`, `metrics.json`, `run_manifest.json`, and `infer.log`, while raw combined files and the readable directory remained absent.
- Ran the resumable full command with `--infer-only --max-checkpoints 50`; the eight smoke predictions were reused and epochs 1-49 were completed without `--force`.
- The first smoke prediction was written at approximately 04:44 HKT and the final epoch-49 prediction at approximately 05:28 HKT.
- GPU dispatcher log: `logs/20260719_patientVali260605v3_exp09_all_epoch_ckpts_rerun_v1_gpu.log`.
- The dispatcher completed with `[done] inference only: checkpoints=50 tasks_per_checkpoint=8`; error scans found no traceback, runtime error, or inference failure.

## Aggregation and Final Artifacts

- After confirming all 400 prediction files, ran the same script locally with `--skip-infer --max-checkpoints 50`.
- Raw output: `outputs/2026-07/2026-07-19/20260719_patientVali260605v3_exp09_all_epoch_ckpts_rerun_v1` (about 172 MB).
- Readable output: `outputs/2026-07/2026-07-19/0719v3_exp09_all_epoch_ckpts_rerun_v1` (about 88 MB).
- Raw output contains epochs `exp09_epoch000` through `exp09_epoch049`, 400 per-task prediction directories, `combined_predictions.csv`, `combined_predictions.parquet`, and `inference_summary.json`.
- Readable output contains 400 per-task CSV files, `README.md`, `manifest.json`, `combined_predictions_readable.csv`, and checkpoint/task/dataset summary CSV files.
- The unrelated historical file `exp07_exp08_all_epochs_all_dataset_results.csv` was not copied.

## Final Validation

- Raw and readable combined row counts: 43,800.
- Treatment counts: 800 single-drug and 43,000 double-drug rows.
- Checkpoint coverage: exactly epochs `0..49` and labels `exp09_epoch000..exp09_epoch049`.
- Task coverage: exactly the three single-drug and five double-drug patientVali tasks.
- `score_name` is uniformly `pred_task_prob`.
- All prediction scores are non-missing, finite, and in `[0,1]`; observed range is approximately `3.770898e-06` to `1.0`.
- Summary coverage: 50 checkpoint rows, 400 checkpoint/task rows, and 250 checkpoint/dataset rows.
- Readable per-task CSV count is 400 and their row counts sum to 43,800.
- Raw and readable manifests are identical, contain 400 inference records, and reference only `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra`; the lr1e5 `20260615_2010` run is absent.
- Raw and readable column schemas match the historical selectedref generator outputs, and coverage/row counts match the historical standard. Prediction probabilities were intentionally not required to equal the historical values.
- Final `py_compile` and `git diff --check` passed for the modified script and session summary.

## Workspace Preservation

- Pre-existing modifications in `infer.py`, model/data modules, experiment scripts, documentation, and prior review records were left unchanged.
- This execution edited only the exp09 patient-validation entry point, appended its change history to the required session summary, and added this execution review record; inference and aggregation created only the two new output directories and the new GPU log.
