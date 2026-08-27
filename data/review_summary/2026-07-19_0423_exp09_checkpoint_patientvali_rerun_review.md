# 2026-07-19 04:23 HKT Exp09 Checkpoint and patientVali Rerun Review

## Scope

- Traced the exp09 initialization artifacts used by the Exp31 RNA-seq PDX fine-tuning benchmark and the Exp32 organoid sensitivity inference.
- Checked the selected exp09 training run and all of its saved checkpoints.
- Reconstructed the script, checkpoint source, and patient-validation data used for `outputs/2026-06/2026-06-15/0615v3_exp09_lr1e5_all_epoch_ckpts`.
- Assessed whether the selected Exp31/Exp32 exp09 run can be inferred again with the patient-validation all-epoch script. No inference was launched and no model or data artifact was changed.

## Exp31 versus Exp32 Checkpoint Identity

- Both experiments use the same exp09 training run:
  `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra`.
- They do not use the same checkpoint file:
  - all four Exp31 fine-tuning run manifests record `last.ckpt` as the initialization checkpoint;
  - the Exp32 result summary and runner record `epoch=5.ckpt` as the inference checkpoint.
- Current SHA-256 values:
  - `last.ckpt`: `e75a2314bb11b698acc48739891578e7de0fddb293118f3e3c0fdc3419a39c2d`;
  - `epoch=5.ckpt`: `a5734682c37d807b2b6d1bb5ea66ad908107f45e15605aa65c40f6bab9b6db97`;
  - `epoch=49.ckpt`: `e75a2314bb11b698acc48739891578e7de0fddb293118f3e3c0fdc3419a39c2d`.
- Therefore, Exp31 `last.ckpt` is byte-identical to `epoch=49.ckpt`, while Exp32 `epoch=5.ckpt` is a different saved model state.

## Selected Exp09 Run Availability

- `run_manifest.json` exists and records `run_status=fit_completed`, `max_epochs=50`, `save_every_n_epochs=1`, and `save_last_ckpt=true`.
- Saved epoch checkpoints `epoch=0.ckpt` through `epoch=49.ckpt` are all present with no gaps.
- `last.ckpt` is also present. The directory therefore retains 50 numbered epoch checkpoints plus `last.ckpt`, for 51 checkpoint files total.

## `0615v3_exp09_lr1e5_all_epoch_ckpts` Reconstruction

- The readable output manifest records the source checkpoint run as
  `checkpoints/20260615_2010_exp09_lr1e5_v1_unified_all_single_double_for_extra`.
- The generating entry point still exists:
  `utils/17_infer_patient_validation_exp09_all_epoch_ckpts.py`.
- It imports shared inference/export helpers from
  `utils/16_infer_patient_validation_all_epoch_ckpts.py`.
- The lr1e5 directory name is not hard-coded in a separate script. The generic script was run with overrides for `--exp09-dir`, `--output-root`, and `--readable-output-dir`.
- Both the raw and readable historical results remain complete at the directory level:
  - 50 checkpoints x 8 tasks = 400 raw `predictions.parquet` files;
  - 400 readable per-checkpoint/per-task CSV files;
  - 43,800 combined prediction rows recorded by the manifest.
- The inference-ready input root still exists at `data/training_ready_patientVali260605v3` with all 8 expected task directories, all 8 test-only split directories, `global_meta.json`, both build summaries, expression/task artifacts, protein/drug embeddings, PPI/PDI/DDI matrices, and cell/cell-type LLM embeddings.
- The source raw-data root `data/rawdata/ptv2drug_patientVali260605` and its five dataset inputs also remain present.
- The script help entry point loads successfully in the required `flow_v2` conda environment.

## Rerun Feasibility

- Repository-side checkpoint, script, and data requirements are present, so the selected exp09 run can be inferred again on patientVali260605v3 when a CUDA GPU is available.
- This was already successfully done historically: `outputs/2026-06/2026-06-15/0615v3_exp09_all_epoch_ckpts` records the same selectedref exp09 run, 50 epoch checkpoints, 8 tasks, and 43,800 predictions.
- To create a genuinely new run, both `--output-root` and `--readable-output-dir` should point to new unused directories. Otherwise the script reuses existing per-task prediction files unless `--force` is supplied.
- The script enumerates `epoch=*.ckpt` and does not directly select `last.ckpt`. This is not a weight-coverage problem for Exp31 because `last.ckpt` is byte-identical to `epoch=49.ckpt`. Exp32 `epoch=5.ckpt` is included in the all-epoch run.
- If only one exact checkpoint is desired, the current all-epoch script has no arbitrary epoch selector: `--max-checkpoints` only keeps the first N sorted epochs. A direct `infer.py` invocation or a checkpoint directory containing only the desired numbered epoch would be needed.
- `infer.py` and several model/data modules have uncommitted changes dated after the original June 15 inference. Their new paths default to legacy-compatible/off modes, so no static incompatibility was found, but exact bit-for-bit reproduction of the historical probabilities was not execution-tested in this review.
