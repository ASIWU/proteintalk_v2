# 2026-05-09 18:35 HKT Script and Hyperparameter Smoke Review

## Scope

- Tested human-readable PTV3 experiment scripts after wandb/hyperparameter support was added.
- Audited generated training/inference manifests for bounded 8-GPU smoke runs.
- Checked logger and checkpoint hyperparameter paths: tensorboard, wandb, both, none, epoch checkpoints, and step checkpoints.

## Findings

- `scripts/run_all_required_ptv3_experiments.sh` completed bounded 8-GPU execution for all 8 required training jobs and all 9 extra-data inference jobs under prefix `20260509_scripts_all_smoke`.
- Hyperparameter overrides were recorded correctly in `checkpoints/20260509_hparam_wandb_smoke_single_pert_stratified_fold0/run_manifest.json`.
- `exp_05_single_no_pdi_5fold.sh` correctly disables real PDI through `pdi_mode=zero`; the manifest still records the matrix path because the model receives an all-zero matrix with the same shape.
- Found a real bug: `LOGGER_BACKEND=none` still attached `LearningRateMonitor`, causing Lightning to fail before training.

## Fix

- Updated `train.py` so `LearningRateMonitor` is only added when a logger exists.
- Re-ran step checkpoint smoke with `LOGGER_BACKEND=none`, `SAVE_EVERY_N_TRAIN_STEPS=1`, and `MONITOR=none`; it completed and produced `epoch=0-step=1.ckpt` plus `last.ckpt`.
- Re-ran combined logger smoke with `LOGGER_BACKEND=both` and `WANDB_MODE=disabled`; it completed.
- Ran the compatibility wrapper `scripts/train_required_ptv3_experiments.sh` with bounded 8-GPU settings and `RUN_INFERENCE=0`; all 8 training rows in `logs/20260509_wrapper_smoke_runtime_summary.tsv` exited with status `0`.

## Verification

- `bash -n` passed for `scripts/ptv3_experiment_common.sh`, all 8 `scripts/exp_*.sh` files, `scripts/run_all_required_ptv3_experiments.sh`, and `scripts/train_required_ptv3_experiments.sh`.
- `py_compile` passed for `train.py`, `infer.py`, `dataset/training_ready_dataset.py`, `model/training_ready_models.py`, and `model/training_ready_lightning.py`.
- Runtime summaries:
  - `logs/20260509_scripts_all_smoke_runtime_summary.tsv`
  - `logs/20260509_hparam_wandb_smoke_runtime_summary.tsv`
  - `logs/20260509_ckpt_step_smoke2_runtime_summary.tsv`
  - `logs/20260509_logger_both_smoke_runtime_summary.tsv`
  - `logs/20260509_wrapper_smoke_runtime_summary.tsv`
