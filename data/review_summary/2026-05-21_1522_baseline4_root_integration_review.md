# 2026-05-21 15:22 HKT Baseline4 Root Integration Review

## Scope

Reviewed the migration of the baseline4 fast model from `new_version` into the
root training and inference workflow.

## Files Reviewed

- `train.py`
- `infer.py`
- `dataset/training_ready_fast_dataset.py`
- `model/fast_delta_model.py`
- `model/fast_lightning.py`
- `model/graph_feature_utils.py`
- `model/training_ready_models.py`
- `scripts/ptv3_experiment_common.sh`
- `scripts/exp_01_single_pert_stratified_5fold.sh`
- `scripts/exp_02_single_cell_type_5fold.sh`
- `scripts/exp_03_single_cell_5fold.sh`
- `scripts/exp_04_single_no_mse_5fold.sh`
- `scripts/exp_05_single_no_pdi_5fold.sh`
- `scripts/exp_06_double_pert_pair_5fold.sh`
- `scripts/exp_07_extra_single_all_train_infer.sh`
- `scripts/exp_08_extra_double_all_train_infer.sh`
- `scripts/0521_baseline4_8gpu_parallel.sh`

## Checks

- No data artifact files were modified.
- Data processing utilities under `utils/` were not modified.
- `train.py` now supports `model_type=fast_delta` at root level and keeps the
  legacy model path available.
- Root default settings are single GPU, `batch_size=256`, bf16 mixed precision,
  baseline4 graph/PCEP options, and wandb logging.
- `exp_05_single_no_pdi_5fold.sh` now performs a matched w/o graph feature
  ablation with `--graph-feature-mode zero`.
- The 8-GPU launcher schedules exp01-06 fold tasks as independent single-GPU
  jobs and then runs exp07/08 after reference folds are available.
- Fast inference aligns extra-task expression matrices to the checkpoint protein
  axis, allowing extra single/double inference when the task protein axis differs
  from the training task axis.

## Verification

Static checks passed:

- `python -m py_compile train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py model/fast_lightning.py model/graph_feature_utils.py model/training_ready_models.py model/training_ready_lightning.py scripts/select_reference_epoch.py`
- `bash -n scripts/ptv3_experiment_common.sh`
- `bash -n scripts/exp_0*.sh`
- `bash -n scripts/0521_baseline4_8gpu_parallel.sh`

Runtime checks passed on two GPUs:

- Root `train.py` dry-run produced full-axis output:
  - expression `(256, 10982)`
  - response logits `(256, 1)`
  - synergy logits `(256, 1)`
- `scripts/0521_baseline4_8gpu_parallel.sh` smoke with `GPU_IDS=0,1`,
  `FOLDS=0`, one train/valid/test batch, and one inference batch completed.
- exp01-06 fold0 manifests are `fit_completed/test_completed`.
- exp07 and exp08 all-train manifests are `fit_completed/skipped`.
- Extra inference wrote 9 `predictions.parquet` files:
  - 6 extra single tasks;
  - 3 extra double tasks.

## Residual Risk

The smoke test intentionally used one epoch and batch limits. Full 8-GPU
validation should be run with the default script settings to confirm final
metrics and wall-clock behavior on the user's standard machine.

## Follow-up: Reference Epoch Aggregation

After user confirmation, the default extra-inference reference epoch aggregation
was changed from `median` to `mean` in `scripts/ptv3_experiment_common.sh`.
Thus exp07/exp08 now use the average best epoch from the corresponding 5-fold
reference runs before training the all-data model and evaluating extra tasks.
