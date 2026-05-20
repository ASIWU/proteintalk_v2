# PTV3 Experiment Scripts

These scripts are thin launchers around `train.py` and `infer.py`.  In normal
use, change hyper-parameters by setting environment variables before `bash`;
you usually do not need to edit the scripts.

```bash
EXP_PREFIX=20260510_lr3e5_bs8 \
LEARNING_RATE=3e-5 \
BATCH_SIZE=8 \
FOLDS="0" \
bash scripts/exp_06_double_pert_pair_5fold.sh
```

The variables are read once when the script starts.  They apply to every fold
run by that command.  Use a new `EXP_PREFIX` for each run unless you
intentionally set `ALLOW_EXISTING_RUN=1`.

Only variables exposed by `scripts/ptv3_experiment_common.sh` can be changed
this way.  For lower-level `train.py` arguments that are not listed here, add
a new environment variable in the common script or run `train.py` directly.

## Environment

The common script activates the repository environment automatically:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
```

It then uses `PYTHON_BIN`, defaulting to:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python
```

## Quick Starts

Run all required PTV3 experiment families:

```bash
EXP_PREFIX=20260510_full MAX_EPOCHS=50 BATCH_SIZE=16 bash scripts/run_all_required_ptv3_experiments.sh
```

Run one experiment family:

```bash
EXP_PREFIX=20260510_single_pert MAX_EPOCHS=50 BATCH_SIZE=16 bash scripts/exp_01_single_pert_stratified_5fold.sh
```

Run one fold as a smoke test:

```bash
EXP_PREFIX=20260510_test FOLDS="0" MAX_EPOCHS=2 bash scripts/exp_01_single_pert_stratified_5fold.sh
```

Limit the smoke test to a few batches.  Use integer `1` for one batch;
use float `1.0` for all batches.

```bash
EXP_PREFIX=20260510_smoke \
FOLDS="0" \
MAX_EPOCHS=1 \
LIMIT_TRAIN_BATCHES=2 \
LIMIT_VAL_BATCHES=2 \
LIMIT_TEST_BATCHES=2 \
RUN_PREFLIGHT=0 \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

## Scripts

| Script | Runs |
|---|---|
| `exp_01_single_pert_stratified_5fold.sh` | single-drug 5-fold, stratified `pert_id` split |
| `exp_02_single_cell_type_5fold.sh` | single-drug 5-fold, `cell_type` split |
| `exp_03_single_cell_5fold.sh` | single-drug 5-fold, `cell` split |
| `exp_04_single_no_mse_5fold.sh` | single-drug no-MSE ablation; adds `--no-mse-loss` |
| `exp_05_single_no_pdi_5fold.sh` | single-drug no-PDI ablation |
| `exp_06_double_pert_pair_5fold.sh` | double-drug canonical pert-pair 5-fold |
| `exp_07_extra_single_all_train_infer.sh` | train all single-drug data, then infer extra single datasets |
| `exp_08_extra_double_all_train_infer.sh` | train all single+double data, then infer extra double datasets |
| `run_all_required_ptv3_experiments.sh` | calls all eight scripts in order |

`run_all_required_ptv3_experiments.sh` runs preflight once before the first
experiment, then sets `RUN_PREFLIGHT=0` for the remaining experiment families.

## How To Change Hyper-Parameters

Use this pattern:

```bash
VAR1=value VAR2=value bash scripts/exp_XX_name.sh
```

For long commands, put one variable per line:

```bash
EXP_PREFIX=20260510_tuned_double \
FOLDS="0 1 2 3 4" \
MAX_EPOCHS=100 \
LEARNING_RATE=5e-5 \
BATCH_SIZE=8 \
HIDDEN_DIM=384 \
NUM_HEADS=8 \
NUM_LAYERS=6 \
DROPOUT=0.2 \
SCHEDULER_NAME=cosine_warmup \
bash scripts/exp_06_double_pert_pair_5fold.sh
```

For a single fold, change only `FOLDS`:

```bash
EXP_PREFIX=20260510_fold2 FOLDS="2" bash scripts/exp_06_double_pert_pair_5fold.sh
```

For multiple selected folds:

```bash
EXP_PREFIX=20260510_folds024 FOLDS="0 2 4" bash scripts/exp_06_double_pert_pair_5fold.sh
```

## Common Run Variables

| Variable | Default | Meaning |
|---|---:|---|
| `EXP_PREFIX` | timestamp + script name | Prefix used in checkpoint, log, output, and runtime-summary paths |
| `FOLDS` | `0 1 2 3 4` | Space-separated fold ids to run |
| `MAX_EPOCHS` | `50` | Number of training epochs |
| `RUN_PREFLIGHT` | `1` | Compile and data-validate before training |
| `ALLOW_EXISTING_RUN` | `0` | Fail if output dirs already contain files; set `1` only to intentionally reuse dirs |
| `CKPT_DIR` | `checkpoints` | Base checkpoint directory |
| `LOG_DIR` | `logs` | Base TensorBoard/runtime log directory |
| `OUTPUT_DIR` | `outputs` | Base inference output directory |
| `TIME_SUMMARY_PATH` | `logs/${EXP_PREFIX}_runtime_summary.tsv` | Runtime summary TSV path |

Each training run writes a manifest under:

```bash
checkpoints/${EXP_PREFIX}_.../run_manifest.json
```

Each script appends runtime rows to:

```bash
logs/${EXP_PREFIX}_runtime_summary.tsv
```

## Training Hyper-Parameters

| Variable | Default | Passed to `train.py` | Notes |
|---|---:|---|---|
| `BATCH_SIZE` | `16` | `--batch-size` | Per-GPU batch size. Effective global batch is roughly `BATCH_SIZE * DEVICES`. |
| `MAX_EPOCHS` | `50` | `--max-epochs` | Training epochs per fold. |
| `LEARNING_RATE` | `1e-4` | `--learning-rate` | Base optimizer learning rate. |
| `OPTIMIZER_NAME` | `adamw` | `--optimizer-name` | Supported: `adam`, `sgd`, `adamw`, `adamw_fused`, `adamw_fused_<multiplier>`. |
| `SCHEDULER_NAME` | empty | `--scheduler-name` | Optional: `cosine`, `step`, `plateau`, `cosine_warmup`. Empty means no scheduler. |
| `GRADIENT_CLIP_VAL` | `1.0` | `--gradient-clip-val` | Lightning gradient clipping value. |
| `LOG_EVERY_N_STEPS` | `1` | `--log-every-n-steps` | Trainer logging interval. Use `1` for every optimizer step in wandb/TensorBoard. |
| `CHECK_VAL_EVERY_N_EPOCH` | `1` | `--check-val-every-n-epoch` | Validation frequency in epochs. |
| `LIMIT_TRAIN_BATCHES` | `1.0` | `--limit-train-batches` | `1.0` means all train batches; `1` means exactly one batch. |
| `LIMIT_VAL_BATCHES` | `1.0` | `--limit-val-batches` | Same integer vs float rule as above. |
| `LIMIT_TEST_BATCHES` | `1.0` | `--limit-test-batches` | Same integer vs float rule as above. |

Common examples:

```bash
# Smaller learning rate and batch size.
EXP_PREFIX=20260510_lr3e5_bs8 \
LEARNING_RATE=3e-5 \
BATCH_SIZE=8 \
bash scripts/exp_06_double_pert_pair_5fold.sh

# Longer training with cosine warmup.
EXP_PREFIX=20260510_100ep_warmup \
MAX_EPOCHS=100 \
SCHEDULER_NAME=cosine_warmup \
bash scripts/exp_06_double_pert_pair_5fold.sh

# Quick debug run on one fold and two batches.
EXP_PREFIX=20260510_debug \
FOLDS="0" \
MAX_EPOCHS=1 \
LIMIT_TRAIN_BATCHES=2 \
LIMIT_VAL_BATCHES=2 \
LIMIT_TEST_BATCHES=2 \
LOGGER_BACKEND=none \
RUN_PREFLIGHT=0 \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

If wandb shows too few train-loss points, check `BATCH_SIZE * DEVICES` and
`LOG_EVERY_N_STEPS`.  Large per-GPU batches can leave only a small number of
optimizer steps per epoch; `LOG_EVERY_N_STEPS=1` records every step.

## Model Architecture Hyper-Parameters

| Variable | Default | Passed to `train.py` | Notes |
|---|---:|---|---|
| `HIDDEN_DIM` | `256` | `--hidden-dim` | Main transformer hidden dimension. |
| `NUM_HEADS` | `8` | `--num-heads` | Attention heads. Keep `HIDDEN_DIM` divisible by `NUM_HEADS`. |
| `NUM_LAYERS` | `4` | `--num-layers` | Transformer layer count. |
| `DROPOUT` | `0.1` | `--dropout` | Dropout probability. |

Example:

```bash
EXP_PREFIX=20260510_deeper \
HIDDEN_DIM=384 \
NUM_HEADS=8 \
NUM_LAYERS=6 \
DROPOUT=0.2 \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

## Loss And Class-Imbalance Hyper-Parameters

| Variable | Default | Passed to `train.py` | Notes |
|---|---:|---|---|
| `MSE_WEIGHT` | `1.0` | `--mse-weight` | Weight on expression MSE loss when MSE is enabled. |
| `BCE_WEIGHT` | empty | `--bce-weight` | Overrides active task label BCE weight for the current task head; empty uses `train.py` defaults. |
| `POSITIVE_WEIGHT` | empty | `--positive-weight` | Positive-class weight for the active BCE loss; empty means no positive-class reweighting. |
| `FOCAL_LOSS` | `0` | `--focal-loss` if `1` | Uses the focal-loss path instead of standard BCE. |

Notes:

- `exp_04_single_no_mse_5fold.sh` disables MSE with `--no-mse-loss`.
- `MSE_WEIGHT=0` makes the weighted MSE contribution zero, but it does not
  change the script identity or ablation naming.
- `BCE_WEIGHT`, `POSITIVE_WEIGHT`, and `FOCAL_LOSS` are useful when label
  imbalance is more important than expression reconstruction.

Examples:

```bash
# Increase expression reconstruction weight.
EXP_PREFIX=20260510_mse2 \
MSE_WEIGHT=2.0 \
bash scripts/exp_01_single_pert_stratified_5fold.sh

# Increase positive-class weight for the active task label.
EXP_PREFIX=20260510_posw3 \
POSITIVE_WEIGHT=3.0 \
bash scripts/exp_06_double_pert_pair_5fold.sh

# Use focal loss.
EXP_PREFIX=20260510_focal \
FOCAL_LOSS=1 \
bash scripts/exp_06_double_pert_pair_5fold.sh
```

## GPU And Performance Variables

| Variable | Default | Meaning |
|---|---:|---|
| `GPU_IDS` | `0,1,2,3,4,5,6,7` | Value assigned to `CUDA_VISIBLE_DEVICES` |
| `DEVICES` | `8` | Lightning `--devices`; keep this equal to the number of visible GPUs for DDP |
| `PRECISION` | `32-true` | Lightning precision string |
| `NUM_WORKERS` | `0` | DataLoader workers |
| `INFER_BATCH_SIZE` | `16` | Inference batch size for scripts 07/08 |
| `INFER_DEVICE` | `cuda:0` | Device passed to `infer.py` |
| `INFER_LIMIT_BATCHES` | empty | Optional inference batch limit |

Single-GPU example:

```bash
EXP_PREFIX=20260510_gpu3_single \
GPU_IDS=3 \
DEVICES=1 \
BATCH_SIZE=16 \
INFER_DEVICE=cuda:0 \
bash scripts/exp_06_double_pert_pair_5fold.sh
```

Because `GPU_IDS=3` becomes `CUDA_VISIBLE_DEVICES=3`, the visible device inside
the process is still `cuda:0`.

Two-GPU example:

```bash
EXP_PREFIX=20260510_gpu45 \
GPU_IDS=4,5 \
DEVICES=2 \
BATCH_SIZE=12 \
bash scripts/exp_06_double_pert_pair_5fold.sh
```

## Checkpoint Variables

| Variable | Default | Meaning |
|---|---:|---|
| `BEST_CKPT_METRIC` | `valid_auprc` | Named validation metric for best-checkpoint selection |
| `SAVE_EVERY_N_EPOCHS` | `1` | Save periodic epoch checkpoints every N epochs |
| `SAVE_EVERY_N_TRAIN_STEPS` | empty | Save periodic train-step checkpoints every N train steps |
| `SAVE_TOP_K` | `-1` | Number of monitored checkpoints to keep; `-1` keeps all |
| `SAVE_LAST_CKPT` | `1` | Save `last.ckpt`; set `0` to pass `--no-save-last-ckpt` |
| `CHECKPOINT_FILENAME` | empty | Custom Lightning checkpoint filename pattern |
| `MONITOR` | empty | Raw Lightning metric override; normally leave empty and use `BEST_CKPT_METRIC` |
| `MONITOR_MODE` | empty | Raw override mode; normally inferred from `BEST_CKPT_METRIC` |
| `ALLOW_NONFINITE_MONITOR` | `0` | Fail if the checkpoint metric is NaN/Inf; set `1` only for debugging |
| `REFERENCE_5FOLD_CKPT_PATH` | empty | Optional reference 5-fold checkpoint path/glob/prefix for scripts 07/08 fixed-epoch extra evaluation |
| `REFERENCE_EPOCH_AGG` | `median` | Aggregate fold best epochs with `median`, `mean`, `min`, or `max` |
| `REFERENCE_EPOCH_ROUNDING` | `nearest` | Round fractional aggregate epochs with `nearest`, `floor`, or `ceil` |
| `REFERENCE_EPOCH_MIN_COUNT` | `5` | Minimum usable reference fold manifests required |
| `REFERENCE_REQUIRE_TEST_COMPLETED` | `1` | Require reference folds to have completed their own test pass |
| `REFERENCE_SPLIT_STRATEGY_REGEX` | script default | Override the expected reference split regex |
| `REFERENCE_ALLOW_MIXED_CONFIG` | `0` | Allow mixed reference model/loss/monitor configs; normally keep `0` |
| `REFERENCE_ALLOW_DUPLICATE_SPLITS` | `0` | Allow duplicate reference split strategies; normally keep `0` |

`BEST_CKPT_METRIC` accepts:

| Value | Lightning monitor | Mode | Meaning |
|---|---|---|---|
| `valid_auprc` or `auprc` | `val/task_auprc` | `max` | Best active-task validation AUPRC; this is the default |
| `valid_total_loss` or `total_loss` | `val/total_loss` | `min` | Previous default behavior |
| `valid_loss1` or `loss1` | `val/loss1` | `min` | Best validation expression MSE loss |
| `valid_loss2` or `loss2` | `val/loss2` | `min` | Best validation active-label BCE loss |

For single-drug runs, active-task AUPRC is response AUPRC.  For double-drug
runs, active-task AUPRC is synergy AUPRC.

If the selected checkpoint metric is missing or non-finite, training fails
instead of silently choosing a meaningless checkpoint.  For AUPRC/AUROC this
usually means the evaluated validation batches contain only one class.  Use
full validation data, or use `BEST_CKPT_METRIC=total_loss` / `loss1` / `loss2`
for tiny smoke tests.

When `SCHEDULER_NAME=plateau`, the plateau scheduler monitors the same metric
and mode as the selected checkpoint metric.

Choose best checkpoint by validation BCE loss:

```bash
EXP_PREFIX=20260510_loss2_best \
BEST_CKPT_METRIC=valid_loss2 \
bash scripts/exp_06_double_pert_pair_5fold.sh
```

Restore the old total-loss checkpoint selection:

```bash
EXP_PREFIX=20260510_total_loss_best \
BEST_CKPT_METRIC=total_loss \
bash scripts/exp_06_double_pert_pair_5fold.sh
```

Save every 25 epochs:

```bash
EXP_PREFIX=20260510_ckpt25 \
SAVE_EVERY_N_EPOCHS=25 \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

Save every train step:

```bash
EXP_PREFIX=20260510_ckpt_step \
SAVE_EVERY_N_TRAIN_STEPS=1 \
MONITOR=none \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

If `SAVE_EVERY_N_TRAIN_STEPS` is set and `MONITOR` is not set by the user, the
wrapper automatically changes `MONITOR` to `none`.  This avoids monitoring a
validation metric at train-step checkpoint time.

Use a custom checkpoint filename:

```bash
EXP_PREFIX=20260510_named_ckpt \
CHECKPOINT_FILENAME="{epoch}-{step}" \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

## Logging Variables

| Variable | Default | Meaning |
|---|---:|---|
| `LOGGER_BACKEND` | `tensorboard` | `tensorboard`, `wandb`, `both`, or `none` |
| `LOG_TO_WANDB` | `0` | Legacy switch; if `1`, uses wandb even if `LOGGER_BACKEND` differs |
| `WANDB_PROJECT` | `aivc_proteintalk` | wandb project name |
| `WANDB_ENTITY` | empty | optional wandb entity |
| `WANDB_GROUP` | empty | optional wandb group |
| `WANDB_TAGS` | empty | space-separated wandb tags |
| `WANDB_MODE` | empty | optional `online`, `offline`, or `disabled` |
| `WANDB_LOG_MODEL` | `0` | log checkpoint artifacts to wandb when `1` |

Wandb only:

```bash
EXP_PREFIX=20260510_wandb \
LOGGER_BACKEND=wandb \
WANDB_PROJECT=aivc_proteintalk \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

TensorBoard and wandb together:

```bash
EXP_PREFIX=20260510_both \
LOGGER_BACKEND=both \
WANDB_PROJECT=aivc_proteintalk \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

Offline wandb:

```bash
EXP_PREFIX=20260510_offline \
LOGGER_BACKEND=wandb \
WANDB_MODE=offline \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

No logger for debugging:

```bash
EXP_PREFIX=20260510_no_logger \
LOGGER_BACKEND=none \
FOLDS="0" \
MAX_EPOCHS=1 \
bash scripts/exp_01_single_pert_stratified_5fold.sh
```

## Extra-Data Inference Variables

Scripts 07 and 08 train on all available training data and then run inference
on extra datasets.  These variables are most relevant there:

| Variable | Default | Meaning |
|---|---:|---|
| `RUN_INFERENCE` | `1` | Run inference after training in scripts 07/08 |
| `INFER_BATCH_SIZE` | `16` | Batch size passed to `infer.py` |
| `INFER_DEVICE` | `cuda:0` | Inference device |
| `INFER_LIMIT_BATCHES` | empty | Optional inference limit for debugging |
| `OUTPUT_DIR` | `outputs` | Base directory for inference outputs |

By default, scripts 07/08 keep their previous behavior: train the all-data
model, read `best_model_path` from that run's `run_manifest.json`, then use it
for extra-data inference.

For leakage-safe extra-data evaluation, pass a reference 5-fold checkpoint path
or prefix.  The script reads the reference folds' `best_model_path` epochs,
aggregates them, retrains the all-data model for `selected_epoch + 1` epochs,
sets `--monitor none`, and uses `last.ckpt` for extra inference.  The extra
test labels are not used to choose the checkpoint.  The all-data training
manifest records this under `reference_epoch_policy`.

The reference selector is intentionally strict.  By default it requires five
completed reference folds, existing checkpoint files, matching task head,
matching model/loss/monitor configuration, distinct split strategies, and
`test_status=test_completed`.  Script 07 expects `pert_stratified_5fold_fold*`;
script 08 expects `pert_id_5fold_fold*`.  Reference policy also rejects
`SCHEDULER_NAME=plateau`, because a validation-driven scheduler would let the
all-data validation split affect the final weights.

Extra single with the median best epoch from a previous single-drug 5-fold run:

```bash
EXP_PREFIX=20260511_extra_single_ref_epoch \
REFERENCE_5FOLD_CKPT_PATH=checkpoints/20260510_single_pert_stratified_5fold \
REFERENCE_EPOCH_MIN_COUNT=5 \
bash scripts/exp_07_extra_single_all_train_infer.sh
```

Use the mean epoch instead:

```bash
REFERENCE_5FOLD_CKPT_PATH=checkpoints/20260510_single_pert_stratified_5fold \
REFERENCE_EPOCH_AGG=mean \
bash scripts/exp_07_extra_single_all_train_infer.sh
```

The same mechanism works for extra double if the reference path points to
double-drug 5-fold checkpoint directories.

Train but skip extra inference:

```bash
EXP_PREFIX=20260510_train_only \
RUN_INFERENCE=0 \
bash scripts/exp_07_extra_single_all_train_infer.sh
```

Debug extra inference on a small number of batches:

```bash
EXP_PREFIX=20260510_extra_debug \
FOLDS="0" \
MAX_EPOCHS=1 \
INFER_LIMIT_BATCHES=2 \
RUN_PREFLIGHT=0 \
bash scripts/exp_08_extra_double_all_train_infer.sh
```

## Recommended Tuning Workflow

1. Start with one fold and short training:

   ```bash
   EXP_PREFIX=20260510_trial_a \
   FOLDS="0" \
   MAX_EPOCHS=5 \
   LEARNING_RATE=1e-4 \
   BATCH_SIZE=8 \
   bash scripts/exp_06_double_pert_pair_5fold.sh
   ```

2. Compare one change at a time with a new `EXP_PREFIX`:

   ```bash
   EXP_PREFIX=20260510_trial_b \
   FOLDS="0" \
   MAX_EPOCHS=5 \
   LEARNING_RATE=5e-5 \
   BATCH_SIZE=8 \
   bash scripts/exp_06_double_pert_pair_5fold.sh
   ```

3. When a setting looks good, run all folds:

   ```bash
   EXP_PREFIX=20260510_final_lr5e5_bs8 \
   FOLDS="0 1 2 3 4" \
   MAX_EPOCHS=50 \
   LEARNING_RATE=5e-5 \
   BATCH_SIZE=8 \
   bash scripts/exp_06_double_pert_pair_5fold.sh
   ```

4. Inspect:

   ```bash
   logs/${EXP_PREFIX}_runtime_summary.tsv
   checkpoints/${EXP_PREFIX}_*/run_manifest.json
   ```

## Gotchas

- `BATCH_SIZE` is per GPU.  If `DEVICES=8` and `BATCH_SIZE=16`, the effective
  global batch is about `128`.
- Keep `DEVICES` consistent with `GPU_IDS`.  For example, use
  `GPU_IDS=4,5 DEVICES=2`.
- Use `LIMIT_TRAIN_BATCHES=1.0` for all batches.  `LIMIT_TRAIN_BATCHES=1`
  means one batch.
- Use a fresh `EXP_PREFIX` for each run.  Existing non-empty checkpoint, log,
  or output directories make the script fail by default.
- Prefer `BEST_CKPT_METRIC` for checkpoint selection.  If you use raw
  `MONITOR`, its metric name must match a metric logged by `train.py`.
- Step checkpointing should normally use `MONITOR=none`.
