# 2026-05-25 20:54 HKT MSE-Gap Training Strategy Review

## Scope

- Reviewed whether training strategy alone can expand the single-drug w/o-MSE ablation gap to 5 AUPRC points.
- Kept the baseline4 architecture and existing data unchanged.
- Added only default-off infrastructure for true EarlyStopping and a reusable 2-GPU strategy screening script.

## Code Changes Reviewed

- `train.py`
  - Added optional Lightning `EarlyStopping`.
  - New flags: `--early-stopping-patience`, `--early-stopping-min-delta`.
  - Defaults leave existing training unchanged.
- `scripts/ptv3_experiment_common.sh`
  - Added `EARLY_STOPPING_PATIENCE` and `EARLY_STOPPING_MIN_DELTA` plumbing.
- `scripts/run_mse_gap_ckpt_strategy_2gpu.sh`
  - Added generic `lastN` fixed-final-epoch strategies.
  - Added `VARIANTS` so MSE-only sweeps can reuse an already computed no-MSE reference.

## Experiments

- Dataset/task: `ptv3_main_singledrug / pert_stratified_5fold`.
- Default runtime setting: 1 GPU per job, batch size 256.
- Compared paired with-MSE and w/o-MSE runs whenever the strategy affected both variants.
- Strategy families tested:
  - best-checkpoint monitor: `valid_auprc`, `valid_auroc`, `loss2`, `total_loss`;
  - fixed final epoch: `last3`, `last5`, `last8`, `last10`, `last20`;
  - MSE weight/schedule under `last8`;
  - true EarlyStopping with `patience=1` and `patience=3`;
  - short-budget best-checkpoint with `MAX_EPOCHS=5` and `MAX_EPOCHS=8`.

## Results

- Best-checkpoint metrics did not help:
  - best AUPRC gap among monitor choices was `valid_auprc`, only `+0.002553`.
- Fixed final epoch had the best training-only gap:
  - `last8`: with/w/o AUPRC `0.664738 / 0.640970`, gap `+0.023768`.
- Increasing MSE weight did not improve the gap:
  - `MSE_WEIGHT=0.5`: gap `+0.020872`;
  - `MSE_WEIGHT=1.0`: gap `-0.002439`.
- Warmup-decay did not improve the gap:
  - `0.5 -> 0.1 effective tail`: gap `+0.020476`;
  - `1.0 -> 0.1 effective tail`: gap `+0.013399`.
- True EarlyStopping did not help:
  - `patience=3`: gap `+0.012671`;
  - `patience=1`: gap `-0.001552`.
- Short-budget best-checkpoint did not help:
  - `MAX_EPOCHS=5`: gap `+0.007895`;
  - `MAX_EPOCHS=8`: gap `+0.014979`.

## Conclusion

- Training strategy alone did not expand the w/o-MSE ablation gap to 5 AUPRC points.
- The most defensible training-only setting is `last8`, but its gap is only about `+2.38` points.
- True EarlyStopping is useful infrastructure, but should not be used as the default MSE-gap claim.
- No data files or data-processing code were changed.
