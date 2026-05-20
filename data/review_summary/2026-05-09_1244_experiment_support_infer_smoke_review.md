# 2026-05-09 12:44 HKT Experiment Support / Inference Smoke Review

## Scope

- Reviewed and updated the training/inference entrypoints for the requested experiment set:
  single-drug 5-fold `pert_stratified`, `cell_type`, `cell`, single-drug no-MSE, single-drug no-PDI, double-drug `pert_id`, all-single training for extra single-drug validation, and all-single+double training for extra double-drug validation.
- Kept the current full protein-axis data contract. No top-2000 protein-axis path was introduced.
- Kept the two-loss contract: `loss1` is expression MSE and `loss2` is one active task label, response for single-drug and synergy for double-drug.

## Code Changes

- `train.py`
  - Added `--pdi-mode {real,zero}` so no-PDI ablation can use an all-zero PDI matrix with the same real artifact shape.
  - Added checkpoint monitor arguments and run-manifest fields.
  - Added split audit fields to the run manifest, including feature membership/source task counts and active-label nonempty/empty counts.
  - Ensured active task config is resolved before dataloader/model construction and recorded.
- `infer.py`
  - Added `--pdi-mode` and `--limit-batches`.
  - Relaxed checkpoint validation to compare only architecture-affecting fields, so external validation can use different task labels without needing `--allow-checkpoint-config-mismatch`.
  - Fixed limited inference output: prediction metadata now uses processed batch `row_index` values, not the full split, so `--limit-batches` writes correctly sized prediction tables.

## Data Support Confirmed

- `ptv3_main_singledrug` `all_train_subset_test`: train `17986`, valid `1798`, test `3597`; all train rows are `primary` single-drug rows with nonempty `PRISM1st_label_total`.
- `ptv3_main_doubledrug` `all_train_subset_test`: train `19777`, valid `179`, test `358`; train contains `17986` `merged_single_drug` rows and `1791` native double-drug rows. Active `synergy` is nonempty only for the `1791` native double rows and empty/masked for the `17986` merged single rows.
- Double-drug `pert_id_5fold_fold0` currently uses the same auxiliary-single-row design: train `19275` includes `17986` merged single-drug rows plus `1289` native double-drug rows; valid/test contain native double-drug rows only.

## Smoke Validation

- `py_compile` passed for:
  `train.py`, `infer.py`, `dataset/training_ready_dataset.py`, `model/training_ready_lightning.py`, `model/training_ready_models.py`.
- 8-GPU DDP 2-epoch bounded training smoke passed for:
  - `20260509_smoke_single_pert_stratified_2ep`
  - `20260509_smoke_single_cell_type_2ep`
  - `20260509_smoke_single_cell_2ep`
  - `20260509_smoke_single_no_mse_2ep`
  - `20260509_smoke_single_no_pdi_2ep`
  - `20260509_smoke_double_pert_pair_2ep`
  - `20260509_smoke_all_single_for_extra_2ep`
  - `20260509_smoke_all_single_double_for_extra_2ep`
- Extra single-drug inference smoke passed with `--limit-batches 2`:
  `mat1_480_faims`, `mat1_qe`, `mat2_480_faims`, `mat2_qe`, `mat3_qe`, `mat4_qe`.
- Extra double-drug inference smoke passed with `--limit-batches 2`:
  `nature`, `nc`, `guomics`.
- Final `nvidia-smi --query-compute-apps` showed no active GPU compute process.

## Notes

- Parallel single-process inference on multiple GPUs produced CUDA initialization failures on this machine. Sequential GPU inference works. This was not a model/data bug.
- Metric values from these runs are only smoke-test values because all training and inference commands used tiny batch limits.
