# 2026-05-27 20:33 Dose Covariate and update_0527 Data Review

## Scope

- Read `docs/Data_Process_1.md` through `docs/Data_Process_4.md`.
- Checked main PTV3 single-drug and double-drug raw info, standardized outputs, and training-ready outputs for dose support.
- Compared `data/rawdata/update_0527` extra test files against existing extra single-drug files and `update_0526` extra double-drug files.

## Findings

1. Main PTV3 single-drug supports one raw dose value.
   - Raw info has `pert_dose` and `pert_dose_unit`.
   - Stage 1 maps `pert_dose -> pert_dose1`; `pert_dose2` is missing.
   - Training-ready writes `pert_dose1_index` and `pert_dose2_index`; current single rows have `pert_dose2_index` mapped to the shared `no` dose bucket.

2. Main PTV3 double-drug supports two raw dose values.
   - Raw info has `pert_dose1` and `pert_dose2`.
   - Stage 1 and training-ready preserve both fields and build indexed columns.
   - In the main double feature table, native double rows have both dose slots; merged main single auxiliary rows only have the first dose slot and `pert_dose2` is missing.

3. The training/dataset stack can consume dose as categorical covariates, but dose is not enabled by default.
   - `dataset/training_ready_fast_dataset.py` maps `pert_dose1` and `pert_dose2` to their indexed columns.
   - `train.py`/`infer.py` category sizing uses the shared `pert_dose` value map for both fields.
   - Default `--batch-cov-list` remains `machineID_new Cell_plate Cell cell_type batch pert_time`, so dose requires an explicit covariate list and retraining/inference with matching checkpoint config.

4. `update_0527` is not wired into the current standardization code.
   - Extra single-drug code still reads `data/rawdata/extra_singledrug`.
   - Extra double-drug code still reads `data/rawdata/update_0526/extra_doubledrug` via `EXTRA_DOUBLE_UPDATE_ROOT`.

5. `update_0527` extra files do not match the old files by only adding dose.
   - Extra single-drug files add `logfold_change` and add rows; `pert_dose` exists but is entirely missing in both old and new files.
   - Extra double-drug files keep `pert_dose`, but it is also entirely missing.
   - New double-drug files add IC50/Emax-related columns and remove the old `test` column.
   - Guomics raw `pert_id1` changed from values such as `116A1` to `High116A1`, and the row set does not match the old Guomics file even on minimal `(Cell, pert_id1, pert_id2)` identity.

## update_0527 Comparison Summary

- Extra single:
  - Old rows are subsets of the new files on common columns.
  - New rows added: mat1 4407, mat2 3447, mat3 2033, mat4 2196.
  - New-only column: `logfold_change`.
  - `pert_dose` non-null count: 0 in all old and new files.

- Extra double:
  - NC: same row count and same common identity rows, but old-only columns `anchor_lib/group/group1/test` are absent and new IC50/Emax columns are added.
  - Nature: old rows are present, but 45,755 rows are added and `test` is absent.
  - Guomics: row count changes from 9,001 to 16,237 and old/new rows do not match on core identity because raw IDs changed.
  - `pert_dose` non-null count: 0 in all checked new double files.

## Risks

- If `update_0527` double-drug files are substituted directly, current split and inference code will not apply the existing `test == 1 and test_label != delete` evaluation filter because the `test` column is missing.
- Current dose covariate support is usable for main PTV3 single/double data, but extra test `pert_dose` values do not currently provide usable dose signal.
