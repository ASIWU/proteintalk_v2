# 2026-06-03 12:12 HKT update_0527 Extra Dose Match Review

## Scope

- Reviewed the user-referenced extra single/double update under `data/rawdata/update_0527`.
- Compared the new files to the currently wired raw inputs:
  - extra single: `data/rawdata/extra_singledrug`
  - extra double: `data/rawdata/update_0526/extra_doubledrug`
- Checked the current standardization and dose-covariate contracts in `utils/00_standardize_rawdata.py`, `utils/02_build_training_ready_data.py`, `train.py`, and `dataset/training_ready_fast_dataset.py`.

## Findings

- The literal path `data/raw_data/updata_0527` does not exist in this checkout. The present update directory is `data/rawdata/update_0527`.
- Current standardization code is not wired to the new files:
  - extra single still reads `data/rawdata/extra_singledrug` with `20260413..._dup.csv` filenames.
  - extra double still uses `EXTRA_DOUBLE_UPDATE_ROOT = data/rawdata/update_0526/extra_doubledrug` and `260525...test_label.csv` filenames.
- Extra single update is a superset, not a full one-to-one match:
  - Old rows are present when excluding the newly filled `pert_dose` and new `logfold_change`.
  - New row deltas: mat1 `+4407`, mat2 `+3447`, mat3 `+2033`, mat4 `+2196`.
  - New `pert_dose` is fully non-empty and constant `10` in all six extra single files.
  - `PRISM2nd_label_total` non-empty counts also increased, so the stage-2 extra single feature rows would change.
- Extra double common non-dose identity is preserved, but the schema and row cardinality changed:
  - New files use `pert_dose1` and `pert_dose2`; the current parser expects a single `pert_dose` and copies it to both dose slots.
  - NC row count and common non-dose rows match exactly.
  - Guomics and Nature have expanded row counts (`+7236` and `+45755`), mostly same common cell-drug-pair rows repeated with different dose values.
  - New test/test_label columns are present in all three new double files.
- Dose is not just a small categorical update for extra double:
  - `pert_dose1` is low-cardinality.
  - `pert_dose2` is high-cardinality continuous and includes extreme values:
    - Guomics max `5.03e+304`, `15542` raw unique values.
    - NC max `1.02e+300`, `16244` raw unique values.
    - Nature max `6104145698.55086`, `68922` raw unique values.
  - The current stage-2 rule maps numeric dose to `ceil(dose)` categorical indices. Applying that directly to new `pert_dose2` would create impractically large category indices and can break dose-covariate training/inference.

## Conclusion

The new `update_0527` data does not fully match the currently used data as a direct replacement. Extra single is close and likely needs only a file-path/filename update plus normal `pert_dose -> pert_dose1` handling, but it still changes the evaluated row set. Extra double needs parser changes for `pert_dose1`/`pert_dose2`, and the `pert_dose2` values require a separate validation/binning/normalization decision before enabling dose covariates.
