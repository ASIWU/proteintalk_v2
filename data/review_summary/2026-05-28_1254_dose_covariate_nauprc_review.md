# 2026-05-28 12:54 HKT Dose Covariate and nAUPRC Review

## Scope

- Reviewed the dose covariate changes in `train.py`, `infer.py`, and shared experiment scripts.
- Reviewed result reporting paths for exp01-exp08 and added nAUPRC to the default report surface.
- Confirmed no data processing or split generation code was changed for this dose experiment.

## Findings

- Dose is default-off. Existing no-dose runs keep the original covariate list: `machineID_new`, `Cell_plate`, `Cell`, `cell_type`, `batch`, `pert_time`.
- Enabling `USE_DOSE_COVARIATE=1` adds `pert_dose1` and `pert_dose2` to both train and infer covariates.
- Dose-enabled run manifests for exp01, exp06, exp07, and exp08 all record `use_dose_covariate=true` and include both dose covariates in `batch_cov_list`.
- `git diff` is empty for `utils/00_standardize_rawdata.py`, `utils/02_build_training_ready_data.py`, and `utils/09_build_data_splits.py`.

## Validation

- `python -m py_compile train.py infer.py scripts/report_ptv3_exp_results.py scripts/model_size_sweep_report.py scripts/covariate_analysis_report.py`
- `bash -n scripts/ptv3_experiment_common.sh scripts/exp_01_single_pert_stratified_5fold.sh scripts/exp_08_extra_double_all_train_infer.sh`
- Full dose run completed with prefix `20260528_dose_h512_lr2e4_v1`.
- `scripts/report_ptv3_exp_results.py --prefix 20260528_dose_h512_lr2e4_v1 --format markdown` produced exp01-exp08 metrics with nAUPRC by default.
