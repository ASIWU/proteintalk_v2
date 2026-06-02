# 2026-05-28 23:09 HKT Dose Parameter Search Review

## Scope

- Reviewed the dose-enabled parameter-search changes and final selected experiment outputs.
- Confirmed the work stayed in training, inference, model, and reporting code plus new search/report helper scripts.
- Confirmed no data processing, training-ready construction, or split-generation code was modified for this search.

## Validation

- `python -m py_compile train.py infer.py scripts/dose_param_search_report.py scripts/report_ptv3_exp_results.py`
- `bash -n scripts/run_dose_param_search.sh scripts/ptv3_experiment_common.sh`
- Empty diff for:
  - `utils/00_standardize_rawdata.py`
  - `utils/02_build_training_ready_data.py`
  - `utils/09_build_data_splits.py`
  - `data/Data_Process_1.md`
  - `data/Data_Process_2.md`
  - `data/Data_Process_3.md`
  - `data/Data_Process_4.md`

## Final Selected Configs

| experiment | config | AUPRC | nAUPRC | AUROC |
|---|---|---:|---:|---:|
| exp01 single unseen pert_id | `mse075_drop010` | 0.660333 | 5.574654 | 0.898136 |
| exp04 w/o MSE | `mse075_drop010` | 0.648277 | 5.492080 | 0.895242 |
| exp05 w/o graph | `mse075_drop010` | 0.596229 | 5.023520 | 0.842970 |
| exp02 unseen cell type | `ctrl_drop020` | 0.817573 | 6.237695 | 0.942154 |
| exp03 unseen cell | `lr1e4` | 0.774338 | 6.493304 | 0.932460 |
| exp06 double unseen drug | `dbl_mse010` | 0.736157 | 1.825010 | 0.808599 |
| exp07 extra single mean | `mse050_lr1e4` | 0.547417 | 2.582783 | 0.751748 |
| exp08 extra double mean | `lr3e4` | 0.079844 | 1.918203 | 0.627824 |

## Notes

- Stage1 final 5-fold gaps were `+0.012056` AUPRC for w/o MSE and `+0.064104` AUPRC for w/o graph. The no-graph gap exceeded 5 points; the no-MSE gap did not.
- Stage3 required extra full-fold confirmation because several 3-fold candidates were unstable once folds 1 and 3 were added. Final selection is based on all compared stage3 candidates having `count=17986`.
- Extra single and extra double searches used the existing data and all-train extra inference scripts; no new extra data was introduced.
