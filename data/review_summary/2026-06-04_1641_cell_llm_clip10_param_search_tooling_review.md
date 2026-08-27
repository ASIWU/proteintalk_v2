# 2026-06-04 16:41 HKT Cell LLM Clip10 Parameter Search Tooling Review

Reviewed the corrected Cell LLM clip10 experiment tooling and historical dose-search/report patterns before adding the new tuning runner.

Files reviewed:

- `scripts/run_cell_llm_clip10_selected_suite.sh`
- `scripts/run_dose_param_search.sh`
- `scripts/dose_param_search_report.py`
- `scripts/ptv3_experiment_common.sh`
- `scripts/report_ptv3_exp_results.py`
- `scripts/report_cell_drug_time_eval.py`
- `scripts/exp_01_single_pert_stratified_5fold.sh`
- `scripts/exp_02_single_cell_type_5fold.sh`
- `scripts/exp_03_single_cell_5fold.sh`
- `scripts/exp_04_single_no_mse_5fold.sh`
- `scripts/exp_05_single_no_pdi_5fold.sh`
- `scripts/exp_06_double_pert_pair_5fold.sh`
- `scripts/exp_07_extra_single_all_train_infer.sh`
- `scripts/exp_08_extra_double_all_train_infer.sh`
- corrected `20260604_cell_llm_dose_clip10_selected_v1` manifests for Cell LLM and graph-mode checks

Findings:

- The selected-suite runner already had the corrected Cell embedding validation and `CELL_LLM_*` environment usage.
- The older dose parameter-search report used an outdated stage1 score and ranked task-specific stages by n-AUPRC, so a separate corrected report was safer than changing historical tooling.
- The exp07/exp08 wrappers already support the required reference-epoch policy through `REFERENCE_EPOCH_AGG=mean`, nearest rounding, and `last.ckpt`.
- Existing corrected manifests confirm `cell_llm_mode=frozen`, `cell_llm_summary.embedding_rows=74`, and exp05 `graph_feature_mode=zero`.

Changes made:

- Added `scripts/run_cell_llm_clip10_param_search.sh`.
- Added `scripts/report_cell_llm_clip10_param_search.py`.
- Updated `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`.
- Updated `docs/2026-04-15_data_standardization_session_summary.md`.

Validation:

- `python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_llm_clip10_param_search.py`
- `bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh`
- `bash scripts/run_cell_llm_clip10_param_search.sh --help`
- Report smoke test against `20260602_llm_dose_param_v1` stage4 artifacts completed and marked old cell-type LLM manifests invalid.

No long tuning or final selected training jobs were launched in this review/update.
