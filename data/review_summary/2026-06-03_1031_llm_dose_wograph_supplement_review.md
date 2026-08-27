# 2026-06-03 10:31 HKT LLM Dose W/o Graph Supplement Review

Reviewed no-graph artifacts to supplement the LLM cell embedding + dose selected-results documentation. This note was updated after a fresh exp05 retrain with the current exp01 parameters completed under `20260603_llm_dose_exp01params_wograph_v1`.

Evidence checked:

- `checkpoints/20260603_llm_dose_exp01params_wograph_v1_exp05_single_no_graph_5fold_single_no_graph_fold*/run_manifest.json`
- `outputs/2026-06/2026-06-03/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.csv`
- `logs/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.md`
- `logs/20260603_llm_dose_exp01params_wograph_v1_runtime_summary.tsv`
- `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`

Findings:

- The fresh comparable w/o graph source is `20260603_llm_dose_exp01params_wograph_v1_exp05_single_no_graph_5fold`.
- All 5 exp05 manifests use `graph_feature_mode=zero`, `cell_type_llm_mode=frozen`, and `use_dose_covariate=True`.
- The exp05 hyperparameters match the selected exp01/exp04 `mse050` setup: lr 2e-4, batch size 256, dropout 0.15, weight decay 1e-4, MSE weight 0.50, hidden dim 512.
- Report CSV mean5 original exp05 metrics are AUROC 0.848554, AUPRC 0.592835, and n-AUPRC 4.992117.
- Documentation was updated to include exp05 as a fresh supplemental w/o graph retrain separate from the primary graph-enabled selected prefix.

No code changes were made.
