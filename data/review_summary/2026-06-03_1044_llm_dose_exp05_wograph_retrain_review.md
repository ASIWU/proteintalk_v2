# 2026-06-03 10:44 HKT LLM Dose exp05 W/o Graph Retrain Review

Reviewed the fresh exp05 w/o graph retrain requested after the initial supplemental result was based on an existing run.

Evidence checked:

- Training prefix: `20260603_llm_dose_exp01params_wograph_v1`
- Checkpoints: `checkpoints/20260603_llm_dose_exp01params_wograph_v1_exp05_single_no_graph_5fold_single_no_graph_fold*/run_manifest.json`
- Report markdown: `logs/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.md`
- Report CSV: `outputs/2026-06/2026-06-03/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.csv`
- Report JSON: `outputs/2026-06/2026-06-03/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.json`
- Runtime summary: `logs/20260603_llm_dose_exp01params_wograph_v1_runtime_summary.tsv`

Findings:

- The fresh run completed all 5 exp05 folds.
- All 5 manifests use `graph_feature_mode=zero`, `cell_type_llm_mode=frozen`, and `use_dose_covariate=True`.
- The fresh exp05 hyperparameters match current exp01 `mse050`: lr 2e-4, batch size 256, dropout 0.15, weight decay 1e-4, MSE weight 0.50, hidden dim 512.
- Fresh report CSV mean5 original metrics are AUROC 0.848554, AUPRC 0.592835, and n-AUPRC 4.992117.
- The selected-results documentation now references the fresh retrain prefix instead of the earlier existing exp05 source.

No code changes were made.
