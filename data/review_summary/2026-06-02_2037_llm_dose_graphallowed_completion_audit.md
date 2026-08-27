# 2026-06-02 20:37 HKT LLM Dose Graph-Allowed Completion Audit

Reviewed the current experiment artifacts for the LLM cell embedding + dose tuning objective after the user constraint that no-graph is not allowed. This audit was later supplemented with a fresh w/o graph retrain section on 2026-06-03.

Evidence checked:

- Recomputed stage1 graph-only ranking from `checkpoints/20260601_llm_celltype_exp01_v1_stage1_*_exp01_*` and `*_exp04_*` manifests.
- Recomputed stage2/stage3/stage4 rankings from `checkpoints/20260602_llm_dose_param_v1_stage{2,3,4}_*` manifests.
- Checked final selected suite report files under prefix `20260602_llm_dose_graphallowed_selected_v1`.
- Checked final training manifests for experiments exp01, exp02, exp03, exp04, exp06, exp07, and exp08.
- Searched final-prefix checkpoints/logs/outputs for no-graph indicators and exp05 entries.
- Later retrained and checked supplemental exp05 w/o graph manifests and report under prefix `20260603_llm_dose_exp01params_wograph_v1`.

Findings:

- Stage1 graph-only ranking selects `mse050`/`mse050_warmdecay` tie; `mse050` is accepted as the simpler equivalent schedule.
- Stage2 exp03 selects `covdrop010` by highest screen AUPRC.
- Stage3 exp02 selects `covdrop010` by highest screen AUPRC.
- Stage4 exp06 selects `drop020` by highest screen AUPRC.
- Final selected suite completed successfully and wrote markdown, CSV, and JSON reports.
- Final selected manifests all use `graph_feature_mode=real`, `cell_type_llm_mode=frozen`, and `use_dose_covariate=True`.
- Final selected prefix contains no exp05/no-graph experiment.
- Fresh supplemental exp05 w/o graph source has 5 folds with `graph_feature_mode=zero`, `cell_type_llm_mode=frozen`, `use_dose_covariate=True`, and the same `mse050` parameters as current exp01/exp04.
- Supplemental exp05 report CSV mean5 original metrics are AUROC 0.848554, AUPRC 0.592835, and n-AUPRC 4.992117.

No code changes were made during this audit, the fresh retrain, or the supplemental documentation update.
