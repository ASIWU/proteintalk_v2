# 2026-06-09 14:08 HKT Cell + Cell-type LLM Clip10 Tuned Completion Review

## Scope

- Monitored the `20260608_cell_celltype_llm_clip10_tune_v1` tuning target through screen completion, full promoted 5-fold completion, and final selected exp01-exp08 completion.
- Verified final selected artifacts and wrote the formal report.

## Outputs

- Full report: `logs/20260608_cell_celltype_llm_clip10_tune_v1_full_param_search_report.md`
- Selected report: `logs/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.md`
- Selected CSV: `outputs/2026-06/2026-06-08/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.csv`
- Selected JSON: `outputs/2026-06/2026-06-08/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.json`
- Formal document: `docs/2026-06-09_cell_celltype_llm_clip10_tuned_results.md`

## Audit

- Active runner processes after completion: none observed.
- Final selected manifests audited: `32`.
- Audit errors: `0`.
- exp05 uses graph-zero as the no-graph diagnostic ablation.
- exp01/02/03/04/06/07/08 use graph-real.
- All final selected manifests include frozen Cell LLM and frozen cell-type LLM summaries with expected row counts and feature dimension.

## Selected Results

- exp01 mean5 AUPRC/AUROC: `0.669365 / 0.896935`.
- exp02 mean5 AUPRC/AUROC: `0.815855 / 0.945795`.
- exp03 mean5 AUPRC/AUROC: `0.784136 / 0.932252`.
- exp04 mean5 AUPRC/AUROC: `0.657075 / 0.906519`.
- exp05 mean5 AUPRC/AUROC: `0.606597 / 0.842547`.
- exp06 mean5 AUPRC/AUROC: `0.767741 / 0.825790`.
- exp07 extra mean AUPRC/AUROC: `0.560998 / 0.760442`.
- exp08 extra mean AUPRC/AUROC: `0.096357 / 0.650258`.
