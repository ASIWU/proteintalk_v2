# 2026-06-04 12:49 HKT Cell LLM Clip10 Full Suite Review

## Scope

- Ran the corrected Cell LLM clip10 selected suite with prefix `20260604_cell_llm_dose_clip10_selected_v1`.
- Covered exp01-exp06 5-fold training, exp07 all-data extra single-drug inference, and exp08 all-data extra double-drug inference.
- Reviewed generated reports, manifests, and corrected Cell LLM artifact alignment.

## Findings

- The full suite completed successfully.
- Corrected reference epochs were recomputed from corrected folds:
  - exp07 selected epoch `11` from exp01 fold epochs `11, 14, 9, 9, 11`;
  - exp08 selected epoch `6` from exp06 fold epochs `1, 2, 7, 17, 1`.
- Fold mean original metrics:
  - exp01 single unseen drug: AUROC/AUPRC/n-AUPRC `0.891681 / 0.661912 / 5.588333`;
  - exp02 single unseen cell type: `0.945987 / 0.811791 / 6.099584`;
  - exp03 single unseen cell: `0.930525 / 0.776145 / 6.475614`;
  - exp04 single w/o MSE: `0.895744 / 0.662743 / 5.591658`;
  - exp05 single w/o graph: `0.844887 / 0.608214 / 5.127502`;
  - exp06 double unseen drug pair: `0.834738 / 0.782355 / 1.941130`.
- Extra mean original metrics:
  - exp07 extra single: AUROC/AUPRC/n-AUPRC `0.770714 / 0.569953 / 2.720985`;
  - exp08 extra double: `0.639328 / 0.094090 / 2.134296`.
- Graph ablation remains meaningful under corrected Cell LLM:
  - exp01 original AUPRC `0.661912`;
  - exp05 original AUPRC `0.608214`.

## Artifacts

- Full Markdown report: `logs/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.md`
- CSV report: `outputs/2026-06/2026-06-04/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.csv`
- JSON report: `outputs/2026-06/2026-06-04/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.json`
- Runtime summary: `logs/20260604_cell_llm_dose_clip10_selected_v1_runtime_summary.tsv`
- Reference epoch summaries:
  - `logs/20260604_cell_llm_dose_clip10_selected_v1_exp07_extra_single_all_train_infer_all_single_for_extra_reference_epoch_summary.json`
  - `logs/20260604_cell_llm_dose_clip10_selected_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra_reference_epoch_summary.json`

## Validation

- 32 corrected manifests were found:
  - exp01-exp06: 5 folds each;
  - exp07/exp08: 1 all-data run each.
- All corrected manifests use `cell_llm_mode=frozen`.
- All corrected manifests report `cell_llm_summary.embedding_rows=74`.
- No corrected manifest contains old `cell_type_llm` naming.
- exp05 manifests use `graph_feature_mode=zero`; all other corrected manifests use `graph_feature_mode=real`.
- Corrected embedding artifact checks passed for shape `(74, 4096)`, row 0 zero vector, and finite nonzero rows.

## Residual Risk

- exp08 extra double AUPRC is low in absolute terms, especially on the Nature combined subset, despite n-AUPRC above baseline.
- The `cell_type_5fold` split name remains part of the dataset split strategy for exp02; this is a split label, not the removed LLM embedding interface.
- Manifests store the corrected embedding path as an absolute path, while the runner is configured with the relative path. Both point to the same artifact.
