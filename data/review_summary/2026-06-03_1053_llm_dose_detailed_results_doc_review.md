# 2026-06-03 10:53 HKT LLM Dose Detailed Results Doc Review

Reviewed and expanded `docs/2026-06-02_llm_dose_graphallowed_selected_results.md` to match the level of detail in `docs/2026-06-01_llm_celltype_embedding_experiment_results.md`.

Evidence checked:

- Reference document structure from `docs/2026-06-01_llm_celltype_embedding_experiment_results.md`
- Primary selected report CSV/JSON/markdown under prefix `20260602_llm_dose_graphallowed_selected_v1`
- Fresh exp05 w/o graph report CSV/JSON/markdown under prefix `20260603_llm_dose_exp01params_wograph_v1`
- Stage1, stage2, stage3, and stage4 tuning manifests
- exp07 and exp08 reference epoch summary JSON files
- Fresh exp05 w/o graph manifests

Documentation updates:

- Added setup and selected-parameter context.
- Added full tuning screen ranking tables for exp01/exp04, exp02, exp03, and exp06.
- Added exp07/exp08 reference epoch table.
- Added full fold mean table for all three evaluation methods.
- Added fresh exp05 w/o graph fold detail.
- Added gap analysis, extra mean results, extra subset results, delta vs the 20260601 LLM-only suite, readout, output files, and validation sections.

Verification:

- Confirmed selected and fresh w/o graph report files exist.
- Confirmed key documented metrics match the CSV reports.
- Confirmed fresh exp05 has 5 manifests with `graph_feature_mode=zero`, `cell_type_llm_mode=frozen`, and `use_dose_covariate=True`.

No code changes were made.
