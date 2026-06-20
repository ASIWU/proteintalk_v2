# 2026-06-11 12:17 HKT Exp07 Transcriptome AnnData Detailed Report Review

- Reviewed `docs/2026-06-10_exp07_transcriptome_anndata_explanation.md`, `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`, `data/transcriptome_anndata/ptv3_single_nonablation/README.md`, and `data/transcriptome_anndata/ptv3_single_nonablation/summary.json`.
- Checked the exp07 execution path in `scripts/exp_07_extra_single_all_train_infer.sh`, shared reference-epoch settings in `scripts/ptv3_experiment_common.sh`, and the selected exp07 run manifest under `checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp07_extra_single_all_train_infer_all_single_for_extra/`.
- Found complete 2026-06-10 exp07 artifacts: `last.ckpt`, reference-epoch summary JSON, six extra single-drug prediction parquet files, and the final cell-drug-dose evaluation CSV/JSON.
- Expanded the exp07 AnnData explanation into a detailed report with export provenance, split semantics, fixed reference epoch policy, final exp07 metrics, subset tables, grouped metrics, deltas versus prior selected runs, inference outputs, and code references.
- No training or inference rerun was needed because existing result artifacts were complete.
