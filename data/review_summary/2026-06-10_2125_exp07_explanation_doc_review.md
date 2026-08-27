# 2026-06-10 21:25 Exp07 Explanation Document Review

- Reviewed `scripts/exp_07_extra_single_all_train_infer.sh`, `scripts/ptv3_experiment_common.sh`, `scripts/select_reference_epoch.py`, and the AnnData export README to prepare a colleague-facing explanation of exp07.
- Confirmed exp07 trains `ptv3_main_singledrug` with `all_train_subset_test` and `--skip-test`.
- Confirmed reference-epoch mode selects epochs from exp01 perturbation-stratified 5-fold `best_model_path` values, aggregates them using configured `REFERENCE_EPOCH_AGG` and `REFERENCE_EPOCH_ROUNDING`, applies `max_epochs=selected_epoch+1`, and uses `last.ckpt` for extra-data inference.
- Confirmed the six external test AnnData files are the `ptv3_extra_singledrug_*` common-gene-axis h5ad files, each using `extra_single_test_only` split metadata.
- Added `docs/2026-06-10_exp07_transcriptome_anndata_explanation.md`.
