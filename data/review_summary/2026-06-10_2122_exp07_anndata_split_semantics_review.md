# 2026-06-10 21:22 Exp07 AnnData Split Semantics Review

- Reviewed `utils/09_build_data_splits.py`, `utils/12_export_single_drug_transcriptome_anndata.py`, `train.py`, and PTV3 split manifests to clarify exp07 AnnData labels.
- `ptv3_main_singledrug/all_train_subset_test` intentionally stores all valid main single-drug anchors in train while drawing internal valid/test monitoring subsets from the same anchors.
- Source pkl counts confirm main `all_train_subset_test`: train `17986`, valid `1798`, test `3597`, with valid/test rows also present in train and valid/test mutually disjoint.
- The AnnData exporter preserves exact overlapping source memberships by writing labels such as `train+valid` and `train+test` in `split_exp07_all_single_for_extra`.
- The actual exp07 external evaluation data are the six `ptv3_extra_singledrug_*` tasks, exported separately with `extra_single_test_only` / source `test_only`; their train and valid splits are empty.
- `train.py` guards `all_train_subset_test` by requiring `--skip-test`, so final claims are expected to use `infer.py` on the external extra-data tasks.
- Conclusion: `train+test` in the main h5ad is a faithful but potentially confusing representation of the internal all-train monitoring split, not evidence that extra exp07 test data were included in training.
