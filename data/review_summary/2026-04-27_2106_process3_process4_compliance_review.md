# Process 3 / Process 4 Compliance Review

Review time: 2026-04-27 21:06 HKT

Reviewed requirement docs:

- `docs/Data_Process_3.md`
- `docs/Data_Process_4.md`

Reviewed implementation files:

- `utils/09_build_data_splits.py`
- `dataset/training_ready_dataset.py`
- `model/training_ready_models.py`
- `model/training_ready_lightning.py`
- `train.py`
- `infer.py`
- `docs/data_process_summary_03.md`
- `docs/data_process_summary_04.md`

## Findings

1. Process 3 is mostly implemented, but label coverage is only audited, not enforced as a hard error.
   - `docs/Data_Process_3.md` says all non-control data need labels.
   - Current manifests show no missing values for the checked label columns on valid anchors.
   - Current code would still write splits if future data had missing labels.

2. Process 3 skips anchors that cannot resolve to a control row.
   - `ptv1_aivc`: 14060 candidate anchors, 13137 valid anchors, 923 skipped missing-control anchors.
   - `ptv3_main_singledrug`: 18144 candidate anchors, 17986 valid anchors, 158 skipped missing-control anchors.
   - This is necessary for the current dataset class, but it may conflict with a strict interpretation of "all data" in Step 3.

3. Process 3 double-drug `pert_id` 5-fold split uses an assumption.
   - Current implementation uses ordered `pert_id1 + pert_id2` pair holdout.
   - The docs do not specify whether double-drug holdout should be ordered pair, unordered pair, first-drug, second-drug, or any-drug holdout.

4. Process 3 PTV1 fixed split relies on training-ready `data_split`.
   - The docs reference `rawdata/ptv1/experiment_type_list`.
   - The current code assumes earlier processing already converted that source into `data_split`.

5. Process 3 PTV3 extra single-drug tasks are test-only by assumption.
   - Step 3 explicitly names extra_guomics/nc/nature.
   - Step 4 names extra_singledrug inference targets, so extra singledrug was also implemented as `test_only`.

6. Process 4 satisfies the runnable training/inference contract, but not exact legacy architecture parity.
   - The selected six model names are present.
   - All models accept double-drug input.
   - Graph code uses PDI only.
   - The implementation is compact and compatible, not a verified line-by-line port of legacy model internals.
   - `baseline_emb_v3` is name-compatible but not verified against historical baseline internals.

7. Process 4 training runs `trainer.test(...)` after fit.
   - This is operationally useful, but can be slow for smoke checks.
   - A future `--skip-test` or `--limit-test-batches` option would improve usability.

## Satisfied Requirements

- Step 3 train/valid/test splits are generated.
- Step 3 supports `random`, `pert_stratified`, `cell`, and `cell_type` for PTV3 main single-drug.
- Step 3 includes valid sets.
- Step 3 supports 5-fold and `all_train_subset_test`.
- Step 3 double-drug only has `pert_id` family 5-fold splits.
- Step 3 extra guomics/nc/nature are test-only.
- Step 3 PTV1 main has fixed split and `pert_id` 5-fold.
- Step 3 PTV1 extra single-drug is test-only.
- Step 4 has `train.py` and `infer.py`.
- Step 4 removed `embedding_methods` and `inverse_machine_id` from the new CLI.
- Step 4 keeps `batch_cov_list` and does not default to `pert_dose`.
- Step 4 keeps only the requested six model names.
- Step 4 graph logic uses PDI only.
- Step 4 uses both `pert_id1` and `pert_id2`, with a shared single/double drug code path.
- Step 4 inference supports extra single/double drug style tasks.

## Verification Rechecked

- Split manifests were inspected for all current PTV1/PTV3 tasks.
- Label coverage entries in current manifests are all `ok` for checked valid anchors.
- Missing-control skipped anchor counts were identified and recorded in `docs/data_process_summary_03.md`.
- Previous smoke checks remain valid: py_compile, CLI help, model dry-runs, one-batch training, and extra double-drug inference.

## Review Conclusion

The code satisfies the first goal of running end-to-end and covers the major requested workflow surfaces. It does not fully satisfy the strictest interpretation of the docs in three areas: hard label enforcement, full candidate-anchor coverage when controls are missing, and exact legacy model architecture reuse. These are now explicitly documented for user decision.
