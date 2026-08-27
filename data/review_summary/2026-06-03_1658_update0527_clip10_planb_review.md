# 2026-06-03 16:58 HKT update_0527 Clip10 Plan B Review

Reviewed and implemented Plan B for update_0527 extra data ingestion and dose clipping.

## Scope

- `utils/00_standardize_rawdata.py`
- `utils/02_build_training_ready_data.py`
- `scripts/report_extra_doubledrug_test_label_auprc.py`
- `scripts/report_cell_drug_time_eval.py`
- Rebuilt standardized/training-ready data, splits, and drug-side derived artifacts.

## Findings And Decisions

- The update_0527 raw files exist under both `extra_singledrug` and `extra_doubledrug`.
- Extra double update_0527 files contain independent `pert_dose1` and `pert_dose2`, so standardization should not copy one raw dose across both slots except for legacy fallback files.
- Numeric dose clipping belongs in stage 1 so all downstream standardized and training-ready artifacts share the same dose semantics.
- Stage 2 should still validate dose range and fail fast on negative or unclipped numeric dose.
- After rebuild, `protein_index_to_id` was unchanged and `pert_index_to_id` changed. Therefore only Morgan drug embedding, DDI, and PDI were regenerated; full protein/PPI/GPU feature regeneration was not needed.

## Validation Summary

- Static compile passed for touched pipeline/report files plus `train.py` and `infer.py`.
- Stage-1 standardized validation passed.
- Stage-2 training-ready validation passed.
- Split rebuild passed for `--dataset-group all`.
- Acceptance checks:
  - `value_to_index["pert_dose"]` max numeric index is `10`;
  - `value_to_index["pert_dose"]["no"]` is `"11"`;
  - standardized/training-ready numeric dose max is `10.0`;
  - extra single/double source paths point at update_0527;
  - extra double `test` and `test_label` columns are present;
  - extra double standardized dose slots match independently clipped raw `pert_dose1` and `pert_dose2`.
- Drug-side artifacts were regenerated and shape-checked:
  - drug embedding `(6131, 2048)`;
  - DDI `(6131, 6131)`;
  - PDI `(6131, 11345)`.
- One-batch single and double training smokes with dose covariates passed using a finite checkpoint monitor and disabled W&B logging.
- One-batch extra single and extra double inference smokes passed.
- Extra double test-label report passed on partial smoke predictions against update_0527 metadata.
- Cell-drug-dose grouping check confirmed the report path uses clipped training-ready dose tokens.
- `bash -n` passed for exp07, exp08, full-suite, and common experiment launcher scripts.
