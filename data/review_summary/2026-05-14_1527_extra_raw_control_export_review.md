# 2026-05-14 15:27 HKT Extra Raw Control Export Review

## Scope

- Read `docs/Data_Process_1.md` through `docs/Data_Process_4.md` to confirm the PTV3 standardization, extra-data control matching, stage-2 filtering, and training-ready contracts.
- Reviewed `utils/00_standardize_rawdata.py` for generated extra sample ids, `source_row_index`, control-pool matching, and extra single/double task handling.
- Reviewed `utils/02_build_training_ready_data.py` for processed-row filtering and appended matched-control handling.

## Findings

- PTV3 extra single/double raw files do not have native `sample_id`; stage 1 generates deterministic ids from task name and raw row index.
- Stage 1 records `source_file_info` and `source_row_index`, which is the stable join key back to each raw CSV.
- Stage 2 filters extra single rows by non-empty `PRISM2nd_label_total` and extra double rows by non-empty `PRISM1st_label_total`.
- The requested exported `control` values should therefore come from stage-2 processed self rows, not directly from stage-1 info, so rows removed by prior filtering keep an empty `control`.

## Outcome

- Implemented `utils/10_export_extra_raw_with_controls.py`.
- Exported annotated raw CSVs and summary mapping files under `data/standardized/ptv3/extra_raw_with_controls/`.
- Verified all 9 PTV3 extra single/double raw files align to stage-1 sample ids and stage-2 processed self rows without row-index mismatches.
