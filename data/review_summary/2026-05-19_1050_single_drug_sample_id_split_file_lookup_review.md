# 2026-05-19 10:50 HKT Single-Drug Sample ID Split File Lookup Review

## Scope

- Re-read `docs/Data_Process_1.md` through `docs/Data_Process_4.md`.
- Located the prior export for sharing PTV3 main single-drug three 5-fold split definitions as raw `sample_id` lists.
- Checked JSON structure and count TSV.

## Located Output

- Split JSON:
  `data/training_ready/ptv3/splits/ptv3_main_singledrug/ptv3_main_singledrug_three_5fold_sample_id_splits.json`
- Count TSV:
  `data/training_ready/ptv3/splits/ptv3_main_singledrug/ptv3_main_singledrug_three_5fold_sample_id_split_counts.tsv`

## JSON Structure

- `dataset_group`: `ptv3`
- `task_name`: `ptv3_main_singledrug`
- `strategies`:
  - `pert_stratified_5fold`
  - `cell_type_5fold`
  - `cell_5fold`
- Each strategy has `fold0` through `fold4`.
- Each fold has `train`, `valid`, and `test` lists containing non-control perturbation anchor `sample_id` values from the original single-drug info table.

## Count Summary

- `pert_stratified_5fold` fold0: train `12297`, valid `2083`, test `3606`.
- `cell_type_5fold` fold0: train `9059`, valid `1177`, test `7750`.
- `cell_5fold` fold0: train `12330`, valid `4287`, test `1369`.
- The count TSV contains all 45 strategy/fold/split count rows and reports `missing_from_raw_info_count=0` plus `duplicate_sample_id_count=0` for every row.
