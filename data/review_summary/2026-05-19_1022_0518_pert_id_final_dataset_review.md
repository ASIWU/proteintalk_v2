# 2026-05-19 10:22 HKT 0518 Pert ID Final Dataset Review

## Scope

- Reviewed `scripts/0518_1.sh`.
- Reviewed `docs/Data_Process_1.md` through `docs/Data_Process_4.md`.
- Reviewed relevant prior records in `data/review_summary`, especially the double-drug single-merge and experiment-support reviews.
- Checked `data/standardized/ptv3/global_meta.json`, `data/training_ready/ptv3/global_meta.json`, Stage-1 info tables, Stage-2 `processed.csv` / `feature_table.csv`, and current split indices for:
  - `L9200_2760`
  - `L9200_20`
  - `L9200_1678`
  - `L9200_1047`

## Findings

- All four IDs are present in both standardized and training-ready PTV3 `global_meta.json` `pert_index`.
- All four IDs are present in the raw and standardized main single-drug table.
- `L9200_20`, `L9200_1678`, and `L9200_1047` have six main single-drug rows each, but all have empty `PRISM1st_label_total` in the raw and standardized tables.
- Stage 2 removes those three IDs from `ptv3_main_singledrug` because the documented and implemented rule is: non-control single-drug rows require non-empty `PRISM1st_label_total`.
- Because `ptv3_main_doubledrug` only merges the filtered `ptv3_main_singledrug` processed rows into its feature table, those three IDs never enter the double-drug training feature table or splits.
- `L9200_2760` has six main single-drug rows with non-empty `PRISM1st_label_total`, so it is retained in:
  - `data/training_ready/ptv3/tasks/ptv3_main_singledrug/processed.csv`
  - `data/training_ready/ptv3/tasks/ptv3_main_singledrug/feature_table.csv`
  - `data/training_ready/ptv3/tasks/ptv3_main_doubledrug/feature_table.csv`
- `L9200_2760` is absent from `ptv3_main_doubledrug/processed.csv` by design, because that file contains native double-drug rows only. The merged single-drug rows live in the double-drug `feature_table.csv` with `feature_membership="merged_single_drug"`.
- Current split indices place the six `L9200_2760` merged single-drug rows in train for every `ptv3_main_doubledrug/pert_id_5fold_fold*` split and in train for `all_train_subset_test`; they are not valid/test rows.
- `L9200_2760` appears in several extra single-drug Stage-1 tables, but those extra rows have empty `PRISM2nd_label_total`; Stage 2 removes them under the documented extra-single rule.

## Relevant Rules

- `docs/Data_Process_2.md`: single-drug non-control rows require non-empty `PRISM1st_label_total`; extra single-drug non-control rows require non-empty `PRISM2nd_label_total`.
- `docs/Data_Process_2.md`: `ptv3_main_doubledrug` feature table must include filtered `ptv3_main_singledrug` processed rows.
- `docs/Data_Process_3.md`: merged main single-drug rows are train-only auxiliary rows for double-drug splits.

## Conclusion

The three IDs `L9200_20`, `L9200_1678`, and `L9200_1047` are missing from final training feature tables because their main single-drug labels are empty and Stage 2 filters them out. `L9200_2760` is not missing from the training feature table or train splits; it is only absent from the native double-drug `processed.csv`, which is expected under the current data contract.
