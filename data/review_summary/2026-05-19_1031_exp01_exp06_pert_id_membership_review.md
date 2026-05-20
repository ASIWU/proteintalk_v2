# 2026-05-19 10:31 HKT Exp01 / Exp06 Pert ID Membership Review

## Scope

- Reviewed `scripts/exp_01_single_pert_stratified_5fold.sh`.
- Reviewed `scripts/exp_06_double_pert_pair_5fold.sh`.
- Checked `L9200_2760`, `L9200_20`, `L9200_1678`, and `L9200_1047` against:
  - raw main single-drug CSV
  - standardized main single-drug info
  - standardized main double-drug info
  - training-ready single-drug `feature_table.parquet`
  - training-ready double-drug `feature_table.parquet`
  - exp01 `pert_stratified_5fold_fold0..4` train/valid/test indices
  - exp06 `pert_id_5fold_fold0..4` train/valid/test indices

## Script Targets

- `exp_01_single_pert_stratified_5fold.sh` trains `ptv3_main_singledrug` with `pert_stratified_5fold_fold${fold}` and task head `response`.
- `exp_06_double_pert_pair_5fold.sh` trains `ptv3_main_doubledrug` with `pert_id_5fold_fold${fold}` and task head `synergy`.

## Findings

- `L9200_2760`
  - Raw main single rows: 6.
  - Raw and standardized `PRISM1st_label_total`: `non-responsive`.
  - Final single-drug feature rows: 6.
  - Exp01 membership: fold0 train 6, fold1 train 6, fold2 train 6, fold3 test 6, fold4 train 6.
  - Native double-drug rows: 0.
  - Final double-drug feature rows: 6, all with `feature_membership="merged_single_drug"`.
  - Exp06 membership: train 6 in every fold.

- `L9200_20`
  - Raw main single rows: 6.
  - Raw and standardized `PRISM1st_label_total`: empty.
  - Final single-drug feature rows: 0.
  - Exp01 membership: absent from all folds.
  - Native double-drug rows: 0.
  - Final double-drug feature rows: 0.
  - Exp06 membership: absent from all folds.
  - Reason: removed by Stage-2 single-drug non-control label filter.

- `L9200_1678`
  - Raw main single rows: 6.
  - Raw and standardized `PRISM1st_label_total`: empty.
  - Final single-drug feature rows: 0.
  - Exp01 membership: absent from all folds.
  - Native double-drug rows: 0.
  - Final double-drug feature rows: 0.
  - Exp06 membership: absent from all folds.
  - Reason: removed by Stage-2 single-drug non-control label filter.

- `L9200_1047`
  - Raw main single rows: 6.
  - Raw and standardized `PRISM1st_label_total`: empty.
  - Final single-drug feature rows: 0.
  - Exp01 membership: absent from all folds.
  - Native double-drug rows: 0.
  - Final double-drug feature rows: 0.
  - Exp06 membership: absent from all folds.
  - Reason: removed by Stage-2 single-drug non-control label filter.

## Conclusion

For the two requested training scripts, the user is correct for `L9200_2760`: it is in the final training-ready feature tables and participates in exp01/exp06, except exp01 fold3 where it is held out in test. The other three perturbation IDs are absent from both requested training workflows because their main single-drug `PRISM1st_label_total` is empty and they have no native double-drug rows to enter exp06 independently.
