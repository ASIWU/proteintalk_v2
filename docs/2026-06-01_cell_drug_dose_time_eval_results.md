# 2026-06-01 Cell-Drug-Dose Time-Collapsed Evaluation Results

Run prefix: `20260601_cell_drug_selected_full_v1`

This report reuses the trained checkpoints and prediction files from the selected-parameter full run. No retraining was needed for this update.

Primary outputs:

- `outputs/2026-06/2026-06-01/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.csv`
- `outputs/2026-06/2026-06-01/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.json`
- `logs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.md`

Evaluation modes:

- `original`: original row/time-point-level evaluation.
- `cell-drug-dose-bylasttime`: one datapoint per cell-drug-dose or cell-drug-dose-combo, using the maximum numeric `pert_time`; ties at the last time are averaged.
- `cell-drug-dose-avgtime`: one datapoint per cell-drug-dose or cell-drug-dose-combo, using the average prediction across all time rows.

Grouping key:

- single-drug tasks: `Cell + pert_id1 + pert_dose1_norm`.
- double-drug tasks: `Cell + sorted((pert_id1, pert_dose1_norm), (pert_id2, pert_dose2_norm))`.

The model uses dose as categorical covariate indices, not direct float values. `pert_dose1` and `pert_dose2` are normalized to text values and mapped through the shared `pert_dose` index; numeric values map to `ceil(dose)`, and missing values map to `no`.

## Fold Summary

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp01 | single unseen drug | mean5 | original | 0.898153 | 0.666963 | 5.632660 | 17986 | 2137 | 15849 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-bylasttime | 0.898841 | 0.670592 | 5.687552 | 9032 | 1070 | 7962 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-avgtime | 0.898938 | 0.668244 | 5.665939 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | original | 0.942208 | 0.818603 | 6.245977 | 17986 | 2137 | 15849 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-bylasttime | 0.941977 | 0.818496 | 6.261922 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-avgtime | 0.943112 | 0.821349 | 6.278684 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | original | 0.932452 | 0.777002 | 6.514967 | 17986 | 2137 | 15849 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-bylasttime | 0.931772 | 0.776499 | 6.521652 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-avgtime | 0.933672 | 0.781928 | 6.566714 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | original | 0.895315 | 0.651119 | 5.515379 | 17986 | 2137 | 15849 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-bylasttime | 0.895131 | 0.651950 | 5.545950 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-avgtime | 0.896577 | 0.655386 | 5.572692 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | original | 0.842961 | 0.599514 | 5.051623 | 17986 | 2137 | 15849 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-bylasttime | 0.842036 | 0.594336 | 5.025268 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-avgtime | 0.844718 | 0.601805 | 5.085172 | 9032 | 1070 | 7962 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | original | 0.810580 | 0.754331 | 1.872959 | 1791 | 723 | 1068 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-bylasttime | 0.808264 | 0.753058 | 1.867977 | 896 | 362 | 534 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-avgtime | 0.812681 | 0.757687 | 1.879982 | 896 | 362 | 534 | 0 | 0 |

## Extra Subset Summary

| exp | subset | group | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | original | 0.700611 | 0.458101 | 2.233243 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.700611 | 0.458101 | 2.233243 | 15834 | 3248 | 12586 | 0 | 15834 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-avgtime | 0.700611 | 0.458101 | 2.233243 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | original | 0.696125 | 0.428855 | 2.090666 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-bylasttime | 0.696125 | 0.428855 | 2.090666 | 15834 | 3248 | 12586 | 0 | 15834 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-avgtime | 0.696125 | 0.428855 | 2.090666 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | original | 0.806186 | 0.668934 | 3.050159 | 12138 | 2662 | 9476 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.806057 | 0.669379 | 3.049674 | 12087 | 2653 | 9434 | 9 | 12087 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-avgtime | 0.806057 | 0.669379 | 3.049674 | 12087 | 2653 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | original | 0.806313 | 0.646476 | 2.947756 | 12138 | 2662 | 9476 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-bylasttime | 0.806434 | 0.647092 | 2.948135 | 12087 | 2653 | 9434 | 9 | 12087 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-avgtime | 0.806434 | 0.647092 | 2.948135 | 12087 | 2653 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | original | 0.701418 | 0.450277 | 2.133153 | 17609 | 3717 | 13892 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-bylasttime | 0.701418 | 0.450277 | 2.133153 | 17609 | 3717 | 13892 | 0 | 17609 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-avgtime | 0.701418 | 0.450277 | 2.133153 | 17609 | 3717 | 13892 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | original | 0.799838 | 0.631861 | 3.041722 | 11072 | 2300 | 8772 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-bylasttime | 0.800319 | 0.632922 | 3.046594 | 11023 | 2290 | 8733 | 10 | 11023 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-avgtime | 0.800319 | 0.632922 | 3.046594 | 11023 | 2290 | 8733 | 10 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | original | 0.977273 | 0.500000 | 11.500000 | 23 | 1 | 22 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.977273 | 0.500000 | 11.500000 | 23 | 1 | 22 | 0 | 23 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.977273 | 0.500000 | 11.500000 | 23 | 1 | 22 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | original | 0.684323 | 0.088628 | 2.706228 | 3084 | 101 | 2983 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.684323 | 0.088628 | 2.706228 | 3084 | 101 | 2983 | 0 | 3084 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.684323 | 0.088628 | 2.706228 | 3084 | 101 | 2983 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | original | 0.686947 | 0.093622 | 2.851785 | 3107 | 102 | 3005 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-bylasttime | 0.686947 | 0.093622 | 2.851785 | 3107 | 102 | 3005 | 0 | 3107 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-avgtime | 0.686947 | 0.093622 | 2.851785 | 3107 | 102 | 3005 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | original | 0.655933 | 0.102577 | 1.635419 | 5341 | 335 | 5006 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.585306 | 0.046255 | 1.350565 | 2949 | 101 | 2848 | 185 | 2949 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.585306 | 0.046255 | 1.350565 | 2949 | 101 | 2848 | 185 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | original | 0.631212 | 0.042283 | 1.548479 | 17615 | 481 | 17134 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.617803 | 0.028604 | 1.487155 | 12062 | 232 | 11830 | 214 | 12062 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.617803 | 0.028604 | 1.487155 | 12062 | 232 | 11830 | 214 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | original | 0.657212 | 0.062684 | 1.763445 | 22956 | 816 | 22140 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-bylasttime | 0.616641 | 0.032805 | 1.478791 | 15011 | 333 | 14678 | 399 | 15011 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-avgtime | 0.616641 | 0.032805 | 1.478791 | 15011 | 333 | 14678 | 399 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | original | 0.602401 | 0.126080 | 1.372208 | 2057 | 189 | 1868 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.622679 | 0.124191 | 1.446508 | 1817 | 156 | 1661 | 28 | 1817 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.622679 | 0.124191 | 1.446508 | 1817 | 156 | 1661 | 28 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | original | 0.522445 | 0.074763 | 1.066711 | 13098 | 918 | 12180 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.520111 | 0.072916 | 1.061229 | 12633 | 868 | 11765 | 47 | 12633 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.520111 | 0.072916 | 1.061229 | 12633 | 868 | 11765 | 47 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | original | 0.539314 | 0.083226 | 1.139381 | 15155 | 1107 | 14048 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-bylasttime | 0.538205 | 0.080619 | 1.137641 | 14450 | 1024 | 13426 | 75 | 14450 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-avgtime | 0.538205 | 0.080619 | 1.137641 | 14450 | 1024 | 13426 | 75 | 0 |

## Extra Mean

This table is only a convenience aggregate across the extra subsets; use `Extra Subset Summary` as the primary exp07/exp08 report.

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | extra single | mean_extra | original | 0.751748 | 0.547417 | 2.582783 | 84625 | 17837 | 66788 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.751827 | 0.547771 | 2.583577 | 84474 | 17809 | 66665 | 28 | 84474 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.751827 | 0.547771 | 2.583577 | 84474 | 17809 | 66665 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.627824 | 0.079844 | 1.918203 | 41218 | 2025 | 39193 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.613931 | 0.069015 | 1.822739 | 32568 | 1459 | 31109 | 474 | 32568 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.613931 | 0.069015 | 1.822739 | 32568 | 1459 | 31109 | 474 | 0 |

## Notes

- exp06 now has `0` collapsed label-conflict groups under the cell-drug-dose key.
- exp07/exp08 must be reported by subset. `mean_extra` is retained only as an aggregate convenience row.
- exp08 subset rows are further split into `unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`, and `combined`, following the `test_label` report contract from the 2026-05-26 review history.
- exp07/exp08 extra rows have missing `pert_time` and missing raw dose values; their normalized dose fields are `no`, so the extra collapsed metrics remain unchanged from the previous cell-drug-only report.
