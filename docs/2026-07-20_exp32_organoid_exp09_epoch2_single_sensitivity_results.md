# Exp32 Organoid Exp09 Single-drug Sensitivity Results

- Generated: `2026-07-20T13:03:55+00:00`
- Experiment prefix: `20260720_exp32_organoid_exp09_epoch2_single_sensitivity`
- Checkpoint: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`
- Query: 3,217 single drugs per sample, represented by identical drug slots.
- Exposure: 24 hours; both dose slots are 10.
- Score: `pred_sensitivity_prob`, taken only from the unified head's `pred_task_prob`.
- This experiment has no ground-truth response labels, so no label-based performance metrics are computed.

## Validation

- Both tasks completed with exactly 41,821 query predictions; combined rows: 83,642.
- All probabilities are finite and within `[0, 1]`.
- The two devices form exactly 41,821 one-to-one sample-drug pairs.
- Protein axis, drug scope, feature indices, covariates, hashes, and exp09 architecture contract passed reporter validation.

## Artifacts

- Combined CSV: `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_predictions.csv`
- Combined Parquet: `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_predictions.parquet`
- Per-sample top-20 CSV: `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_top20_by_sample.csv`
- Summary JSON: `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_summary.json`

## Device Probability Distributions

| device | n | mean | median | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 41821 | 0.068240 | 0.005048 | 0.132767 | 0.620466 | 0.000111 | 0.994713 |
| CAC | 41821 | 0.061047 | 0.003241 | 0.088380 | 0.567849 | 0.000081 | 0.996415 |

## Sample Probability Distributions

| device | sample | pat_ID | tissue | mean | median | P90 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | PTV2_1 | KOPC-P105 | Pancreas | 0.048448 | 0.003580 | 0.062268 | 0.334564 |
| B | PTV2_10 | KOLU-P1589 | Lung | 0.070186 | 0.005293 | 0.139615 | 0.638822 |
| B | PTV2_11 | KOLU-P1586 | Lung | 0.070306 | 0.005223 | 0.140299 | 0.641151 |
| B | PTV2_12 | KOLU-P1596 | Lung | 0.070015 | 0.005243 | 0.138883 | 0.637270 |
| B | PTV2_13 | KOPC-P102 | Pancreas | 0.049211 | 0.003572 | 0.063283 | 0.345513 |
| B | PTV2_2 | KOLU-P1583 | Lung | 0.070111 | 0.005255 | 0.139392 | 0.638106 |
| B | PTV2_3 | KOLU-P1585 | Lung | 0.071690 | 0.005430 | 0.147650 | 0.655764 |
| B | PTV2_4 | KOLU-P1587 | Lung | 0.070988 | 0.005274 | 0.143756 | 0.648960 |
| B | PTV2_5 | KOCO-P281 | Colon | 0.085469 | 0.007093 | 0.242199 | 0.785635 |
| B | PTV2_6 | KOLU-P1594 | Lung | 0.069692 | 0.005270 | 0.136817 | 0.632762 |
| B | PTV2_7 | KOLU-P1593 | Lung | 0.070978 | 0.005282 | 0.143659 | 0.648465 |
| B | PTV2_8 | KOLU-P1584 | Lung | 0.070584 | 0.005305 | 0.141815 | 0.643497 |
| B | PTV2_9 | KOLU-P1614 | Lung | 0.069441 | 0.005293 | 0.135878 | 0.629823 |
| CAC | PTV2_1 | KOPC-P105 | Pancreas | 0.039408 | 0.002445 | 0.036329 | 0.221222 |
| CAC | PTV2_10 | KOLU-P1589 | Lung | 0.064072 | 0.003393 | 0.099690 | 0.606307 |
| CAC | PTV2_11 | KOLU-P1586 | Lung | 0.063658 | 0.003333 | 0.096555 | 0.601170 |
| CAC | PTV2_12 | KOLU-P1596 | Lung | 0.064002 | 0.003327 | 0.097914 | 0.608401 |
| CAC | PTV2_13 | KOPC-P102 | Pancreas | 0.040022 | 0.002459 | 0.036708 | 0.230429 |
| CAC | PTV2_2 | KOLU-P1583 | Lung | 0.063459 | 0.003338 | 0.095799 | 0.595784 |
| CAC | PTV2_3 | KOLU-P1585 | Lung | 0.064343 | 0.003432 | 0.101539 | 0.609036 |
| CAC | PTV2_4 | KOLU-P1587 | Lung | 0.064691 | 0.003371 | 0.101859 | 0.620098 |
| CAC | PTV2_5 | KOCO-P281 | Colon | 0.076250 | 0.004091 | 0.171870 | 0.774790 |
| CAC | PTV2_6 | KOLU-P1594 | Lung | 0.063237 | 0.003439 | 0.096875 | 0.584731 |
| CAC | PTV2_7 | KOLU-P1593 | Lung | 0.064126 | 0.003329 | 0.098727 | 0.610106 |
| CAC | PTV2_8 | KOLU-P1584 | Lung | 0.063655 | 0.003414 | 0.098377 | 0.595056 |
| CAC | PTV2_9 | KOLU-P1614 | Lung | 0.062681 | 0.003365 | 0.092965 | 0.577079 |

## B versus CAC Agreement

Overall paired Pearson: `0.995239`; Spearman: `0.998963`; MAE: `0.008309`.

| sample | pat_ID | tissue | Pearson | Spearman | MAE | mean CAC-B |
| --- | --- | --- | --- | --- | --- | --- |
| PTV2_1 | KOPC-P105 | Pancreas | 0.991829 | 0.999019 | 0.009040 | -0.009040 |
| PTV2_10 | KOLU-P1589 | Lung | 0.996370 | 0.999352 | 0.007648 | -0.006114 |
| PTV2_11 | KOLU-P1586 | Lung | 0.995988 | 0.999247 | 0.007959 | -0.006648 |
| PTV2_12 | KOLU-P1596 | Lung | 0.996278 | 0.999334 | 0.007728 | -0.006013 |
| PTV2_13 | KOPC-P102 | Pancreas | 0.991776 | 0.998986 | 0.009189 | -0.009189 |
| PTV2_2 | KOLU-P1583 | Lung | 0.995961 | 0.999261 | 0.007996 | -0.006652 |
| PTV2_3 | KOLU-P1585 | Lung | 0.995686 | 0.999204 | 0.008414 | -0.007346 |
| PTV2_4 | KOLU-P1587 | Lung | 0.996302 | 0.999308 | 0.007753 | -0.006297 |
| PTV2_5 | KOCO-P281 | Colon | 0.995198 | 0.998909 | 0.010397 | -0.009218 |
| PTV2_6 | KOLU-P1594 | Lung | 0.996328 | 0.999364 | 0.007635 | -0.006455 |
| PTV2_7 | KOLU-P1593 | Lung | 0.995858 | 0.999225 | 0.008160 | -0.006852 |
| PTV2_8 | KOLU-P1584 | Lung | 0.995973 | 0.999278 | 0.008022 | -0.006929 |
| PTV2_9 | KOLU-P1614 | Lung | 0.995843 | 0.999280 | 0.008078 | -0.006760 |

## Top-drug Agreement

| sample | top-50 overlap | top-50 Jaccard | top-100 overlap | top-100 Jaccard |
| --- | --- | --- | --- | --- |
| PTV2_1 | 48 | 0.923077 | 100 | 1.000000 |
| PTV2_2 | 49 | 0.960784 | 98 | 0.960784 |
| PTV2_3 | 48 | 0.923077 | 98 | 0.960784 |
| PTV2_4 | 49 | 0.960784 | 98 | 0.960784 |
| PTV2_5 | 45 | 0.818182 | 94 | 0.886792 |
| PTV2_6 | 49 | 0.960784 | 96 | 0.923077 |
| PTV2_7 | 49 | 0.960784 | 98 | 0.960784 |
| PTV2_8 | 49 | 0.960784 | 98 | 0.960784 |
| PTV2_9 | 48 | 0.923077 | 96 | 0.923077 |
| PTV2_10 | 48 | 0.923077 | 98 | 0.960784 |
| PTV2_11 | 49 | 0.960784 | 98 | 0.960784 |
| PTV2_12 | 48 | 0.923077 | 98 | 0.960784 |
| PTV2_13 | 48 | 0.923077 | 100 | 1.000000 |

## Overall Device Shift

The signed difference is defined as CAC minus B over `41821` exact pairs.

- Mean signed difference: `-0.007193`
- Median signed difference: `-0.001519`
- Mean absolute difference: `0.008309`
- P90/P95 absolute difference: `0.020937` / `0.046686`

## Drugs With Largest Device Differences

| rank | drug | mean B | mean CAC | mean CAC-B | mean abs diff | max abs diff |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | L9200_430 | 0.438812 | 0.316504 | -0.122308 | 0.122308 | 0.139040 |
| 2 | 160 | 0.438267 | 0.317387 | -0.120880 | 0.120880 | 0.136171 |
| 3 | extid::brdk43187018001033 | 0.456294 | 0.338886 | -0.117408 | 0.117408 | 0.129201 |
| 4 | extid::brdk60997853001023 | 0.428388 | 0.312772 | -0.115615 | 0.115615 | 0.126148 |
| 5 | extid::brdk85920262001021 | 0.645638 | 0.534194 | -0.111443 | 0.111443 | 0.135784 |
| 6 | extid::brdk45293975001020 | 0.358922 | 0.251316 | -0.107606 | 0.107606 | 0.137165 |
| 7 | 142 | 0.670221 | 0.562896 | -0.107325 | 0.107325 | 0.151947 |
| 8 | L9200_1754 | 0.670221 | 0.562896 | -0.107325 | 0.107325 | 0.151947 |
| 9 | extid::l92001754 | 0.674662 | 0.569459 | -0.105203 | 0.105203 | 0.152555 |
| 10 | extid::brdk62008436001221 | 0.674662 | 0.569459 | -0.105203 | 0.105203 | 0.152554 |
| 11 | extid::brdk77008974001032 | 0.385007 | 0.280298 | -0.104709 | 0.104709 | 0.121677 |
| 12 | extid::l92001324 | 0.437654 | 0.335503 | -0.102151 | 0.102151 | 0.111049 |
| 13 | extid::1299a19 | 0.437654 | 0.335503 | -0.102151 | 0.102151 | 0.111049 |
| 14 | extid::brdk81418486001475 | 0.437654 | 0.335503 | -0.102151 | 0.102151 | 0.111049 |
| 15 | extid::l186 | 0.437654 | 0.335503 | -0.102151 | 0.102151 | 0.111049 |
| 16 | extsmiles::0ac4f812df4a | 0.437654 | 0.335503 | -0.102151 | 0.102151 | 0.111049 |
| 17 | 129 | 0.498156 | 0.398703 | -0.099452 | 0.099452 | 0.108638 |
| 18 | extsmiles::fd35df255e2e | 0.322575 | 0.223654 | -0.098921 | 0.098921 | 0.133163 |
| 19 | extid::brdk12251893065047 | 0.322575 | 0.223654 | -0.098921 | 0.098921 | 0.133163 |
| 20 | L9200_800 | 0.303079 | 0.205026 | -0.098052 | 0.098052 | 0.138972 |

## Interpretation Notes

- Higher scores indicate higher predicted sensitivity under the standardized exposure condition.
- B and CAC are retained as separate instrument measurements; predictions are not averaged.
- Applying 24 hours and dose 10 to all 3,217 drugs is an intentional standardized extrapolation, including drugs without that exact historical condition.
- The top-20 CSV contains 20 ranked drugs for each of 13 samples on each device (520 rows).
