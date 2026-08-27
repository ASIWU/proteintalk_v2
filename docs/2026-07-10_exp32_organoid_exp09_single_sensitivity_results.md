# Exp32 Organoid Exp09 Single-drug Sensitivity Results

- Generated: `2026-07-10T07:04:24+00:00`
- Experiment prefix: `20260710_exp32_organoid_exp09_single_sensitivity`
- Checkpoint: `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=5.ckpt`
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

- Combined CSV: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_predictions.csv`
- Combined Parquet: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_predictions.parquet`
- Per-sample top-20 CSV: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_top20_by_sample.csv`
- Summary JSON: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_summary.json`

## Device Probability Distributions

| device | n | mean | median | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 41821 | 0.072649 | 0.002208 | 0.154170 | 0.726599 | 0.000031 | 0.996785 |
| CAC | 41821 | 0.069915 | 0.000983 | 0.090937 | 0.814324 | 0.000015 | 0.998846 |

## Sample Probability Distributions

| device | sample | pat_ID | tissue | mean | median | P90 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | PTV2_1 | KOPC-P105 | Pancreas | 0.049435 | 0.001549 | 0.062592 | 0.434719 |
| B | PTV2_10 | KOLU-P1589 | Lung | 0.074874 | 0.002251 | 0.161684 | 0.737827 |
| B | PTV2_11 | KOLU-P1586 | Lung | 0.075134 | 0.002260 | 0.162882 | 0.741123 |
| B | PTV2_12 | KOLU-P1596 | Lung | 0.074902 | 0.002225 | 0.160362 | 0.740462 |
| B | PTV2_13 | KOPC-P102 | Pancreas | 0.049998 | 0.001512 | 0.063246 | 0.449569 |
| B | PTV2_2 | KOLU-P1583 | Lung | 0.074429 | 0.002160 | 0.156489 | 0.737357 |
| B | PTV2_3 | KOLU-P1585 | Lung | 0.075271 | 0.002124 | 0.163680 | 0.749956 |
| B | PTV2_4 | KOLU-P1587 | Lung | 0.075153 | 0.002197 | 0.162907 | 0.745093 |
| B | PTV2_5 | KOCO-P281 | Colon | 0.096642 | 0.004627 | 0.354770 | 0.858251 |
| B | PTV2_6 | KOLU-P1594 | Lung | 0.073776 | 0.002110 | 0.152551 | 0.733184 |
| B | PTV2_7 | KOLU-P1593 | Lung | 0.076100 | 0.002413 | 0.170826 | 0.744937 |
| B | PTV2_8 | KOLU-P1584 | Lung | 0.074897 | 0.002114 | 0.160046 | 0.744926 |
| B | PTV2_9 | KOLU-P1614 | Lung | 0.073826 | 0.002156 | 0.152625 | 0.729959 |
| CAC | PTV2_1 | KOPC-P105 | Pancreas | 0.044798 | 0.000668 | 0.025559 | 0.386985 |
| CAC | PTV2_10 | KOLU-P1589 | Lung | 0.073295 | 0.001015 | 0.106565 | 0.828124 |
| CAC | PTV2_11 | KOLU-P1586 | Lung | 0.073129 | 0.001168 | 0.108259 | 0.819345 |
| CAC | PTV2_12 | KOLU-P1596 | Lung | 0.073383 | 0.001089 | 0.106604 | 0.827287 |
| CAC | PTV2_13 | KOPC-P102 | Pancreas | 0.045325 | 0.000656 | 0.025874 | 0.393480 |
| CAC | PTV2_2 | KOLU-P1583 | Lung | 0.072919 | 0.001028 | 0.104890 | 0.823371 |
| CAC | PTV2_3 | KOLU-P1585 | Lung | 0.073611 | 0.000935 | 0.109524 | 0.834914 |
| CAC | PTV2_4 | KOLU-P1587 | Lung | 0.073545 | 0.000964 | 0.107778 | 0.834597 |
| CAC | PTV2_5 | KOCO-P281 | Colon | 0.088074 | 0.001369 | 0.256425 | 0.927174 |
| CAC | PTV2_6 | KOLU-P1594 | Lung | 0.072437 | 0.001011 | 0.103708 | 0.817231 |
| CAC | PTV2_7 | KOLU-P1593 | Lung | 0.073336 | 0.001087 | 0.106544 | 0.826382 |
| CAC | PTV2_8 | KOLU-P1584 | Lung | 0.072831 | 0.000969 | 0.106223 | 0.825152 |
| CAC | PTV2_9 | KOLU-P1614 | Lung | 0.072210 | 0.001061 | 0.101093 | 0.814934 |

## B versus CAC Agreement

Overall paired Pearson: `0.992097`; Spearman: `0.993484`; MAE: `0.010684`.

| sample | pat_ID | tissue | Pearson | Spearman | MAE | mean CAC-B |
| --- | --- | --- | --- | --- | --- | --- |
| PTV2_1 | KOPC-P105 | Pancreas | 0.985673 | 0.995670 | 0.009857 | -0.004637 |
| PTV2_10 | KOLU-P1589 | Lung | 0.992986 | 0.995366 | 0.010696 | -0.001579 |
| PTV2_11 | KOLU-P1586 | Lung | 0.993963 | 0.995277 | 0.009750 | -0.002005 |
| PTV2_12 | KOLU-P1596 | Lung | 0.993410 | 0.995071 | 0.010251 | -0.001520 |
| PTV2_13 | KOPC-P102 | Pancreas | 0.986783 | 0.995760 | 0.009636 | -0.004673 |
| PTV2_2 | KOLU-P1583 | Lung | 0.993455 | 0.995076 | 0.010209 | -0.001510 |
| PTV2_3 | KOLU-P1585 | Lung | 0.992959 | 0.994712 | 0.010657 | -0.001660 |
| PTV2_4 | KOLU-P1587 | Lung | 0.992790 | 0.995107 | 0.010804 | -0.001608 |
| PTV2_5 | KOCO-P281 | Colon | 0.987816 | 0.993442 | 0.016246 | -0.008568 |
| PTV2_6 | KOLU-P1594 | Lung | 0.993798 | 0.995606 | 0.009957 | -0.001340 |
| PTV2_7 | KOLU-P1593 | Lung | 0.993109 | 0.995377 | 0.010658 | -0.002764 |
| PTV2_8 | KOLU-P1584 | Lung | 0.993460 | 0.995134 | 0.010186 | -0.002067 |
| PTV2_9 | KOLU-P1614 | Lung | 0.993683 | 0.995232 | 0.009989 | -0.001617 |

## Top-drug Agreement

| sample | top-50 overlap | top-50 Jaccard | top-100 overlap | top-100 Jaccard |
| --- | --- | --- | --- | --- |
| PTV2_1 | 40 | 0.666667 | 90 | 0.818182 |
| PTV2_2 | 47 | 0.886792 | 95 | 0.904762 |
| PTV2_3 | 49 | 0.960784 | 97 | 0.941748 |
| PTV2_4 | 47 | 0.886792 | 98 | 0.960784 |
| PTV2_5 | 48 | 0.923077 | 96 | 0.923077 |
| PTV2_6 | 46 | 0.851852 | 95 | 0.904762 |
| PTV2_7 | 47 | 0.886792 | 95 | 0.904762 |
| PTV2_8 | 46 | 0.851852 | 95 | 0.904762 |
| PTV2_9 | 46 | 0.851852 | 95 | 0.904762 |
| PTV2_10 | 47 | 0.886792 | 96 | 0.923077 |
| PTV2_11 | 47 | 0.886792 | 95 | 0.904762 |
| PTV2_12 | 47 | 0.886792 | 95 | 0.904762 |
| PTV2_13 | 40 | 0.666667 | 91 | 0.834862 |

## Overall Device Shift

The signed difference is defined as CAC minus B over `41821` exact pairs.

- Mean signed difference: `-0.002734`
- Median signed difference: `-0.000833`
- Mean absolute difference: `0.010684`
- P90/P95 absolute difference: `0.029332` / `0.067280`

## Drugs With Largest Device Differences

| rank | drug | mean B | mean CAC | mean CAC-B | mean abs diff | max abs diff |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | L9200_430 | 0.504281 | 0.303280 | -0.201001 | 0.201001 | 0.210008 |
| 2 | 200 | 0.667016 | 0.466272 | -0.200744 | 0.200744 | 0.243184 |
| 3 | L9200_679 | 0.216009 | 0.403906 | 0.187897 | 0.187897 | 0.239986 |
| 4 | 160 | 0.520858 | 0.334321 | -0.186537 | 0.186537 | 0.193906 |
| 5 | L9200_1486 | 0.418901 | 0.605331 | 0.186431 | 0.186431 | 0.224056 |
| 6 | L9200_1790 | 0.317501 | 0.503921 | 0.186420 | 0.186420 | 0.240951 |
| 7 | extid::brdk83988098001020 | 0.335018 | 0.156043 | -0.178975 | 0.178975 | 0.266254 |
| 8 | extid::brdk83988098003034 | 0.335018 | 0.156043 | -0.178975 | 0.178975 | 0.266254 |
| 9 | L9200_133 | 0.371546 | 0.201127 | -0.170419 | 0.170419 | 0.222420 |
| 10 | L9200_2314 | 0.341834 | 0.490987 | 0.149154 | 0.155153 | 0.196336 |
| 11 | L9200_2646 | 0.238035 | 0.391313 | 0.153279 | 0.154318 | 0.214135 |
| 12 | extid::brda27376179001013 | 0.305472 | 0.156621 | -0.148852 | 0.148852 | 0.213646 |
| 13 | extid::brdk60997853001023 | 0.373255 | 0.226793 | -0.146462 | 0.146462 | 0.170322 |
| 14 | L9200_1492 | 0.491432 | 0.622638 | 0.131206 | 0.144789 | 0.183488 |
| 15 | L9200_2312 | 0.278978 | 0.414596 | 0.135619 | 0.142359 | 0.187841 |
| 16 | L9200_2575 | 0.231308 | 0.090220 | -0.141088 | 0.141088 | 0.283565 |
| 17 | L9200_2091 | 0.303808 | 0.165691 | -0.138116 | 0.138116 | 0.186934 |
| 18 | extid::brdk85920262001021 | 0.729203 | 0.591818 | -0.137385 | 0.137385 | 0.226838 |
| 19 | L9200_2707 | 0.682461 | 0.817197 | 0.134737 | 0.134737 | 0.162247 |
| 20 | extid::brdk90947825001027 | 0.245134 | 0.112585 | -0.132549 | 0.132549 | 0.240980 |

## Interpretation Notes

- Higher scores indicate higher predicted sensitivity under the standardized exposure condition.
- B and CAC are retained as separate instrument measurements; predictions are not averaged.
- Applying 24 hours and dose 10 to all 3,217 drugs is an intentional standardized extrapolation, including drugs without that exact historical condition.
- The top-20 CSV contains 20 ranked drugs for each of 13 samples on each device (520 rows).
