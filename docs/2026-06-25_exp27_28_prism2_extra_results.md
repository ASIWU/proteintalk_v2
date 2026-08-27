# exp_27-exp_28 PRISM2 Extra Inference Results

Date: 2026-06-25 HKT

## Completion Summary

The exp27/exp28 PRISM2 extra-inference suite completed successfully under the final run prefix:

- Run prefix: `20260625_1940_prism2_exp27_28_nowandb`
- Data root: `data/training_ready_prism2_main`
- Runtime summary: `logs/20260625_1940_prism2_exp27_28_nowandb_runtime_summary.tsv`
- Cell-drug-dose grouped report CSV: `outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_reports/exp27_28_cell_drug_dose_time_eval.csv`

Completion and validation:

- Training jobs: `2/2` successful.
- Extra single-drug inference jobs: `6/6` successful.
- Extra double-drug inference jobs: `3/3` successful.
- Runtime summary rows: `11/11` have status `0`.
- exp27 manifest uses `task=ptv3_main_singledrug_prism2`, `split=all_train_subset_test`, `key1=PRISM2nd_label_total`, and response head.
- exp28 manifest uses `task=ptv3_main_doubledrug_prism2aux`, `split=all_train_subset_test`, `key1=synergy`, and synergy head, with the PRISM2 single-drug auxiliary key recorded as `PRISM2nd_label_total`.

## Setup

This follows the exp07/exp08 mechanics, but uses the PRISM2 main-label branch created for exp21-exp26.

| exp | source logic | train task | train label/head | inference targets |
|---|---|---|---|---|
| exp27 | exp07 | `ptv3_main_singledrug_prism2` | `PRISM2nd_label_total` / response | `mat1_480_faims`, `mat1_qe`, `mat2_480_faims`, `mat2_qe`, `mat3_qe`, `mat4_qe` |
| exp28 | exp08 | `ptv3_main_doubledrug_prism2aux` | `synergy` / synergy, with PRISM2 single auxiliary rows | `nature`, `nc`, `guomics` |

Command used in `gpu2`:

```bash
TRAINING_READY_ROOT=data/training_ready_prism2_main \
EXP_PREFIX=20260625_1940_prism2_exp27_28_nowandb \
LOGGER_BACKEND=tensorboard \
LOG_TO_WANDB=0 \
RUN_PREFLIGHT=1 \
bash scripts/run_exp_27_28_prism2_extra.sh
```

Added scripts:

| script | purpose |
|---|---|
| `scripts/exp_27_extra_single_prism2_all_train_infer.sh` | train PRISM2 all-single and infer mat extra single-drug datasets |
| `scripts/exp_28_extra_double_prism2aux_all_train_infer.sh` | train PRISM2-aux all-single+double and infer extra double-drug datasets |
| `scripts/run_exp_27_28_prism2_extra.sh` | sequential runner for exp27 then exp28 |

## Configs

The exp27/28 runs reuse the corresponding exp07/08 style hyperparameters, but swap the single-drug main label to `PRISM2nd_label_total`.

| exp | role | LR | batch | dropout | weight decay | graph mode | additional settings |
|---|---|---:|---:|---:|---:|---|---|
| exp27 | extra single, all-data train | 2e-4 | 256 | 0.15 | 1e-4 | real | single response head, `PRISM2nd_label_total` |
| exp28 | extra double, all-data train with PRISM2 auxiliary single rows | 2e-4 | 256 | 0.20 | 1e-4 | real | double synergy head, `PAIR_FUSION_MODE=dual`, `PAIR_TYPE_FEATURES=1`, `USE_DDI=1`, `GRAPH_PAIR_ADD_SCALE=0.5` |

## Checkpoints

| exp | checkpoint dir | best checkpoint |
|---|---|---|
| exp27 | `checkpoints/20260625_1940_prism2_exp27_28_nowandb_all_single_prism2_for_extra` | `epoch=27-step=840.ckpt` |
| exp28 | `checkpoints/20260625_1940_prism2_exp27_28_nowandb_all_single_double_prism2aux_for_extra` | `epoch=12-step=481.ckpt` |

Checkpoint policy:

- exp27/28 are all-data extra-inference runs, so the validation split is the convenience `all_train_subset_test` split rather than an independent held-out benchmark.
- Both runs used the best checkpoint produced by the training script in the run directory.
- The all-data validation metrics can reach `1.0` and should not be read as external generalization. The meaningful readout for this workflow is the extra-data inference table below.

## Runtime

| step | dataset/task | status | duration_sec |
|---|---|---:|---:|
| train | exp27 all-single PRISM2 | 0 | 83 |
| infer | `ptv3_extra_singledrug_mat1_480_faims` | 0 | 15 |
| infer | `ptv3_extra_singledrug_mat1_qe` | 0 | 14 |
| infer | `ptv3_extra_singledrug_mat2_480_faims` | 0 | 12 |
| infer | `ptv3_extra_singledrug_mat2_qe` | 0 | 12 |
| infer | `ptv3_extra_singledrug_mat3_qe` | 0 | 14 |
| infer | `ptv3_extra_singledrug_mat4_qe` | 0 | 12 |
| train | exp28 all-single+double PRISM2aux | 0 | 78 |
| infer | `ptv3_extra_doubledrug_nature` | 0 | 27 |
| infer | `ptv3_extra_doubledrug_nc` | 0 | 13 |
| infer | `ptv3_extra_doubledrug_guomics` | 0 | 8 |

Prediction row counts:

| exp | output target | rows |
|---|---|---:|
| exp27 | `ptv3_extra_singledrug_mat1_480_faims` | 17140 |
| exp27 | `ptv3_extra_singledrug_mat1_qe` | 17140 |
| exp27 | `ptv3_extra_singledrug_mat2_480_faims` | 13882 |
| exp27 | `ptv3_extra_singledrug_mat2_qe` | 13882 |
| exp27 | `ptv3_extra_singledrug_mat3_qe` | 18332 |
| exp27 | `ptv3_extra_singledrug_mat4_qe` | 12295 |
| exp28 | `ptv3_extra_doubledrug_guomics` | 5633 |
| exp28 | `ptv3_extra_doubledrug_nature` | 68182 |
| exp28 | `ptv3_extra_doubledrug_nc` | 15155 |

## Extra Mean Results

The table below reports mean extra metrics with the same grouping styles used by the earlier cell-drug-dose/time reports.

- `original`: row/timepoint-level metrics from the raw inference output.
- `cell-drug-dose-bylasttime`: one score per cell-drug-dose group, using the last available timepoint when time exists.
- `cell-drug-dose-avgtime`: one score per cell-drug-dose group, averaging scores across timepoints.
- `conflicts`: grouped rows where duplicate cell-drug-dose labels disagree.
- `missing_time`: rows/groups without a usable time value for the by-last-time grouping. For these extra datasets, by-last-time and avg-time are numerically identical because time is unavailable in the standardized inference rows.

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp27 | extra single | mean_extra | original | 0.808905 | 0.633075 | 0.209041 | 3.018387 | 92671 | 19345 | 73326 | 0 | 0 |
| exp27 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.808772 | 0.634595 | 0.212230 | 2.976682 | 86450 | 18286 | 68164 | 28 | 86450 |
| exp27 | extra single | mean_extra | cell-drug-dose-avgtime | 0.808772 | 0.634595 | 0.212230 | 2.976682 | 86450 | 18286 | 68164 | 28 | 0 |
| exp28 | extra double | mean_extra | original | 0.621895 | 0.081156 | 0.047592 | 1.925263 | 88970 | 3696 | 85274 | 0 | 0 |
| exp28 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.620954 | 0.081700 | 0.047788 | 1.931812 | 66275 | 2926 | 63349 | 15 | 66275 |
| exp28 | extra double | mean_extra | cell-drug-dose-avgtime | 0.620954 | 0.081700 | 0.047788 | 1.931812 | 66275 | 2926 | 63349 | 15 | 0 |

## exp27 Extra Single-Drug Subset Results

Output root:

```text
outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_prism2_for_extra
```

Primary metrics CSV:

```text
outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_prism2_for_extra/extra_singledrug_metrics.csv
```

Original row/timepoint-level aggregate over the 6 mat datasets:

| metric | value |
|---|---:|
| mean AUROC | 0.808905 |
| mean AUPRC | 0.633075 |
| mean baseline AUPRC | 0.209041 |
| mean nAUPRC | 3.018387 |
| mean ACC | 0.853505 |
| valid total | 92671 |

Full subset table:

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp27 | ptv3_extra_singledrug_mat1_480_faims | extra | original | 0.726164 | 0.489476 | 0.202917 | 2.412194 | 17140 | 3478 | 13662 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.723420 | 0.488554 | 0.205230 | 2.380525 | 16445 | 3375 | 13070 | 0 | 16445 |
| exp27 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-avgtime | 0.723420 | 0.488554 | 0.205230 | 2.380525 | 16445 | 3375 | 13070 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat1_qe | extra | original | 0.718936 | 0.474887 | 0.202917 | 2.340300 | 17140 | 3478 | 13662 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-bylasttime | 0.716286 | 0.473707 | 0.205230 | 2.308181 | 16445 | 3375 | 13070 | 0 | 16445 |
| exp27 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-avgtime | 0.716286 | 0.473707 | 0.205230 | 2.308181 | 16445 | 3375 | 13070 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat2_480_faims | extra | original | 0.901044 | 0.791167 | 0.215819 | 3.665880 | 13882 | 2996 | 10886 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.902410 | 0.796470 | 0.221938 | 3.588702 | 12125 | 2691 | 9434 | 9 | 12125 |
| exp27 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-avgtime | 0.902410 | 0.796470 | 0.221938 | 3.588702 | 12125 | 2691 | 9434 | 9 | 0 |
| exp27 | ptv3_extra_singledrug_mat2_qe | extra | original | 0.892565 | 0.772182 | 0.215819 | 3.577913 | 13882 | 2996 | 10886 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-bylasttime | 0.893808 | 0.776328 | 0.221938 | 3.497949 | 12125 | 2691 | 9434 | 9 | 12125 |
| exp27 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-avgtime | 0.893808 | 0.776328 | 0.221938 | 3.497949 | 12125 | 2691 | 9434 | 9 | 0 |
| exp27 | ptv3_extra_singledrug_mat3_qe | extra | original | 0.720974 | 0.500879 | 0.210834 | 2.375711 | 18332 | 3865 | 14467 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-bylasttime | 0.720951 | 0.501203 | 0.211298 | 2.372024 | 18287 | 3864 | 14423 | 0 | 18287 |
| exp27 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-avgtime | 0.720951 | 0.501203 | 0.211298 | 2.372024 | 18287 | 3864 | 14423 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat4_qe | extra | original | 0.893743 | 0.769860 | 0.205937 | 3.738323 | 12295 | 2532 | 9763 | 0 | 0 |
| exp27 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-bylasttime | 0.895756 | 0.771307 | 0.207747 | 3.712714 | 11023 | 2290 | 8733 | 10 | 11023 |
| exp27 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-avgtime | 0.895756 | 0.771307 | 0.207747 | 3.712714 | 11023 | 2290 | 8733 | 10 | 0 |

Original row/timepoint-level per-dataset metrics with ACC:

| task | AUROC | AUPRC | baseline | nAUPRC | ACC | valid |
|---|---:|---:|---:|---:|---:|---:|
| `ptv3_extra_singledrug_mat1_480_faims` | 0.726164 | 0.489476 | 0.202917 | 2.412194 | 0.820362 | 17140 |
| `ptv3_extra_singledrug_mat1_qe` | 0.718936 | 0.474887 | 0.202917 | 2.340300 | 0.804201 | 17140 |
| `ptv3_extra_singledrug_mat2_480_faims` | 0.901044 | 0.791167 | 0.215819 | 3.665880 | 0.901527 | 13882 |
| `ptv3_extra_singledrug_mat2_qe` | 0.892565 | 0.772182 | 0.215819 | 3.577913 | 0.894180 | 13882 |
| `ptv3_extra_singledrug_mat3_qe` | 0.720974 | 0.500879 | 0.210834 | 2.375711 | 0.802913 | 18332 |
| `ptv3_extra_singledrug_mat4_qe` | 0.893743 | 0.769860 | 0.205937 | 3.738323 | 0.897845 | 12295 |

## exp28 Extra Double-Drug Subset Results

Output root:

```text
outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_double_prism2aux_for_extra
```

Primary metrics files:

```text
outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_double_prism2aux_for_extra/extra_doubledrug_test_label_auprc.csv
outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_double_prism2aux_for_extra/extra_doubledrug_test_label_auprc.json
```

Original row/timepoint-level combined-group aggregate over guomics/nc/nature:

| metric | value |
|---|---:|
| mean AUROC | 0.621895 |
| mean AUPRC | 0.081156 |
| mean baseline AUPRC | 0.047592 |
| mean nAUPRC | 1.925263 |
| valid total | 88970 |

Full test-label grouping:

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp28 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | original | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 42 |
| exp28 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | original | 0.687572 | 0.099805 | 0.034520 | 2.891234 | 5591 | 193 | 5398 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.687572 | 0.099805 | 0.034520 | 2.891234 | 5591 | 193 | 5398 | 0 | 5591 |
| exp28 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.687572 | 0.099805 | 0.034520 | 2.891234 | 5591 | 193 | 5398 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_guomics | combined | original | 0.689628 | 0.106948 | 0.034617 | 3.089424 | 5633 | 195 | 5438 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-bylasttime | 0.689628 | 0.106948 | 0.034617 | 3.089424 | 5633 | 195 | 5438 | 0 | 5633 |
| exp28 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-avgtime | 0.689628 | 0.106948 | 0.034617 | 3.089424 | 5633 | 195 | 5438 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | original | 0.648831 | 0.084924 | 0.059693 | 1.422697 | 16652 | 994 | 15658 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.630717 | 0.088793 | 0.063162 | 1.405803 | 10576 | 668 | 9908 | 4 | 10576 |
| exp28 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.630717 | 0.088793 | 0.063162 | 1.405803 | 10576 | 668 | 9908 | 4 | 0 |
| exp28 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | original | 0.615910 | 0.038872 | 0.027169 | 1.430782 | 51530 | 1400 | 50130 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.623893 | 0.040282 | 0.027384 | 1.471004 | 34911 | 956 | 33955 | 11 | 34911 |
| exp28 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.623893 | 0.040282 | 0.027384 | 1.471004 | 34911 | 956 | 33955 | 11 | 0 |
| exp28 | ptv3_extra_doubledrug_nature | combined | original | 0.646337 | 0.055265 | 0.035112 | 1.573980 | 68182 | 2394 | 65788 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-bylasttime | 0.643515 | 0.056896 | 0.035703 | 1.593626 | 45487 | 1624 | 43863 | 15 | 45487 |
| exp28 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-avgtime | 0.643515 | 0.056896 | 0.035703 | 1.593626 | 45487 | 1624 | 43863 | 15 | 0 |
| exp28 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | original | 0.600402 | 0.128426 | 0.091881 | 1.397739 | 2057 | 189 | 1868 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.600402 | 0.128426 | 0.091881 | 1.397739 | 2057 | 189 | 1868 | 0 | 2057 |
| exp28 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.600402 | 0.128426 | 0.091881 | 1.397739 | 2057 | 189 | 1868 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | original | 0.509829 | 0.070716 | 0.070087 | 1.008968 | 13098 | 918 | 12180 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.509829 | 0.070716 | 0.070087 | 1.008968 | 13098 | 918 | 12180 | 0 | 13098 |
| exp28 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.509829 | 0.070716 | 0.070087 | 1.008968 | 13098 | 918 | 12180 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nc | combined | original | 0.529720 | 0.081254 | 0.073045 | 1.112385 | 15155 | 1107 | 14048 | 0 | 0 |
| exp28 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-bylasttime | 0.529720 | 0.081254 | 0.073045 | 1.112385 | 15155 | 1107 | 14048 | 0 | 15155 |
| exp28 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-avgtime | 0.529720 | 0.081254 | 0.073045 | 1.112385 | 15155 | 1107 | 14048 | 0 | 0 |

Original row/timepoint-level combined rows with ACC:

| task | AUROC | AUPRC | baseline | nAUPRC | ACC | valid | pos | neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ptv3_extra_doubledrug_guomics` | 0.689628 | 0.106948 | 0.034617 | 3.089424 | 0.814841 | 5633 | 195 | 5438 |
| `ptv3_extra_doubledrug_nature` | 0.646337 | 0.055265 | 0.035112 | 1.573980 | 0.652196 | 68182 | 2394 | 65788 |
| `ptv3_extra_doubledrug_nc` | 0.529720 | 0.081254 | 0.073045 | 1.112385 | 0.696998 | 15155 | 1107 | 14048 |

## Output Files

| file | content |
|---|---|
| `logs/20260625_1940_prism2_exp27_28_nowandb_runtime_summary.tsv` | train/inference runtime status summary |
| `outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_reports/exp27_28_cell_drug_dose_time_eval.csv` | unified exp27/28 original and cell-drug-dose grouped metrics |
| `outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_prism2_for_extra/extra_singledrug_metrics.csv` | exp27 original extra single-drug metrics |
| `outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_double_prism2aux_for_extra/extra_doubledrug_test_label_auprc.csv` | exp28 original extra double-drug test-label metrics |
| `outputs/2026-06/2026-06-25/20260625_1940_prism2_exp27_28_nowandb_all_single_double_prism2aux_for_extra/extra_doubledrug_test_label_auprc.json` | exp28 original extra double-drug metrics JSON |

## Readout

1. exp27/exp28 PRISM2 extra inference is complete and matches the exp07/exp08 data flow, with the requested PRISM2 label branch.
2. exp27 is stronger than the earlier exp07 extra-single result on the same external mat targets: original mean AUPRC `0.633075` versus the tuned exp07 reference AUPRC `0.561044` reported in `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`.
3. exp27 gains are concentrated on mat2 and mat4. The weaker external subsets remain mat1 and mat3, with original AUPRC around `0.47-0.50`.
4. exp28 is weaker than the earlier exp08 tuned extra-double reference on original mean AUPRC: `0.081156` versus `0.091853`. Its strongest combined subset is guomics by n-AUPRC (`3.089424`), while nc is close to baseline (`1.112385` n-AUPRC).
5. The grouped metrics do not materially change the conclusion. exp27 grouped AUPRC is `0.634595`, slightly above the original `0.633075`; exp28 grouped AUPRC is `0.081700`, slightly above the original `0.081156`.
6. The all-data validation checkpoint metrics should not be used as benchmark evidence because `all_train_subset_test` is not an independent held-out split. The external extra tables above are the relevant evaluation.
