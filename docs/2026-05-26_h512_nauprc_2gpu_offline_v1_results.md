# h512 Baseline4 2-GPU Full-Suite Results

Date: 2026-05-26 17:15 HKT

Run prefix: `20260526_h512_nauprc_2gpu_offline_v1`

Command:

```bash
WANDB_MODE=offline EXP_PREFIX=20260526_h512_nauprc_2gpu_offline_v1 GPU_IDS=0,1 \
  bash scripts/0526_baseline4_task_specific_2gpu_parallel.sh
```

Status:

- Full suite completed successfully.
- Runtime summary: `41/41` rows returned `status=0`.
- GPU mode: two one-GPU workers, `GPU_IDS=0,1`, `DEVICES=1`.
- Wandb mode: `offline`, because the online wandb server returned an HTTP error during the first launch attempt.

Key output paths:

- Runtime summary: `logs/20260526_h512_nauprc_2gpu_offline_v1_runtime_summary.tsv`
- Launcher log: `logs/20260526_h512_nauprc_2gpu_offline_v1_launcher.log`
- Extra single outputs: `outputs/20260526_h512_nauprc_2gpu_offline_v1_exp07_extra_single_all_train_infer_all_single_for_extra`
- Extra double outputs: `outputs/20260526_h512_nauprc_2gpu_offline_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra`
- Extra double grouped report: `outputs/20260526_h512_nauprc_2gpu_offline_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra/extra_doubledrug_test_label_auprc.csv`

## Active Model Defaults

| setting | value |
| --- | ---: |
| `HIDDEN_DIM` | `512` |
| `EXPRESSION_LATENT_DIM` | `768` |
| `COVARIATE_EMBEDDING_DIM` | `96` |
| `LEARNING_RATE` | `2e-4` |
| `BATCH_SIZE` | `256` |
| `INFER_BATCH_SIZE` | `256` |
| `MODEL_TYPE` | `fast_delta` |
| `PROTEIN_CONCAT_MODE` | `pcep` |
| `GRAPH_FEATURE_MODE` | `real` |

Task-specific launcher defaults:

| setting | single drug | double drug |
| --- | ---: | ---: |
| pair fusion | `symmetric` | `dual` |
| pair type features | `0` | `1` |
| DDI | `0` | `1` |
| graph pair add scale | `0.0` | `0.5` |
| MSE inactive label weight | `1.0` | `0.2` |

## 5-Fold Summary

| experiment | task | AUROC | AUPRC | baseline | n-AUPRC | ACC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exp01 | single unseen drug | 0.903166 | 0.677846 | 0.118838 | 5.723701 | 0.919744 |
| exp02 | single unseen cell type | 0.933192 | 0.792645 | 0.173980 | 6.098465 | 0.901885 |
| exp03 | single unseen cell | 0.927191 | 0.751375 | 0.154930 | 6.228606 | 0.900473 |
| exp04 | single w/o MSE | 0.893265 | 0.651609 | 0.118838 | 5.499016 | 0.917120 |
| exp05 | single w/o graph | 0.853110 | 0.579863 | 0.118838 | 4.876731 | 0.911721 |
| exp06 | double unseen drug pair | 0.736534 | 0.616861 | 0.404342 | 1.526420 | 0.694366 |

## 5-Fold Detail

### exp01 single unseen drug

| fold | AUROC | AUPRC | baseline | n-AUPRC | ACC | best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.864744 | 0.578386 | 0.120910 | 4.783626 | 0.900166 | 3 |
| 1 | 0.936036 | 0.781086 | 0.120022 | 6.507840 | 0.940407 | 6 |
| 2 | 0.894365 | 0.579320 | 0.108467 | 5.340973 | 0.902324 | 10 |
| 3 | 0.913331 | 0.770133 | 0.112594 | 6.839919 | 0.939950 | 4 |
| 4 | 0.907354 | 0.680304 | 0.132197 | 5.146148 | 0.915875 | 6 |
| avg | 0.903166 | 0.677846 | 0.118838 | 5.723701 | 0.919744 | - |

### exp02 single unseen cell type

| fold | AUROC | AUPRC | baseline | n-AUPRC | ACC | best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.943295 | 0.757451 | 0.145419 | 5.208732 | 0.902323 | 4 |
| 1 | 0.916346 | 0.701553 | 0.106180 | 6.607201 | 0.931047 | 7 |
| 2 | 0.928674 | 0.822307 | 0.224599 | 3.661226 | 0.874332 | 4 |
| 3 | 0.918975 | 0.847685 | 0.326531 | 2.596035 | 0.831633 | 7 |
| 4 | 0.958668 | 0.834231 | 0.067173 | 12.419134 | 0.970093 | 7 |
| avg | 0.933192 | 0.792645 | 0.173980 | 6.098465 | 0.901885 | - |

### exp03 single unseen cell

| fold | AUROC | AUPRC | baseline | n-AUPRC | ACC | best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.910795 | 0.832442 | 0.303871 | 2.739455 | 0.844412 | 5 |
| 1 | 0.930702 | 0.772996 | 0.105141 | 7.351987 | 0.943177 | 6 |
| 2 | 0.887698 | 0.594460 | 0.183423 | 3.240928 | 0.834229 | 10 |
| 3 | 0.956540 | 0.822112 | 0.112056 | 7.336606 | 0.945636 | 8 |
| 4 | 0.950221 | 0.734866 | 0.070161 | 10.474056 | 0.934911 | 0 |
| avg | 0.927191 | 0.751375 | 0.154930 | 6.228606 | 0.900473 | - |

### exp04 single w/o MSE

| fold | AUROC | AUPRC | baseline | n-AUPRC | ACC | best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.818366 | 0.519265 | 0.120910 | 4.294657 | 0.903772 | 7 |
| 1 | 0.940099 | 0.771237 | 0.120022 | 6.425782 | 0.941799 | 6 |
| 2 | 0.876399 | 0.547811 | 0.108467 | 5.050480 | 0.901494 | 12 |
| 3 | 0.930911 | 0.747640 | 0.112594 | 6.640152 | 0.933000 | 2 |
| 4 | 0.900548 | 0.672089 | 0.132197 | 5.084007 | 0.905534 | 5 |
| avg | 0.893265 | 0.651609 | 0.118838 | 5.499016 | 0.917120 | - |

### exp05 single w/o graph

| fold | AUROC | AUPRC | baseline | n-AUPRC | ACC | best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.830974 | 0.497214 | 0.120910 | 4.112282 | 0.895729 | 1 |
| 1 | 0.892555 | 0.726797 | 0.120022 | 6.055521 | 0.931217 | 9 |
| 2 | 0.796698 | 0.422730 | 0.108467 | 3.897308 | 0.903154 | 6 |
| 3 | 0.890871 | 0.640458 | 0.112594 | 5.688214 | 0.926884 | 3 |
| 4 | 0.854452 | 0.612115 | 0.132197 | 4.630330 | 0.901621 | 3 |
| avg | 0.853110 | 0.579863 | 0.118838 | 4.876731 | 0.911721 | - |

### exp06 double unseen drug pair

| fold | AUROC | AUPRC | baseline | n-AUPRC | ACC | best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.744139 | 0.674238 | 0.434174 | 1.552922 | 0.700280 | 11 |
| 1 | 0.728485 | 0.562388 | 0.363158 | 1.548605 | 0.692105 | 20 |
| 2 | 0.752255 | 0.619074 | 0.402235 | 1.539086 | 0.715084 | 19 |
| 3 | 0.756166 | 0.611472 | 0.397727 | 1.537416 | 0.707386 | 12 |
| 4 | 0.701622 | 0.617136 | 0.424419 | 1.454073 | 0.656977 | 8 |
| avg | 0.736534 | 0.616861 | 0.404342 | 1.526420 | 0.694366 | - |

## Extra Single-Drug Inference

| task | AUROC | AUPRC | baseline | n-AUPRC | ACC | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ptv3_extra_singledrug_mat1_480_faims | 0.7134 | 0.4684 | 0.2051 | 2.2833 | 0.8175 | 15834 |
| ptv3_extra_singledrug_mat1_qe | 0.7121 | 0.4529 | 0.2051 | 2.2077 | 0.8150 | 15834 |
| ptv3_extra_singledrug_mat2_480_faims | 0.8180 | 0.6725 | 0.2193 | 3.0665 | 0.8557 | 12138 |
| ptv3_extra_singledrug_mat2_qe | 0.8176 | 0.6674 | 0.2193 | 3.0432 | 0.8482 | 12138 |
| ptv3_extra_singledrug_mat3_qe | 0.7103 | 0.4645 | 0.2111 | 2.2004 | 0.8099 | 17609 |
| ptv3_extra_singledrug_mat4_qe | 0.8016 | 0.6393 | 0.2077 | 3.0774 | 0.8527 | 11072 |
| avg | 0.7621 | 0.5608 | 0.2113 | 2.6464 | 0.8332 | 84625 |

## Extra Double-Drug Standard Inference

This table is the standard synergy evaluation from `scripts/show_extra_results.py`.

| task | AUROC | AUPRC | baseline | n-AUPRC | ACC | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ptv3_extra_doubledrug_guomics | 0.6850 | 0.0220 | 0.0113 | 1.9390 | 0.8461 | 9001 |
| ptv3_extra_doubledrug_nature | 0.6576 | 0.0624 | 0.0349 | 1.7854 | 0.6832 | 23389 |
| ptv3_extra_doubledrug_nc | 0.5174 | 0.0814 | 0.0724 | 1.1243 | 0.6777 | 16394 |
| avg | 0.6200 | 0.0552 | 0.0396 | 1.6162 | 0.7357 | 48784 |

## Extra Double-Drug Grouped Evaluation

This table uses `scripts/report_extra_doubledrug_test_label_auprc.py` to join predictions to `data/rawdata/update_0526/extra_doubledrug` by `feature_row_index`, filter `test=1`, remove `test_label=delete`, and report the two requested groups plus their combined result.

| task | group | AUROC | AUPRC | baseline | n-AUPRC | ACC | valid | pos | neg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | 1.000000 | 1.000000 | 0.043478 | 23.000000 | 0.782609 | 23 | 1 | 22 |
| ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | 0.697009 | 0.079047 | 0.032750 | 2.413676 | 0.863489 | 3084 | 101 | 2983 |
| ptv3_extra_doubledrug_guomics | combined | 0.699703 | 0.084473 | 0.032829 | 2.573122 | 0.862890 | 3107 | 102 | 3005 |
| ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | 0.600865 | 0.134981 | 0.091881 | 1.469076 | 0.555664 | 2057 | 189 | 1868 |
| ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | 0.495598 | 0.069959 | 0.070087 | 0.998177 | 0.694304 | 13098 | 918 | 12180 |
| ptv3_extra_doubledrug_nc | combined | 0.518687 | 0.081984 | 0.073045 | 1.122378 | 0.675487 | 15155 | 1107 | 14048 |
| ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | 0.638126 | 0.092268 | 0.062722 | 1.471061 | 0.582101 | 5341 | 335 | 5006 |
| ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | 0.644530 | 0.047848 | 0.027306 | 1.752273 | 0.714732 | 17615 | 481 | 17134 |
| ptv3_extra_doubledrug_nature | combined | 0.657552 | 0.063506 | 0.035546 | 1.786568 | 0.683873 | 22956 | 816 | 22140 |

## Validation Notes

- Shell syntax checks passed for:
  - `scripts/0526_baseline4_task_specific_2gpu_parallel.sh`
  - `scripts/0521_baseline4_8gpu_parallel.sh`
  - `scripts/exp_08_extra_double_all_train_infer.sh`
  - `scripts/ptv3_experiment_common.sh`
- Python compile checks passed for:
  - `train.py`
  - `infer.py`
  - `model/fast_delta_model.py`
  - `model/fast_lightning.py`
  - `model/training_ready_lightning.py`
  - `scripts/report_extra_doubledrug_test_label_auprc.py`
  - `scripts/show_extra_results.py`
- Current training-ready extra double feature tables do not yet contain `test` / `test_label`; therefore the native `infer.py` grouped metric appears only after data regeneration. The grouped report above is computed by joining current predictions back to the updated raw CSV files with `feature_row_index`.
