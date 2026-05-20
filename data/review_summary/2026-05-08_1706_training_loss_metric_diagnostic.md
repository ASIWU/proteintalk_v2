# 2026-05-08 17:06 HKT Training Loss / Metric Diagnostic

> 2026-05-08 17:36 HKT correction: the original interpretation in this note used the ported head names `bce_loss1` / `bce_loss2`, which made the loss contract confusing. The corrected trainer now defines `loss1 = expression MSE` and `loss2 = active task BCE`. See `data/review_summary/2026-05-08_1736_loss_contract_legacy_recheck.md` for the authoritative corrected review and rerun results.

## Scope

Ran bounded real-data multi-epoch diagnostics on the H200 GPU to check whether the current data and training pipelines produce sensible loss and metric trends.

Both runs used:

- model: `attention_v10_hetero_cls_ee`
- batch size: `2`
- epochs: `8`
- train batches per epoch: `50`
- validation batches per epoch: `50`
- test skipped
- accelerator: `gpu`
- devices: `1`

## Runs

### Main double-drug

Command shape:

```bash
python train.py \
  --task-name ptv3_main_doubledrug \
  --split-strategy pert_id_5fold_fold0 \
  --model-type attention_v10_hetero_cls_ee \
  --batch-size 2 \
  --max-epochs 8 \
  --limit-train-batches 50 \
  --limit-val-batches 50 \
  --skip-test \
  --accelerator gpu \
  --devices 1
```

Outputs:

- logs: `/tmp/proteintalk_v2_diagnostic_logs/20260508_diagnostic_ptv3_dd_8ep_50b/version_0/`
- checkpoints: `/tmp/proteintalk_v2_diagnostic_checkpoints/20260508_diagnostic_ptv3_dd_8ep_50b/`
- best checkpoint: `epoch=6.ckpt`
- best validation total loss: `1.2083688974380493`

Epoch-level trend:

| epoch | train_total | train_mse | train_bce1 | train_bce2 | val_total | val_mse | val_bce1 | val_bce2 | val_auroc2 | val_auprc2 | val_pcc_all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38.271 | 37.513 | 0.000 | 0.758 | 3.617 | 2.548 | 0.000 | 1.069 | 0.422 | 0.257 | 0.344 |
| 1 | 4.002 | 3.289 | 0.000 | 0.714 | 2.612 | 1.953 | 0.000 | 0.659 | 0.402 | 0.277 | 0.353 |
| 2 | 3.248 | 2.434 | 0.000 | 0.814 | 2.229 | 1.529 | 0.000 | 0.701 | 0.453 | 0.262 | 0.388 |
| 3 | 2.865 | 2.134 | 0.000 | 0.731 | 1.980 | 1.351 | 0.000 | 0.629 | 0.541 | 0.303 | 0.440 |
| 4 | 2.881 | 2.080 | 0.000 | 0.801 | 1.974 | 1.277 | 0.000 | 0.697 | 0.562 | 0.334 | 0.507 |
| 5 | 2.509 | 1.807 | 0.000 | 0.702 | 1.672 | 0.958 | 0.000 | 0.714 | 0.514 | 0.327 | 0.652 |
| 6 | 2.242 | 1.521 | 0.000 | 0.721 | 1.208 | 0.584 | 0.000 | 0.625 | 0.671 | 0.457 | 0.804 |
| 7 | 1.930 | 1.176 | 0.000 | 0.753 | 1.258 | 0.548 | 0.000 | 0.709 | 0.395 | 0.245 | 0.779 |

### Main single-drug

Command shape:

```bash
python train.py \
  --task-name ptv3_main_singledrug \
  --split-strategy random \
  --model-type attention_v10_hetero_cls_ee \
  --batch-size 2 \
  --max-epochs 8 \
  --limit-train-batches 50 \
  --limit-val-batches 50 \
  --skip-test \
  --accelerator gpu \
  --devices 1
```

Outputs:

- logs: `/tmp/proteintalk_v2_diagnostic_logs/20260508_diagnostic_ptv3_sd_8ep_50b/version_0/`
- checkpoints: `/tmp/proteintalk_v2_diagnostic_checkpoints/20260508_diagnostic_ptv3_sd_8ep_50b/`
- best checkpoint: `epoch=7.ckpt`
- best validation total loss: `1.1667859554290771`

Epoch-level trend:

| epoch | train_total | train_mse | train_bce1 | train_bce2 | val_total | val_mse | val_bce1 | val_bce2 | val_auroc | val_auprc | val_pcc_all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 40.043 | 39.552 | 0.490 | 0.000 | 3.332 | 2.385 | 0.947 | 0.000 | 0.530 | 0.244 | 0.370 |
| 1 | 3.612 | 3.077 | 0.535 | 0.000 | 2.727 | 1.848 | 0.879 | 0.000 | 0.503 | 0.321 | 0.397 |
| 2 | 2.734 | 2.375 | 0.359 | 0.000 | 2.302 | 1.503 | 0.799 | 0.000 | 0.515 | 0.287 | 0.447 |
| 3 | 2.262 | 2.069 | 0.194 | 0.000 | 2.420 | 1.509 | 0.911 | 0.000 | 0.496 | 0.313 | 0.533 |
| 4 | 2.435 | 2.005 | 0.430 | 0.000 | 1.885 | 1.164 | 0.721 | 0.000 | 0.479 | 0.267 | 0.610 |
| 5 | 2.157 | 1.737 | 0.419 | 0.000 | 1.906 | 0.992 | 0.914 | 0.000 | 0.555 | 0.272 | 0.763 |
| 6 | 2.000 | 1.572 | 0.428 | 0.000 | 1.369 | 0.588 | 0.782 | 0.000 | 0.532 | 0.261 | 0.783 |
| 7 | 1.608 | 1.264 | 0.344 | 0.000 | 1.167 | 0.470 | 0.697 | 0.000 | 0.539 | 0.246 | 0.771 |

## Label Audit

Current supervised classification heads are task-specific:

- `ptv3_main_doubledrug`: `PRISM1st_label_total` and `PRISM2nd_label_total` are empty in train/valid/test; `synergy` is populated.
- `ptv3_main_singledrug`: `PRISM1st_label_total` is populated; `synergy` is empty.

Full split counts:

- double-drug train: `1289` synergy labels, no PRISM labels
- double-drug valid: `142` synergy labels, no PRISM labels
- single-drug train: `12950` PRISM labels, no synergy labels
- single-drug valid: `1438` PRISM labels, no synergy labels

The bounded validation windows used here are imbalanced:

- double-drug first 100 valid anchors: `71` non-syn, `29` syn
- single-drug first 100 valid anchors: `76` non-responsive, `24` sensitive

## Interpretation

What looks healthy:

- The real data loader, graph model, optimizer, loss, validation, checkpointing, and TensorBoard logging all ran on GPU.
- Expression train/validation loss dropped strongly in both tasks.
- Expression correlation improved strongly:
  - double-drug `val/pcc_all`: `0.344 -> 0.804` at epoch 6, `0.779` at epoch 7
  - single-drug `val/pcc_all`: `0.370 -> 0.783` at epoch 6, `0.771` at epoch 7
- `val/mse_loss` dropped strongly:
  - double-drug `2.548 -> 0.584` at best epoch 6
  - single-drug `2.385 -> 0.470` at epoch 7

What does not yet look convincing:

- `bce_loss1` cannot be evaluated on main double-drug because task-1 labels are fully masked.
- `bce_loss2` cannot be evaluated on main single-drug because synergy labels are fully masked.
- Active classification metrics are noisy under the 100-sample validation cap:
  - double-drug synergy AUROC improved to `0.671` at epoch 6, then fell to `0.395` at epoch 7
  - single-drug response AUROC stayed near `0.5`
- Accuracy is mostly reflecting class imbalance in the capped validation window, not real discrimination.

## Conclusion

The training and data pipelines are functional for real GPU training, and the expression objective is learning. The current short diagnostics do not prove that the classification heads are learning robustly. For main double-drug, only `loss2` is active; for main single-drug, only `loss1` is active. A production-quality check should run larger validation coverage and probably tune class imbalance handling before judging AUROC/AUPRC.

## 2026-05-08 17:18 HKT Follow-up: Full Validation and Class-only Checks

Full-validation inference was run from the best mixed-objective checkpoints:

| run | checkpoint | active task | full valid AUROC | full valid AUPRC | full valid ACC | probability spread |
| --- | --- | --- | ---: | ---: | ---: | --- |
| double-drug mixed | `/tmp/proteintalk_v2_diagnostic_checkpoints/20260508_diagnostic_ptv3_dd_8ep_50b/epoch=6.ckpt` | synergy / `loss2` | 0.651 | 0.525 | 0.662 | `pred_synergy_prob` mean 0.3914, std 0.00012, all below 0.5 |
| single-drug mixed | `/tmp/proteintalk_v2_diagnostic_checkpoints/20260508_diagnostic_ptv3_sd_8ep_50b/epoch=7.ckpt` | response / `loss1` | 0.412 | 0.097 | 0.883 | `pred_response_prob` mean 0.0687, std 0.00008, all below 0.5 |

The full-validation result shows that classification accuracy is mostly the majority-class baseline. The mixed double-drug AUROC/AUPRC have some ranking signal, but the absolute probabilities are collapsed near a constant value. The mixed single-drug classifier is not discriminating.

Class-only diagnostic runs were also executed with `--no-mse-loss` for 8 epochs, 50 train batches, and 50 validation batches:

| run | active loss trend | validation loss trend | validation metric trend |
| --- | --- | --- | --- |
| double-drug class-only | train `bce_loss2`: 0.804, 0.773, 1.003, 0.732, 0.865, 0.788, 0.752, 0.777 | val `bce_loss2`: 0.763, 1.088, 0.712, 0.648, 0.628, 0.644, 0.624, 0.617 | capped val AUROC2 ends 0.624, AUPRC2 ends 0.449 |
| single-drug class-only | train `bce_loss1`: 0.779, 0.759, 0.445, 0.241, 0.645, 0.517, 0.619, 0.487 | val `bce_loss1`: 1.434, 1.298, 1.336, 1.450, 1.272, 0.864, 1.235, 1.184 | capped val AUROC stays around 0.43-0.48, AUPRC ends 0.293 |

Full-validation inference from the class-only checkpoints:

| run | checkpoint | active task | full valid AUROC | full valid AUPRC | full valid ACC | probability spread |
| --- | --- | --- | ---: | ---: | ---: | --- |
| double-drug class-only | `/tmp/proteintalk_v2_diagnostic_checkpoints/20260508_diagnostic_ptv3_dd_classonly_8ep_50b/epoch=7.ckpt` | synergy / `loss2` | 0.635 | 0.460 | 0.662 | `pred_synergy_prob` mean 0.3729, std 0.00003, all below 0.5 |
| single-drug class-only | `/tmp/proteintalk_v2_diagnostic_checkpoints/20260508_diagnostic_ptv3_sd_classonly_8ep_50b/epoch=5.ckpt` | response / `loss1` | 0.543 | 0.131 | 0.883 | `pred_response_prob` mean 0.0301, std 0.000006, all below 0.5 |

Updated conclusion:

- The real data pipeline and model dimensions are operational: both mixed-objective tasks trained on GPU and wrote checkpoints/logs without shape failures.
- Expression regression is the healthy part of the current pipeline: MSE drops and `pcc_all` rises strongly.
- Classification masking is behaving as coded: double-drug has no active `loss1`; single-drug has no active `loss2`.
- Classification learning is not yet reliable. The active heads produce near-constant probabilities on full validation, so ACC is dominated by the negative majority class and should not be interpreted as true classifier quality.
- Before trusting classification metrics, train with explicit imbalance handling such as task-specific positive weights, focal loss tuning, oversampling, stronger BCE weights, or staged classification training, and evaluate on the full validation split rather than capped validation batches.
