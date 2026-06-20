# PTV1 Frozen Cell LLM Fine-tune Search Report

Run date: 2026-06-08 HKT

Formal prefix: `20260608_ptv1_cell_llm_tune_v1`

Baseline reference: `20260608_ptv1_cell_llm_v1`, exp_11 AUPRC `0.893452`

Official exp_13 interpretation: evaluate the exp_11-selected parameterization with two settings:

1. direct inference from the exp_11 best checkpoint;
2. train with the exp_11 parameters on all PTV1 data, then infer exp_13.

The cross-candidate exp_13 leaderboard is retained only as diagnostic analysis. It is not the official exp_13 selection criterion because it can select a different architecture from exp_11.

## Scope

- Searched 32 frozen Cell LLM candidates.
- Every candidate ran:
  - exp_11 fixed experiment type random split;
  - exp_12 unseen-drug `pert_id` folds 0-4;
  - exp_13 `ptv1_extra_singledrug` direct from exp_11 best checkpoint;
  - exp_13 all-PTV1 train replay using the exp_11 best epoch plus one epoch count, then extra inference from `last.ckpt`.
- Common formal settings:
  - `CELL_LLM_MODE=frozen`;
  - `BEST_CKPT_METRIC=valid_auprc`;
  - `LOGGER_BACKEND=none`, `LOG_TO_WANDB=0`, `PROGRESS_BAR=0`;
  - full batches: `limit_train_batches=1.0`, `limit_val_batches=1.0`, `limit_test_batches=1.0`;
  - GPU training/inference on one H200 device via `GPU_IDS=0`.

Runtime window: 2026-06-08 13:19:08 to 17:03:05 HKT. Total wall time was about 3.73 hours.

## Final Selections

| Objective | Selected candidate | Metric | Value | Notes |
|---|---|---:|---:|---|
| exp_11 best | `mse050_drop010` | exp_11 AUPRC | `0.926324` | `+0.032872` over baseline exp_11 |
| exp_12 best | `mse050_target_pdi` | exp_12 mean5 AUPRC | `0.584798` | `+0.073163` over baseline exp_12 mean5 |
| exp_13 tied to exp_11, direct | `mse050_drop010` | extra direct AUPRC | `0.561480` | same parameters and weights as exp_11 best |
| exp_13 tied to exp_11, all_train | `mse050_drop010` | extra all_train AUPRC | `0.579478` | same parameters as exp_11 best, retrained on all PTV1 |

Under this corrected interpretation, exp_13 uses `mse050_drop010`, not `mse025_graph_off`. The earlier graph-off candidate was the best result in a cross-candidate exp_13 diagnostic leaderboard, but it does not satisfy the requirement that exp_13 evaluates the exp_11-selected parameterization.

## Exp_13 for Exp_11 Best Parameters

`mse050_drop010` parameters:

- `MSE_WEIGHT=0.50`
- `DROPOUT=0.10`
- graph enabled: `GRAPH_FEATURE_MODE=real`, `GRAPH_STRUCTURAL_RP=1`, `GRAPH_DRUG_CONCAT=1`, `GRAPH_LOGIT_SCALE=2.0`
- `MSE_TARGET_MODE=all`
- `CELL_LLM_MODE=frozen`

| exp_13 setting | AUPRC | AUROC | nAUPRC | rows |
|---|---:|---:|---:|---:|
| direct from exp_11 best checkpoint | `0.561480` | `0.580062` | `1.092882` | 218 |
| all_train with exp_11 parameters | `0.579478` | `0.603900` | `1.127913` | 218 |

## Top Exp_11 Candidates

| Candidate | exp_11 AUPRC | exp_12 mean5 | exp_13 best mode | exp_13 best AUPRC |
|---|---:|---:|---|---:|
| `mse050_drop010` | `0.926324` | `0.524204` | all_train | `0.579478` |
| `mse025_pcep_additive` | `0.920494` | `0.566838` | all_train | `0.586986` |
| `mse025_pcep_off` | `0.920252` | `0.560799` | all_train | `0.565346` |
| `mse025` | `0.919742` | `0.560868` | all_train | `0.587619` |
| `mse025_label_smooth005` | `0.917003` | `0.523133` | all_train | `0.589597` |
| `mse025_no_plate_cov` | `0.915377` | `0.579369` | direct | `0.572792` |
| `pos_auto_mse050` | `0.910329` | `0.568242` | all_train | `0.581378` |
| `mse025_ctrl_drop005` | `0.910006` | `0.571625` | direct | `0.573412` |

## Top Exp_12 Candidates

| Candidate | exp_11 AUPRC | exp_12 mean5 AUPRC | exp_12 std | exp_13 best AUPRC |
|---|---:|---:|---:|---:|
| `mse050_target_pdi` | `0.893894` | `0.584798` | `0.054942` | `0.582517` |
| `mse025_target_expr_pdi` | `0.900940` | `0.581388` | `0.083893` | `0.591647` |
| `mse025_no_plate_cov` | `0.915377` | `0.579369` | `0.097652` | `0.572792` |
| `mse050_drop020` | `0.904759` | `0.576603` | `0.085287` | `0.581371` |
| `mse025_graph_logit1` | `0.899613` | `0.576119` | `0.091236` | `0.582374` |
| `mse025_graph_logit4` | `0.888153` | `0.575851` | `0.059068` | `0.595581` |
| `mse050_lr3e4` | `0.907171` | `0.573283` | `0.054860` | `0.583502` |
| `mse025_drop020` | `0.864835` | `0.573280` | `0.075931` | `0.593277` |

## Diagnostic Cross-candidate Exp_13 Leaderboard

| Candidate | exp_11 AUPRC | exp_12 mean5 AUPRC | exp_13 best mode | exp_13 best AUPRC |
|---|---:|---:|---|---:|
| `mse025_graph_off` | `0.890020` | `0.527913` | direct | `0.604631` |
| `mse025_drop010` | `0.888632` | `0.568124` | direct | `0.604476` |
| `mse025_graph_logit4` | `0.888153` | `0.575851` | all_train | `0.595581` |
| `mse025_target_expr_pdi` | `0.900940` | `0.581388` | direct | `0.591647` |
| `mse025_lr3e4` | `0.889428` | `0.551117` | all_train | `0.590139` |
| `mse025_label_smooth005` | `0.917003` | `0.523133` | all_train | `0.589597` |
| `mse025` | `0.919742` | `0.560868` | all_train | `0.587619` |
| `mse025_pcep_additive` | `0.920494` | `0.566838` | all_train | `0.586986` |

This table is diagnostic only. It compares exp_13 performance across all searched candidates and therefore can select a different parameterization from exp_11.

## Audit

Formal artifact audit passed.

| Check | Result |
|---|---:|
| Candidate rows in consolidated TSV | 32 |
| Candidates selectable for exp_13 | 22 |
| Candidates below exp_11 threshold | 10 |
| Training manifests | 224 / 224 |
| exp_11 random split manifests | 32 / 32 |
| exp_12 complete five-fold candidates | 32 / 32 |
| exp_13 all-train manifests | 32 / 32 |
| Inference manifests | 64 / 64 |
| exp_13 prediction files with 218 rows | 64 / 64 |
| Runtime summary rows | 288 / 288 |
| Runtime rows with nonzero status | 0 |
| Checkpoint directories with `.ckpt` files | 224 / 224 |
| Manifest checkpoint paths present on disk | 224 / 224 |

All formal training manifests had `dataset_group=ptv1`, `cell_llm_mode=frozen`, `accelerator=gpu`, and full-batch limits equal to `1.0`.

## Artifacts

- Full markdown table: `logs/20260608_ptv1_cell_llm_tune_v1_fine_tune_results.md`
- Full TSV table: `outputs/20260608_ptv1_cell_llm_tune_v1_fine_tune_results.tsv`
- Runtime summaries: `logs/20260608_ptv1_cell_llm_tune_v1_*_runtime_summary.tsv`
- Checkpoints: `checkpoints/20260608_ptv1_cell_llm_tune_v1_*`
- Extra predictions: `outputs/20260608_ptv1_cell_llm_tune_v1_*/ptv1_extra_singledrug/predictions.parquet`

## Recommendation

- Use `mse050_drop010` when optimizing strictly for exp_11.
- Use `mse050_target_pdi` when optimizing exp_12 unseen-drug mean5.
- For official exp_13 evaluation tied to exp_11, use `mse050_drop010` for both direct and all_train settings.
