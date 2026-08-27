# 2026-07-08 exp_04_v2 Random Control-Proteome No-MSE Baseline Results

## Scope

This report summarizes `exp_04_v2`, a no-MSE single-drug ablation that replaces the real paired `control_expression` with a saved seed-42 random control proteome matrix.

Final run prefix:

`20260708_exp04_v2`

The run used W&B-disabled settings:

- `LOGGER_BACKEND=none`
- `LOG_TO_WANDB=0`
- `WANDB_MODE=disabled`
- `GPU_IDS=0`
- `DEVICES=1`

## Completion

Runtime summary:

`logs/20260708_exp04_v2_runtime_summary.tsv`

Completion counts:

| kind | count | failed |
|---|---:|---:|
| train | 5 | 0 |
| total | 5 | 0 |

The `gpu2` tmux session returned to a shell prompt after:

`[done] runtime summary: logs/20260708_exp04_v2_runtime_summary.tsv`

Runtime:

- Start: `2026-07-08T07:02:10Z` / `2026-07-08 15:02:10 HKT`
- End: `2026-07-08T07:10:41Z` / `2026-07-08 15:10:41 HKT`
- Total fold runtime: `511` seconds
- Mean fold runtime: `102.2` seconds

All five formal fold manifests have:

- `run_status=fit_completed`
- `test_status=test_completed`
- `have_mse_loss=false`
- `control_expression_mode=random_saved`

## Random Control Artifact

Saved artifact:

`data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.npy`

Metadata:

`data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.meta.json`

Artifact summary:

| field | value |
|---|---:|
| seed | 42 |
| shape | `(18359, 10982)` |
| control rows used for stats | 424 |
| fallback proteins | 97 |
| mean-fallback proteins | 28 |
| std-fallback proteins | 97 |
| clipped negatives | 0 |

Policy:

- One random control row is generated for every `feature_table` row.
- Rows are aligned by `perturb_row`: `random_control_expression_matrix[perturb_row]`.
- Each protein is sampled independently from real control-row `nanmean/nanstd`.
- Per-protein missing or zero std falls back to global control std.
- Missing per-protein mean falls back to global control mean.
- Negative sampled values are clipped to `0`.
- The same saved artifact is used for train, valid, and test.

## Experiment Setup

| field | value |
|---|---|
| script | `scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh` |
| task | `ptv3_main_singledrug` |
| split | `pert_stratified_5fold_fold0..4` |
| head | `response` |
| label | `PRISM1st_label_total` |
| MSE loss | disabled with `--no-mse-loss` |
| control expression | saved random seed-42 artifact |
| graph features | real |
| model | `fast_delta` |
| LR | `2e-4` |
| batch size | `256` |
| max epochs | `50` |
| precision | `bf16-mixed` |

## Evaluation Notes

- Metrics are the row-level `test_results` saved in each fold `run_manifest.json`.
- AUROC, AUPRC, nAUPRC, ACC, and loss2 mean values are unweighted means across five folds.
- The mean-table baseline uses aggregate positives divided by aggregate count, matching the convention used in `docs/2026-06-23_update0623_exp01_08_results.md`.
- The real-control comparison below uses the update_0623 exp04 no-MSE row from `20260623_130336_update0623_nowandb`.

## Fold Mean Results

| exp | task | split | AUROC | AUPRC | baseline | nAUPRC | ACC | loss2 | count | pos | neg | best epochs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exp04_v2 | single no-MSE random-control baseline | mean5 | 0.905289 | 0.673665 | 0.119242 | 5.717265 | 0.913943 | 0.288991 | 17779 | 2120 | 15659 | 1,1,2,3,10 |

## Comparison With Real-Control exp04

| run | control expression | AUROC | AUPRC | baseline | nAUPRC | ACC | best epochs |
|---|---|---:|---:|---:|---:|---:|---|
| exp04 real-control no-MSE (`20260623_130336_update0623_nowandb`) | real paired control | 0.900805 | 0.669301 | 0.119242 | 5.671871 | 0.912080 | 3,1,2,3,6 |
| exp04_v2 random-control no-MSE (`20260708_exp04_v2`) | saved seed-42 random control | 0.905289 | 0.673665 | 0.119242 | 5.717265 | 0.913943 | 1,1,2,3,10 |
| delta random - real | - | +0.004484 | +0.004364 | +0.000000 | +0.045394 | +0.001863 | - |

## Fold Detail

| exp | fold | AUROC | AUPRC | baseline | nAUPRC | ACC | loss2 | count | pos | neg | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp04_v2 | 0 | 0.927187 | 0.708677 | 0.136389 | 5.195983 | 0.919921 | 0.250434 | 3534 | 482 | 3052 | 1 |
| exp04_v2 | 1 | 0.908305 | 0.590959 | 0.124542 | 4.745053 | 0.887856 | 0.254310 | 3549 | 442 | 3107 | 1 |
| exp04_v2 | 2 | 0.934226 | 0.812005 | 0.138757 | 5.851977 | 0.919476 | 0.265089 | 3589 | 498 | 3091 | 2 |
| exp04_v2 | 3 | 0.897202 | 0.687516 | 0.098388 | 6.987766 | 0.926774 | 0.251411 | 3537 | 348 | 3189 | 3 |
| exp04_v2 | 4 | 0.859528 | 0.569171 | 0.098039 | 5.805545 | 0.915686 | 0.423711 | 3570 | 350 | 3220 | 10 |

## Fold Delta vs Real-Control exp04

| fold | delta AUROC | delta AUPRC | delta nAUPRC | delta ACC | random best epoch | real best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.016047 | +0.005952 | +0.043637 | +0.006508 | 1 | 3 |
| 1 | -0.006198 | -0.022435 | -0.180138 | -0.001972 | 1 | 1 |
| 2 | +0.002950 | +0.008696 | +0.062666 | +0.004737 | 2 | 2 |
| 3 | +0.006191 | +0.033310 | +0.338551 | +0.004523 | 3 | 3 |
| 4 | +0.003431 | -0.003701 | -0.037747 | -0.004482 | 10 | 6 |

## Interpretation

- The random-control no-MSE run did not degrade the update_0623 no-MSE baseline. Mean AUPRC is `+0.004364` and mean AUROC is `+0.004484` relative to real paired controls.
- Fold-level movement is small and mixed: folds 0, 2, and 3 improve in AUPRC; folds 1 and 4 decline.
- Because MSE is disabled, the control proteome is not supervised through the reconstruction loss. Under this specific no-MSE single-drug setup, the response head appears to get little incremental benefit from the biologically paired real control expression.
- This comparison should be read only against the same update_0623 exp04 setup. It is not directly comparable to older selected-suite reports that used different data artifacts, tuning choices, or extra grouped evaluations.

## Artifacts

- Runtime summary: `logs/20260708_exp04_v2_runtime_summary.tsv`
- Checkpoints: `checkpoints/20260708_exp04_v2_exp04_v2_random_no_mse_fold{0..4}/`
- Random control matrix: `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.npy`
- Random control metadata: `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.meta.json`
- Review summary: `data/review_summary/2026-07-08_1458_exp04_v2_random_baseline_review.md`
