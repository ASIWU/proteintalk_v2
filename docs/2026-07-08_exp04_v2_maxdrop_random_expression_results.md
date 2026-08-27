# exp_04_v2 Max-Drop Random Expression Results

- Generated: `2026-07-08T12:57:27+00:00`
- Final prefix: `20260708_exp04_v2_maxdrop_random_expr_clean`
- Current seed42 prefix: `20260708_exp04_v2`
- Task: `ptv3_main_singledrug` / `response`
- Winner artifact: `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_per_row_gene_permutation_seed42.npy`

## Validation
- Final and current seed42 manifests are present and match expected settings.

## Screen Selection
- Winner policy: `per_row_gene_permutation`
- Screen fold0 real-control AUPRC: `0.702725`
- Screen fold0 winner AUPRC: `0.654977`
- Screen fold0 drop vs real: `0.047748`

## 5-Fold Summary
| set | folds | mean AUPRC | std AUPRC | mean nAUPRC | mean AUROC | mean ACC |
|---|---:|---:|---:|---:|---:|---:|
| maxdrop_random | 5 | 0.592230 | 0.093408 | 5.036277 | 0.891713 | 0.903330 |
| current_seed42_random | 5 | 0.673665 | 0.087538 | 5.717265 | 0.905289 | 0.913943 |

## Comparison
- Final mean AUPRC: `0.592230`
- Current seed42 random mean AUPRC: `0.673665`
- Mean AUPRC drop vs current seed42 random: `0.081436`
- Final fold0 drop vs screen real-control reference: `0.047748`

## Fold Metrics
| fold | final AUPRC | current seed42 AUPRC | final-current | final nAUPRC | final AUROC | final count |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.654977 | 0.708677 | -0.053699 | 4.802262 | 0.912988 | 3534 |
| 1 | 0.528327 | 0.590959 | -0.062632 | 4.242157 | 0.890222 | 3549 |
| 2 | 0.658974 | 0.812005 | -0.153031 | 4.749112 | 0.891045 | 3589 |
| 3 | 0.679847 | 0.687516 | -0.007668 | 6.909826 | 0.911761 | 3537 |
| 4 | 0.439023 | 0.569171 | -0.130149 | 4.478030 | 0.852547 | 3570 |
