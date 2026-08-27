# exp_04_v2 Global Mean/Std Random Expression Results

- Generated: `2026-07-09T04:06:52+00:00`
- Global mean/std prefix: `20260709_exp04_v2_global_meanstd_random_expr`
- Current seed42 baseline prefix: `20260708_exp04_v2`
- Max-drop comparison prefix: `20260708_exp04_v2_maxdrop_random_expr_clean`
- Task: `ptv3_main_singledrug` / `response`
- Split: `pert_stratified_5fold_fold0..4`
- Artifact: `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_global_normal_clip_seed42.npy`

## Validation
- All expected manifests are present and match the global mean/std random-expression setting.

## Artifact
- Policy: `global_normal_clip`
- Global control mean: `14.529332`
- Global control std: `1.553119`
- Shape: `[18359, 10982]`
- Dtype: `float32`

## 5-Fold Summary
| set | folds | mean AUPRC | std AUPRC | mean nAUPRC | mean AUROC | mean ACC |
|---|---:|---:|---:|---:|---:|---:|
| global_meanstd_random | 5 | 0.581342 | 0.114177 | 4.951581 | 0.886019 | 0.903646 |
| current_seed42_random | 5 | 0.673665 | 0.087538 | 5.717265 | 0.905289 | 0.913943 |
| maxdrop_random | 5 | 0.592230 | 0.093408 | 5.036277 | 0.891713 | 0.903330 |

## Comparison
- Global mean/std mean AUPRC: `0.581342`
- Current seed42 random mean AUPRC: `0.673665`
- Global minus current seed42 mean AUPRC: `-0.092323`
- Max-drop random mean AUPRC: `0.592230`
- Global minus max-drop mean AUPRC: `-0.010887`
- Fold0 screen real-control AUPRC: `0.702725`
- Fold0 screen global mean/std AUPRC: `0.659943`
- Fold0 screen global drop vs real: `0.042782`

## Fold Metrics
| fold | global AUPRC | current seed42 AUPRC | max-drop AUPRC | global-current | global-maxdrop | global nAUPRC | global AUROC | global ACC | count |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.659943 | 0.708677 | 0.654977 | -0.048734 | 0.004966 | 4.838670 | 0.911880 | 0.902943 | 3534 |
| 1 | 0.430835 | 0.590959 | 0.528327 | -0.160124 | -0.097493 | 3.459350 | 0.851887 | 0.883911 | 3549 |
| 2 | 0.680870 | 0.812005 | 0.658974 | -0.131134 | 0.021896 | 4.906915 | 0.895767 | 0.906659 | 3589 |
| 3 | 0.681689 | 0.687516 | 0.679847 | -0.005826 | 0.001842 | 6.928549 | 0.916277 | 0.916596 | 3537 |
| 4 | 0.453375 | 0.569171 | 0.439023 | -0.115797 | 0.014352 | 4.624420 | 0.854284 | 0.908123 | 3570 |
