# 2026-07-09 12:07 HKT exp_04_v2 Global Mean/Std Random Expression Review

- Reviewed the exp_04_v2 random-control runner and confirmed `RANDOM_CONTROL_EXPRESSION_PATH` can safely override the current max-drop default without changing training code.
- Validated the global mean/std random artifact header against `feature_expression_matrix.npy`: both are `(18359, 10982)` float32; artifact metadata policy is `global_normal_clip`, with global mean `14.529332` and std `1.553119`.
- Launched the full unseen-drug 5-fold no-MSE run only through `gpu2` tmux using prefix `20260709_exp04_v2_global_meanstd_random_expr` and `GPU_IDS=0 DEVICES=1`.
- Added `scripts/report_exp04_v2_global_meanstd_random_expression.py` for CPU-side manifest validation and summary generation.
- Verified reporter output has zero validation errors; all five global mean/std manifests are `fit_completed/test_completed`, use `random_saved`, and point to `random_control_expression_global_normal_clip_seed42.npy`.
- Final 5-fold metrics: mean AUPRC `0.581342`, mean nAUPRC `4.951581`, mean AUROC `0.886019`, mean ACC `0.903646`.
- Main comparison: global mean/std random is `0.092323` mean AUPRC below the current seed42 random baseline and `0.010887` below the previous row-wise max-drop run.
