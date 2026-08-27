# 2026-07-09 12:10 HKT exp_04_v2 Default Global Mean/Std Setting Review

- Updated `scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh` to make the completed global mean/std random-expression experiment the default exp_04_v2 setting.
- The runner now defaults to `random_control_expression_global_normal_clip_seed42.npy`, keeps `CONTROL_EXPRESSION_MODE=random_saved`, and keeps the no-MSE unseen-drug 5-fold loop unchanged.
- The script title and `EXPERIMENT_SET_NAME` were renamed from max-drop wording to global mean/std wording to avoid future run-name confusion.
- Updated `scripts/README_ptv3_experiments.md` so the generation command includes `--policy global_normal_clip` and the default artifact path matches the runner.
- Verification passed with `bash -n scripts/ptv3_experiment_common.sh scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh`.
- Confirmed the default artifact and its `.meta.json` exist under `data/training_ready/ptv3/tasks/ptv3_main_singledrug/`.
