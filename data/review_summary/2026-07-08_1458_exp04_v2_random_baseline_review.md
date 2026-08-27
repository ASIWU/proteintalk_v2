# 2026-07-08 14:58 HKT exp_04_v2 Random Baseline Review

## Scope

- Reviewed the fast PTV3 dataset/training/inference path for introducing a saved random control proteome baseline.
- Compared the new launcher against `scripts/exp_04_single_no_mse_5fold.sh` and the shared `scripts/ptv3_experiment_common.sh` interface.

## Implementation Summary

- Added CPU generator `scripts/generate_random_control_proteome.py`.
- Added `control_expression_mode={real,random_saved}` and `--random-control-expression-path` to fast training and inference.
- `random_saved` control expressions are aligned by `perturb_row`, so every feature-table row receives its own pre-sampled random control vector.
- Training/inference manifests record the control-expression mode, resolved artifact path, and compact artifact metadata.
- Added `scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh` for the no-MSE random-control 5-fold baseline.

## Review Notes

- The implementation keeps the existing real-control behavior as the default.
- The random matrix is validated for 2D shape, exact match to `feature_expression_matrix.npy`, and `float32` dtype before use.
- Inference checkpoint validation includes the control-expression mode/path to avoid accidentally evaluating a random-control checkpoint with real controls.
- The experiment launcher checks for the saved artifact but does not generate it inside the GPU session.

## Verification

- Generated the seed-42 artifact in a CPU shell:
  - shape `(18359, 10982)`;
  - control rows `424`;
  - fallback proteins `97`;
  - clipped negatives `0`.
- Confirmed artifact row count matches `feature_table`, meta seed is `42`, and no sampled value is negative.
- Checked a 64-column sample of generated means/stds against real control-row target stats.
- Confirmed a fast dataset item in `random_saved` mode returns `random_control_expression_matrix[perturb_row]`.
- `python -m py_compile train.py infer.py dataset/training_ready_fast_dataset.py scripts/generate_random_control_proteome.py` passed.
- `bash -n scripts/ptv3_experiment_common.sh scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh` passed.
- GPU smoke `20260708_exp04_v2_smoke` completed on tmux `gpu2` fold0 with `fit_completed/test_completed`, `have_mse_loss=false`, and `control_expression_mode=random_saved`.
- Full run `20260708_exp04_v2` completed on tmux `gpu2` for folds `0 1 2 3 4`.
- All five formal manifests have `fit_completed/test_completed`, `have_mse_loss=false`, and `control_expression_mode=random_saved`.
- Formal mean5 results: AUROC `0.905289`, AUPRC `0.673665`, aggregate baseline `0.119242`, nAUPRC `5.717265`, ACC `0.913943`, total count `17779`, positives `2120`, negatives `15659`, best epochs `1,1,2,3,10`.
- Results report added at `docs/2026-07-08_exp04_v2_random_control_baseline_results.md`.
