# 2026-05-27 MSE Gap Experiment Results

## Goal

Increase the unseen single-drug 5-fold gap between the base model and the w/o MSE-loss ablation to more than 5 AUPRC points, without changing data or data-processing code.

## Code Added

All new switches are default-off, so the maintained h512 baseline remains unchanged unless the corresponding env/CLI flags are set.

- `response_base_logit_scale`: scales the ordinary fusion-hidden classification head.
- `zero_init_delta_head`: initializes the expression delta decoder output layer to zero.
- `delta_logit_learnable`: lets the delta-response logit scale be learned.
- `control_expression_dropout`: training-only inverted dropout on control expression.
- `mse_pretrain_epochs` / `mse_pretrain_bce_weight`: optional MSE-first warmup.
- `delta_teacher_loss_weight`: trains the delta-response branch on true perturbed-control expression deltas while inference still uses predicted deltas.
- `scripts/run_mse_gap_delta_screen_2gpu.sh`: extended with the screened method presets.

## Full 5-Fold Results

Current h512 reference:

| model | AUROC | AUPRC | nAUPRC |
| --- | ---: | ---: | ---: |
| h512 baseline | 0.903166 | 0.677846 | 5.723701 |
| h512 w/o MSE | 0.893265 | 0.651609 | 5.499016 |
| gap | +0.009901 | +0.026237 | +0.224685 |

Best conservative candidate, `learn_delta_b075_g2_init01`:

| variant | AUROC | AUPRC | nAUPRC | fold AUPRCs |
| --- | ---: | ---: | ---: | --- |
| with MSE | 0.895665 | 0.652089 | 5.510165 | 0.55019, 0.76409, 0.56253, 0.74525, 0.63838 |
| w/o MSE | 0.889544 | 0.644282 | 5.434896 | 0.52469, 0.77290, 0.51636, 0.75025, 0.65721 |
| gap | +0.006121 | +0.007807 | +0.075269 | - |

Extreme expression-mediated pressure test, `zdelta_b0_g0_dim32`:

| variant | AUROC | AUPRC | nAUPRC | fold AUPRCs |
| --- | ---: | ---: | ---: | --- |
| with MSE | 0.726397 | 0.244446 | 2.049123 | 0.22358, 0.30363, 0.19500, 0.21754, 0.28248 |
| w/o MSE | 0.508356 | 0.120776 | 1.015213 | 0.12165, 0.12026, 0.10938, 0.11300, 0.13960 |
| gap | +0.218040 | +0.123670 | +1.033910 | - |

## Interpretation

The 5-point gap target is reachable only by making classification almost entirely depend on the predicted expression-delta branch. That creates a large and real MSE dependency, but the resulting base model is not usable because with-MSE AUPRC drops from the h512 baseline `0.677846` to `0.244446`.

Every conservative attempt that preserved ordinary hidden/graph classification capacity failed to keep a 5-point full 5-fold gap. The best such full run (`learn_delta_b075_g2_init01`) had only `+0.007807` AUPRC gap and also underperformed the current h512 baseline.

## Recommendation

Do not replace the current h512 baseline with any MSE-gap variant from this round. The current h512 baseline remains the best tradeoff for effect and ablation credibility. The extreme `zdelta_b0_g0_dim32` result can be cited only as a diagnostic showing that MSE can be made essential, not as a model candidate.
