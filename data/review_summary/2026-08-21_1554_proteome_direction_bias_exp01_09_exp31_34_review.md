# 2026-08-21 15:54 HKT Proteome Direction Bias Review (Exp01-09, Exp31-34)

## Scope

- Request: determine whether the current model can predict only perturbation-upregulated proteins and fails on downregulated proteins, by checking Exp01-09 and Exp31-34.
- Reviewed the formal experiment documents, checkpoint/output manifests, model/loss/reporting code, training-ready expression matrices, and published Exp34 expression artifacts.
- Recomputed signed expression predictions in the user-designated `gpu1` tmux session with the original checkpoints, split strategies, task axes, and inference configurations. No checkpoint was trained or modified.
- Existing Exp01-09 and Exp31-33 outputs did not persist `expression_pred.npy`; their proteome direction was therefore reproduced temporarily with `--save-expression-pred`. Exp34 already persisted its complete `28 x 11092` prediction matrix.

## Executive conclusion

The literal statement "the model cannot output a downregulation" is false: the final delta decoder is unconstrained, Exp07 produces `21.33%` predictions below `-0.10`, and even the strongly biased runs emit some small negative deltas.

The practical concern is nevertheless confirmed:

1. On Exp01-03/05/06, where true perturbed proteomes exist, downregulation recall is severely below upregulation recall. At `|delta log1p| > 0.10`, down recall is `3.59%-22.59%`, versus `45.53%-58.63%` for upregulation.
2. Exp09 epoch 2 and its downstream Exp31-34 runs are nearly one-directional for meaningful deltas. On proteins with a measured control, only `0.03%-0.14%` of predictions are below `-0.10`, while `54.42%-70.71%` exceed `+0.10`.
3. The dominant causal defect is asymmetric missing-value handling: a missing control is converted to zero, while MSE validity checks only whether the perturbed target is finite. This creates large artificial positive training targets; the inverse missingness pattern is excluded from loss.
4. The Exp34 top-change report amplifies the defect by ranking deltas after replacing missing controls with zero. Of its 560 published top-20 changes, all are positive and 516 (`92.14%`) have an originally missing control measurement.

## Audit definition and coverage

- Scientific perturbation delta: `predicted perturbed expression - matched control expression`.
- Accuracy metrics use only protein/sample pairs where both observed control and observed perturbed expression are finite.
- Prediction-only metrics for unlabeled/query-only experiments use only proteins with finite control expression. This conservative mask excludes the obvious `NaN -> 0` reporting artifact.
- Primary threshold: `|delta log1p| > 0.10`; additional checks used `0`, `0.25`, and `0.50`.
- Exp01-06: all five formal folds; Exp01/03/04/05 used 512 test rows per fold, Exp02 used all 1,922 available test rows, and Exp06 used all 1,791 available test rows. This evaluates 12.0-15.8 million finite protein/sample pairs per experiment.
- Exp07: 512 rows from each of six formal external tasks (`3,072` total).
- Exp08: 512 rows from each of three formal external tasks (`1,536` total).
- Exp09: six single-drug tasks at formal epoch 2 and three double-drug tasks at formal epoch 8, 512 rows each (`4,608` total).
- Exp31: all four endpoints, both zero-shot and fine-tuned, 1,024 rows each (`8,192` total).
- Exp32: both device tasks, 2,048 rows each (`4,096` total).
- Exp33: all three tissues, 2,048 rows each (`6,144` total).
- Exp34: all 28 published rows from both target branches.

## Ground-truth directional results: Exp01-06

All fractions and recalls below use `|delta log1p| > 0.10` and finite observed control/perturb pairs.

| Exp | Evaluated rows | True down / up | Predicted down / up | Down recall | Up recall | Predicted mean delta | True mean delta | Model MSE | Copy-control MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 2,560 | 34.86% / 27.26% | 10.87% / 43.05% | 11.83% | 54.50% | +0.5386 | -0.0286 | 1.7750 | 0.1419 |
| 02 | 1,922 | 35.34% / 27.61% | 8.03% / 45.26% | 8.20% | 55.54% | +0.6494 | -0.0296 | 2.0736 | 0.1460 |
| 03 | 2,560 | 34.58% / 27.00% | 5.69% / 45.93% | 6.50% | 57.04% | +0.6211 | -0.0280 | 2.0390 | 0.1381 |
| 04 | 2,560 | 34.86% / 27.26% | 0.00% / 0.00% | 0.00% | 0.00% | -0.0002 | -0.0286 | 0.1423 | 0.1419 |
| 05 | 2,560 | 34.86% / 27.26% | 3.44% / 47.74% | 3.59% | 58.63% | +0.6107 | -0.0286 | 1.8888 | 0.1419 |
| 06 | 1,791 | 25.64% / 20.04% | 18.04% / 29.37% | 22.59% | 45.53% | +0.2496 | -0.0177 | 0.8775 | 0.0688 |

Interpretation:

- The observed test data are not up-only; true downregulated proteins are slightly more common than true upregulated proteins.
- Exp01-03/05/06 have a real positive prediction bias and poor downregulation recovery.
- On finite observed pairs, these checkpoints are much worse at expression reconstruction than simply copying the control (`model MSE / copy-control MSE` about `12.5x-14.8x`). Their classification results must not be interpreted as validated perturbed-proteome results.
- Exp04 is the no-MSE ablation. Its expression decoder has no direct expression supervision, so the near-zero random-scale deltas are not a valid proteome prediction result.

## Prediction direction only: Exp07-09 and Exp31-34

These experiments do not contain measured post-treatment proteome ground truth for their query rows. The table therefore describes model output direction only and makes no accuracy claim. Fractions are restricted to measured control proteins.

| Exp/configuration | Evaluated rows | Pred down `< -0.10` | Pred up `> +0.10` | Pred down `< -0.25` | Pred up `> +0.25` | Mean predicted delta |
|---|---:|---:|---:|---:|---:|---:|
| 07 | 3,072 | 21.33% | 41.41% | 9.46% | 34.84% | +0.7992 |
| 08 | 1,536 | 6.28% | 46.30% | 1.75% | 36.55% | +0.7356 |
| 09, epoch 2 single | 3,072 | 0.08% | 55.51% | 0.00% | 47.16% | positive-biased |
| 09, epoch 8 double | 1,536 | 10.05% | 43.13% | 1.34% | 36.04% | positive-biased |
| 09, combined formal selection | 4,608 | 3.31% | 51.50% | 0.43% | 43.56% | +0.7262 |
| 31, 4 endpoints x zero-shot/FT | 8,192 | 0.14% | 70.71% | 0.03% | 64.99% | +1.5217 |
| 32 | 4,096 | 0.05% | 54.42% | 0.00% | 46.37% | +0.8659 |
| 33 | 6,144 | 0.03% | 57.25% | 0.00% | 48.07% | +0.6954 |
| 34, complete published output | 28 | 0.05% | 54.83% | 0.00% | 46.38% | +0.8199 |

Additional interpretation:

- Exp07 disproves an absolute architectural inability to emit downregulation, but it has no post-treatment truth with which to establish correctness.
- Exp09 epoch 2 is the important failure mode: it is the checkpoint inherited by Exp31-34. The epoch-8 double-drug checkpoint is less extreme but remains substantially up-biased.
- Exp31 fine-tuning is explicitly label-only/no-MSE because there is no post-treatment expression. It modifies shared representation layers without any expression-preservation objective, so its emitted proteome must be considered uncontrolled.
- Exp32-34 are virtual/OOD inference tasks. They can expose the output bias but cannot validate either up- or downregulation accuracy.

## Root cause evidence

### 1. The decoder is not sign constrained

- `model/fast_delta_model.py:732-738` ends the delta decoder with an ordinary linear layer; there is no ReLU, sigmoid, absolute value, or clamp.
- `model/fast_delta_model.py:1084-1088` computes `expression_pred = control_expression + delta_scale * delta` in residual mode.
- Therefore negative deltas are representable. The failure is learned/objective-driven rather than a hard architectural restriction.

### 2. Asymmetric NaN handling creates artificial positive targets

- `model/fast_delta_model.py:958` maps missing control expression to zero before the residual prediction.
- `model/fast_lightning.py:409-426` declares an MSE element valid using only `torch.isfinite(true)`. It does not require the matched control to be finite.
- Consequence:
  - control `NaN`, perturb finite: included in MSE with a residual baseline of zero, so the target behaves like a large positive delta equal to the absolute log1p intensity;
  - control finite, perturb `NaN`: excluded from MSE completely.

Measured training-data effect:

| Training task | True down / up among finite pairs (`|delta|>0.10`) | Control-missing share of MSE-valid elements | Mean true delta when both finite | Mean target at control-missing/perturb-finite positions | Effective supervised target-delta mean |
|---|---:|---:|---:|---:|---:|
| main single-drug | 32.02% / 30.76% | 6.31% | -0.00254 | +12.8864 | +0.8114 |
| main double-drug | 31.37% / 29.67% | 6.00% | -0.00408 | +12.8623 | +0.7680 |

Thus the biological finite-pair target is centered near zero and is not up-biased, while the actual supervised contract is strongly positive-biased.

The inverse missingness is also material: in the single-drug training set, `4.50%` of all matrix elements are control-finite/perturb-missing and receive no expression loss. Missing proteomics values are not proof of downregulation, but this asymmetric contract can preferentially discard proteins that fall below detection after treatment while strongly supervising newly detected proteins as large positive deltas.

### 3. Loss and checkpoint selection do not protect directionality

- The expression objective is ordinary absolute-expression MSE plus classification BCE (`model/fast_lightning.py:257-276`); there is no delta-sign, balanced up/down, or direction-recall objective.
- Exp01-03/05/06 use MSE weight `0.25`; Exp09 uses `0.5`, with inactive-label expression weight `0.2`.
- Exp01-08 select checkpoints by validation task AUPRC, not a directional expression metric. Exp09 publishes epoch 2/8 based on downstream classification selection. The fast validation collector reports classification metrics but no downregulation recall.
- Exp04 has MSE disabled. Exp31 fine-tuning also has MSE disabled. Their expression outputs are not directly supervised in those runs.

### 4. Exp34 reporting makes the symptom look even stronger

- `scripts/report_exp34_update0819_ood_epoch2.py:150-157` converts control NaNs to zero.
- `scripts/report_exp34_update0819_ood_epoch2.py:198-216` ranks the largest absolute deltas without excluding originally missing controls.
- The 14 matched controls have 3,836-4,655 missing proteins each; across Exp34, `38.21%` of control-axis positions are missing.
- All 560 published top-20 changes are positive; 516 (`92.14%`) came from originally missing control positions. Even after restricting to finite controls, however, all per-row top-20 absolute changes remain positive, confirming that reporting is an amplifier rather than the sole cause.

## Recommended correction order

1. **Fix the scientific validity mask first.** Carry a control-observed mask and compute expression/delta loss only where both control and perturbed target are finite. If newly detected proteins should be modeled, treat detection/missingness as a separate explicitly supervised task; do not equate missing control with biological abundance zero.
2. **Fix expression reporting.** Preserve the original control-missing mask; exclude unmeasured controls from signed-delta ranking by default; publish separate top-up and top-down tables and include their eligible denominators.
3. **Train and monitor delta direction.** Add finite-pair delta reconstruction (preferably robust loss), sign-balanced up/down sampling or weighting, and validation metrics at fixed thresholds: predicted up/down fraction, up/down recall, sign recall, delta PCC, and MSE skill versus copy-control.
4. **Select a proteome-valid checkpoint.** Do not use classification AUPRC alone when publishing perturbed proteomes. Require a checkpoint that improves over copy-control and does not collapse down recall.
5. **Do not interpret no-MSE outputs as proteome predictions.** This applies to Exp04 and Exp31 fine-tuned checkpoints unless an explicit expression-preservation/freeze policy is added and validated.
6. **Retrain after the mask fix.** Post-hoc thresholding cannot recover missing negative signal from an already biased decoder. Re-run at least Exp01/02/03/06 and evaluate the same signed metrics before using the retrained epoch in Exp31-34.

## Repository impact

- Added this review record only.
- No production code, checkpoint, formal output, split, or training-ready artifact was changed.
- Temporary inference matrices and audit helpers were used only for diagnosis and are not repository deliverables.
- The documented old-code reference `/mnt/shared-storage-user/beam/wuhao/H100/proteintalk/ProteinTalkv2` was not present in the current execution environment, so the review establishes the defect in the current repository but does not assign when it was introduced.

## 2026-08-21 16:17 HKT follow-up: why Exp09 epoch 2 and Exp31-34 are especially severe

### Exp09 epoch 2 is an early-training checkpoint, not an isolated broken epoch

The same 512 rows from `ptv3_extra_singledrug_mat1_qe` were replayed through nine checkpoints with the exact Exp09 inference configuration. Direction statistics below exclude proteins whose matched control is missing.

| Epoch | Pred down `< -0.10` | Pred up `> +0.10` | Any negative | Any positive | Mean delta | Median delta | Single-extra classification AUPRC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1200% | 57.9461% | 18.3440% | 81.6552% | +0.2380 | +0.2203 | 0.588463 |
| 1 | 0.0528% | 56.2781% | 14.0135% | 85.9854% | +0.6128 | +0.1919 | 0.594956 |
| 2 | 0.0865% | 55.8301% | 18.7270% | 81.2721% | +0.8508 | +0.1951 | 0.596656 |
| 3 | 0.0592% | 54.5162% | 22.7608% | 77.2383% | +0.7792 | +0.1620 | 0.578816 |
| 5 | 2.0359% | 45.1315% | 37.1302% | 62.8693% | +0.6522 | +0.0502 | 0.592539 |
| 8 | 6.0560% | 49.5110% | 40.2028% | 59.7967% | +0.7746 | +0.0904 | 0.589168 |
| 12 | 13.9421% | 43.8825% | 47.1631% | 52.8365% | +0.7122 | +0.0160 | 0.560156 |
| 20 | 17.8661% | 43.0803% | 49.1221% | 50.8778% | +0.7326 | +0.0050 | 0.523938 |
| 49 | 23.5015% | 39.3109% | 53.2575% | 46.7422% | +0.6762 | -0.0148 | 0.539717 |

This establishes the following sequence:

1. The defective MSE mask gives the residual decoder a strongly positive initial aggregate gradient. Although only `6.31%` of single-drug MSE-valid elements have a missing control and finite perturbed target, their mean pseudo-delta is `+12.8864`; this shifts the aggregate supervised target-delta mean from approximately zero to `+0.8114`.
2. During epochs 0-3, the decoder first learns this easy global positive direction. Meaningful negative, protein/drug/cell-specific structure appears much later: the down fraction rises from about `0.1%` to `2.0%` at epoch 5, `6.1%` at epoch 8, and `23.5%` at epoch 49.
3. The single-extra classification AUPRC follows a different trajectory and reaches its maximum at epoch 2 (`0.596656`). Therefore choosing epoch 2 for the single-drug downstream branch freezes the model at the early, most one-directional stage. Epoch 2 is not uniquely corrupted; it is the selected member of a severe early-epoch cluster.

The later increase in predicted negative deltas does not prove that those deltas are biologically correct because these extra rows lack post-treatment proteome truth. It does prove that the epoch-2 severity is training-stage dependent rather than a fixed sign constraint or a peculiarity of the downstream datasets.

### Exact meaning of “selected by classification AUPRC”

The original wording needs one correction for Exp09:

- Exp09 training used `monitor=none`, `save_top_k=-1`, and retained every epoch. Lightning did **not** automatically call epoch 2 the best checkpoint.
- A later oracle sweep evaluated all 50 checkpoints on the external labeled tasks and selected the maximum **classification** AUPRC: epoch 2 for six single-extra tasks (`0.596656`) and epoch 8 for three double-extra tasks (`0.101437`).
- The measured quantity is the precision-recall ranking of binary response/synergy probabilities. It does not measure expression delta, down recall, expression MSE, or skill over copying the control.
- In this checkpoint, `response_delta_mode=off` and `delta_logit_scale=0`. Classification logits are produced from the shared hidden representation rather than from the predicted expression delta. Thus a checkpoint can rank sensitive samples well while its expression head is nearly one-directional.
- The validation collector stores labels and classification probabilities only; it does not compute any signed proteome metric. Consequently there is no selection gate capable of rejecting epoch 2 for `0.0865%` down predictions.

For Exp01-08, the normal checkpoint monitor is `val/task_auprc`. For Exp31 fine-tuning, the selected checkpoint is likewise the maximum BRCA validation classification AUPRC. Exp32-34 perform no new checkpoint selection; they directly use the already selected Exp09 epoch 2.

### Why Exp31-34 inherit or amplify the symptom

- **Exp32-34:** these are direct epoch-2 inference branches, so they inherit the early decoder almost unchanged. Their rows have no perturbed-proteome ground truth and therefore cannot correct or validate the direction bias. OOD drugs, cells, covariates, and targets make the outputs extrapolations, but are not required to explain the shared near-zero down fraction: the fixed-input trajectory already shows that epoch 2 itself has this behavior.
- **Exp31 zero-shot:** it also starts from epoch 2, but its baseline is RNA-seq transformed with `log1p`, whereas Exp09 was trained on proteomics `log1p` intensity. On finite control values, the Exp31 RNA input has mean `2.3622` and median `2.4248`; the main-double proteomics training controls have mean `14.3135` and median `14.1843`. This severe modality/scale shift sends the control encoder far outside its training distribution and is consistent with Exp31's larger mean predicted delta (`+1.5217`) and up fraction (`70.71%`). The checkpoint axis also contains 933 proteins absent from RNA, represented as `NaN` and converted to zero inside the model.
- **Exp31 fine-tuning:** MSE is disabled and `response_delta_mode=off`. The classification loss updates the shared control/fusion representation, while the delta head receives no expression-preservation signal. Its output can therefore drift as an uncontrolled side effect even though the fine-tuning checkpoint improves classification AUPRC. Selecting the fine-tuned epoch by classification AUPRC again cannot reject that drift.
- **Exp34 reporting:** on top of the inherited epoch-2 bias, the report converts missing controls to zero and ranks absolute deltas. This makes missing-control proteins appear as very large positive changes; `516/560` published top-20 entries (`92.14%`) have a missing original control. The reporting rule amplifies the visual symptom but is not its root cause, because finite-control results remain strongly positive.

## 2026-08-21 18:50 HKT clarification: absolute expression versus signed perturbation delta

The phrase "mapping a missing control to zero creates a positive target" refers to the **residual/delta target**, not to a claim that every possible perturbed-expression representation must be positive.

Let `p` be the observed perturbed absolute expression and `c` the true but unobserved control expression. The scientifically intended target is:

```text
true_delta = p - c
```

After the current model maps the missing control to zero, the effective residual target becomes:

```text
pseudo_delta = p - 0 = p
```

Therefore:

```text
pseudo_delta - true_delta = c
```

Whenever the underlying control expression is positive, zero imputation shifts the residual target upward by the full control abundance. For example, a biologically downregulated protein with `control=14` and `perturb=13` has `true_delta=-1`; if the control is missing and replaced by zero, the supervised target becomes `pseudo_delta=+13`, reversing the direction.

If a different centered expression representation allowed `p<0`, the pseudo-delta would not be guaranteed to be positive. For example, `p=-1` gives `pseudo_delta=-1`; however, compared with `true_delta=-15` when `c=14`, it is still shifted in the positive direction by `+14`. Thus "positive shift" is the general statement; "almost all pseudo-deltas are positive" is specific to the current data representation.

In this repository, protein intensities are represented as nonnegative absolute abundance followed by `log1p`, not as zero-centered signed expression. Biological downregulation is therefore `perturb < control`, not `perturb < 0`. The empirical training-data audit confirms the consequence:

- control-missing / perturb-finite positions have mean perturbed absolute expression, and hence mean pseudo-delta, `+12.8864` in the main single-drug task;
- finite control/perturb pairs have mean true delta `-0.00254`;
- the asymmetric missingness contract shifts the effective supervised target-delta mean to `+0.8114`.

The loss gradient makes the direction explicit. For a missing control converted to zero and a finite target `p=13`, the residual loss near an initial predicted delta of zero is:

```text
L = (pred_delta - 13)^2
dL / d(pred_delta) = 2 * (pred_delta - 13) = -26
```

Gradient descent subtracts this negative gradient and therefore increases the predicted delta. The target is not zero: only the residual baseline was replaced by zero, while the finite perturbed target remains 13.

The required correction is consequently to construct the reconstruction/delta-loss mask from the raw tensors before imputation:

```python
valid = torch.isfinite(control_raw) & torch.isfinite(perturb_true)
```

Any protein position with a non-finite control or non-finite perturbed target must be excluded from signed-delta loss and signed-delta validation. Missingness may still be supplied to the conditioning encoder through an observation mask or a learned missing-value embedding; that input representation does not replace the joint-finite loss mask.
