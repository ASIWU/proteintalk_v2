# exp_04_v2 Max-Drop Random Expression Detailed Analysis

- Date: 2026-07-09 HKT
- Task: `ptv3_main_singledrug` / `response`
- Formal split: `pert_stratified_5fold_fold0..4`
- Screen prefix: `20260708_exp04_v2_random_expr_screen`
- Final clean prefix: `20260708_exp04_v2_maxdrop_random_expr_clean`
- Final setting written to: `scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh`
- Selected artifact: `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_per_row_gene_permutation_seed42.npy`

## Executive Summary

The strongest random-expression setting found in this experiment is `per_row_gene_permutation`.

This setting does not merely replace real control expression with plausible random expression. It makes the control-expression vector internally inconsistent with the ordered protein axis on every individual sample: each row receives a fresh random gene-column permutation. That breaks the gene-to-protein alignment used by PCEP and prevents the model from learning one stable alternate mapping.

The final 5-fold result shows a clear degradation relative to the original seed42 random-control baseline:

| setting | mean AUPRC | mean nAUPRC | mean AUROC | mean ACC |
|---|---:|---:|---:|---:|
| current seed42 random | 0.673665 | 5.717265 | 0.905289 | 0.913943 |
| max-drop random, `per_row_gene_permutation` | 0.592230 | 5.036277 | 0.891713 | 0.903330 |
| drop | 0.081436 | 0.680987 | 0.013577 | 0.010612 |

The fold0 screen also shows why this is a targeted stress test rather than a generic random baseline. The original per-protein random expression did not degrade performance; it slightly improved AUPRC (`0.708677` vs real-control no-MSE `0.702725`). Zero control also did not collapse performance (`0.716967`). The only random policies that reliably reduced fold0 AUPRC were policies that destroy the expression vector's gene identity structure: global sampling and row-wise gene permutation.

## Question Being Answered

The original question was not "does any random expression matrix differ from real control expression?" The practical question was:

Can we construct a random control-expression input which produces the largest measurable drop under the existing `exp_04_v2` no-MSE response setting, without changing the model architecture?

The answer is yes, but the mechanism matters:

- Randomizing values while preserving per-gene distribution is too weak.
- Zeroing control expression is also too easy for the model to ignore.
- A fixed wrong gene mapping can be learned around.
- A row-specific wrong gene mapping is harder to ignore because it injects sample-specific noise into the PCEP pooling path.

## Model Path Relevant To This Experiment

The important implementation detail is how `random_saved` control expression is consumed.

In `dataset/training_ready_fast_dataset.py`, when `control_expression_mode=random_saved`, the dataset does not use the matched real control row. It returns:

```text
random_control_expression_matrix[perturb_row]
```

as `batch["control_expression"]`.

In `model/fast_delta_model.py`, the model always encodes this control expression through the normal control-expression pathway. With `PROTEIN_CONCAT_MODE=pcep`, it also uses the normalized control-expression vector in PCEP:

```text
scores = query @ protein_features.T
scores = scores * normalized_expression
top-k scores select expression-weighted protein features
pooled PCEP vector is added back into control_hidden
```

So control expression enters the model in two related ways:

- As a dense expression vector encoded into control/cell context.
- As a gene-aligned weighting vector for PCEP's protein-aware pooling.

This is why the gene order matters. PCEP assumes the `i`th expression value corresponds to the `i`th ordered protein feature. If values are shifted to the wrong protein positions, the PCEP score computation becomes biologically and structurally inconsistent.

## Why The Original Random Control Did Not Collapse

The original seed42 random artifact used the `per_protein_normal_clip` policy:

```text
For each gene independently:
sample Normal(control-row per-gene mean, control-row per-gene std)
clip negatives to zero
```

This preserves much of the per-gene marginal distribution:

- genes that are usually high remain high;
- genes that are usually low remain low;
- the vector remains aligned to the ordered protein axis;
- PCEP still sees gene values in the right protein positions.

That means the model can still use broad expression scale and gene identity statistics even though the exact cell/sample-specific control is fake. In fold0 this setting reached AUPRC `0.708677`, slightly above the real-control no-MSE reference `0.702725`.

This result says the current `fast_delta` response model is not primarily using fine-grained matched real-control expression in a way that is destroyed by per-gene random sampling. It is either relying more on other feature groups, or using only coarse gene-wise distributional information from the control vector.

## Screen Policy Interpretations

The screen compared one real-control reference, six random candidates, and one zero diagnostic. All were no-MSE, PCEP on, graph/target/covariates default.

| policy | fold0 AUPRC | drop vs real-control | interpretation |
|---|---:|---:|---|
| real_control_full_nomse | 0.702725 | 0.000000 | Reference with true control expression. |
| per_protein_normal_clip | 0.708677 | -0.005952 | Preserves per-gene marginals and alignment; not destructive. |
| global_normal_clip | 0.659943 | 0.042782 | Removes gene-specific distributions; PCEP sees weak/no gene identity. |
| global_value_bootstrap | 0.659719 | 0.043006 | Similar to global normal, preserves global value histogram but destroys gene identity. |
| fixed_gene_permutation | 0.743994 | -0.041269 | Stable wrong mapping can be learned as a consistent alternate basis. |
| per_row_gene_permutation | 0.654977 | 0.047748 | Fresh wrong mapping per row; strongest candidate drop. |
| cross_cell_real_control | 0.687633 | 0.015092 | Real expression from wrong cell is only mildly harmful. |
| zero_control | 0.716967 | -0.014242 | Constant zero input can be ignored or treated as a stable bias. |

The ranking is informative:

1. `per_row_gene_permutation` is worst because it breaks gene/protein alignment differently for each row.
2. `global_normal_clip` and `global_value_bootstrap` are almost tied because both remove gene-specific expression structure.
3. `cross_cell_real_control` only mildly hurts, so "wrong biological cell" is less destructive than "wrong gene identity".
4. `fixed_gene_permutation` improves performance, which strongly suggests the model can adapt to stable but non-biological expression transformations.
5. `zero_control` improves over real-control reference on fold0, showing that simply removing expression values is not an adversarial perturbation in this setup.

## Why Row-Wise Gene Permutation Is The Selected Setting

`per_row_gene_permutation` starts from the same per-gene normal sampling used by the original random control, then applies an independent random permutation over gene columns for every row.

This has three important effects:

1. It preserves plausible numeric scale.

   The expression values remain in the same overall range as the original random expression. This avoids an easy out-of-distribution artifact such as huge values, NaNs, or negative values.

2. It destroys gene identity.

   A value sampled for one gene is fed to another protein's position. PCEP's protein feature lookup and expression weighting no longer refer to the same biological entity.

3. It prevents a stable workaround.

   A fixed permutation creates one consistent wrong mapping; the model can learn around it during training. A row-wise permutation changes the wrong mapping per sample, so the model receives noisy, sample-specific PCEP weights that cannot be decoded as one alternate coordinate system.

This is why `fixed_gene_permutation` performed well (`0.743994` AUPRC) while `per_row_gene_permutation` performed worst (`0.654977` AUPRC). The difference is not just "permuted versus not permuted"; it is "stable wrong basis" versus "unstable wrong basis".

## Why Zero Control Does Not Collapse

A common first guess was: if random control expression should hurt, zeroing control expression should hurt even more. The screen shows the opposite: zero-control fold0 AUPRC was `0.716967`.

This is plausible for this architecture and task:

- With no MSE loss, there is no reconstruction objective forcing the model to preserve biologically meaningful expression values.
- A constant all-zero control vector is easy for the network to identify as a stable condition.
- PCEP with a constant/zero expression vector becomes a mostly stable learned bias rather than random sample-specific noise.
- Other features remain available: graph features, Morgan drug embedding, target proteins, and categorical covariates.

So zero control is not an adversarial perturbation here. It removes one variable signal, but it does not inject misleading row-specific signal. The model can route around it.

## Relationship To Feature Attribution Results

The fold0 feature-attribution matrix is important context. In exp01 unseen-drug fold0:

| comparison | AUPRC | drop vs full no-MSE |
|---|---:|---:|
| full no-MSE | 0.702725 | 0.000000 |
| graph_zero | 0.597593 | 0.105132 |
| pcep_off | 0.665255 | 0.037470 |
| per_row_gene_permutation screen | 0.654977 | 0.047748 |
| original random control | 0.708677 | -0.005952 |

This suggests:

- Graph features are still the most important group in exp01 fold0 among the tested leave-one-out ablations.
- PCEP/control-expression manipulation can hurt, but it does not dominate the full model.
- The selected row-wise permutation is slightly more harmful than simply turning PCEP off on fold0 (`0.654977` vs `0.665255`), which means it is not just removing the pathway; it likely injects misleading signal into the pathway.
- The model still performs far above the label baseline after row-wise permutation, so the prediction is not solely dependent on control expression/PCEP.

For exp03 unseen-cell fold0, PCEP off had a larger drop (`0.070487`) than in exp01. This max-drop experiment was formally run on exp04_v2's pert-stratified setting, so conclusions should not be over-generalized to cell split without a matching screen.

## Final 5-Fold Result

The final clean 5-fold run used:

```text
EXP_PREFIX=20260708_exp04_v2_maxdrop_random_expr_clean
RANDOM_CONTROL_EXPRESSION_PATH=data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_per_row_gene_permutation_seed42.npy
CONTROL_EXPRESSION_MODE=random_saved
PROTEIN_CONCAT_MODE=pcep
GRAPH_FEATURE_MODE=real
TARGET_PROTEIN_MAX_LENGTH=32
no MSE loss
```

Fold-level results:

| fold | max-drop AUPRC | current seed42 AUPRC | delta | max-drop AUROC | count |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.654977 | 0.708677 | -0.053699 | 0.912988 | 3534 |
| 1 | 0.528327 | 0.590959 | -0.062632 | 0.890222 | 3549 |
| 2 | 0.658974 | 0.812005 | -0.153031 | 0.891045 | 3589 |
| 3 | 0.679847 | 0.687516 | -0.007668 | 0.911761 | 3537 |
| 4 | 0.439023 | 0.569171 | -0.130149 | 0.852547 | 3570 |

The degradation is consistent in direction across all folds:

- every fold has lower AUPRC than the original seed42 random baseline;
- fold2 and fold4 have the largest drops;
- fold3 has only a small drop, indicating the effect is not uniformly strong across split composition;
- the mean AUPRC drop is `0.081436`, roughly a 12.1% relative decrease from the current seed42 random mean AUPRC.

The AUROC drop is smaller (`0.013577` mean), while AUPRC and nAUPRC move more. This indicates the perturbation mainly affects positive-class ranking precision under class imbalance, not just global separability.

## What This Does And Does Not Prove

This experiment proves:

- The model is vulnerable to random expression inputs that break row-wise gene/protein alignment.
- The original per-gene random expression was too weak to test worst-case dependence on the control/PCEP path.
- The selected `per_row_gene_permutation` setting is a stronger stress test than `zero_control`, `cross_cell_real_control`, or global-value randomization.
- The final exp04_v2 script now points to the strongest observed random-expression artifact by default.

This experiment does not prove:

- That control expression is the primary driver of response prediction.
- That PCEP is more important than graph features in exp01/exp04.
- That all random-expression policies will behave similarly across unseen-cell splits.
- That the model "collapses" completely under corrupted control expression.

The strongest feature-attribution evidence still points to graph features as a major driver for exp01 fold0. The random-expression stress test is about making the control/PCEP branch harmful enough to reveal sensitivity, not about proving it is globally dominant.

## Recommended Interpretation For Reporting

Use this wording:

```text
For exp_04_v2, the strongest random-expression stress test was row-wise gene permutation of the saved random control-expression matrix. This setting preserves expression scale but destroys per-sample gene/protein alignment, producing a 5-fold mean AUPRC drop of 0.0814 versus the original seed42 random-control baseline.
```

Avoid this wording:

```text
Random control expression makes the model collapse.
```

That is too strong. Even under the selected max-drop setting, mean AUPRC remains `0.592230`, and fold0 AUPRC remains `0.654977`. The model is degraded but not collapsed, likely because graph/drug/target/covariate features remain predictive.

Also avoid calling this a neutral "random baseline" without qualification. `per_row_gene_permutation` is better described as an adversarial or max-drop random-expression stress test, because it was selected after screening policies for maximal degradation.

## Final Setting

The final default in `scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh` is:

```bash
RANDOM_CONTROL_EXPRESSION_PATH="${RANDOM_CONTROL_EXPRESSION_PATH:-data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_per_row_gene_permutation_seed42.npy}"
```

The environment override is preserved, so other artifacts can still be tested without editing the script.

## Validation Notes

- CPU artifact generation and reporting were run locally.
- GPU training/testing was run through tmux session `gpu2`.
- The formal final run used clean prefix `20260708_exp04_v2_maxdrop_random_expr_clean`.
- A previous run with prefix `20260708_exp04_v2_maxdrop_random_expr` was interrupted because it inherited `RUN_PREFLIGHT=1`; it is not used for analysis.
- Screen reporter validation errors: `0`.
- Final reporter validation errors: `0`.
- All formal final manifests have `run_status=fit_completed` and `test_status=test_completed`.

## Follow-Up Experiments

If the goal is to make a stronger mechanistic claim about PCEP/control expression, the next tests should be:

1. Repeat the same random-expression screen on the unseen-cell split.
2. Run multiple random seeds for `per_row_gene_permutation` to estimate variance.
3. Combine `per_row_gene_permutation` with graph ablations to test whether graph features are masking larger PCEP sensitivity.
4. Add a "train real, test corrupted" inference-only test to separate training-time adaptation from inference-time reliance.
5. Log PCEP top-k gene/protein selections to verify that row-wise permutation changes the selected proteins rather than only changing expression magnitude.
