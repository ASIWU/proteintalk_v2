# Exp34 Protein Attribution Implementation and Execution Review

- Review time: 2026-08-19 14:44 HKT (UTC+08:00)
- Source inference: `20260819_exp34_update0819_ood_epoch2`
- Checkpoint SHA-256: `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076`
- GPU target: outer tmux `gpu1_deep:0`, H200 worker `cuda:0`
- Scope: per-sample protein attribution only; no Morgan, graph, target,
  covariate, or cross-sample consensus attribution.

## Method

- Attribution target is the response logit, not sigmoid probability.
- All non-protein batch inputs remain fixed at their formal inference values.
- The reference vector is the per-protein `nanmedian` over 498 unique control
  expression rows referenced by the checkpoint `all_train_subset_test` train
  set-info.
- The baseline contains 11,092 values. There are 132 proteins with no finite
  value in any referenced training control; those baseline entries are zero.
- Fast local importance is `gradient x (control - training_control_median)`.
- Formal importance is midpoint Integrated Gradients with 32 steps. Samples
  would rerun at 64 steps only if both absolute completeness error exceeded
  `1e-3` and relative error exceeded `5%`.
- Positive scores push the model toward a larger sensitivity logit; negative
  scores push it toward a smaller sensitivity logit.
- Absolute ranks use protein index as a deterministic tie-break.

## Implementation and Tests

- `utils/attribution_utils.py` isolates attribution math from Lightning so it
  can be tested with analytic linear models.
- `scripts/attribute_exp34_update0819_ood_epoch2.py` reconstructs the exact
  formal model and datasets from persisted manifests, verifies checkpoint and
  axes, requires `gpu1_deep:0`, and refuses to publish results unless the formal
  probabilities are reproduced within `1e-7`.
- Tests cover analytic Gradient x Delta and IG equality, completeness, batched
  sample independence, all-NaN baseline behavior, deterministic ranking, and
  invalid integration steps.
- Combined test result: `14 passed, 4 subtests passed`.

## Smoke and Formal Execution

- Smoke included two samples from each target branch and used IG-4 with
  adaptive IG-8. All four samples upgraded to eight steps, published
  probability maximum error was `1.12e-08`, maximum absolute completeness error
  was `0.00218`, and no final warning remained.
- Formal execution covered 28 samples and used IG-32. No sample required IG-64.
- Formal published-probability reproduction error was exactly zero.
- Formal output contains 310,576 rows: 28 samples x 11,092 proteins.
- All gradients, Gradient x Delta values, IG values, and expression/baseline
  values are finite. Every sample has exactly 11,092 unique absolute ranks.
- Maximum absolute completeness error is `0.000182`; maximum relative error is
  `0.0631`, occurring where the underlying logit difference is near zero. The
  configured joint absolute/relative warning rule produced zero warnings.
- Gradient x Delta and IG agree strongly:
  - Top-100 overlap: minimum `0.88`, mean `0.927`, maximum `0.96`;
  - full-axis absolute-value Spearman: minimum `0.981`, mean `0.988`, maximum
    `0.992`.

## Outputs

The formal output directory is
`outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_protein_attribution`.

- `protein_attributions.parquet`: all 310,576 signed attribution rows;
- `top200_absolute_ig_per_sample.csv`: requested per-sample importance ranking;
- `top100_positive_ig_per_sample.csv` and
  `top100_negative_ig_per_sample.csv`: signed directions;
- `sample_diagnostics.csv`: completeness and Gradient x Delta/IG agreement;
- `published_probability_reproduction.csv`: exact inference replay audit;
- saved training-control baseline, protein axes, summary, run manifest, and
  Markdown interpretation.

The rankings explain this checkpoint relative to the chosen training-control
median. They are not causal biomarkers or experimental evidence, and the two
drugs remain chemical OOD.
