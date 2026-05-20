# 2026-05-07 17:33 HKT Legacy Trainer Options, Model Consolidation, and Inference Metrics Review

## Scope

This review records the follow-up implementation after the user accepted some
remaining differences and requested these concrete changes:

- port legacy trainer options: focal loss, positive weights, `adamw_fused`,
  `step`, `cosine_warmup`, and `UnfreezeCallback`;
- keep only two model choices: the baseline and one consolidated PDI hetero
  graph model;
- control target-token use and gene/protein gate use by parameters;
- skip top-k protein selection;
- extend `infer.py` to compute full expression metrics when expression
  predictions are requested.

## Review Details

### Trainer

Implemented in `model/training_ready_lightning.py`:

- Added legacy `FocalLossWithAlpha`.
- Added task-specific positive weights:
  - `positive_weight1` for the response head;
  - `positive_weight2` for the synergy head;
  - `--positive-weight` in `train.py` applies the same value to both heads.
- Preserved focal-loss precedence over positive-weight BCE, matching the old
  trainer behavior.
- Added `adamw_fused` optimizer support:
  - exact `adamw_fused` uses `embedding_proj` at `learning_rate / 10.0`;
  - suffixed `adamw_fused_<factor>` uses `embedding_proj` at
    `learning_rate * factor`.
- Added legacy scheduler options:
  - `step`: `StepLR(step_size=30, gamma=0.01)`;
  - `cosine_warmup`: 25-epoch `LinearLR` warmup from `0.01` to `1.0`, then
    cosine annealing to epoch 250.
- Added `UnfreezeCallback` for `model.embedding_proj`, wired through
  `train.py --unfreeze-at-epoch`.

Remaining trainer note:

- `adamw_fused` and `UnfreezeCallback` intentionally require a model layer named
  `embedding_proj`; this exists on the PDI hetero graph model and does not
  exist on the baseline model.

### Model Consolidation

Implemented in `model/training_ready_models.py`:

- The active model registry is now only:
  - `attention_v10_hetero_cls_ee`
  - `baseline_emb_v3`
- `attention_v10_hetero_cls_ee` is the consolidated PDI hetero graph model.
- `--use-target` controls whether target protein tokens are included.
- `--target-protein-fusion-model gate` controls the graph gene/protein gate.
- `--target-protein-fusion-model concat` keeps the non-gated graph protein
  addition path.
- The no-target graph path keeps the legacy double-drug perturbation behavior:
  non-`mlp` perturb fusion keeps both graph drug tokens; `mlp` projects each
  drug token.
- When `--use-target` is enabled, the graph path uses target protein tokens and
  the target-style double-drug perturbation handling.

The user explicitly accepted the prior single-drug-forward adaptation for
target/gate behavior, so no per-model legacy adapter split was implemented.

### train.py

Implemented in `train.py`:

- Reduced `--model-type` choices to `attention_v10_hetero_cls_ee` and
  `baseline_emb_v3`.
- Added trainer flags:
  - `--focal-loss` / `--focal_loss`
  - `--positive-weight` / `--positive_weight`
  - `--positive-weight1`
  - `--positive-weight2`
  - `--optimizer-name` / `--optimizer_name` with dynamic names such as
    `adamw_fused_0.5`
  - `--scheduler-name` / `--scheduler_name`
  - `--unfreeze-at-epoch`
  - `--unfreeze-layer-name`
  - `--gradient-clip-val`
- Added `--use-target` for the consolidated graph model.
- Run manifests now record the new trainer/model options.

### infer.py

Implemented in `infer.py`:

- Reduced `--model-type` choices to the same two active models.
- Added `--use-target`.
- Metrics collection no longer re-reads every dataset item after inference; it
  collects labels and masks inside the inference loop.
- When `--save-expression-pred` is set, inference now also computes
  `legacy_validation_metrics` using the same full metric suite as validation
  and test:
  - task1/task2 AUROC, AUPRC, ACC;
  - all-gene and top-50 expression MSE/MAE/PCC/R2/direction accuracy;
  - delta metrics;
  - MMD and energy distance.

## Verification

Executed in `flow_v2`:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python -m py_compile model/training_ready_lightning.py model/training_ready_models.py dataset/training_ready_dataset.py train.py infer.py
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python train.py --help
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python infer.py --help
```

Focused smoke checks passed:

- model registry contains only `attention_v10_hetero_cls_ee` and
  `baseline_emb_v3`;
- `adamw`, `adamw_fused`, and `adamw_fused_0.5` optimizer configuration works;
- `step`, `plateau`, and `cosine_warmup` scheduler configuration works;
- focal-loss masked BCE returns a finite loss;
- consolidated PDI graph forward works for:
  - no target, no gate;
  - target tokens, no gate;
  - target tokens plus gate.

## Current Next Plan

No more code changes are planned for top-k protein selection or per-model
legacy adapter splitting, per the user's decision.

Recommended next action is a real-data dry-run or one-epoch smoke training with
the desired graph configuration, for example:

- default consolidated graph: no `--use-target`, `--target-protein-fusion-model concat`;
- target-token graph: add `--use-target`;
- gated graph: add `--target-protein-fusion-model gate`, with or without
  `--use-target` depending on the experiment.

Because top-k protein selection was explicitly skipped, full PTV3 training will
still run full Transformer attention over roughly 11k protein tokens.
