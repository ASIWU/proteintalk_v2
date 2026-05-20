# 2026-05-07 17:09 HKT Training Metrics, NaN, and Forward Review

Update note, 2026-05-07 17:33 HKT: the remaining trainer-option gaps
listed in this review were implemented, model choices were consolidated to
`attention_v10_hetero_cls_ee` and `baseline_emb_v3`, top-k protein selection
was skipped per user decision, and inference expression metrics were extended.
See
`data/review_summary/2026-05-07_1733_legacy_trainer_options_model_consolidation_infer_metrics.md`.

## Scope

This review continues from
`data/review_summary/2026-05-07_1643_legacy_model_port_and_entrypoint_review.md`
and focuses on the remaining training-code gaps the user called out:

- trainer metrics versus legacy `model/trainer_dd.py` + `utils/eval_dd.py`
- dataset NaN handling versus legacy `dataset/dataset_csv.py` /
  `dataset/dataset_csv_dd.py`
- selected model forward behavior versus legacy `attention.py` /
  `attention_dd.py`

## Review Details

### Dataset and Dataloader

Previous finding: `ProteinTalkDataset.__getitem__` matched the legacy
`{control, perturb}` sampling contract but sanitized control expression NaNs to
zero before model input.

Current fix:

- `dataset/training_ready_dataset.py` no longer calls
  `np.nan_to_num(..., 0.0)` for control rows.
- Control and perturb expressions now preserve NaNs from
  `feature_expression_matrix.npy`.
- This lets `model.training_ready_models.ValueEmbedding` handle NaN values via
  its learned `nan_embedding`, as in legacy `model/basemodel.py`.

Remaining differences:

- Storage is still training-ready feature table + expression `.npy`, not the
  legacy monolithic parquet row object.
- Batch covariates and perturb indices are still pre-tokenized by the new data
  pipeline, not encoded by legacy `embedding_methods` at dataset time.

### Trainer and Metrics

Previous finding: the current Lightning wrapper only logged AUROC/AUPRC for
task 1 and task 2, while legacy validation/test called
`compute_validation_metrics`.

Current fix in `model/training_ready_lightning.py`:

- Restored the legacy-style double-drug metric suite:
  - task 1: `auroc`, `auprc`, `acc`
  - task 2: `auroc2`, `auprc2`, `acc2`
  - expression all-gene metrics: `mse_all`, `mae_all`, `pcc_all`, `r2_all`,
    `direction_acc_all`
  - expression top-50 metrics: `mse_top50`, `mae_top50`, `pcc_top50`,
    `r2_top50`, `direction_acc_top50`
  - delta metrics: `delta_pcc_all`, `delta_r2_all`, `delta_pcc_top50`,
    `delta_r2_top50`
  - distribution metrics: `mmd`, `energy_distance`
- Validation/test now collect first group member predictions and labels the
  same way legacy trainer code does.
- Loss handling is closer to legacy:
  - MSE masks perturb expression label NaNs.
  - BCE uses `1 - label_mask` as weights and divides by valid weight sum.

Remaining trainer differences:

- `focal_loss`, positive class weights, `adamw_fused`, `step`,
  `cosine_warmup`, and `UnfreezeCallback` are still not ported into the new
  entrypoint.
- Current `infer.py` still computes classification metrics only in
  `metrics.json`; full expression metrics during inference would require saving
  or computing expression predictions in the inference loop.

### Model Forward

Previous finding: model internals had been ported to legacy-style token/CLS
Transformer paths, but exact forward behavior still needed targeted review.

Current fixes:

- `ValueEmbedding` already matched legacy NaN handling; the dataset fix now
  allows this code path to run for control NaNs.
- `attention_v10_hetero_cls_ee_no_target` graph double-drug forward now matches
  legacy `attention_dd.py` perturb token behavior:
  - non-`mlp` mode keeps both graph drug embeddings as perturb tokens;
  - `mlp` mode applies `pert_proj` per drug token.
- Non-graph target model `fusion_mode=add` now matches legacy
  `attention.py`: target protein tokens are not added into the context vector
  for `attention_v4_cls_ee_target` / `attention_v4_cls_ee_target_proemb`.
  The graph gate path keeps target mean in add context, matching the legacy
  gate model.

Remaining model-forward caveat:

- The old repo has double-drug classes for `attention_v4_cls_ee` and
  `attention_v10_hetero_cls_ee_no_target`, but not for every selected target /
  gate model. For those selected models, the current code keeps the legacy
  single-drug forward structure and adapts `pert_id` to the new required
  two-drug input contract.
- `baseline_emb_v3` has an added synergy head for the new double-drug trainer;
  this head does not exist in the legacy single-task baseline.

## Verification

Executed in `flow_v2`:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python -m py_compile model/training_ready_lightning.py model/training_ready_models.py dataset/training_ready_dataset.py train.py infer.py
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python train.py --help
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python infer.py --help
```

Focused smoke checks passed:

- dataset preserves NaNs in control and perturb expressions;
- `ValueEmbedding` accepts NaNs and returns finite embeddings;
- legacy-style validation metric helper returns the expected 22 metric keys;
- `ProteinTalkLightning.validation_step` and epoch-end metric logging run on a
  tiny batch with NaNs;
- `attention_v10_hetero_cls_ee_no_target` graph double-drug forward returns
  `(expression, response_logits, synergy_logits)` with expected shapes.

## Next Plan

1. Port remaining legacy trainer controls if needed:
   `focal_loss`, positive weights, `adamw_fused`, `step`,
   `cosine_warmup`, and `UnfreezeCallback`.
2. Make selected model adapters more explicit per model name, especially for
   target/gate models whose old repo implementation is single-drug only.
3. Add a top-k protein option before full PTV3 legacy Transformer training,
   because full attention over about 11k protein tokens is likely expensive.
4. Extend `infer.py` to optionally compute the full expression metric suite
   when expression predictions are requested.
