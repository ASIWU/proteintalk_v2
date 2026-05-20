# 2026-05-07 19:31 HKT Legacy Behavior Recheck: Dataset, Trainer, train.py, infer.py

## Scope

This review rechecked the current training-ready implementation against the legacy ProteinTalk v2 code, with emphasis on:

- current `dataset/training_ready_dataset.py`
- current `model/training_ready_lightning.py`
- current `train.py`
- current `infer.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/dataset/dataset_csv.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/dataset/dataset_csv_dd.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/trainer.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/trainer_dd.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train_dd.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/infer.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/infer_dd.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/eval.py`
- legacy `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/eval_dd.py`

I also re-read:

- `docs/2026-04-15_data_standardization_session_summary.md`
- `docs/Data_Process_4.md`
- `docs/data_process_summary_04.md`

## Bottom Line

The current implementation is not a line-for-line legacy copy. It preserves the high-level training contract needed for the new training-ready data:

- `{control, perturb}` batch structure;
- train-time random control/perturb group sampling;
- eval/test group repetition;
- double-drug output heads for expression, response, and synergy;
- legacy-style masked MSE/BCE loss;
- legacy-style validation/test metric names from `utils/eval_dd.py`;
- focal loss, positive weights, `adamw_fused`, `step`, `cosine_warmup`, and `UnfreezeCallback`;
- PDI-only legacy hetero graph path for the consolidated graph model.

Important differences remain. Some are intentional based on previous decisions; others are behavior changes that should be treated as known deviations.

## Dataset And Dataloader Differences

### 1. Storage/read path differs by design

Legacy:

- `DatasetCSV` receives a dataframe where `expressions_hvg` is already a per-row expression vector.
- Legacy split loading is done through `DataSpliter`, which reads `row_to_set_index.pkl`, split index pickles, and split-specific `set_info` files from the legacy `preprocess_path`.
- Relevant legacy lines:
  - `dataset_csv.py:11-46`
  - `dataset_csv.py:48-109`
  - `dataset_csv_dd.py:11-37`
  - `dataset_csv_dd.py:39-103`

Current:

- `TrainingReadyArtifacts` reads `feature_table.parquet` / `.pkl` / `.csv`, `feature_expression_matrix.npy`, `feature_ordered_protein_index.json`, `feature_sample_ids.json`, and `global_meta.json`.
- Expression values are fetched by row index from the aligned `.npy` matrix.
- Relevant current lines:
  - `dataset/training_ready_dataset.py:37-60`
  - `dataset/training_ready_dataset.py:145-160`
  - `dataset/training_ready_dataset.py:304-311`

Assessment: intentional divergence. This follows the current training-ready artifact design.

### 2. Feature encoding differs by design

Legacy:

- Uses `embedding_methods` and `DataSpliter._embedding_column()` at dataloader setup time.
- Single-drug `pert_id` can be mapped to embedding or graph index through `get_drug_embedding` / `get_drug_index`.
- Double-drug `pert_id` can be a merged string such as `drug1+drug2`; `dataset_csv_dd.py` splits it through `merged_feature_list`.
- Relevant legacy lines:
  - `dataset_csv.py:26-32`
  - `dataset_csv.py:178-199`
  - `dataset_csv_dd.py:23-29`
  - `dataset_csv_dd.py:172-205`

Current:

- Consumes pre-tokenized columns like `machineID_new_index`, `Cell_index`, `pert_index1`, `pert_index2`.
- Does not use `embedding_methods`.
- Always emits double-drug `pert_id` shape:
  - graph model: `[group_size, 2]`
  - baseline embedding model: `[group_size, 2, drug_embedding_dim]`
- Single-drug rows use `pert_index2` as the metadata `no` index.
- Relevant current lines:
  - `dataset/training_ready_dataset.py:16-24`
  - `dataset/training_ready_dataset.py:312-323`
  - `dataset/training_ready_dataset.py:343-346`

Assessment: intentional divergence, consistent with `docs/Data_Process_4.md`.

### 3. Train sampling is close to legacy

Legacy:

- Train mode picks the anchor by `idx % dataset_len`.
- Samples `group_size` control rows with replacement.
- Samples `group_size - 1` perturb rows with replacement, then appends the anchor row.
- Relevant legacy lines:
  - `dataset_csv.py:134-146`
  - `dataset_csv_dd.py:128-140`

Current:

- Same anchor/control/perturb strategy in train mode.
- Relevant current lines:
  - `dataset/training_ready_dataset.py:281-291`

Assessment: behavior is preserved here.

### 4. Eval/test control sampling is intentionally different

Legacy:

- Eval/test mode randomly picks one control row with `np.random.choice`, then repeats it to `group_size`.
- Relevant legacy lines:
  - `dataset_csv.py:154-174`
  - `dataset_csv_dd.py:148-168`

Current:

- Eval/test mode deterministically uses the sorted first control row, then repeats it to `group_size`.
- Relevant current lines:
  - `dataset/training_ready_dataset.py:292-298`

Assessment: not exact legacy behavior. This was changed to make validation/inference deterministic. If exact stochastic legacy eval is required, this should become a flag.

### 5. Label and mask behavior differs from the old dataloader

Legacy:

- Single-drug `Sensitive_label_encoder` returns positive only for `sensitive`; everything else becomes `0`.
- Double-drug `Synergy_label_encoder` returns positive for `1` / `'1'`; everything else becomes `0`.
- The double-drug trainer can consume `sensitive_label_mask` and `synergy_label_mask`, but legacy `train_dd.py` feature list includes `synergy` and does not include `synergy_label_mask`.
- Relevant legacy lines:
  - `dataset_csv.py:222-232`
  - `dataset_csv_dd.py:246-268`
  - `train_dd.py:292-306`
  - `model/trainer_dd.py:238-249`

Current:

- `encode_binary_label` distinguishes valid positives, valid negatives, and missing/unknown labels.
- Missing/unknown labels are emitted as label `0.0` with mask `1.0`.
- Response label values recognize `sensitive/responsive/yes/1/true` and numeric `1.0`; negatives recognize `non-responsive/nonresponsive/no/0/false` and numeric `0.0`.
- Synergy label values recognize `syn/synergy/synergistic/yes/1/true`; negatives recognize `non-syn/nonsyn/non-synergy/non_synergy/no/0/false`.
- Relevant current lines:
  - `dataset/training_ready_dataset.py:94-142`
  - `dataset/training_ready_dataset.py:335-340`

Assessment: current behavior is safer for mixed current labels and extra tasks, but it is not exactly the old encoder behavior. It prevents missing labels from silently becoming negatives.

### 6. Missing-control behavior is stricter upstream

Legacy:

- Assumes split artifacts and `set_info` are valid.

Current:

- `build_pairing_from_table` raises if inferred anchors cannot resolve to a control row.
- Step 3 split generation skips anchors with no matched controls, per user decision.
- Relevant current lines:
  - `dataset/training_ready_dataset.py:208-230`

Assessment: compatible with current workflow, not a legacy behavior copy.

### 7. Dataloader construction differs

Legacy:

- Single-drug training uses `drop_last=True`.
- Double-drug training does not set `drop_last`, so default is false.
- Legacy train uses the test dataloader as validation.
- Relevant legacy lines:
  - `train.py:462-465`
  - `train_dd.py:384-387`

Current:

- Builds train, valid, and test loaders separately.
- Train loader default `drop_last` is false unless `--drop-last` is passed.
- Valid/test loaders are always `shuffle=False`.
- Relevant current lines:
  - `train.py:107-182`
  - `train.py:157-175`

Assessment: current double-drug default matches legacy double-drug `drop_last=False`, but single-drug `drop_last=True` is not preserved by default. Validation/test separation is intentional per user request.

## Trainer And Metrics Differences

### 1. Loss behavior mostly matches legacy double-drug trainer

Legacy double:

- Model returns expression prediction, response logits, synergy logits.
- MSE masks NaNs in perturb expression labels.
- BCE loss uses mask semantics `1=ignore`, `0=keep`.
- Total loss is `mse_weight * mse + bce_weight1 * bce1 + bce_weight2 * bce2`.
- Relevant legacy lines:
  - `model/trainer_dd.py:147-231`
  - `model/trainer_dd.py:233-266`

Current:

- Same three-output contract.
- MSE masks NaNs in perturb expression labels.
- BCE mask semantics match `1=ignore`, `0=keep`.
- Total loss uses separate task weights and optional MSE.
- Relevant current lines:
  - `model/training_ready_lightning.py:405-448`
  - `model/training_ready_lightning.py:450-471`
  - `model/training_ready_lightning.py:473-479`

Assessment: behavior is close to legacy double-drug trainer.

### 2. Focal loss and UnfreezeCallback match the legacy pattern

Legacy:

- `FocalLossWithAlpha` returns unreduced per-element loss so masks can be applied later.
- `UnfreezeCallback` freezes `embedding_proj` at fit start and unfreezes at the chosen epoch.
- Relevant legacy lines:
  - `model/trainer_dd.py:17-35`
  - `model/trainer_dd.py:37-56`

Current:

- Same focal-loss formula and delayed reduction.
- Same freeze/unfreeze behavior, with an additional explicit error if the requested layer is absent.
- Relevant current lines:
  - `model/training_ready_lightning.py:302-315`
  - `model/training_ready_lightning.py:317-343`

Assessment: behavior is preserved, with stricter error handling.

### 3. Positive weights differ from invoked legacy `train_dd.py`

Legacy double trainer class:

- Supports `positive_weight1` and `positive_weight2` through `BCEWithLogitsLoss(pos_weight=...)`.
- Relevant legacy lines:
  - `model/trainer_dd.py:69-70`
  - `model/trainer_dd.py:116-134`

Legacy `train_dd.py` wrapper:

- Has a `positive_weight` argument, but `train_model()` does not pass it into `MultiTaskLightningModel`.
- Relevant legacy lines:
  - `train_dd.py:39-41`
  - `train_dd.py:52-65`

Current:

- Exposes `--positive-weight`, `--positive-weight1`, and `--positive-weight2`.
- Actually passes those values into `ProteinTalkLightning`.
- Relevant current lines:
  - `train.py:219-222`
  - `train.py:279-280`
  - `train.py:318-332`

Assessment: current behavior preserves the trainer class capability, not the old `train_dd.py` wrapper omission.

### 4. Optimizer and scheduler behavior is mostly preserved

Legacy:

- Supports `adam`, `adamw`, `sgd`, `adamw_fused`, `step`, `plateau`, `cosine`, `cosine_warmup`.
- Single-drug exact `adamw_fused` uses `embedding_proj` at learning-rate `/10`.
- Double-drug trainer uses suffix parsing for `adamw_fused_<factor>`.
- Relevant legacy lines:
  - `model/trainer.py:530-641`
  - `model/trainer_dd.py:638-722`

Current:

- Supports `adam`, `adamw`, `sgd`, exact `adamw_fused`, and `adamw_fused_<factor>`.
- Exact `adamw_fused` uses `/10`; suffix uses `lr * factor`.
- Supports `step`, `plateau`, `cosine`, and `cosine_warmup`.
- Relevant current lines:
  - `model/training_ready_lightning.py:560-624`

Assessment: behavior is preserved or made more explicit. Current exact `adamw_fused` is more robust than legacy double-drug suffix parsing.

### 5. Validation/test metrics are close to legacy `eval_dd.py`

Legacy double:

- Validation/test collect only the first repeated group member.
- Metrics are computed through `utils.eval_dd.compute_validation_metrics`.
- Expression labels are converted with `np.nan_to_num(..., nan=0.0)` before metric computation.
- Metrics include task1/task2 AUROC/AUPRC/ACC, expression MSE/MAE/PCC/R2, top50 metrics, delta metrics, MMD, and energy distance.
- Relevant legacy lines:
  - `model/trainer_dd.py:298-380`
  - `model/trainer_dd.py:430-510`
  - `utils/eval_dd.py:328-506`

Current:

- Also collects only the first repeated group member.
- Uses local `compute_validation_metrics` ported from `utils/eval_dd.py`.
- Also uses `np.nan_to_num(..., nan=0.0)` before metric computation.
- Logs the same metric-name family.
- Relevant current lines:
  - `model/training_ready_lightning.py:495-558`
  - `model/training_ready_lightning.py:214-299`

Assessment: close to legacy double-drug behavior. Minor implementation differences are that current uses a local copy instead of importing `utils/eval_dd.py`, uses `ValueError` instead of `assert` for prediction/target shape mismatch, and does not include the legacy `verbose` debug-print path.

### 6. Inference metrics differ from legacy predict hooks

Legacy single:

- `predict_step` returns `ny_prob` and `Cell`.
- `on_predict_epoch_end` can all-gather across devices and prints AUROC/AUPRC, top-k recall, enrichment factor, and last-positive rank.
- Legacy `infer.py` also prints metrics by cell.
- Relevant legacy lines:
  - `model/trainer.py:389-528`
  - `infer.py:105-121`
  - `infer.py:155-271`

Legacy double:

- `predict_step` returns `expression_pred`, `ny_prob1`, and `ny_prob2`.
- `on_predict_epoch_end` computes printed metrics mainly for task 2 when labels are available.
- Relevant legacy lines:
  - `model/trainer_dd.py:530-637`

Current:

- `infer.py` runs a manual `torch.no_grad()` loop instead of `trainer.predict()`.
- Writes structured `predictions.parquet`/CSV and `metrics.json`.
- Computes task1 and task2 binary metrics.
- Computes full expression validation metrics only when `--save-expression-pred` is enabled.
- Does not compute per-cell top-k recall/enrichment/rank reports.
- Does not use DDP/all-gather for inference.
- Relevant current lines:
  - `infer.py:275-369`
  - `model/training_ready_lightning.py:627-648`

Assessment: current inference is more artifact-oriented and less print/report oriented. It is not equivalent to legacy distributed predict hooks or legacy per-cell reporting.

## train.py Differences

### 1. Unified entrypoint instead of separate single/double scripts

Legacy:

- Single-drug and double-drug use separate entrypoints: `train.py` and `train_dd.py`.
- Relevant legacy lines:
  - `train.py:167-224`
  - `train_dd.py:170-225`

Current:

- One `train.py` handles both single-drug and double-drug tasks through the same double-drug perturbation contract.
- Relevant current lines:
  - `train.py:197-269`

Assessment: intentional divergence required by current Process 4.

### 2. Model registry is reduced by design

Legacy:

- Contains many model branches.
- Relevant legacy lines:
  - `train.py:336-458`
  - `train_dd.py:326-380`

Current:

- Active choices are only `attention_v10_hetero_cls_ee` and `baseline_emb_v3`.
- Relevant current lines:
  - `train.py:204`
  - `model/training_ready_models.py:29-38`

Assessment: intentional divergence based on user decision.

### 3. Validation policy differs by user request

Legacy:

- `train_dd.py` passes the test dataloader as validation dataloader and does not run a separate post-fit test.
- Relevant legacy lines:
  - `train_dd.py:384-394`

Current:

- Uses `valid_indices` for validation when present.
- Falls back to test only if valid is empty.
- After fit, runs `trainer.test()` with the best validation checkpoint unless `--skip-test` is set.
- Relevant current lines:
  - `train.py:107-123`
  - `train.py:432-446`

Assessment: intentional divergence; this follows the user-confirmed requirement.

### 4. Batch covariate defaults differ

Legacy single:

- Default batch covariates include `machineID_new`, `Cell_plate`, `Cell`, `cell_type`, `batch`, `pert_time`.
- Relevant legacy lines:
  - `train.py:317-323`

Legacy double:

- Default batch covariates include `machineID_new`, `Cell_plate`, `Cell`, `batch`, `pert_time`; `cell_type` only when `--add_cell_type`.
- Relevant legacy lines:
  - `train_dd.py:292-301`

Current:

- Default batch covariates are `machineID_new`, `Cell_plate`, `Cell`, `cell_type`, `pert_time`.
- `batch` is not included by default because the current training-ready tables do not expose `batch_index`.
- Relevant current lines:
  - `train.py:240`
  - `dataset/training_ready_dataset.py:16-24`

Assessment: intentional format adaptation, but not exact legacy covariate behavior.

### 5. Graph handling is reduced to PDI only

Legacy:

- Supports `hetero`, `ppi`, `pdi`, `hetero_onlypos`, `hetero_ddi`, and thresholding.
- Relevant legacy lines:
  - `train.py:151-164`
  - `train_dd.py:154-167`

Current:

- Only the consolidated PDI hetero graph model is active.
- PDI matrix defaults to current `[drug, protein]` orientation and is transposed for the legacy graph.
- Relevant current lines:
  - `train.py:232`
  - `train.py:286-287`
  - `model/training_ready_models.py:209-256`

Assessment: intentional divergence.

### 6. Checkpoint behavior differs

Legacy:

- Training entrypoints do not have the current strict initialization checkpoint path.
- Inference uses `load_from_checkpoint`.
- Relevant legacy lines:
  - `infer.py:60-75`
  - `infer_dd.py:48-80`

Current:

- `train.py --checkpoint-path` loads model weights before training with strict key matching by default.
- Partial loading requires `--allow-partial-checkpoint-load`.
- Relevant current lines:
  - `train.py:185-194`
  - `train.py:263-268`
  - `train.py:333`

Assessment: intentional safety improvement, not legacy behavior.

### 7. Other legacy train options not carried over

Current `train.py` does not preserve these legacy options:

- `--training_percentage`
- `--filter_sensitive`
- `--delete_cov`
- `--graph_threshold`
- `--log_to_wandb`
- top-k protein selection
- multiple graph types beyond PDI

Assessment: mostly intentional or previously scoped out. If exact legacy training experiments need these, they must be explicitly reintroduced.

## infer.py Differences

### 1. Inference input source differs

Legacy:

- Uses separate `--inference_folder` with `row_to_set_index.pkl`, `set_info.pkl`, and `inference_indices.pkl`.
- Uses separate `--inference_parquet_path`.
- Relevant legacy lines:
  - `infer.py:320-323`
  - `infer.py:418-430`
  - `infer_dd.py:188-190`
  - `infer_dd.py:290-299`

Current:

- Uses training-ready task artifacts and Step 3 split artifacts by default.
- Can infer all anchors by setting `--split-strategy ""`.
- Relevant current lines:
  - `infer.py:141-150`
  - `infer.py:153-203`
  - `infer.py:216-236`

Assessment: intentional divergence for the new data layout.

### 2. Inference execution differs

Legacy:

- Wraps model in Lightning and uses `trainer.predict()`.
- Prediction hooks can perform distributed all-gather.
- Relevant legacy lines:
  - `infer.py:105-121`
  - `infer_dd.py:84-95`

Current:

- Uses manual `torch.no_grad()` over a single `DataLoader`.
- Moves nested batches to one requested device.
- Relevant current lines:
  - `infer.py:50-55`
  - `infer.py:272-304`

Assessment: not equivalent for distributed inference. Fine for single-device inference.

### 3. Output format differs

Legacy:

- Primarily prints metrics; structured file output is not the main behavior.

Current:

- Writes `predictions.parquet` or CSV fallback, `metrics.json`, optional `expression_pred.npy`, and `run_manifest.json`.
- Relevant current lines:
  - `infer.py:58-66`
  - `infer.py:311-369`
  - `infer.py:370-401`

Assessment: intentional improvement for reproducibility.

### 4. Checkpoint validation is stricter

Legacy:

- Loads through Lightning `load_from_checkpoint` and relies on class/key compatibility.

Current:

- Strict state dict loading by default.
- Checks sibling `run_manifest.json` for config mismatch unless explicitly bypassed.
- Relevant current lines:
  - `infer.py:69-76`
  - `infer.py:79-138`
  - `infer.py:193-202`
  - `infer.py:270-271`

Assessment: intentional safety improvement, not exact legacy behavior.

### 5. Legacy per-cell and top-k classification reports are absent

Legacy single:

- Prints per-cell AUROC/AUPRC and top-k recall / enrichment factor.
- Relevant legacy lines:
  - `infer.py:155-271`

Current:

- Computes global binary metrics for task1/task2.
- Does not group by cell or compute top-k recall/enrichment/rank.
- Relevant current lines:
  - `infer.py:331-342`

Assessment: remaining behavior gap if those reports are required.

## Current Single-Drug And Double-Drug Train/Test Readiness

The current data/code path can construct train/valid/test dataloaders for both single-drug and double-drug tasks under the unified double-drug contract:

- single-drug rows are represented as `pert_index1 + no`;
- double-drug rows are represented as `pert_index1 + pert_index2`;
- `train.py` rejects `test_only` splits for training;
- `infer.py` supports `test_only` splits for extra tasks.

This is behavior-compatible with the current training-ready data, but not identical to the old single-drug `pert_id` one-token behavior.

## Main Known Deviations To Keep On Record

1. Eval/test control sampling is deterministic now, not random like legacy.
2. Label/mask encoding is stricter and treats missing labels as masked, instead of silently converting many unknown values to negative.
3. Current train uses true validation splits and post-fit best-checkpoint testing; legacy double used test-as-validation.
4. Current train default `drop_last=False`; legacy single default was `drop_last=True`.
5. Current default batch covariates exclude legacy `batch`.
6. Current inference is single-device/manual-loop and does not preserve legacy Lightning `predict_step` all-gather behavior.
7. Current inference does not reproduce legacy per-cell top-k recall/enrichment/rank print reports.
8. Current checkpoint loading and manifest validation are stricter than legacy.
9. Current positive weights are actually wired into the wrapper; legacy `train_dd.py` had a CLI argument but did not pass it through.

## Recommendation

I would not describe the current stack as "exact original behavior preserved." A more accurate statement is:

> The current stack preserves the legacy training batch shape, the legacy-style double-drug loss/metric contract, and the requested legacy model internals where they are still in scope, while intentionally adapting storage, feature encoding, split usage, active model choices, PDI-only graph behavior, and artifact-oriented inference to the new training-ready pipeline.

If the next goal is stricter behavioral parity, the highest-impact remaining choices are:

1. Add a `--eval-control-sampling {deterministic,random}` option and default it according to the desired parity/reproducibility tradeoff.
2. Decide whether missing/unknown labels should stay masked, or whether any task should reproduce the old "unknown becomes negative" encoder behavior.
3. Decide whether legacy per-cell/top-k inference reports are required in addition to current `metrics.json`.
4. Decide whether current single-drug training should default `drop_last=True` when `task_name` is a single-drug task.
5. Decide whether distributed inference support is required now.
