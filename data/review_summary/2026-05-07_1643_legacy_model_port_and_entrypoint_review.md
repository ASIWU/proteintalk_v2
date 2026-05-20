# 2026-05-07 16:43 HKT Legacy Model Port and Entrypoint Review

Update note, 2026-05-07 17:09 HKT: the trainer-metric and
control-expression NaN caveats in this review were addressed in
`data/review_summary/2026-05-07_1709_training_metrics_nan_forward_review.md`.
The remaining caveat is practical full-attention cost and explicit double-drug
adapter review for selected target/gate models that only existed as single-drug
legacy classes.

## Scope

Reviewed and updated the current training-ready code against the legacy
ProteinTalk v2 implementation:

- Current dataset/dataloader: `dataset/training_ready_dataset.py`, `train.py`
- Current model/trainer: `model/training_ready_models.py`,
  `model/training_ready_lightning.py`
- Current entrypoints: `train.py`, `infer.py`
- Legacy references:
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/dataset/dataset_csv.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/dataset/dataset_csv_dd.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/attention.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/attention_dd.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/basemodel.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/graph.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train_dd.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/infer.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/infer_dd.py`

## Code Update Implemented

1. `model/training_ready_models.py` no longer uses the previous compact
   context implementation.
   - Legacy-style `ValueEmbedding` with NaN embedding is restored.
   - Legacy stacked batch-covariate embeddings are restored.
   - The selected attention models now use gene / perturb / batch / target
     tokens, token-type embeddings, CLS token, and `nn.TransformerEncoder`.
   - Target-token models preserve the target protein token path.
   - `attention_v4_cls_ee_target_proemb` and
     `attention_v10_hetero_cls_ee_gate` preserve the gene/protein gate path.
   - All selected models return `(expression, response_logits, synergy_logits)`
     for the current double-drug trainer.

2. The PDI graph implementation now uses the legacy PyG path.
   - `create_pdi_only_graph` creates legacy `protein -> drug` `binds` edges and
     `drug -> protein` `rev_binds` edges.
   - `PDIOnlyProteinDrugNet` is ported from the legacy two-layer
     `HeteroConv`/`WeightedSAGEConv_hetero` implementation.
   - Current PDI artifacts are `[drug, protein]`, so
     `reverse_pdi=True` is the default and transposes to legacy
     `[protein, drug]` before graph construction.

3. `baseline_emb_v3` now uses the legacy baseline pattern instead of the
   compact Transformer replacement.
   - It loads a gene embedding `.npy` dataset, projects gene and perturbation
     features, fuses them with an MLP, and produces expression/classification
     heads.
   - It has an added synergy head for the current double-drug training contract.
   - Correctness depends on the gene embedding dataset row order matching the
     current feature table row order.

4. `train.py` and `infer.py` now expose the legacy model knobs needed to
   instantiate the restored internals:
   - `--fusion-mode`
   - `--num-heads`
   - `--num-layers`
   - `--cls-type`
   - `--graph-dropout`
   - `--target-protein-fusion-model`
   - `--gate-weight`
   - `--pdi-input-orientation`
   - `--emb-dataset-path`
   - `--gene-emb-dim`

Relevant current code:

- `model/training_ready_models.py:168` `PDIOnlyProteinDrugNet`
- `model/training_ready_models.py:218` `create_pdi_only_graph`
- `model/training_ready_models.py:268` `LegacyDoubleDrugTransformer`
- `model/training_ready_models.py:593` `BaselineEmbV3`
- `model/training_ready_models.py:694` `build_model`
- `train.py:187` legacy model CLI knobs
- `infer.py:108` legacy inference CLI knobs

## Dataset Class Check

Current dataset class:

- `TrainingReadyArtifacts` loads `feature_table.parquet/.pkl/.csv`,
  `feature_expression_matrix.npy`, `feature_ordered_protein_index.json`,
  `feature_sample_ids.json`, and `global_meta.json`
  (`dataset/training_ready_dataset.py:125`).
- `ProteinTalkDataset.__getitem__` preserves the legacy `{control, perturb}`
  batch contract and the train/eval group sampling shape
  (`dataset/training_ready_dataset.py:261`).
- `_format_rows` builds expression, row index, batch covariates, perturbation
  ids/embeddings, target protein list, response labels, synergy labels, and
  masks (`dataset/training_ready_dataset.py:284`).
- `_perturbation_indices` always returns `[pert_index1, pert_index2]`, so both
  single-drug and double-drug tasks use the same double-drug model contract
  (`dataset/training_ready_dataset.py:325`).

Differences from legacy:

1. Storage differs by design.
   - Current: split feature table plus expression `.npy`.
   - Legacy: monolithic parquet with row-level `expressions_hvg` objects.

2. Embedding/token conversion moved earlier in the pipeline.
   - Current: process-2 creates indexed columns and the dataset reads them.
   - Legacy: `DataSpliter` plus `embedding_methods` converted strings to
     embeddings/ids at dataset time.

3. Drug representation is unified.
   - Current: every perturb sample has two drug slots.
   - Legacy: single-drug and double-drug used separate dataset modules; double
     drug split merged ids from a `+` string.

4. Label handling is stricter and mask-aware.
   - Current: unknown/empty labels become mask `1`.
   - Legacy double-drug synergy treated only `1` as positive and most other
     values as `0`.

5. Control expression NaNs are sanitized to zero in the current dataset before
   model input.
   - Legacy dataset did not do this; legacy `ValueEmbedding` handled NaNs.
   - If exact NaN semantics are required, this is the remaining dataset-level
     behavior difference to change.

## Dataloader Check

Current dataloaders are created in `train.build_data_loaders` rather than a
separate dataloader class (`train.py:83`).

What matches legacy:

- Train mode samples controls and perturb rows by `set_info`.
- Eval mode repeats one control and one anchor row to `group_size`.
- Batch output is a nested `{control, perturb}` dictionary consumed directly by
  the models.

Differences from legacy:

1. Current uses process-3 split artifacts:
   - `train_indices_<strategy>.pkl`
   - `valid_indices_<strategy>.pkl`
   - `test_indices_<strategy>.pkl`
   - split-specific `*_set_info_<strategy>.pkl`

2. Current has a true validation dataloader when valid indices exist.
   - Legacy `train_dd.py` used the test dataloader as validation.
   - The current valid/test separation is intentional per user decision.

3. Current extra tasks that are `test_only` cannot be trained directly.
   - `train.py` raises when train indices are empty.
   - They remain inference/test-only tasks as requested.

Real-data dataloader smoke check:

```text
ptv3_main_singledrug batch control (1, 2, 10982), pert_id (1, 2, 2, 2048)
ptv3_main_doubledrug batch control (1, 2, 11092), pert_id (1, 2, 2)
```

This confirms the current dataset/dataloader can load both single-drug and
double-drug training batches under the shared two-drug contract.

## train.py Check

Current `train.py`:

- Loads training-ready artifacts and split artifacts.
- Builds `ProteinTalkDataset` train/valid/test dataloaders.
- Builds one of the six selected model names.
- Validates during `trainer.fit`.
- Tests only after training using the best validation checkpoint
  (`train.py:358`, `train.py:369`).

Differences from legacy:

1. Current is a unified PTV1/PTV3 single/double entrypoint.
   - Legacy used separate `train.py` and `train_dd.py`.

2. Current no longer uses `DataSpliter` or `embedding_methods`.
   - This matches the new data-process requirement that token/index work is
     already done before training.

3. Some trainer options remain not ported:
   - focal loss
   - positive class weights
   - `adamw_fused`
   - `step` and `cosine_warmup`
   - `UnfreezeCallback`

4. The restored legacy Transformer models attend over all feature proteins.
   - Current PTV3 batches have about 11k protein tokens.
   - Full real-data forward/training is therefore much heavier than the prior
     compact implementation and may require top-k protein selection or large GPU
     memory for practical runs.

## infer.py Check

Current `infer.py`:

- Reads the same training-ready task artifacts as training.
- Can infer from an existing split or from all anchor rows.
- Builds the same selected model class as `train.py`.
- Loads checkpoints with `strict=False`.
- Runs a manual torch loop and writes:
  - predictions parquet/csv
  - metrics JSON
  - optional expression prediction `.npy`
  - run manifest

Differences from legacy:

1. Legacy inference used Lightning `trainer.predict`; current inference uses an
   explicit torch loop.
2. Legacy inference expected separate inference folders; current inference
   reuses training-ready task/split artifacts.
3. Current writes structured outputs, which legacy inference mostly printed.
4. Checkpoint compatibility depends on the restored model architecture and
   matching CLI knobs. Old compact checkpoints should not be reused as legacy
   model checkpoints.

## Verification

Executed:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python -m py_compile model/training_ready_models.py train.py infer.py dataset/training_ready_dataset.py model/training_ready_lightning.py
```

Result: passed.

Synthetic forward smoke test:

- Built all six selected model names.
- Ran forward on a small double-drug synthetic batch.
- Verified each returns:
  - expression `(2, 2, 8)`
  - response logits `(2, 2, 1)`
  - synergy logits `(2, 2, 1)`

Real-data dataloader/model-construction smoke test:

- Built a PTV3 single-drug dataloader batch and non-graph model.
- Built a PTV3 double-drug dataloader batch and PDI graph model.
- Did not run full real-data Transformer forward over 11k protein tokens in the
  smoke test because that is a full training-scale attention workload.

## Current Answer to Train/Test Readiness

Current data and code can load train/valid/test batches for both main
single-drug and main double-drug tasks, and the selected model classes now
instantiate with legacy-style internals.

Remaining practical caveat:

- The legacy attention architecture is much heavier than the previous compact
  model because it attends over every protein token in the feature matrix. For
  PTV3 this is roughly 11k tokens per sample. That may be impractical without a
  top-k protein option, reduced feature set, or sufficiently large GPU memory.

Behavior caveats updated by the 17:09 review:

- Trainer metrics were restored to the legacy-style validation/test metric
  suite.
- Control-expression NaN sanitization was removed so `ValueEmbedding` handles
  NaNs.
- `baseline_emb_v3` requires a gene embedding dataset aligned to the current
  feature table row order.
