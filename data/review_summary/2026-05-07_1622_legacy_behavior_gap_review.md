# 2026-05-07 16:22 HKT Legacy Behavior Gap Review

## Scope

Reviewed the current training-ready implementation against the legacy ProteinTalk v2 code under:

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/dataset`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train.py`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train_dd.py`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/infer.py`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/infer_dd.py`

Also re-read the current process docs, especially `docs/Data_Process_4.md` and
`docs/2026-04-15_data_standardization_session_summary.md`.

## Bottom Line

The current codebase can execute training/inference smoke tests on the new
training-ready data, but it does **not** preserve the original legacy model
behavior. The biggest mismatch is `model/training_ready_models.py`: it exposes
the requested model names, but most names are backed by one compact shared
`DoubleDrugContextModel`, not by the legacy internal architectures.

This is a blocking compatibility issue if the goal is to reproduce or continue
from the original ProteinTalk v2 behavior.

## Label Coverage Policy Update

Implemented the user decision for missing labels in
`utils/09_build_data_splits.py`.

Current behavior:

- Partial missing labels are recorded in `split_manifest.json`.
- If the checked label column is missing entirely, or every checked anchor has
  an empty value, the split builder emits `RuntimeWarning`.
- The manifest now records `checked_anchor_count`, `missing_anchor_fraction`,
  `all_labels_missing`, and `warning`.

Current regenerated manifests show no all-missing checked label columns:

- `ptv1/ptv1_aivc`: `PRISM1st_label_total`, `13137` checked, `0` missing.
- `ptv1/ptv1_extra_singledrug`: `PRISM2nd_label_total`, `182` checked, `0` missing.
- `ptv3/ptv3_main_singledrug`: `PRISM1st_label_total`, `17986` checked, `0` missing.
- `ptv3/ptv3_main_doubledrug`: `synergy`, `1791` checked, `0` missing.
- All PTV3 extra single/double tasks: checked label column status `ok`.

Relevant current code:

- `utils/09_build_data_splits.py:458` implements label coverage auditing.
- `utils/09_build_data_splits.py:462` warns when the checked label column is absent.
- `utils/09_build_data_splits.py:480` detects all-empty labels.
- `utils/09_build_data_splits.py:489` records `all_missing_labels`.

Verification:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python -m py_compile utils/09_build_data_splits.py dataset/training_ready_dataset.py model/training_ready_lightning.py model/training_ready_models.py train.py infer.py
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python utils/09_build_data_splits.py --dataset-group all
```

## Dataset / Dataloader Differences

### What Matches

The sampling contract is close to legacy:

- Current `ProteinTalkDataset.__getitem__` samples a control group and a
  perturb group, and always includes the anchor perturb in train mode
  (`dataset/training_ready_dataset.py:261`).
- Legacy single and double `DatasetCSV._sample_pair` follow the same high-level
  sampling shape (`dataset_csv.py:127`, `dataset_csv_dd.py:121`).
- Evaluation repeats one control and one anchor to `group_size` in both current
  and legacy implementations (`dataset/training_ready_dataset.py:272`,
  `dataset_csv.py:154`, `dataset_csv_dd.py:148`).

### Important Differences

1. Storage format:
   - Current reads `feature_table.parquet/.pkl/.csv` plus
     `feature_expression_matrix.npy` (`dataset/training_ready_dataset.py:37`,
     `dataset/training_ready_dataset.py:125`).
   - Legacy reads a monolithic parquet where `expressions_hvg` is already a
     row-level object column (`dataset_csv.py:145`, `dataset_csv_dd.py:139`).

2. Feature encoding:
   - Current assumes process-2 has already created index columns for batch
     covariates and pert ids (`dataset/training_ready_dataset.py:294`).
   - Legacy uses `embedding_methods` at dataloader time to convert strings into
     embeddings or ids (`dataset_csv.py:26`, `dataset_csv_dd.py:23`).

3. Drug representation:
   - Current always returns a two-drug tensor: `pert_id = [pert_index1, pert_index2]`
     for graph models or `[2, drug_embedding_dim]` for non-graph models
     (`dataset/training_ready_dataset.py:300`, `dataset/training_ready_dataset.py:325`).
   - Legacy double-drug stores a merged `pert_id` value and splits `+` inside the
     dataset formatter (`dataset_csv_dd.py:187`).
   - Legacy single-drug and double-drug have separate dataset modules.

4. Label encoding:
   - Current encodes response and synergy labels centrally and marks unknown
     labels with mask `1` (`dataset/training_ready_dataset.py:94`,
     `dataset/training_ready_dataset.py:317`).
   - Legacy uses simpler task-specific encoders; for example double-drug synergy
     is positive only for `1`, otherwise `0` (`dataset_csv_dd.py:252`).

5. Expression NaN handling:
   - Current sanitizes control expression values to zero before model input
     (`dataset/training_ready_dataset.py:287`).
   - Legacy does not sanitize control expression in the dataset; the legacy
     `ValueEmbedding` has explicit NaN handling inside the model.

6. Batch covariates:
   - Current default is `machineID_new`, `Cell_plate`, `Cell`, `cell_type`,
     `pert_time` (`train.py:189`).
   - Legacy double-drug default is `machineID_new`, `Cell_plate`, `Cell`,
     `batch`, `pert_time`, with optional `cell_type` (`train_dd.py:292`).

## Trainer and Metrics Differences

### What Matches

- Current and legacy double-drug wrappers both optimize expression MSE plus two
  BCE losses (`model/training_ready_lightning.py:53`,
  legacy `model/trainer_dd.py:147`).
- Both use mask semantics where `1` means ignore and `0` means keep
  (`model/training_ready_lightning.py:73`, legacy `model/trainer_dd.py:188`).
- Both select the first group element for epoch-end metrics
  (`model/training_ready_lightning.py:113`, legacy `model/trainer_dd.py:298`).

### Important Differences

1. Metrics:
   - Current logs only loss, AUROC, and AUPRC for task 1 and task 2
     (`model/training_ready_lightning.py:133`, `model/training_ready_lightning.py:171`).
   - Legacy logs expression and classification metrics from
     `utils.eval_dd.compute_validation_metrics`: AUROC/AUPRC/accuracy,
     MSE/MAE/PCC/R2/direction accuracy, top-50 expression metrics, delta metrics,
     MMD, and energy distance (`model/trainer_dd.py:368`).

2. Optimizer / scheduler support:
   - Current supports `adamw`, `adam`, `sgd`, plus `cosine` and `plateau`
     schedulers (`model/training_ready_lightning.py:147`).
   - Legacy supports focal loss, positive class weights, `adamw_fused` parameter
     groups, `step`, and `cosine_warmup` in addition to basic optimizers
     (`model/trainer_dd.py:117`, `model/trainer_dd.py:530`).

3. Unfreeze behavior:
   - Legacy has `UnfreezeCallback` for `embedding_proj`
     (`model/trainer_dd.py:37`).
   - Current has no unfreeze callback.

4. Test policy:
   - Current now validates during fit and tests from the best validation
     checkpoint after fit (`train.py:325`, `train.py:332`).
   - Legacy `train_dd.py` calls `trainer.fit(...)` and returns; the old training
     path does not run `trainer.test` after fit (`train_dd.py:112`).
   - This current difference is intentional per user decision 9.

## train.py Differences

1. Entry point structure:
   - Current has one unified `train.py` for PTV1/PTV3 training-ready tasks
     (`train.py:164`).
   - Legacy uses separate single-drug and double-drug entry points
     (`train.py` and `train_dd.py`).

2. Data loading:
   - Current loads task artifacts from `data/training_ready/<group>/tasks/<task>`
     and split artifacts from `data/training_ready/<group>/splits/<task>`
     (`train.py:211`, `train.py:83`).
   - Legacy loads `data_path`, `preprocess_path`, `row_to_set_index.pkl`, and
     `train/test` split pickles manually (`train_dd.py:274`, `train_dd.py:316`).

3. Validation:
   - Current uses true `valid_indices` when present (`train.py:85`).
   - Legacy `train_dd.py` builds only train/test datasets and passes the test
     dataloader as validation (`train_dd.py:317`, `train_dd.py:387`).

4. Model registry:
   - Current exposes exactly the six requested model names (`model/training_ready_models.py:32`).
   - Legacy has many explored model branches; selected branches are only a
     subset (`train_dd.py:326`, legacy single `train.py` around the selected
     model branches).

5. Checkpoint compatibility:
   - Current loads checkpoint state dicts with `strict=False` (`train.py:156`).
   - Legacy inference/training uses Lightning `load_from_checkpoint`, which
     expects the legacy class structure (`infer_dd.py:48`).
   - Because current models are compact and not legacy architectures, current
     checkpoints and legacy checkpoints should not be treated as compatible.

## infer.py Differences

1. Current inference:
   - Builds the current compact model, runs a manual torch loop, writes
     `predictions.parquet`/CSV, `metrics.json`, optional `expression_pred.npy`,
     and `run_manifest.json` (`infer.py:151`, `infer.py:180`, `infer.py:209`).

2. Legacy inference:
   - Uses Lightning `trainer.predict` and each model wrapper's `predict_step`
     (`infer_dd.py:84`, `infer_dd.py:92`).
   - The double-drug legacy path computes/prints effective-key-2 metrics inside
     `on_predict_epoch_end`, but it does not write the same structured parquet
     and JSON outputs (`model/trainer_dd.py:530`).
   - The single-drug legacy path also includes per-cell printed metrics
     (`infer.py` legacy `evaluate_predictions_by_cell`).

3. Inference data:
   - Current reads the same training-ready task/split artifacts used for
     training (`infer.py:79`).
   - Legacy reads a separate `inference_folder` containing
     `row_to_set_index.pkl`, `set_info.pkl`, and `inference_indices.pkl`
     (`infer_dd.py:290`).

## Model Internal Differences

This is the critical compatibility gap.

1. `attention_v4_cls_ee`:
   - Legacy double-drug class uses `ValueEmbedding`, perturb tokens, batch
     covariate tokens, token-type embeddings, a CLS token, and
     `nn.TransformerEncoder` (`attention_dd.py:9`, `attention_dd.py:35`,
     `attention_dd.py:52`, `attention_dd.py:57`).
   - Current compact model uses one shared context model with a simple linear
     value projection, mean batch context, no token sequence, no CLS token, and
     no transformer (`model/training_ready_models.py:50`,
     `model/training_ready_models.py:143`, `model/training_ready_models.py:236`).

2. Target models:
   - Legacy `attention_v4_cls_ee_target` and `attention_v4_cls_ee_target_proemb`
     append target protein tokens and, for proemb, use a gene/protein gate in
     the gene-token path.
   - Current target support mean-pools target proteins into one context vector
     (`model/training_ready_models.py:299`), which is not equivalent to legacy
     target token behavior.

3. Graph/PDI models:
   - Legacy graph construction creates a PyG `HeteroData` graph with
     `protein -> drug` `binds` edges and `drug -> protein` `rev_binds` edges
     (`model/graph.py:136`).
   - Legacy graph network uses `PDIOnlyProteinDrugNet`, two `HeteroConv` layers,
     and `WeightedSAGEConv_hetero` with edge weights (`model/basemodel.py:1051`).
   - Current graph support is `PDIEncoder`, a compact sparse one-hop aggregator
     over a `[drug, protein]` matrix (`model/training_ready_models.py:78`).
   - Therefore the current graph model is not the original PDI graph network.

4. PDI orientation:
   - Current artifacts are `[drug, protein]`, as recorded in
     `docs/data_process_summary_02.md`.
   - Legacy `create_pdi_only_graph` interprets `pdi_matrix.nonzero()` as
     `protein -> drug` edge indices, so it expects `[protein, drug]`
     orientation (`model/graph.py:144`).
   - To preserve the original graph implementation while using current
     artifacts, the default graph build must reverse/transpose current PDI to
     `[protein, drug]` before calling legacy `create_pdi_only_graph`.

5. `baseline_emb_v3`:
   - Legacy `Baseline_emb_v3` loads an external gene embedding dataset and uses
     a fusion MLP plus expression and classification heads.
   - Current `BaselineEmbV3` is only a name-compatible subclass of the shared
     compact model and does not implement that legacy baseline
     (`model/training_ready_models.py:324`).

6. Missing direct double-drug legacy variants:
   - Direct legacy double-drug implementations exist for `attention_v4_cls_ee`
     and `attention_v10_hetero_cls_ee_no_target`.
   - The selected names `attention_v4_cls_ee_target`,
     `attention_v4_cls_ee_target_proemb`, `attention_v10_hetero_cls_ee_gate`,
     and `baseline_emb_v3` are primarily present as single-drug legacy classes
     and need careful double-drug adaptation.
   - The adaptation should keep the original internal logic but use a two-drug
     perturb input by default.

## Environment Note

The `flow_v2` environment currently has:

- `torch 2.7.1+cu126`
- `torch_geometric 2.7.0`

So porting the original PyG graph implementation is feasible from a dependency
standpoint.

## Required Follow-up Implementation

To satisfy the user's confirmed requirement, the next code update should:

1. Replace the compact shared model internals with legacy-equivalent internals.
2. Preserve one shared public double-drug input contract, but route each selected
   model name to a legacy-faithful implementation.
3. Use the original PyG PDI graph path:
   - build `HeteroData` with `create_pdi_only_graph`;
   - use `PDIOnlyProteinDrugNet`;
   - transpose current `[drug, protein]` PDI artifacts to legacy
     `[protein, drug]` by default.
4. Restore legacy expression metrics in `model/training_ready_lightning.py`, or
   explicitly add them under new metric names while keeping current AUROC/AUPRC.
5. Preserve the current user-requested training policy: validate during training
   and test only from the best validation checkpoint after training.

