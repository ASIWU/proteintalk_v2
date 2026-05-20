# Group-Size Dimension Removal Review

Date: 2026-05-07 20:25 HKT

## Scope

Reviewed and modified the current training-ready stack to remove the historical `group_size` tensor dimension:

- `dataset/training_ready_dataset.py`
- `model/training_ready_models.py`
- `model/training_ready_lightning.py`
- `train.py`
- `infer.py`
- `scripts/0507_training_stack_smoke.py`
- `docs/data_process_summary_04.md`
- `docs/2026-04-15_data_standardization_session_summary.md`

No `attention.py` or standalone `dataset.py` exists in the current repo; the active equivalents are `model/training_ready_models.py` and `dataset/training_ready_dataset.py`.

## Resulting Data Contract

Dataset item, before DataLoader collation:

- `control["expressions_hvg"]`: `(n_genes,)`
- `perturb["expressions_hvg"]`: `(n_genes,)`
- graph `perturb["pert_id"]`: `(2,)`
- embedding `perturb["pert_id"]`: `(2, drug_embedding_dim)`
- `target_protein_list`: `(target_protein_max_length,)`
- labels and masks: scalar arrays
- batch covariates: scalar arrays

After DataLoader collation:

- `control["expressions_hvg"]`: `(batch_size, n_genes)`
- `perturb["expressions_hvg"]`: `(batch_size, n_genes)`
- graph `perturb["pert_id"]`: `(batch_size, 2)`
- embedding `perturb["pert_id"]`: `(batch_size, 2, drug_embedding_dim)`
- `target_protein_list`: `(batch_size, target_protein_max_length)`
- labels and masks: `(batch_size,)`
- batch covariates: `(batch_size,)`

Model outputs:

- expression prediction: `(batch_size, n_genes)`
- response logits: `(batch_size, 1)`
- synergy logits: `(batch_size, 1)`

## Code Changes Reviewed

Dataset:

- Removed the `group_size` constructor argument and stored attribute.
- Train mode now samples one random matched control row for the anchor perturb row.
- Eval/test mode now uses the deterministic sorted first matched control row.
- The dataset no longer samples `group_size - 1` extra perturb rows.
- `_format_row` now emits one row without a leading group axis.

Model:

- Removed `group_size` from `LegacyDoubleDrugTransformer`, `BaselineEmbV3`, and `build_model`.
- Removed all `bs * gs` flattening and reshape logic.
- Transformer path now builds token tensors as `(batch_size, n_tokens, hidden_dim)`.
- Graph perturb tokens are now `(batch_size, 2, hidden_dim)` before fusion.
- Batch covariate tokens are now `(batch_size, n_covariates, hidden_dim)`.
- Target protein tokens are now `(batch_size, target_protein_max_length, hidden_dim)`.
- Baseline model now reads embedding rows as `(batch_size, gene_emb_dim)`.

Trainer and metrics:

- `_losses` consumes expression labels `(batch_size, n_genes)`, logits `(batch_size, 1)`, labels `(batch_size,)`, masks `(batch_size,)`.
- Validation/test collection no longer indexes `[:, 0]`.
- Metric aggregation concatenates batch rows directly.
- BCE mask shape mismatch now raises a `ValueError` instead of trying to broadcast an old group-style mask.

Entry points:

- `train.py` no longer exposes `--group-size`, no longer records `group_size` in the manifest, and no longer passes it into dataset/model builders.
- `infer.py` no longer exposes `--group-size`, no longer passes it into dataset/model builders, and no longer slices `[:, 0]` from logits, labels, masks, or expression outputs.

## Verification

Passed:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python -m py_compile dataset/training_ready_dataset.py model/training_ready_models.py model/training_ready_lightning.py train.py infer.py scripts/0507_training_stack_smoke.py
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python train.py --help
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python infer.py --help
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python scripts/0507_training_stack_smoke.py
```

Search check:

- `rg -n "group_size|group-size|group size" train.py infer.py dataset model scripts utils` returned no code matches.

Real-data dry runs:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python train.py --dataset-group ptv3 --task-name ptv3_main_doubledrug --split-strategy pert_id_5fold_fold0 --batch-size 2 --hidden-dim 8 --num-heads 2 --num-layers 1 --dry-run-batches 1 --batch-cov-list machineID_new Cell_plate Cell cell_type pert_time --accelerator cpu --devices 1
```

Output:

- expression `(2, 11092)`
- response logits `(2, 1)`
- synergy logits `(2, 1)`

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python train.py --dataset-group ptv3 --task-name ptv3_main_singledrug --split-strategy random --batch-size 2 --hidden-dim 8 --num-heads 2 --num-layers 1 --dry-run-batches 1 --batch-cov-list machineID_new Cell_plate Cell cell_type pert_time --accelerator cpu --devices 1
```

Output:

- expression `(2, 10982)`
- response logits `(2, 1)`
- synergy logits `(2, 1)`

Real-data one-batch CPU fit:

- Command used `ptv3_main_doubledrug`, one train batch, one validation batch, `hidden_dim=8`, `num_layers=1`, CPU.
- Lightning initialization, sanity validation, and metric aggregation completed.
- The process was killed during the first training backward pass with exit 137.
- Interpretation: this is consistent with local CPU memory pressure from backward through the full PDI graph/full-token model. It did not expose a shape mismatch in the loader, forward, loss, or validation metric path.

## Remaining Risk

Full PTV3 training still uses about 11k protein tokens and the full PDI graph. Removing `group_size` reduces one tensor axis, but the full-token Transformer and graph backward are still memory-heavy. Real training should be run on the intended GPU environment with the desired batch size and monitored for memory.
