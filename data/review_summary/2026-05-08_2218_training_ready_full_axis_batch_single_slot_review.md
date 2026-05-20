# 2026-05-08 22:18 HKT Training-Ready Full-Axis / Batch / Single-Slot Review

## Scope

- Fixed single-drug perturbation slots so `pert_id2 == pert_id1` instead of blank / `no`.
- Added `batch` to Stage-2 categorical indexing and default train/infer batch covariates.
- Restored regenerated `data/training_ready` artifacts to full task protein axes instead of stale x2000 matrices.
- Made `use_target` default for the active graph model.
- Kept the new no-group contract; legacy `group_size` remains permanently ignored.
- Used CPU only.

## Files Changed

- `utils/00_standardize_rawdata.py`
- `utils/02_build_training_ready_data.py`
- `dataset/training_ready_dataset.py`
- `model/training_ready_models.py`
- `train.py`
- `infer.py`
- `scripts/0507_training_stack_smoke.py`
- `docs/Data_Process.md`
- `docs/Data_Process_2.md`
- `docs/Data_Process_4.md`
- `docs/2026-04-15_data_standardization_session_summary.md`

## Regeneration

CPU-only commands run with `CUDA_VISIBLE_DEVICES=`:

- `utils/00_standardize_rawdata.py`
- `utils/02_build_training_ready_data.py`
- `utils/03_validate_training_ready_outputs.py`
- `utils/09_build_data_splits.py --dataset-group all`

## Validation

- `utils/03_validate_training_ready_outputs.py` passed.
- Main regenerated shapes:
  - `ptv3_main_singledrug`: `18568 x 10982`
  - `ptv3_main_doubledrug`: `20764 x 11092`
  - `ptv1_aivc`: `15002 x 5576`
  - `ptv1_extra_singledrug`: `186 x 5576`
- `batch_index` exists in checked regenerated feature tables.
- Single-drug perturb slot checks:
  - `ptv3_main_singledrug`: `0` mismatches
  - `ptv3_extra_singledrug_mat1_480_faims`: `0` mismatches
  - `ptv1_extra_singledrug`: `0` mismatches
  - `ptv1_aivc`: `0` nonempty `pert_id1` rows with blank/`no` `pert_id2`
  - `ptv3_main_doubledrug` merged single-drug rows: `0` mismatches
- `py_compile` passed for the touched code paths.
- `scripts/0507_training_stack_smoke.py` passed on CPU; smoke output reported `GPU available: False, used: False`.

## Embedding / Graph Artifact Check

Existing PTV3 derived artifacts still match regenerated PTV3 metadata:

- protein embedding: `(11345, 1280)`
- drug embedding: `(6113, 2048)`
- PPI: `(11345, 11345)`
- PDI: `(6113, 11345)`
- DDI: `(6113, 6113)`

Because this fix does not add/remove/reorder protein IDs or perturbation IDs, PTV3 protein embeddings, ligand/drug embeddings, PPI, PDI, and DDI do not need to be regenerated.
