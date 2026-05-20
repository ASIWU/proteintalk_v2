# 2026-05-09 13:37 HKT Strategy Audit Guard Review

## Scope

- Re-audit the current experiment strategy after the user asked whether the strategy was fully reliable.
- Search for loopholes in training/inference support for:
  - single-drug 5-fold splits by pert_id, cell_type, and cell;
  - single-drug no-MSE and no-PDI ablations;
  - double-drug pert_id split;
  - all-single training to extra single inference;
  - all-single+double training to extra double inference.

## Loopholes Found

- `infer.py` checkpoint validation did not compare `task_head`; this allowed a response checkpoint to be passed to a synergy inference job without a hard error.
- Extra datasets use task-specific full protein axes. This is valid for `attention_v10_hetero_cls_ee`, because protein tokens are rebuilt from each task ordered protein index and shared protein embeddings, but it would be unsafe for axis-fixed models if not guarded.

## Fixes

- Added `task_head` to inference checkpoint config validation.
- Added resolved dataset/meta/protein embedding/drug embedding/PDI artifact paths and `pdi_mode` to inference checkpoint validation.
- Added exact ordered protein axis validation for non-graph/axis-fixed models.
- Kept graph model variable-axis support explicit for `attention_v10_hetero_cls_ee`.

## Evidence

- `python -m py_compile train.py dataset/training_ready_dataset.py model/training_ready_models.py model/training_ready_lightning.py infer.py` passed.
- Deliberate wrong-head test failed as intended:
  - current inference: extra double synergy;
  - checkpoint: all-single response;
  - error: `task_head: current='synergy' checkpoint='response'`.
- Guarded extra inference passed for:
  - `ptv3_extra_singledrug_mat1_480_faims`
  - `ptv3_extra_singledrug_mat1_qe`
  - `ptv3_extra_singledrug_mat2_480_faims`
  - `ptv3_extra_singledrug_mat2_qe`
  - `ptv3_extra_singledrug_mat3_qe`
  - `ptv3_extra_singledrug_mat4_qe`
  - `ptv3_extra_doubledrug_nature`
  - `ptv3_extra_doubledrug_nc`
  - `ptv3_extra_doubledrug_guomics`
- Guarded inference output audit passed: each output manifest has expected task, expected `task_head`, expected active label key, no checkpoint mismatch override, `limit_batches=2`, and 4 prediction rows.
- Strategy data audit passed. Notes:
  - extra single mat1/mat2 axes: 10169 proteins vs main single 10982;
  - extra single mat3/mat4 axes: 11092 proteins vs main single 10982;
  - extra double nature/nc axes: 11343 proteins vs main double 11092;
  - extra double guomics axis: 11092 proteins, same length as main double;
  - all-single+double train split has 17986 merged single rows and 1791 active synergy rows.
- Final GPU process query had no active compute processes.

## Remaining Assumptions

- The current final strategy is for the graph attention model `attention_v10_hetero_cls_ee`.
- Internal `all_train_subset_test` valid/test splits are smoke-monitor subsets from the training universe; final external claims should use the dedicated extra-data inference tasks.
- Generated embeddings and graph matrices are treated as fixed upstream artifacts and were not regenerated in this audit.
