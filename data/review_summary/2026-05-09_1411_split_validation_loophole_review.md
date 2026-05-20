# 2026-05-09 14:11 HKT Split Validation Loophole Review

## Scope

- Re-run the strategy audit loop after the previous inference-guard fix.
- Check whether every requested experiment family has safe train/valid/test split artifacts and can run without hidden fallback behavior.

## Loophole Found

- `ptv3_main_singledrug/cell_type_5fold_fold*` had empty validation splits.
- `train.py` intentionally falls back from empty valid to test for test-only style tasks, but for real 5-fold training this would select checkpoints using the test fold.

## Fix

- Updated `utils/09_build_data_splits.py` with `validation_item_count()`.
- The split builder now keeps validation non-empty when there are enough non-test groups/rows while preserving at least one train group/row.
- Regenerated all split artifacts with:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/09_build_data_splits.py --dataset-group all
```

## Evidence

- After regeneration, `ptv3_main_singledrug/cell_type_5fold_fold0..4` valid counts are:
  - fold0: `1177`
  - fold1: `1865`
  - fold2: `1865`
  - fold3: `1865`
  - fold4: `1865`
- All corrected cell-type folds have zero train/valid/test row overlap.
- Full strategy audit passed. It checks:
  - full protein axes are not 2000;
  - expression matrix columns match each task ordered protein axis;
  - `batch_index` exists;
  - single-drug valid anchors satisfy `pert_id2 == pert_id1` and `pert_index2 == pert_index1`;
  - main double merged single rows have masked/empty active `synergy`;
  - extra single and extra double test-only splits cover all valid anchors;
  - requested 5-fold split families have non-empty train/valid/test sets and no group leakage.
- 8-GPU bounded training smoke passed:
  - experiment: `20260509_strategy_cell_type_valid_fix_smoke`
  - split: `ptv3_main_singledrug/cell_type_5fold_fold0`
  - limits: `max_epochs=1`, `limit_train_batches=1`, `limit_val_batches=1`
  - manifest result: `valid_source=valid`, counts `train=9059`, `valid=1177`, `test=7750`, overlaps all `0`.
- `py_compile` passed for the split builder, training, inference, dataset, Lightning module, and model files.
- Final GPU process query had no active compute processes.

## Explicit Strategy Definition

- The current double-drug `pert_id_5fold_fold*` strategy is pair-level: the fold key is ordered `pert_id1+pert_id2`.
- It is not an individual-drug-cold split. If individual-drug-cold evaluation is needed, it should be added as a separate strategy rather than changing the meaning of the current pair-level strategy.
