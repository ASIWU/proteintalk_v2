# 2026-05-07 15:20 HKT User Decision Follow-up Review

## Scope

Reviewed and updated the Step 3/4 data split and training behavior after user decisions on the open items from the 2026-05-07 training-readiness review.

## Decisions Applied

- Missing-control anchors remain skipped.
- Double-drug `pert_id` 5-fold keeps ordered `pert_id1 + pert_id2` as the unseen combination.
- All extra tasks remain `test_only`.
- Target UniProt IDs missing from `protein_index` continue to be dropped.
- PTV1 training remains out of scope for now.

## Code Changes

- `utils/09_build_data_splits.py`
  - PTV1 `fixed_experiment_type` now parses `data/rawdata/ptv1/experiment_type_list` directly in Step 3.
  - PTV1 `ptv1_aivc` now also gets a `random` train/valid/test split.
  - PTV1 split manifest records fixed-split pair counts and unmatched valid anchors.
- `train.py`
  - Added `--skip-test`.
  - Added `--limit-test-batches`.
  - After `trainer.fit`, testing now restores the best validation checkpoint when available.
  - Run manifest records best checkpoint and test checkpoint path.

## Generated Artifacts

Regenerated split artifacts with:

```bash
python utils/09_build_data_splits.py --dataset-group all
```

Current `ptv1_aivc` split strategies:

- `fixed_experiment_type`
- `random`
- `pert_id_5fold_fold0..4`
- `all_train_subset_test`

Current `ptv1_aivc` fixed split counts:

- train: 7041
- valid: 1481
- test: 799
- unmatched valid anchors from `experiment_type_list`: 3816

## Verification

Commands run:

```bash
python -m py_compile utils/09_build_data_splits.py train.py
python train.py --help
python utils/09_build_data_splits.py --dataset-group all
python train.py --task-name ptv3_main_doubledrug --split-strategy pert_id_5fold_fold0 --model-type attention_v4_cls_ee --batch-size 1 --group-size 2 --hidden-dim 8 --max-epochs 1 --limit-train-batches 1 --limit-val-batches 1 --limit-test-batches 1 --checkpoint-dir /tmp/proteintalk_v2_checkpoints --log-dir /tmp/proteintalk_v2_logs --experiment-name smoke_best_ckpt_test --batch-cov-list machineID_new Cell_plate Cell cell_type pert_time
```

Result:

- Compilation passed.
- `train.py --help` exposes `--skip-test` and `--limit-test-batches`.
- Split regeneration completed for all PTV1/PTV3 tasks.
- Smoke training completed and restored `/tmp/proteintalk_v2_checkpoints/smoke_best_ckpt_test/epoch=0.ckpt` for testing.

## Remaining Clarifications

- User asked for more detail before deciding whether missing checked labels should hard-fail split generation or remain audit-only.
- User asked for more detail on compact current model implementations versus exact legacy architectures before deciding whether legacy internals must be restored.

