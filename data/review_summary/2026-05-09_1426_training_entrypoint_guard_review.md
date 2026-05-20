# 2026-05-09 14:26 HKT Training Entrypoint Guard Review

## Scope

- Re-run the loophole audit after fixing the generated split artifacts.
- Focus on runtime paths that could still make a correct artifact strategy unsafe.

## Loopholes Found

- `train.py` still had an empty-valid fallback that reused the test split as validation. Current generated formal splits no longer triggered this, but a bad/custom split could silently reintroduce checkpoint-selection leakage.
- `all_train_subset_test` intentionally draws valid/test subsets from train anchors. If `train.py` ran `trainer.test` on that strategy, the resulting internal test metric could be mistaken for a final held-out result.

## Fixes

- `train.py` now raises an error when a training split has no valid indices.
- `train.py` now requires `--skip-test` for `--split-strategy all_train_subset_test`; final evaluation for all-data training must use `infer.py` on external extra-data tasks.

## Evidence

- `py_compile` passed for:
  - `train.py`
  - `utils/09_build_data_splits.py`
  - `infer.py`
  - `dataset/training_ready_dataset.py`
  - `model/training_ready_lightning.py`
  - `model/training_ready_models.py`
- Formal corrected split dry run passed:
  - task: `ptv3_main_singledrug`
  - split: `cell_type_5fold_fold0`
  - output: expression `(1, 10982)`, response logits `(1, 1)`, synergy logits `(1, 1)`.
- Artificial empty-valid split failed as intended:
  - error: formal training requires a non-empty validation split to avoid selecting checkpoints on the test split.
- `all_train_subset_test` without `--skip-test` failed as intended.
- `all_train_subset_test` with `--skip-test` dry run passed.
- Full strategy audit passed.
- Final GPU process query had no active compute processes.

## Final Strategy Guardrails

- Formal 5-fold training requires non-empty train/valid/test splits.
- `all_train_subset_test` is only for fitting all available train anchors before external validation; its internal test subset must not be used for final claims.
- Extra-data validation should be run through `infer.py` with checkpoint config validation enabled.
