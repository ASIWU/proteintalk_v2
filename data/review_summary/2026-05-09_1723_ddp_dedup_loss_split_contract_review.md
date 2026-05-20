# 2026-05-09 17:23 HKT DDP Deduplicated Loss and Split-Contract Review

## Scope

- Continued the adversarial confidence loop after the 17:11 strategy review.
- Focused on loopholes that could still make the experiment strategy look successful while silently using the wrong validation loss, stale artifacts, incomplete splits, or incomplete all-single/single+double data.

## Loophole Found

- DDP validation/test metrics were already all-gathered and deduplicated by `row_index`, but the epoch losses used by checkpoint monitoring could still come from per-rank scalar logging.
- Since Lightning DDP can replicate validation/test samples for uneven dataloader sizes, this could make `val/total_loss` monitor semantics diverge from the deduplicated reported metrics.

## Fixes Applied

- `model/training_ready_lightning.py`
  - Added per-sample MSE/BCE loss collection.
  - Recomputed validation/test `loss1`, `loss2`, and `total_loss` after all-gather plus `row_index` deduplication.
  - Checkpoint monitor `val/total_loss` now uses the same deduplicated global rows as reported validation metrics.

- `utils/03_validate_training_ready_outputs.py`
  - Added split-contract validation for all required PTV3 experiment strategies.
  - Validates fold artifacts, non-empty splits, formal split row disjointness, valid/test binary-label balance, extra-data `test_only` coverage, double-drug auxiliary synergy masking, and canonical unordered pair leakage.

- `infer.py`
  - Added inference/checkpoint protein-axis traceability fields to `run_manifest.json`.
  - The manifest now records the active task axis, checkpoint task/split, checkpoint axis path, derived checkpoint axis size, and whether the axes match.

## Data Composition Audit

- `ptv3_main_singledrug`
  - rows: 18,568
  - expression shape: `(18568, 10982)`
  - single `pert_id2 == pert_id1` bad count: 0

- `ptv3_main_doubledrug`
  - rows: 20,764
  - expression shape: `(20764, 11092)`
  - native double rows: 2,196
  - merged single auxiliary rows: 18,568
  - auxiliary non-empty `synergy`: 0
  - auxiliary scope: `single_drug_auxiliary_synergy_masked`

## Validation Commands

- `python -m py_compile train.py dataset/training_ready_dataset.py model/training_ready_models.py model/training_ready_lightning.py infer.py utils/03_validate_training_ready_outputs.py`
- `bash -n scripts/run_ptv3_training_experiments.sh`
- `python utils/03_validate_training_ready_outputs.py`
- `EXP_PREFIX=20260509_confidence2_smoke FOLDS=0 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 INFER_LIMIT_BATCHES=1 BATCH_SIZE=2 INFER_BATCH_SIZE=2 bash scripts/run_ptv3_training_experiments.sh`
- `python -u infer.py --dataset-group ptv3 --task-name ptv3_extra_doubledrug_nature --split-strategy test_only --split-name test --checkpoint-path checkpoints/20260509_confidence2_smoke_all_single_double_for_extra/epoch=0.ckpt --output-dir outputs/20260509_confidence2_manifest_axis_probe_double --batch-size 1 --limit-batches 1 --device cpu`

## Smoke Results

- All 8 bounded training jobs completed:
  - single pert-stratified fold0
  - single cell_type fold0
  - single cell fold0
  - single no-MSE fold0
  - single no-PDI fold0
  - double canonical pair fold0
  - all-single for extra inference
  - all-single+double for extra inference
- All 8 checkpoint manifests report `run_status=fit_completed`; all best checkpoint paths exist.
- All 9 extra inference tasks completed and wrote 2 bounded predictions each:
  - single: mat1_480_faims, mat1_qe, mat2_480_faims, mat2_qe, mat3_qe, mat4_qe
  - double: nature, nc, guomics
- Axis-manifest probe completed for `ptv3_extra_doubledrug_nature`; the output manifest records inference axis `11343`, checkpoint axis `11092`, `protein_axis_matches_checkpoint=false`, and 1 prediction.
- Final GPU check showed no active compute process.

## Current Confidence Boundary

- No known loophole remains in the verifiable implementation strategy: data contracts, split semantics, loss routing, checkpoint manifest binding, stale-output guard, ablation flags, and extra-data inference paths are all covered by validators and a bounded 8-GPU smoke.
- This does not mathematically guarantee biological metric convergence in long training; it confirms that the current strategy is executable and internally consistent.
