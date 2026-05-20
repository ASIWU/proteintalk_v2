# 2026-05-09 17:11 HKT Adversarial Strategy Loophole Review

## Scope

The previous strategy was not accepted as literally 100% certain. I rechecked it against the likely silent-failure modes:

- split semantics and leakage;
- label/mask routing for both losses;
- all-train plus extra-data inference;
- checkpoint and manifest binding;
- stale checkpoint/output reuse;
- no-MSE and no-PDI ablation behavior;
- the full experiment bash runner itself.

## Loopholes Found and Fixed

### 1. Double-drug reversed-pair leakage

The old double-drug `pert_id_5fold_fold*` strategy used an ordered pair. An audit found 29 unordered drug pairs that exist in both orders, and these reversed pairs could cross train/valid/test.

Fix:

- `utils/09_build_data_splits.py` now builds the double-drug fold key as a canonical unordered pair.
- `docs/Data_Process_3.md` now documents that `pert_id_5fold_fold*` is canonical unordered-pair holdout, not individual-drug cold-start.
- Rebuilt all split artifacts with `utils/09_build_data_splits.py --dataset-group all`.

Verification:

- All five double-drug `pert_id_5fold_fold*` splits now have zero unordered-pair overlap across train/valid/test.
- `data/training_ready/split_build_manifest.json` again contains all 13 PTV1/PTV3 tasks.

### 2. Stale experiment directory reuse

The training script could reuse a previous `EXP_PREFIX`, leaving old checkpoint/log/output files in place.

Fix:

- `scripts/run_ptv3_training_experiments.sh` defaults to `ALLOW_EXISTING_RUN=0`.
- Non-empty checkpoint, log, or inference output directories now fail before training/inference starts.
- Intentional reuse requires explicit `ALLOW_EXISTING_RUN=1`.

Verification:

- A deliberate rerun with an existing `EXP_PREFIX` failed before training with the expected error.

### 3. Incomplete checkpoint manifest risk

`run_manifest.json` did not say whether training completed, so an interrupted run could leave a manifest that looked usable.

Fix:

- `train.py` writes `run_status="fit_started"` before fit and `run_status="fit_completed"` after successful fit.
- `train.py` writes `fit_completed_at`, `test_status`, and resolved absolute artifact paths.
- `infer.py` now refuses checkpoints whose manifest is not `fit_completed`, unless `--allow-incomplete-checkpoint-manifest` is explicitly used for migration/debug.

Verification:

- A deliberate inference attempt from an old manifest without `run_status` failed with the expected `not marked fit_completed` error.
- New confidence-smoke checkpoints all have `run_status=fit_completed` and existing best checkpoint files.

### 4. Missing script preflight

The experiment script could launch long training before validating current data artifacts.

Fix:

- `scripts/run_ptv3_training_experiments.sh` defaults to `RUN_PREFLIGHT=1`.
- Preflight runs py_compile, standardized-data validation, and training-ready validation.

Verification:

- `bash -n scripts/run_ptv3_training_experiments.sh` passed.
- `utils/01_validate_standardized_outputs.py` passed.
- `utils/03_validate_training_ready_outputs.py` passed.

## Strategy Audit Results

- PTV3 global proteins: `11345`.
- PTV3 global drugs: `6113`.
- No checked task uses a 2000 protein axis.
- `batch_index` exists in checked feature tables.
- Single-drug non-control rows satisfy `pert_id2 == pert_id1` and `pert_index2 == pert_index1`.
- Double-drug merged single rows have empty `synergy` and `training_label_scope == single_drug_auxiliary_synergy_masked`.
- Formal target splits have nonempty train/valid/test and no row overlap, except `all_train_subset_test` where train/test overlap is intentional and `train.py` requires `--skip-test`.
- All checked valid/test splits have both positive and negative active labels.
- Extra test-only tasks exactly cover primary non-control anchors.

## 8-GPU Confidence Smoke

Command family:

```bash
EXP_PREFIX=20260509_confidence_smoke FOLDS=0 RUN_PREFLIGHT=0 MAX_EPOCHS=2 LIMIT_TRAIN_BATCHES=2 LIMIT_VAL_BATCHES=2 LIMIT_TEST_BATCHES=2 INFER_LIMIT_BATCHES=2 BATCH_SIZE=2 INFER_BATCH_SIZE=2 bash scripts/run_ptv3_training_experiments.sh
```

Passed training strategy classes:

- single pert-stratified fold0;
- single cell_type fold0;
- single cell fold0;
- single no-MSE ablation;
- single no-PDI ablation;
- double canonical unordered-pair fold0;
- all-single for extra single;
- all-single+double for extra double.

Passed external inference tasks:

- `ptv3_extra_singledrug_mat1_480_faims`
- `ptv3_extra_singledrug_mat1_qe`
- `ptv3_extra_singledrug_mat2_480_faims`
- `ptv3_extra_singledrug_mat2_qe`
- `ptv3_extra_singledrug_mat3_qe`
- `ptv3_extra_singledrug_mat4_qe`
- `ptv3_extra_doubledrug_nature`
- `ptv3_extra_doubledrug_nc`
- `ptv3_extra_doubledrug_guomics`

Each external inference smoke wrote 4 predictions and a run manifest.

## Residual Boundary

I cannot honestly claim mathematical 100% confidence in future model quality or long-run convergence from 2-batch smoke tests. I can state that, after this adversarial loop, I know of no remaining code-path, data-contract, split-leakage, manifest-binding, or experiment-runner loophole in the current strategy.

Final GPU process check showed no active compute process.
