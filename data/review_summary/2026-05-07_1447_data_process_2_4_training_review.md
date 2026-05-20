# 2026-05-07 14:47 HKT Data Process 2-4 Training Review

## Scope

Reviewed:

- `docs/Data_Process_2.md`
- `docs/Data_Process_3.md`
- `docs/Data_Process_4.md`
- `docs/data_process_summary_01.md`
- `docs/data_process_summary_02.md`
- `docs/data_process_summary_03.md`
- `docs/data_process_summary_04.md`
- Current `data/training_ready` task outputs and split manifests
- Current dataset/model/training/inference code paths

## Verification

Commands run in `flow_v2`:

```bash
python -m py_compile dataset/training_ready_dataset.py model/training_ready_models.py model/training_ready_lightning.py train.py infer.py utils/09_build_data_splits.py utils/02_build_training_ready_data.py utils/03_validate_training_ready_outputs.py
python utils/03_validate_training_ready_outputs.py
python train.py --task-name ptv3_main_singledrug --split-strategy random --model-type attention_v4_cls_ee --batch-size 1 --group-size 2 --hidden-dim 8 --dry-run-batches 1 --batch-cov-list machineID_new Cell_plate Cell cell_type pert_time
python train.py --task-name ptv3_main_doubledrug --split-strategy pert_id_5fold_fold0 --model-type attention_v4_cls_ee --batch-size 1 --group-size 2 --hidden-dim 8 --dry-run-batches 1 --batch-cov-list machineID_new Cell_plate Cell cell_type pert_time
```

Results:

- Python compilation passed.
- Stage-2 training-ready validation passed.
- Single-drug dry run produced `expression=(1, 2, 10982)`, response logits `(1, 2, 1)`, synergy logits `(1, 2, 1)`.
- Double-drug dry run produced `expression=(1, 2, 11092)`, response logits `(1, 2, 1)`, synergy logits `(1, 2, 1)`.

## Current Train/Test Status

- `ptv3_main_singledrug` has train/valid/test split artifacts and can enter the current training code path.
- `ptv3_main_doubledrug` has train/valid/test split artifacts and can enter the current training code path.
- PTV3 derived artifacts required by the default training code are present: drug embedding, protein embedding, and PDI matrix.
- Extra PTV3 single-drug and double-drug tasks are currently `test_only`; they are suitable as inference/evaluation targets after a checkpoint exists, not as training tasks.
- Current checkout checkpoint directories under `checkpoints/` contain run manifests but no `.ckpt` files. A checkpoint must be produced or supplied before regular `infer.py` can run from those folders.
- PTV1 task data and splits exist, but PTV1 derived embeddings/graph artifacts are not present except UniProt export files, so default model training for PTV1 is not ready without building those artifacts.

## Decisions Needed

1. Decide whether missing-control anchors should remain skipped or whether Process 2 should be changed so every candidate anchor maps to a control.
   Current skipped counts: `ptv3_main_singledrug` skips 158 anchors; `ptv1_aivc` skips 923 anchors.
2. Decide the exact double-drug holdout definition for `pert_id` 5-fold.
   Current implementation uses the ordered pair `pert_id1 + pert_id2`.
3. Decide whether PTV3 extra single-drug tasks should remain `test_only`.
4. Decide whether missing labels should hard-fail split generation or remain manifest-only audit entries.
5. Decide whether the feature table plus aligned `.npy` expression matrix contract is acceptable, or whether expression vectors must be embedded directly into `feature_table.csv` as originally requested in `Data_Process_2.md`.
6. Decide whether target UniProt IDs missing from `protein_index` should continue to be dropped from `target_protein_list` or mapped to a synthetic/special index.
7. Decide whether PTV1 fixed splits may rely on the training-ready `data_split` column or must directly parse `rawdata/ptv1/experiment_type_list` during split generation.
8. Decide whether exact legacy model internals are required for the selected model names. Current models are compact compatible implementations with the requested public names and double-drug input contract.
9. Decide whether `train.py` should add `--skip-test` and `--limit-test-batches`; current training always runs `trainer.test(...)` after fit.
10. Decide whether PTV1 should be made training-ready in the same way as PTV3 by building PTV1 drug/protein embeddings and graph artifacts.

