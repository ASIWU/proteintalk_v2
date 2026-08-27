# PTV1 Pipeline And Experiment Review

Review time: 2026-06-03 21:04 HKT.

## Scope Reviewed

- New isolated PTV1 utilities in `utils/ptv1/`.
- New PTV1 experiment wrappers and reporter in `scripts/ptv1/`.
- Generated PTV1 outputs in `data/standardized/ptv1`, `data/training_ready/ptv1`, and `data/training_ready/ptv1/derived`.
- Formal PTV1 experiment artifacts for prefix `20260603_2037_ptv1`.

## Data Findings

- `ptv1_aivc` standardized rows: 15002.
- `ptv1_aivc` missing matched controls: 923.
- `ptv1_aivc` self-control rows: 942.
- `ptv1_extra_singledrug` standardized rows: 218.
- `ptv1_extra_singledrug` empty target lists: 4.
- `ptv1_extra_singledrug` controls were matched from PTV1 AIVC by cell plate; training-ready rows are 222 after adding 4 matched controls.
- PTV1 training-ready global meta has 5578 proteins and 128 perturbations.

## Split Findings

- `fixed_experiment_type` split has train/valid/test counts 7041/1481/799 with no overlap.
- PTV1 unseen-drug folds have zero pairwise test-drug overlap.
- Fold test-drug counts are 13, 13, 13, 13, and 12.
- `ptv1_extra_singledrug` uses `test_only` with 218 test anchors.

## Derived Artifact Findings

- Protein ESM embedding shape is 5578 x 1280 with 102 sequence fallback rows and no unresolved rows.
- Morgan drug embedding shape is 128 x 2048 with 2 SMILES fallback rows and no unresolved rows.
- PPI, DDI, and PDI matrix shapes match the PTV1 metadata ordering:
  - PPI 5578 x 5578;
  - DDI 128 x 128;
  - PDI 128 x 5578.
- Derived matrices were built from PTV1 index ordering, not sliced from PTV3 matrices.

## Experiment Findings

- Exp_11 completed on `fixed_experiment_type`:
  - AUROC 0.957636;
  - AUPRC 0.915971;
  - n-AUPRC 3.127609;
  - count 799.
- Exp_12 completed all five PTV1 unseen-drug folds:
  - mean AUROC 0.707712;
  - mean AUPRC 0.545776;
  - mean n-AUPRC 2.212531;
  - total count 13137.
- Exp_13 selected epoch 6 from exp_12 best epochs 9, 4, 4, 7, and 5, trained all PTV1 with max epochs 7, and wrote 218 extra single-drug predictions.
- Exp_13 extra single-drug overall metrics:
  - AUROC 0.593455;
  - AUPRC 0.576296;
  - n-AUPRC 1.121719;
  - count 218.

## Validation Performed

- `python -m py_compile utils/ptv1/*.py scripts/ptv1/report_ptv1_exp_results.py`
- `bash -n scripts/ptv1/*.sh`
- `python utils/ptv1/01_validate_ptv1_standardized.py`
- `python utils/ptv1/03_validate_ptv1_training_ready.py`
- one-batch exp_11 smoke with real graph features.
- one-batch exp_12 fold0 smoke with real graph features.
- one-batch exp_13 all-train plus extra inference smoke.
- formal exp_11, exp_12 all folds, and exp_13.

## Residual Risks

- PTV1 extra single-drug target annotations depend on PTV3 perturbation metadata lookup, by design for this task.
- PTV1 extra single-drug has no native expression matrix; training-ready expression rows come from matched PTV1 AIVC cell controls.
- The exp_11 user-facing name is "random split", but the implementation strategy remains `fixed_experiment_type` to match the existing artifact semantics.

