# 2026-06-23 13:04 HKT update_0623 Single-Drug Reprocess Review

## Scope

- Reviewed the Data_Process 1-4 workflow and current Stage 1/2/split scripts for replacing PTV3 main single-drug raw metadata with `data/rawdata/update_0623`.
- Compared old and new main single-drug sample metadata against the existing expression matrix.
- Rebuilt and validated standardized/training-ready artifacts and splits.

## Findings

- The new `update_0623` file has the same 28,602 `sample_id` values as the old main single-drug metadata and the expression matrix, so the existing proteome matrix can be retained safely.
- `update_0623` adds `PRISM2nd_*` columns. `PRISM2nd_label_total` is now preserved in main single-drug standardized and training-ready outputs for audit/merged-feature use.
- Stage 1 previously could not complete in this checkout because `ptv1_extra_singledrug/test12091214_sample_predictions_E115id.csv` and `ptds4_84drug_E115ID.csv` are absent from that directory but present under `data/rawdata/old`. A narrow fallback was added.
- The first GPU launch with online W&B stalled after W&B initialization. The active relaunch disables W&B logging and is training under prefix `20260623_130336_update0623_nowandb`.

## Evidence

- `utils/01_validate_standardized_outputs.py`: passed.
- `utils/03_validate_training_ready_outputs.py`: passed after split regeneration.
- `ptv3_main_singledrug` standardized output: 28,602 rows and 10,982 proteins, source metadata path is `data/rawdata/update_0623/260513ptv3_EGH_28602sampinfo_with_smiles_check_prism1_label_add_prism2_label_add_machine_details.csv`.
- `ptv3_main_singledrug` training-ready processed output: 18,359 rows and 10,982 proteins.
- GPU run log: `logs/20260623_130336_update0623_nowandb_exp01_08_tmux.log`.
