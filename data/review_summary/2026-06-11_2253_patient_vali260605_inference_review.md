# 2026-06-11 22:53 HKT Patient Validation 260605 Inference Review

Reviewed the PTV3 data-processing and inference path for `data/rawdata/ptv2drug_patientVali260605`.

Scope:
- `docs/Data_Process_1.md` through `docs/Data_Process_4.md`
- `docs/Training_guideline.md`
- `utils/00_standardize_rawdata.py`
- `utils/02_build_training_ready_data.py`
- `utils/09_build_data_splits.py`
- `infer.py`
- `dataset/training_ready_fast_dataset.py`
- exp_07 and exp_08 checkpoint manifests

Findings:
- The new patient validation files have one info table and one baseline proteome matrix per cohort, with sample IDs aligned between info and matrix for all five cohorts.
- Drug tokens in the single/two-drug rows are covered by the current PTV3 `pert_index`.
- Protein columns are parseable as UniProt IDs; columns absent from the current PTV3 `protein_index` must be dropped for checkpoint-compatible inference.
- P2 and P5 contain multi-drug regimens with more than two unique drug tokens. These are not representable by the current two-slot exp_07/exp_08 models and were skipped.
- P3 contains negative normalized expression values, so it is not compatible with the standard log1p raw-abundance transform. P3 tasks were emitted and inferred only as non-standard/exploratory outputs.

Actions:
- Added `utils/13_build_patient_validation_inference_tasks.py`.
- Added `scripts/run_patient_vali260605_infer.sh`.
- Generated inference-only training-ready tasks under `data/training_ready/ptv3/tasks/ptv3_patientVali260605_*`.
- Generated corresponding test-only splits under `data/training_ready/ptv3/splits/ptv3_patientVali260605_*`.
- Ran exp_07 for single-drug tasks and exp_08 for double-drug tasks.
- Wrote merged predictions to `outputs/20260611_patientVali260605_combined/`.
- Wrote strict standard-compatible predictions to `outputs/20260611_patientVali260605_combined/combined_predictions_standard_only.csv`.

Verification:
- `python -m py_compile utils/13_build_patient_validation_inference_tasks.py`
- `bash -n scripts/run_patient_vali260605_infer.sh`
- `python utils/13_build_patient_validation_inference_tasks.py`
- `python utils/03_validate_training_ready_outputs.py`
- `bash scripts/run_patient_vali260605_infer.sh`

Result counts:
- Single-drug inference: P2 7 rows, P3 2 rows non-standard, P5 7 rows.
- Double-drug inference: P1 468 rows, P2 12 rows, P3 63 rows non-standard, P4 202 rows, P5 115 rows.
- Standard-compatible prediction rows: 811.
- Skipped as unsupported multi-drug rows: P2 8 rows, P5 362 rows.
