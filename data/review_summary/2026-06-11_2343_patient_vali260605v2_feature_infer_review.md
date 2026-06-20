# 2026-06-11 23:43 HKT Patient Validation 260605 v2 Feature + Inference Review

## Scope

- Reviewed the existing data workflow in `docs/Data_Process_[1-4].md`, `docs/Training_guideline.md`, the training-ready builders, split/task loaders, `train.py`, `infer.py`, and exp_07/exp_08 inference settings.
- Reprocessed `data/rawdata/ptv2drug_patientVali260605` as `patientVali260605v2`.

## Changes

- Added/updated patient validation builders:
  - `utils/13_build_patient_validation_inference_tasks.py`;
  - `utils/14_build_patient_validation_features.py`.
- Updated inference feature loaders:
  - `train.py` and `infer.py` now support `--cell-llm-index-column` and `--cell-type-llm-index-column`.
- Updated runner:
  - `scripts/run_patient_vali260605_infer.sh` now points to `data/training_ready_patientVali260605v2` and the v2 derived artifacts.

## Feature Coverage

- Protein: 12,678 raw unique protein columns, 1,901 appended patient-only proteins, 0 unresolved columns.
- Drug: 23 raw unique drugs, 0 missing drugs; v2 Morgan fingerprint shape `[6131, 2048]`.
- Protein ESM: shape `[13246, 1280]`; 1,901 appended protein rows generated.
- Graph features: PPI `[13246, 13246]`, DDI `[6131, 6131]`, PDI `[6131, 13246]`.
- Cell: 1,116 raw patient Cell labels; generated patient Cell LLM embedding shape `[1117, 4096]` with reserved `no`.
- Cell-type: 0 missing cell types; reused existing PTV3 cell-type LLM artifact, shape `[14, 4096]`.
- Batch/machine: unseen values mapped to `no`, matching the current non-generalized design.

## Data Audit

- Negative expression handling: finite negative values are mapped to `NaN` before `log1p`.
- `P3_lungCancer_2024cell` contained 290,916 finite negative values and all were mapped to `NaN`.
- All generated task matrices have finite negative count 0.
- Skipped rows: 370, all with reason `current_model_has_two_drug_slots`; exp_07/exp_08 can only consume one- or two-drug inputs.

## Inference

- Single-drug tasks used exp_07:
  - `checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp07_extra_single_all_train_infer_all_single_for_extra/epoch=5-step=426.ckpt`.
- Double-drug tasks used exp_08:
  - `checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra/epoch=2-step=234.ckpt`.
- Per-task outputs:
  - `outputs/20260611_patientVali260605v2_exp07_single/`;
  - `outputs/20260611_patientVali260605v2_exp08_double/`.
- Combined outputs:
  - `outputs/20260611_patientVali260605v2_combined_predictions.parquet`;
  - `outputs/20260611_patientVali260605v2_combined_predictions.csv`;
  - `outputs/20260611_patientVali260605v2_inference_summary.json`.
- Total prediction rows: 876.

## Verification

- `python -m py_compile train.py infer.py utils/13_build_patient_validation_inference_tasks.py utils/14_build_patient_validation_features.py`
- `python utils/13_build_patient_validation_inference_tasks.py`
- `proxy_on2; CUDA_VISIBLE_DEVICES=0 python -u utils/14_build_patient_validation_features.py --protein-batch-size 8 --embedding-batch-size 32`
- `bash scripts/run_patient_vali260605_infer.sh`
- Post-run audits confirmed feature shapes, skip reasons, output rows, and finite negative count 0 for every generated task matrix.

## Notes

- `Cell_index` remains checkpoint-compatible because expanding trained categorical embedding tables would invalidate exp_07/exp_08 checkpoint shapes. Patient Cell generalization is supplied through the new `cell_llm_index` and patient Cell LLM embedding artifact.
