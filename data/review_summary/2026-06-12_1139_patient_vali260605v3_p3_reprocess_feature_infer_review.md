# 2026-06-12 11:39 HKT Patient Validation 260605 v3 P3 Reprocess + Feature + Inference Review

## Scope

- Reprocessed `data/rawdata/ptv2drug_patientVali260605` as `patientVali260605v3`.
- Corrected P3 expression handling from `negative_to_nan_then_log1p` to log2-scale handling.
- Regenerated v3 derived features and reran exp_07/exp_08 inference.

## Data Processing

- Output root: `data/training_ready_patientVali260605v3`.
- `P1/P2/P4/P5`: raw abundance scale, transformed by direct `log1p`.
- `P3_lungCancer_2024cell`: log2-relative scale, transformed by `log1p(exp2(x))`.
- P3 audit:
  - finite negative values before transform: `290916`;
  - negative values mapped to `NaN`: `0`;
  - negative values converted by `exp2`: `290916`;
  - transformed min/max: approximately `0.00252` / `5.98289`.
- All generated task matrices have finite negative count `0` and infinite count `0`.
- Skipped rows: `370`, all with reason `current_model_has_two_drug_slots`.

## Feature Generation

- Generated with `proxy_on2`.
- Artifacts:
  - drug Morgan fingerprint: `[6131, 2048]`;
  - protein ESM: `[13246, 1280]`;
  - PPI: `[13246, 13246]`;
  - DDI: `[6131, 6131]`;
  - PDI: `[6131, 13246]`;
  - patient Cell LLM: `[1117, 4096]`;
  - cell-type LLM: `[14, 4096]`.
- Feature-space audit:
  - raw unique proteins: `12678`;
  - appended protein count: `1901`;
  - unresolved protein columns: `0`;
  - raw unique drugs: `23`;
  - missing drugs: `0`;
  - raw unique patient Cell labels: `1116`;
  - missing cell types: `0`.

## Per-Task Coverage

- Every task's ordered protein index list is covered by:
  - `protein_embedding_esm.pkl`;
  - `ppi_matrix.npy`;
  - PDI matrix columns.
- Every task's perturbation indices are covered by:
  - `drug_embedding_morgan_2048.pkl`;
  - `ddi_matrix.npy`;
  - PDI matrix rows.
- Every task's `cell_llm_index` is covered by `cell_llm_embedding_patientVali260605v3_qwen3_4096.npz`.
- Every task's `cell_type_llm_index` is covered by `cell_type_llm_embedding_qwen3_4096_v2.npz`.
- Per-task coverage check returned `ALL_COVERAGE_PASS`.

## Inference

- Single-drug tasks used exp_07:
  - `checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp07_extra_single_all_train_infer_all_single_for_extra/epoch=5-step=426.ckpt`.
- Double-drug tasks used exp_08:
  - `checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra/epoch=2-step=234.ckpt`.
- Outputs:
  - `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_exp07_single/`;
  - `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_exp08_double/`;
  - `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_combined_predictions.parquet`;
  - `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_combined_predictions.csv`;
  - `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_inference_summary.json`.
- Total prediction rows: `876`.

## Verification

- `python -m py_compile utils/13_build_patient_validation_inference_tasks.py utils/14_build_patient_validation_features.py train.py infer.py`
- `bash -n scripts/run_patient_vali260605_infer.sh`
- `python -u utils/13_build_patient_validation_inference_tasks.py`
- `proxy_on2; CUDA_VISIBLE_DEVICES=0 python -u utils/14_build_patient_validation_features.py --protein-batch-size 8 --embedding-batch-size 32`
- `RUN_NAME=patientVali260605v3 bash scripts/run_patient_vali260605_infer.sh`
- Post-run audits confirmed transform metadata, feature shapes, per-task coverage, prediction row counts, and output files.
