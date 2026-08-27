# 2026-06-12 12:00 HKT PatientVali260605v3 Infer/Feature/Code Review

## Scope

Reviewed whether the 5 patient validation datasets were fully inferenced, whether generated features were present and coverage-safe, and whether newly generated code had actionable bugs.

## Inference Coverage

- Latest run reviewed: `outputs/2026-06/2026-06-12/20260612_patientVali260605v3_inference_summary.json`.
- Prediction files: 8/8 expected tasks present.
- Total prediction rows: 876.
- Per-dataset supported rows inferenced:
  - P1 ovarian: 468 double-drug rows.
  - P2 lung2020: 7 single-drug rows, 12 double-drug rows.
  - P3 lung2024: 2 single-drug rows, 63 double-drug rows.
  - P4 colon: 202 double-drug rows.
  - P5 breast: 7 single-drug rows, 115 double-drug rows.
- Raw rows skipped by task builder: 370, all documented as unsupported >2-drug combinations for the current two-slot model.

## Feature Coverage

- Reviewed `data/training_ready_patientVali260605v3/ptv3/patientVali260605v3_build_summary.json` and `patientVali260605v3_feature_build_summary.json`.
- Derived features present with expected shapes:
  - protein ESM `[13246, 1280]`;
  - drug Morgan `[6131, 2048]`;
  - PPI `[13246, 13246]`;
  - PDI `[6131, 13246]`;
  - DDI `[6131, 6131]`;
  - patient Cell LLM `[1117, 4096]`;
  - cell-type LLM `[14, 4096]`.
- Independent coverage check found no out-of-range protein, drug, cell LLM, or cell-type LLM indices for all 8 tasks.

## Code Review Result

- `python -m py_compile` passed for the reviewed Python files.
- One actionable code issue found: the new default PTV3 Cell LLM embedding path points to `cell_llm_embedding_qwen3_4096_v2.npz`, while the checked/generated PTV3 default artifact is `cell_llm_embedding_qwen3_4096.npz`; PTV3 training/inference with `--cell-llm-mode frozen` and no explicit embedding path will fail.
