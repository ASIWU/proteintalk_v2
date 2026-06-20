# PTV1 Frozen Cell LLM Fine-tune Full Search Review

Reviewed at: 2026-06-08 17:05 HKT

Prefix: `20260608_ptv1_cell_llm_tune_v1`

## Summary

- Completed the formal 32-candidate frozen Cell LLM fine-tune search.
- Wrote the final report to `docs/2026-06-08_ptv1_frozen_cell_llm_fine_tune_report.md`.
- Consolidated reporter outputs:
  - `logs/20260608_ptv1_cell_llm_tune_v1_fine_tune_results.md`;
  - `outputs/20260608_ptv1_cell_llm_tune_v1_fine_tune_results.tsv`.

## Results

- exp_11 best: `mse050_drop010`, AUPRC `0.926324`.
- exp_12 best: `mse050_target_pdi`, mean5 AUPRC `0.584798`.
- Official exp_13 tied to exp_11 best:
  - `mse050_drop010` direct extra AUPRC/AUROC `0.561480 / 0.580062`;
  - `mse050_drop010` all_train extra AUPRC/AUROC `0.579478 / 0.603900`.
- Diagnostic cross-candidate exp_13 best: `mse025_graph_off`, direct extra AUPRC `0.604631`.
- The diagnostic cross-candidate result is not the official exp_13 selection because it changes graph parameters relative to exp_11.

## Audit

- Candidates: 32.
- Selectable under exp_11 threshold `0.883452`: 22.
- Training manifests: 224/224.
- exp_12 five-fold coverage: 32/32 candidates.
- Inference manifests: 64/64.
- exp_13 prediction files with 218 rows: 64/64.
- Runtime summary rows: 288/288, all status 0.
- Checkpoint directories with `.ckpt` files: 224/224.
- Formal training manifest settings passed: `dataset_group=ptv1`, `cell_llm_mode=frozen`, `accelerator=gpu`, and `limit_*_batches=1.0`.

## Residual Risk

- The cross-candidate exp_13 leaderboard can identify extra-set performance improvements, but it must not be treated as the official exp_13 result when the requirement is to evaluate the exp_11-selected parameterization.
