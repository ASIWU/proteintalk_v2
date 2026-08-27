# 2026-06-21 13:42 HKT PTV1/PTV3 Label Mismatch Report Review

## Scope

- Reviewed PTV1-flow `cell_5fold` label construction from raw `loo_label.csv` and `pheno.csv`.
- Reviewed original PTV3 `ptv3_main_singledrug` response label source, centered on `PRISM1st_label_total`.
- Rechecked corrected key-level comparison behavior and label-aligned PTV3 vs PTV1-flow result artifacts.

## Findings

- PTV1-flow and original PTV3 use different label namespaces for the same `(Cell, pert_id1, pert_id2)` keys: PTV1 uses numeric `pheno.csv`, while original PTV3 uses standardized `PRISM1st_label_total`.
- Across five PTV1 `cell_5fold` test folds, `3673` valid common keys were comparable and `469` labels disagreed, for a `12.77%` mismatch rate.
- PTV1 test folds contain many masked labels: `8756` keys have both 6h and 24h time points, but `5083` are `-1` and only `3673` enter 0/1 evaluation.
- PTV1 `predictions.csv` has an unreliable `experiment_type` column after `-1` masking; corrected key-level comparisons should reconstruct valid keys from raw split files.
- Label-aligned PTV3 comparison has `common_label_mismatch=0` across all folds and reaches mean AP/AUPRC `0.862943`, exceeding PTV1-flow corrected mean AP/AUPRC `0.860341`.

## Validation

- Confirmed per-fold original PTV1 vs PTV3 mismatch counts in `outputs/2026-06/2026-06-21/20260621_ptv1_ptv3_original_label_mismatch_cell_5fold.csv`.
- Confirmed label-aligned final metrics in `outputs/2026-06/2026-06-20/20260620_ptv3_ptv1label_cell_v1_covdrop020_mse075_vs_ptv1_5fold.csv`.
- Wrote the detailed report to `docs/2026-06-21_ptv1_ptv3_cell_label_mismatch_report.md`.

## Changes

- Added a documentation-only report. No code or experiment runner behavior was changed in this review.
