# 2026-06-02 13:59 HKT Unseen-Cell Evaluation Metric Review

## Scope
- Audited active-task AUPRC implementations in `model/fast_lightning.py`, `infer.py`, and `scripts/report_cell_drug_time_eval.py`.
- Checked whether evaluating exp03 unseen-cell at unique `(cell, pert, dose)` granularity changes 5-fold mean AUPRC enough to affect the target.

## Findings
- The row-level AUPRC implementation uses `sklearn.metrics.average_precision_score` on sigmoid probabilities after applying the label mask. No formula-level AUPRC bug was found in the audited fast-model and inference paths.
- `scripts/report_cell_drug_time_eval.py` already supports a unique single-drug `(Cell, pert_id1, pert_dose)` evaluation through `cell-drug-dose-avgtime`. It groups by `["_cell", "_drug_a", "_dose_a", "_drug_b", "_dose_b"]`; for exp03 single-drug evaluation, `_drug_b` and `_dose_b` are empty, so this is unique `(cell, pert, dose)`.
- No label conflicts were observed for exp03 in the audited 5-fold reports, so unique aggregation does not inflate AUPRC by dropping conflicting groups.
- For `20260601_llm_celltype_full_v1` exp03, row-level mean5 AUPRC is `0.777465924`; unique `(cell, pert, dose)` average-time mean5 AUPRC is `0.782651431` (`+0.005185507`).
- For `20260601_cell_drug_selected_full_v1` exp03, row-level mean5 AUPRC is `0.777001569`; unique `(cell, pert, dose)` average-time mean5 AUPRC is `0.781927599` (`+0.004926029`).

## Conclusion
- Changing the reporting granularity to unique `(pert, cell, dose)` is directionally effective because it removes duplicate time-point/replicate weighting, but the effect is small. It does not move current single-model 5-fold mean unseen-cell AUPRC close to `0.85`.
