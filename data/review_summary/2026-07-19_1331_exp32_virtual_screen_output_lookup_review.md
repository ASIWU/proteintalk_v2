# 2026-07-19 13:31 HKT Exp32 Virtual-screen Output Lookup Review

## Scope

- Traced the output paths of `scripts/exp_32_organoid_exp09_single_sensitivity_infer.sh` and its reporter `scripts/report_exp32_organoid_exp09_single_sensitivity.py`.
- Located and inspected the formal raw task predictions, combined virtual-screen tables, per-sample top-20 table, and the 3,217-drug input scope.
- This was a read-only lookup; no code, data, inference, or output artifact was changed.

## Main Virtual-screen Results

- Combined CSV: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_predictions.csv`.
- Combined Parquet: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_predictions.parquet`.
- Both contain 83,642 rows and 18 columns, covering two devices (`B`, `CAC`), 13 sample pairs, and 3,217 drugs.
- The screening score column is `pred_sensitivity_prob`; rows also include patient/sample identifiers, tissue/cell type, drug ID, SMILES, dose, and time.

## Raw Inference Outputs

- QE task: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity/ptv3_exp32_organoid_qe_single/predictions.parquet`.
- 480 FAIMS task: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity/ptv3_exp32_organoid_480_faims_single/predictions.parquet`.
- Each raw task file contains 41,821 rows. The raw score is `pred_task_prob`; the reporter enriches and renames it to `pred_sensitivity_prob` in the combined screening tables.

## Supporting Files

- Per-sample top-20 screening table: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_top20_by_sample.csv`.
- Summary: `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_summary.json`.
- Input drug universe: `data/training_ready_exp32_organoid/ptv3/exp32_organoid_drug_scope.csv`, containing 3,217 drugs plus SMILES, target, source-task, Morgan, and graph coverage metadata.

## Conclusion

- For downstream virtual-screen analysis, use the combined CSV or Parquet rather than the raw per-task prediction files. The Parquet contains the same 83,642 rows while being substantially smaller on disk.
