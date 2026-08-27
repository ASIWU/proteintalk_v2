# Exp32 Drug-ID To Raw-Metadata Mapping Review

- Review time: 2026-07-22 17:09 HKT (UTC+08:00)
- Scope: build a colleague-facing mapping from every drug ID in the epoch-2 Exp32 prediction CSV to raw drug name and raw SMILES fields.
- Existing predictions, training-ready artifacts, standardized data, and raw data were read only and were not modified.

## Outputs

- Shareable mapping: `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_drug_id_mapping.csv`
  - 3,217 rows and 6 columns.
  - Columns: `drug_id`, `raw_drug_id`, `drug_name`, `smiles`, `Smiles_no_chiral`, and `Smiles_with_chiral`.
  - UTF-8 with BOM was used for convenient spreadsheet opening.
- Detailed audit mapping: `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_drug_id_mapping_audit.csv`
  - 3,217 rows and 26 columns.
  - Preserves all distinct raw names/SMILES as JSON arrays, value counts, ID-resolution methods, exact Exp32 SMILES, source tasks, and raw source files.

## Mapping Method

- The authoritative ID universe and order came from `data/training_ready_exp32_organoid/ptv3/exp32_organoid_drug_scope.csv`.
- Standardized task rows supplied the exact resolved `pert_id1` / `pert_id2` used by Exp32.
- Standardized rows were aligned back to their raw rows by exact row index for extra tasks and by exact sample-ID order for main tasks.
- Eleven Exp32 source tasks were covered: main single, main double, six extra-single tasks, and three extra-double tasks.
- Raw side-specific columns were mapped independently for double-drug tables so anchor and library metadata were not mixed.
- The simple file uses the first non-empty value in deterministic Exp32 source-task order. The audit file retains every distinct value when the raw sources disagree.

## ID Interpretation

- 111 IDs are numeric main PTV3 perturbation IDs.
- 2,100 IDs are `L9200_*` main PTV3 perturbation IDs.
- 931 IDs are `extid::*`, derived from normalized raw external IDs.
- 75 IDs are `extsmiles::*`, derived from the first 12 SHA-1 characters of a raw SMILES because no explicit raw drug ID was available. Their `raw_drug_id` is intentionally blank.

## Validation

- Mapping rows / unique IDs: `3217 / 3217`.
- Mapping ID set equals both the Exp32 scope and the prediction `drug_id` set.
- Prediction rows per mapped drug: exactly 26 (13 samples x 2 devices).
- Non-empty `drug_name`: `3217 / 3217`.
- Non-empty raw `smiles`: `3216 / 3217`.
- Non-empty `Smiles_no_chiral`: `3217 / 3217`.
- Non-empty `Smiles_with_chiral`: `3217 / 3217`.
- The only missing generic raw `smiles` value is `L9200_1822` (Everolimus). Its raw source table does not have a generic side-SMILES column for this record, but both chiral SMILES fields and the exact Exp32 SMILES are present.
- Exact Exp32 SMILES found among the corresponding raw `smiles` / `Smiles_no_chiral` / `Smiles_with_chiral` candidates: `3217 / 3217`.
- Every detailed-audit JSON-list column parsed successfully.

## Multi-Source Values

- IDs with more than one raw drug-name value: 1,409.
- IDs with more than one generic raw `smiles` value: 1,075.
- IDs with more than one `Smiles_no_chiral` value: 158.
- IDs with more than one `Smiles_with_chiral` value: 299.
- These are retained in the audit file rather than silently discarded. The shareable file intentionally exposes one deterministic primary value per ID.

## Checksums

- Shareable mapping SHA-256: `6b853b0459fbc6f67214dfbd7c7a06521e35f96ac3dc70fa3afa182f94017838`.
- Detailed audit mapping SHA-256: `3f83c2d9eaad7ca749e3b8cbb7531e0f115820c109e469c72a07169143beaba6`.
