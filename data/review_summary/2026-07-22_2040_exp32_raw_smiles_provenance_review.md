# Exp32 Raw-SMILES Provenance Review

- Review time: 2026-07-22 20:40 HKT (UTC+08:00)
- Scope: identify the raw-data origin and semantics of the SMILES fields in the Exp32 drug-ID mapping, and publish an unambiguous corrected mapping.

## Finding

- The lower-case raw `smiles` field is a legacy source field. It is not uniformly defined as either chiral or non-chiral and must not be presented as the Exp32 model SMILES.
- Exp32 takes its model SMILES from `data/training_ready_exp32_organoid/ptv3/exp32_organoid_drug_scope.csv`, whose values come from the standardized global `pertid_to_smiles` registry.
- Standardization uses the priority `Smiles_with_chiral > smiles > Smiles_no_chiral`, followed by drug-ID-level canonicalization in source-task order.
- For the current 3,217-drug Exp32 scope, every exact model SMILES equals the corresponding selected raw `Smiles_with_chiral` value (`3217 / 3217`).
- Exact-string comparison of the legacy primary field found 268 values equal to the selected no-chiral string, 355 equal to the selected with-chiral string, and 2,862 equal to neither serialization. Different SMILES strings can still encode the same chemical structure; this comparison is about provenance and textual identity, not chemical inequivalence.

## Raw Sources

- Eleven raw CSVs contribute the Exp32 drug universe: the PTV3 main single table, main double table, six update-0527 extra-single tables, and three update-0527 extra-double tables.
- The complete path and column inventory is recorded in `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_drug_id_rawdata_source_inventory.csv`.
- Main and extra single tables expose lower-case `smiles` plus `Smiles_no_chiral` and `Smiles_with_chiral`; extra-double tables expose the corresponding side-specific `smiles1/2`, `Smiles1/2_no_chiral`, and `Smiles1/2_with_chiral` fields. The main double table has only the side-specific no-chiral and with-chiral fields.

## Corrected Output

- `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_drug_id_rawdata_mapping.csv`
  - 3,217 rows, 3,217 unique IDs, and 11 columns.
  - Uses the explicit names `raw_legacy_smiles` and `exp32_model_smiles` so the two meanings cannot be confused.
  - Includes raw ID/name, selected raw no-chiral/with-chiral values, exact matching field, raw task/file provenance, and raw occurrence count.
- The earlier `..._drug_id_mapping.csv` and `..._drug_id_mapping_audit.csv` were preserved unchanged as historical artifacts. The corrected `..._drug_id_rawdata_mapping.csv` is the recommended colleague-facing file.

## Validation

- Corrected mapping ID set equals both the 83,642-row prediction CSV drug-ID set and the 3,217-row Exp32 scope ID set.
- Corrected `exp32_model_smiles` equals the scope `smiles` for every ID.
- Corrected `exp32_model_smiles` equals `raw_Smiles_with_chiral` for every ID.
- All 11 inventory raw paths exist; standardized and raw row counts agree for every task.
- JSON task/file provenance fields parse successfully.
- `python -m py_compile scripts/build_exp32_drug_id_rawdata_mapping.py` and the script help check passed.

## Checksums

- Corrected mapping SHA-256: `4e886337c6255f4508d5edfe856babd50bc28695b7bbca025c187f8a947506e9`.
- Source inventory SHA-256: `e11a6d51298b91b50093f8fddc9dd6bab4c07c62a186abf2fed985b849ded6c5`.
