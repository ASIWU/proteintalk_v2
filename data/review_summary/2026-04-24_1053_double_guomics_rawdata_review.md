# Review Summary

- Time: 2026-04-24 10:53
- Scope: review code compatibility with updated main double-drug rawdata and updated Guomics extra double-drug rawdata.

## Findings

- `utils/00_standardize_rawdata.py` still needed explicit path support for the new main double-drug files dated `20260422`.
- The new main double-drug expression matrix uses direct UniProt IDs and `sample_id`; this is compatible after adding the new filename to the direct-UniProt parsing rule.
- The new main double-drug info table includes expression-backed control samples whose raw `control` field is blank, with `pert_id1 == control` and `pert_id2 == control`; these must be normalized as self-control rows.
- The new main double-drug info table also provides side-specific raw smiles columns, so the standardizer should use them as fallback when the main single-drug registry lacks a pert_id mapping.
- Guomics extra double-drug now uses `260423ptv3_Guomics_drug_combo_unique_with_smlies.csv`; the old `260417` path is no longer present in the checkout.
- Guomics `Library_Primary.Pathway` is pathway metadata, not a direct target protein list, so it should be retained for audit rather than forced into target mapping.

## Outcome

- Patched `utils/00_standardize_rawdata.py`.
- Rebuilt `data/standardized/` and `data/training_ready/`.
- `utils/01_validate_standardized_outputs.py` passed.
- `utils/03_validate_training_ready_outputs.py` passed.
