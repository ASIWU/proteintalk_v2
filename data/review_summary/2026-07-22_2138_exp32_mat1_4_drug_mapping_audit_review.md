# Exp32 3,217-drug SMILES Origin and Mat1-4 Mapping Audit

- Review time: 2026-07-22 21:38 HKT (UTC+08:00)
- Scope: trace the exact raw origin of every Exp32 screened-drug SMILES; replay the mat1-4 extra-single mapping onto the main drug registry; detect inconsistent or chemically risky mappings.
- This review did not modify raw data, standardized data, model features, checkpoints, or prediction outputs.

## Exp32 Scope and Exact Raw SMILES Origins

- Exp32 does not read one standalone “3,000-drug file.” Its exact 3,217-ID scope is the non-control union of the 11 main/extra single- and double-drug task tables listed in `utils/32_build_exp32_organoid_training_ready.py::SOURCE_TASKS`, ordered by the global `pert_index`.
- The canonical per-ID SMILES is first-nonempty by task order. Within each raw single-drug row the selection priority is `Smiles_with_chiral`, legacy `smiles`, then `Smiles_no_chiral`; double-drug fields use their slot-specific equivalents.
- All 3,217 current Exp32 model SMILES were reconstructed exactly from raw with-chirality fields:
  - 3,023 from `Smiles_with_chiral`;
  - 101 from `Smiles1_with_chiral`;
  - 93 from `Smiles2_with_chiral`.
- Exact canonical-origin task counts are:
  - main single: 2,210;
  - mat1 480_FAIMS extra single: 745;
  - mat2 480_FAIMS extra single: 68;
  - NC extra double: 75;
  - Guomics extra double: 65;
  - Nature extra double: 53;
  - main double: 1.
- The origin CSV contains one row per Exp32 ID, the exact raw CSV, zero-based raw row, one-based physical CSV line, raw ID/name, all three SMILES variants, selected raw column, and exact equality to `global_meta.json["pertid_to_smiles"]`.
- The summary JSON inventories all 11 raw files with row counts and SHA-256 hashes.

## Mat1-4 Raw Sources

- mat1:
  - `data/rawdata/update_0527/extra_singledrug/260527ptv3_PRISM1st_validation_phenotype_mat1_480_faims_add_PRISM2nd_label.csv`;
  - `data/rawdata/update_0527/extra_singledrug/260527ptv3_PRISM1st_validation_phenotype_mat1_qe_add_PRISM2nd_label.csv`.
- mat2:
  - `data/rawdata/update_0527/extra_singledrug/260527ptv3_PRISM1st_validation_phenotype_mat2_480_faims_add_PRISM2nd_label.csv`;
  - `data/rawdata/update_0527/extra_singledrug/260527ptv3_PRISM1st_validation_phenotype_mat2_qe_add_PRISM2nd_label.csv`.
- mat3:
  - `data/rawdata/update_0527/extra_singledrug/260527ptv3_PRISM1st_validation_phenotype_mat3_add_PRISM2nd_label.csv`.
- mat4:
  - `data/rawdata/update_0527/extra_singledrug/260527ptv3_PRISM1st_validation_phenotype_mat4_add_PRISM2nd_label.csv`.

## Mapping Contract Replayed

- The only existing-drug registry used for mat1-4 resolution is the update-0623 main-single raw table:
  - `data/rawdata/update_0623/260513ptv3_EGH_28602sampinfo_with_smiles_check_prism1_label_add_prism2_label_add_machine_details.csv`.
- Exact strings from main `smiles`, `Smiles_no_chiral`, and `Smiles_with_chiral` are indexed. A SMILES is usable only when it resolves to exactly one main `pert_id`; ambiguous exact strings are excluded.
- Main `drugname` and semicolon/pipe/comma-split `synonyms` are normalized to lowercase alphanumerics. A normalized name is usable only when it resolves to exactly one main `pert_id`.
- Each mat row is then resolved in this exact order:
  1. exact legacy `smiles`;
  2. exact `Smiles_with_chiral`;
  3. exact `Smiles_no_chiral`;
  4. exact normalized `drug_name` against unique main names/synonyms;
  5. otherwise create `extid::<normalized drug_ID>`.
- This ID-resolution order is intentionally distinguished from the model-SMILES selection order. Model SMILES remains with-chirality first.

## Mapping Counts and Consistency

- Per-mat unique raw IDs: mat1 2,916; mat2 1,590; mat3 2,916; mat4 1,514.
- The union contains 4,506 unique raw drug IDs and 4,393 resolved IDs.
- Union-level resolution methods:
  - 2,820 external IDs not mapped to the main registry;
  - 716 exact `Smiles_with_chiral` mappings;
  - 654 exact legacy-`smiles` mappings;
  - 131 exact `Smiles_no_chiral` mappings;
  - 185 normalized-name mappings.
- The 1,686 raw IDs mapped to 1,573 existing main IDs. A total of 110 main IDs receive multiple raw IDs; the maximum is three. These collapses include aliases, stereochemical records, and formulation variants and are not automatically errors.
- Exact replay mismatch: 0 rows.
- Within-mat raw-ID conflict: 0 rows.
- Cross-mat/device inconsistency: 0 rows. All 4,506 raw IDs retain one resolved ID, one resolution method, one raw name, and one selected raw SMILES wherever repeated.
- mat1 and mat3 contain the same drug library/mapping. The QE and 480_FAIMS copies for mat1/mat2 also agree exactly.

## Chemical-Risk Audit

- Union-level outcome:
  - 2,820 `info`: intentionally external, so no main-structure comparison applies;
  - 1,353 `pass`: exact isomeric structure or an exact-connectivity name match;
  - 321 `review`;
  - 12 `high_risk` unique raw IDs.
- The 321 review items comprise:
  - 159 SMILES mappings with identical non-stereo connectivity but a stereochemical difference;
  - 72 name mappings explained by salt/charge/fragment-parent normalization;
  - 68 name mappings explained by tautomer-parent normalization;
  - 22 remaining name mappings with structural differences but Morgan Tanimoto at least 0.5.
- Current Morgan generation has `includeChirality=False`, so the 159 connectivity-equal stereo differences do not change the radius-2/2048 Morgan row. They may still matter to InChIKey/PDI or future chirality-aware features.
- The 12 high-risk IDs are all name-fallback mappings with Morgan Tanimoto below 0.5. They are exported separately and must not all be interpreted as confirmed identity errors:
  - `BRD-K64178227-001-09-0` (`BTS`) maps only through the main synonym `BTS` to `L9200_1320` (`Pyruvic acid`), while the raw aromatic sulfonamide and main pyruvic-acid structures are unrelated. This is a clear false-positive synonym mapping under the local evidence.
  - `BRD-K46386702-001-02-1` (`ARRY-334543`) maps only through a synonym to `L9200_2538` (`Xevinapant`) despite a major structure mismatch. This is a strong likely synonym error and should be blocked pending registry correction.
  - `BRD-K25570267-001-01-4` (`ezutromid`) and the two raw `tyloxapol` IDs match main primary names but have major raw/main structure conflicts. These are upstream metadata conflicts rather than proof that the name-to-ID association itself is wrong. In particular, main `L9200_40` is named Tyloxapol but its stored model SMILES is the tyramine-like `NCCc1ccc(O)cc1`.
  - Dextrose, N-acetyl-D-glucosamine, pralidoxime chloride, minoxidil, gadobutrol, and lubiprostone include open/closed-chain, resonance, salt/coordination, or related representation differences. They require chemical curation but should not be automatically rejected as wrong identities.

## Artifacts

- Exact Exp32 origin table:
  - `outputs/2026-07/2026-07-22/20260722_exp32_mat1_4_drug_mapping_audit_exp32_3217_smiles_origin.csv`
- Full per-mat unique mapping audit:
  - `outputs/2026-07/2026-07-22/20260722_exp32_mat1_4_drug_mapping_audit_mat1_4_unique_drug_mapping.csv`
- Deduplicated manual-review list (333 rows):
  - `outputs/2026-07/2026-07-22/20260722_exp32_mat1_4_drug_mapping_audit_manual_review_unique.csv`
- Deduplicated high-risk list (12 rows):
  - `outputs/2026-07/2026-07-22/20260722_exp32_mat1_4_drug_mapping_audit_high_risk_unique.csv`
- Machine-readable summary and raw-source hash inventory:
  - `outputs/2026-07/2026-07-22/20260722_exp32_mat1_4_drug_mapping_audit_summary.json`

## Verification and Limitations

- `scripts/audit_exp32_mat_drug_mapping.py` passed Python compilation and a full deterministic rerun in `flow_v2`.
- The 3,217 origin IDs exactly equal the Exp32 scope ID set, every reconstructed model SMILES equals the stored global-meta SMILES, and all mapping consistency/replay assertions passed.
- Structural risk is a curation signal, not an authoritative chemical registry verdict. The audit uses RDKit canonical structures, fragment/charge and tautomer normalization, and radius-2/2048 Morgan similarity. Polymer mixtures, coordination compounds, equilibrating forms, and incorrectly populated upstream SMILES require expert/manual adjudication.
