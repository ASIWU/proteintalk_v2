# 2026-05-19 10:48 HKT Extra Raw With Controls File Lookup Review

## Scope

- Re-read `docs/Data_Process_1.md` through `docs/Data_Process_4.md`.
- Located the prior output for exporting PTV3 extra raw rows with generated `sample_id` and matched `control`.
- Checked output manifest, file names, row counts, and mapping status counts.

## Located Output

- Output directory: `data/standardized/ptv3/extra_raw_with_controls/`
- Export script: `utils/10_export_extra_raw_with_controls.py`
- Manifest: `data/standardized/ptv3/extra_raw_with_controls/export_manifest.json`
- Full mapping file: `data/standardized/ptv3/extra_raw_with_controls/extra_sample_id_control_mapping.csv`
- Non-empty mapping file: `data/standardized/ptv3/extra_raw_with_controls/extra_sample_id_control_mapping_nonempty.csv`

## Output CSVs

Single-drug extra raw files:

- `20260413ptv3_PRISM1st_validation_phenotype_mat1_480_faims_add_PRISM2nd_label_dup_with_sample_id_control.csv`
- `20260413ptv3_PRISM1st_validation_phenotype_mat1_qe_add_PRISM2nd_label_dup_with_sample_id_control.csv`
- `20260413ptv3_PRISM1st_validation_phenotype_mat2_480_faims_add_PRISM2nd_label_dup_with_sample_id_control.csv`
- `20260413ptv3_PRISM1st_validation_phenotype_mat2_qe_add_PRISM2nd_label_dup_with_sample_id_control.csv`
- `20260413ptv3_PRISM1st_validation_phenotype_mat3_add_PRISM2nd_label_dup_with_sample_id_control.csv`
- `20260413ptv3_PRISM1st_validation_phenotype_mat4_add_PRISM2nd_label_dup_with_sample_id_control.csv`

Double-drug extra raw files:

- `260423ptv3_Guomics_drug_combo_unique_with_smlies_with_sample_id_control.csv`
- `260424nc_drugComb_info_unique_with_smiles_with_sample_id_control.csv`
- `260424nature_drugComb_info_unique_with_smiles_with_sample_id_control.csv`

## Verification

- Full mapping rows: `398726`.
- Non-empty mapping rows: `133409`.
- Full mapping `control_export_status` counts:
  - `matched_allowed_control`: `133409`
  - `filtered_out_blank_control`: `265317`
- The annotated raw CSVs start with audit columns including `sample_id`, `control`, `source_row_index`, `kept_after_filter`, `stage2_filter_status`, and `control_export_status`.
- Rows filtered before Stage 2 are represented with blank `control`, matching the requested export behavior.
