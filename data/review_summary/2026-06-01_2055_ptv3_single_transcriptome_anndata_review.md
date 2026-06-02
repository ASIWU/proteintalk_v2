# 2026-06-01 20:55 PTV3 Single-Drug Transcriptome AnnData Review

- Added `utils/12_export_single_drug_transcriptome_anndata.py`.
- Exported exp01, exp02, exp03, and exp07 single-drug non-ablation data to AnnData under `data/transcriptome_anndata/ptv3_single_nonablation`.
- Wrote both common-gene-axis h5ad files for cross-task train/inference and native-gene-axis h5ad files for per-task inspection.
- UniProt-to-HGNC mapping audit is in `mapping/uniprot_to_hgnc_gene_mapping.csv`; `33` ambiguous/unmapped UniProt features were dropped and duplicate gene mappings were collapsed by row-wise `nanmean`.
- Exact split membership is stored in `adata.uns["splits"]` and row-level `is_train/is_valid/is_test` columns. This preserves the intentional overlap in `all_train_subset_test`.
- Controls are encoded with `obs["control"]`, `matched_control_obs_name`, and `control_sample_id`; SMILES are stored in `drug_smiles` and `drug_smiles_primary`.
- Validation passed: script py_compile, h5ad backed reads, common-axis `var_names` equality, split count checks, and zero missing matched controls for checked test rows.
- No `rm` command was used.
