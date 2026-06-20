# 2026-06-02 11:00 PTV3 Single-Drug Transcriptome AnnData Format Review

- Reviewed `data/transcriptome_anndata/ptv3_single_nonablation` against `docs/EXTRA_DATA_FORMAT_README.md` and the user request for exp01/02/03/07 single-drug non-ablation AnnData exports.
- Used an independent subagent plus local read-only validation.
- PASS:
  - All 14 h5ad files are readable with `anndata.read_h5ad`.
  - Required `mix_pert` obs columns are present: `gene_pt`, `drug_pt`, `env_pt`, `CRISPR`, `control`, `split`, `cell_cluster`, and `dataset`.
  - Control rows have empty perturbation tokens; non-control rows have nonempty `drug_pt`; non-control rows have nonempty `drug_smiles_primary`.
  - `common_gene_axis` files share identical `var_names` with `9843` HGNC gene symbols.
  - `adata.uns["splits"]` contains exp01/02/03 fivefold splits plus exp07 in the main file and `extra_single_test_only` in extra files.
  - Source split alignment passed for 132 train/valid/test checks across common and native h5ad files: count, set, and order all match source split pkl sample IDs.
- FAIL / risk under strict format:
  - Main h5ad default `obs["split"]` includes `unused` for 158 perturbation rows that are not part of the valid source split anchors, while the format doc expects `train`/`val`/`test`.
  - Extra single-drug perturbation rows are inference-only and have all-NaN observed `X`; test matrices can be selected, but they are not observed post-perturbation expression matrices.
  - Some `drug_pt` values contain literal `+` characters, which can be misread as combination delimiter by default mix_pert settings.
  - Extra h5ad `uns["splits"]["extra_single_test_only"]` has empty `train_obs_names` and `valid_obs_names`; controls are recorded separately in `control_obs_names` and as compatibility `obs["split"] == "train"`.
- No files were deleted and no `rm` command was used.
