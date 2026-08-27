# 2026-06-26 15:39 Official Fixed Residual Transcriptome Output Review

Scope:
- Reviewed `/mnt/shared-storage-gpfs2/beam-gpfs02/maoxinjie/AIVC/ptv3_single_nonablation/output/official_fixed_residual_20260605`.
- Compared outputs against `data/transcriptome_anndata/ptv3_single_nonablation/common_gene_axis`.

Findings:
- The output root contains two model branches: `cpa` and `biolord`.
- Both models have completed directories for all exp01, exp02, and exp03 fold splits plus exp07 all-train-for-extra inference.
- All 90 exp01/02/03 train/val/test prediction h5ad files exist and match the expected split counts from the transcriptome AnnData export.
- All exp01/02/03 prediction h5ad files use the 9843-gene common HGNC axis, and their non-control `obs_names` order matches the source common-gene-axis AnnData split membership.
- All 12 exp07 extra single-drug prediction h5ad files exist, use the same 9843-gene axis, and match the source extra AnnData non-control row order.
- `obs` contains `PRISM1st_label_total` for exp01/02/03 binary labels with no blank labels in the checked train/test splits. `PRISM2nd_label_total` is blank for exp01/02/03.
- Prediction h5ad files contain generated transcriptome matrices in `.X`; they do not contain model latent embeddings in `obsm` or layers.
- Sampled first/middle/last rows from all 102 prediction h5ad files and found no non-finite values in `.X`.

Risk:
- Several manifests and completion metadata still reference stale `/mnt/shared-storage-user/maoxinjie/...` absolute paths. Those paths do not exist in the current environment. Downstream scripts should use the actual gpfs output paths or rewrite the manifests before relying on them.
