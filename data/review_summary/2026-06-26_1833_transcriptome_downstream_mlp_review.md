# 2026-06-26 18:33 HKT Transcriptome Downstream MLP Review

## Scope

- Reviewed the generated-transcriptome usage contract in `docs/2026-06-26_official_fixed_residual_transcriptome_usage.md`.
- Checked the Morgan fingerprint artifact structure at `data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl`.
- Read existing reporting patterns for mean5 metric aggregation and result Markdown/CSV output.

## Findings

- CPA/BioLord prediction h5ad files expose generated transcriptome features in `.X` with dimension `9843`; no latent `obsm` embedding is present.
- `PRISM1st_label_total` is the usable exp01/02/03 binary label for this downstream task.
- The Morgan pickle contains `item_to_index`, `index_to_item`, and an `embedding_matrix` with shape `(6131, 2048)`, suitable for direct lookup by string-normalized `drug_id`.
- The downstream experiment should remain separate from the PTV3 trainer because this task intentionally excludes controls, covariates, graph inputs, reconstruction MSE, and dose/time features.

## Action

- Added standalone train, orchestration, and reporting scripts for the CPA/BioLord transcriptome MLP experiment.
- Verified static compilation, full cpa/biolord exp01 fold0 data loading, and one-epoch smoke training before launching the full run in tmux `gpu2`.
