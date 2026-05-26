# 2026-05-25 12:01 HKT Foundation Cell Model Research Review

## Scope

- Reviewed whether single-cell foundation models or perturbation foundation models can provide explicit cell/cell-type information for unseen-cell prediction.
- Checked local compatibility with current `ptv3_main_singledrug` training-ready data.
- No data files or data-processing code were modified.

## Local Findings

- `ptv3_main_singledrug/feature_table.parquet` has 18,568 rows, 48 `Cell` values, 8 `cell_type` values, 2 `pert_time` values, and 2,152 single-drug perturbations.
- Existing external Geneformer row embedding file is present:
  `/mnt/shared-storage-user/beam/wuhao/H100/proteintalk/baseline/Geneformer/data/prot2gene_new/embed/geneformer_emb.npy`
  with shape `(28602, 768)`.
- Existing `baseline_emb_v3` already consumes that Geneformer-style row embedding, but the current fast baseline does not use it.

## Research Conclusion

- The lowest-risk next experiment is not full PerturbDiff integration. It is to add a frozen foundation-cell embedding path into `fast_delta` as explicit cell-state information.
- PerturbDiff is relevant as a later teacher/generative model because it explicitly conditions on cell type/covariates and models distribution-level perturbation response, but its released data/checkpoint workflow is large and scRNA-oriented.
- scFoundation/Geneformer/scGPT/UCE-style cell embeddings are more practical first because they can be generated once per control row or reused from the existing Geneformer embedding file.

## Proposed Experiment Order

1. Add `--cell-fm-embedding-path` to fast training and project the row/control embedding into `control_hidden`.
2. Add an ablation that replaces categorical `Cell` with frozen foundation embedding plus `cell_type`/time/batch covariates.
3. Add supervised cell-type contrastive or CE loss on the foundation embedding projection.
4. If useful, generate fresh scFoundation/Geneformer/UCE embeddings from mapped control proteome profiles.
5. Treat PerturbDiff/scGen-style models as teacher models only after the fixed embedding route shows signal.
