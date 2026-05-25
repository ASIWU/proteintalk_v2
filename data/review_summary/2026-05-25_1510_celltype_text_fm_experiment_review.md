# 2026-05-25 15:10 HKT Cell-Type Text Foundation Experiment Review

## Scope

- Deployed an independent cell-type text foundation experiment under `celltype_text_fm/`.
- Tested whether biomedical semantic embeddings for `cell_type` labels can replace or supplement categorical `cell_type` embeddings for unseen-cell prediction.
- Did not modify data files, data-processing code, or root baseline model/training files.

## Implementation

- `celltype_text_fm/text_features.py`
  - Uses `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`.
  - Expands each cell type into 3 biomedical prompts.
  - Averages prompt CLS embeddings per cell type and L2-normalizes them.
  - Builds row-level `prior_features` with shape `(N, 768)`.
- `celltype_text_fm/train_text_celltype.py`
  - Reuses the root fast dataset/model/trainer utilities.
  - Injects SapBERT cell-type features through the existing `prior_features` path.
  - Supports replacing categorical `cell_type` with text embedding via `--drop-cell-type-covariate`.
- `celltype_text_fm/run_bottleneck_2gpu.sh` and `run_folds_2gpu.sh`
  - Run bottleneck or arbitrary folds across two GPUs.

## Download/Deployment

- Initial direct HuggingFace download made no progress for several minutes.
- `proxy_on2` was available from `~/.bashrc`; after enabling it, the SapBERT model downloaded and cached successfully.
- Dry-run passed with:
  - expression `(8, 10982)`;
  - response/synergy logits `(8, 1)`;
  - `prior_features=(8, 768)`.

## Results

All experiments used `single_cell_5fold`, batch size 256, 1 GPU per fold, full covariate UNK dropout `0.15`, and `MSE_WEIGHT=0.075`.

| Setting | Folds | Avg AUROC | Avg AUPRC | Notes |
| --- | --- | ---: | ---: | --- |
| Baseline | 0-4 | 0.934443 | 0.767008 | Existing reference |
| Add SapBERT feature, keep categorical `cell_type` | 2,4 | 0.924423 | 0.696546 | Fold4 improves, fold2 drops |
| Replace categorical `cell_type` with SapBERT feature | 0-4 | 0.929224 | 0.767230 | AUPRC flat, AUROC lower |
| Replace categorical `cell_type` + text logit scale `0.5` | 2,4 | 0.919530 | 0.665745 | Rejected |

Per-fold AUPRC for the best text setting:

- fold0: `0.829289`
- fold1: `0.753980`
- fold2: `0.630274`
- fold3: `0.819977`
- fold4: `0.802628`

## Conclusion

- This method is deployable and cleanly answers the question "input a cell type, get a generalizable embedding".
- It is not a meaningful performance improvement yet: AUPRC is only `+0.00022` over baseline and AUROC is lower.
- The useful interpretation is that SapBERT can replace categorical `cell_type` without losing AUPRC, but it does not solve the unseen-cell bottleneck.
