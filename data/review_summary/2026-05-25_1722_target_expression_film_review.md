# 2026-05-25 17:22 HKT Target-Expression and FiLM Review

## Scope

- Implemented and reviewed default-off experiments for the two proposed unseen-cell directions:
  - method 1: PDI/PPI-guided drug-specific target-expression context;
  - method 2: control-expression-driven pair FiLM.
- Files touched:
  - `model/fast_delta_model.py`
  - `train.py`
  - `infer.py`
  - `scripts/ptv3_experiment_common.sh`
  - `scripts/run_unseen_cell_targetexpr_film_fold0_2gpu.sh`
  - `scripts/run_unseen_cell_targetexpr_pairadd_5fold_2gpu.sh`

## Implementation Notes

- Target-expression weights are built from existing graph artifacts only:
  - direct PDI weights over the current expression gene axis;
  - optional one-hop PPI expansion from top PDI proteins;
  - row normalization and top-k filtering before caching under `graph_cache/`.
- The model converts cached dense weights into sparse top-k buffers at initialization, so training forward only gathers target/neighborhood genes for the two drugs in each sample.
- The target-expression encoder is zero-initialized to avoid random extra features perturbing the baseline at epoch 0.
- Supported target-expression fusion modes:
  - `piece`: append as an extra fusion input;
  - `control_add`: add to `control_hidden`;
  - `pair_add`: add to `pair_hidden`.
- Existing behavior is unchanged unless `--target-expression-mode` or `--cell-pair-film-scale` is explicitly enabled.

## Validation

- Static checks passed:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py dataset/training_ready_fast_dataset.py model/fast_lightning.py`
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_unseen_cell_targetexpr_film_fold0_2gpu.sh scripts/run_unseen_cell_targetexpr_pairadd_5fold_2gpu.sh`
- Dry-run forward passed for `target_expression_mode=pdi_ppi` and `target_expression_fusion_mode=control_add`:
  - expression output `(16, 10982)`
  - response logits `(16, 1)`
  - synergy logits `(16, 1)`
- Inference smoke test passed with the best pair-add checkpoint on `cell_5fold_fold0`, `limit_batches=1`:
  - wrote 8 predictions to `outputs/debug_targetexpr_pairadd_infer_smoke/predictions.parquet`
  - checkpoint config validation passed with target-expression args enabled.

## Results

- Fresh fold0 baseline (`single_cell_5fold_fold0`, full covariate UNK dropout `0.15`, `MSE_WEIGHT=0.075`):
  - AUROC `0.914478`
  - AUPRC `0.846185`
- FiLM-only was consistently worse:
  - best tested fold0 AUPRC `0.834991`
- Target-expression as extra fusion piece was worse:
  - best tested fold0 AUPRC `0.837257`
- Target-expression `control_add` was worse:
  - best tested fold0 AUPRC `0.840263`
- Target-expression `pair_add` was the only positive fold0 variant:
  - fold0 AUROC `0.922204`
  - fold0 AUPRC `0.847389`

Full 5-fold for `target_expression_mode=pdi_ppi`, `target_expression_fusion_mode=pair_add`, top-k `256`, PPI top-k `32`, alpha `0.5`, init scale `0.5`:

| Fold | AUROC | AUPRC |
| --- | ---: | ---: |
| 0 | 0.922204 | 0.847389 |
| 1 | 0.929321 | 0.759097 |
| 2 | 0.882643 | 0.663508 |
| 3 | 0.959585 | 0.816293 |
| 4 | 0.967115 | 0.782669 |
| Avg | 0.932174 | 0.773791 |

Compared with the current full covariate UNK + `MSE_WEIGHT=0.075` unseen-cell baseline (`0.934443 / 0.767008`), this is:

- AUROC: `-0.002269`
- AUPRC: `+0.006783`

## Conclusion

- Do not adopt FiLM as a default unseen-cell change.
- The PDI/PPI target-expression branch is only worth keeping as `pair_add`; it gives a small AUPRC improvement but does not approach the `0.85` target.
- The implementation is suitable as an optional ablation/candidate setting because it is explicit about PDI/PPI use, default-off, and does not modify data-processing code or data files.
