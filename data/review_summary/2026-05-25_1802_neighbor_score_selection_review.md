# 2026-05-25 18:02 HKT Neighbor Score Selection Review

## Scope

- Continued the target-expression branch by testing score-based PDI/PPI neighbor selection ideas.
- Files touched:
  - `model/fast_delta_model.py`
  - `train.py`
  - `infer.py`
  - `scripts/ptv3_experiment_common.sh`
  - `scripts/run_unseen_cell_targetexpr_pairadd_5fold_2gpu.sh`
  - `scripts/run_unseen_cell_neighbor_score_fold0_2gpu.sh`

## Implemented Controls

- Graph-side score controls:
  - `--target-expression-ppi-norm {raw,row,symmetric}`
  - `--target-expression-degree-penalty`
- Cell-aware candidate reweighting:
  - `--target-expression-cell-gate-mode {off,magnitude,signed}`
  - `--target-expression-cell-gate-scale`
  - `--target-expression-cell-gate-temperature`
- The cell-aware gate reweights only the cached top-k target-neighborhood candidates in the model forward pass. It does not change data files or data-processing code.

## Validation

- Static checks passed:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py dataset/training_ready_fast_dataset.py model/fast_lightning.py`
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_unseen_cell_neighbor_score_fold0_2gpu.sh scripts/run_unseen_cell_targetexpr_pairadd_5fold_2gpu.sh`
- Dry-run forward passed for:
  - `target_expression_mode=pdi_ppi`
  - `target_expression_fusion_mode=pair_add`
  - `target_expression_ppi_norm=symmetric`
  - `target_expression_degree_penalty=0.5`
  - `target_expression_cell_gate_mode=magnitude`
- Inference smoke test passed for the magnitude-gate checkpoint on `cell_5fold_fold0`, `limit_batches=1`:
  - wrote 8 predictions to `outputs/debug_neighbor_cellmag_infer_smoke/predictions.parquet`

## Fold0 Screening

All runs used `single_cell_5fold_fold0`, batch size 256, `MSE_WEIGHT=0.075`, full covariate UNK dropout `0.15`, PCEP, graph feature, and target-expression `pair_add`.

| Variant | AUROC | AUPRC | Conclusion |
| --- | ---: | ---: | --- |
| raw pair-add | 0.922204 | 0.847389 | reference |
| symmetric PPI norm | 0.915830 | 0.834186 | worse |
| degree penalty 0.5 | 0.921243 | 0.844042 | worse |
| magnitude gate 1.0 | 0.922484 | 0.848715 | best fold0 |
| magnitude gate 0.5 | 0.916506 | 0.838883 | worse |
| magnitude gate 1.5 | 0.914691 | 0.838733 | worse |
| magnitude gate 2.0 | 0.918693 | 0.841134 | worse |
| signed gate 1.0 | 0.918085 | 0.841246 | worse |
| magnitude gate 1.0, temp 0.5 | 0.916511 | 0.841335 | worse |
| magnitude gate 1.0, temp 2.0 | 0.918523 | 0.840110 | worse |
| top-k 128 + magnitude gate | 0.918404 | 0.838637 | worse |
| top-k 512 + magnitude gate | 0.915256 | 0.835685 | worse |

## Full 5-Fold Check

Best fold0 variant, `raw pair-add + magnitude gate scale 1.0`, full unseen-cell 5-fold:

| Fold | AUROC | AUPRC |
| --- | ---: | ---: |
| 0 | 0.922484 | 0.848715 |
| 1 | 0.931808 | 0.762540 |
| 2 | 0.879066 | 0.634370 |
| 3 | 0.956883 | 0.813937 |
| 4 | 0.957701 | 0.737487 |
| Avg | 0.929588 | 0.759410 |

Compared with raw target-expression pair-add (`0.932174 / 0.773791`), the magnitude gate is worse:

- AUROC: `-0.002585`
- AUPRC: `-0.014382`

Compared with the full covariate UNK + `MSE_WEIGHT=0.075` baseline (`0.934443 / 0.767008`), it is also worse:

- AUROC: `-0.004855`
- AUPRC: `-0.007598`

## Conclusion

- Raw PDI/PPI target-neighborhood scores are currently better than degree-normalized or hub-penalized scores.
- Cell-expression magnitude gating has a small fold0 gain but fails full 5-fold due to fold2/fold4 degradation.
- Recommended setting remains the previous raw PDI/PPI target-expression `pair_add` without cell gate.
- The neighbor score controls should remain default-off ablations for method discussion and diagnostics.
