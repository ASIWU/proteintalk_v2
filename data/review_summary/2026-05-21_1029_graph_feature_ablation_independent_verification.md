# 2026-05-21 10:29 HKT Graph Feature Ablation Independent Verification

Scope: independently verify the `new_version` graph-feature architecture and reported graph vs zero ablation results for `graph128_struct_drugcat_logit2_no_pos`.

Independent subagent: `019e4859-9067-7132-b1c3-e0636dedb3d1` (`Zeno`).

Findings:
- No evidence of fabricated metrics, test-selected checkpoints, or direct label leakage was found.
- The real graph run explicitly builds features from `data/training_ready/ptv3/derived/ppi_matrix.npy`, `pdi_matrix.npy`, and `ddi_matrix.npy`.
- The zero ablation keeps the same architecture and checkpointing path, but returns zero graph tensors and `graph_feature_mask=0`.
- Train/valid/test row overlap, sample_id overlap, and perturbation-id train-test overlap are zero in the checked manifests.
- The cached graph feature metadata reports shape `(6113, 774)` with slices for `pdi_direct`, `pdi_ppi`, `ddi_context`, `pdi_struct`, `pdi_ppi_struct`, `ddi_struct`, `pdi_stats`, and `ddi_stats`.

Recomputed 5-fold DDP manifest summary:
- real graph: AUPRC 0.656369, AUROC 0.900316, ACC 0.917409
- zero graph: AUPRC 0.598797, AUROC 0.849282, ACC 0.913561
- gap: AUPRC +0.057573, AUROC +0.051034, ACC +0.003848

Single-GPU exact reevaluation of best checkpoints:
- real graph: AUPRC 0.656371, AUROC 0.900312, ACC 0.917402
- zero graph: AUPRC 0.598797, AUROC 0.849261, ACC 0.913553
- gap: AUPRC +0.057574, AUROC +0.051051, ACC +0.003849

Risks and caveats:
- DDP evaluation padded fold1/fold3 by one sample in the original distributed test aggregation. The single-GPU reevaluation removes this concern and preserves the same conclusion.
- The graph features are global/transductive graph features: held-out perturbation nodes still have PPI/PDI/DDI-derived features. This is not label leakage, but the paper should state the setting accurately as graph-assisted/transductive unseen-drug generalization unless the benchmark definition already permits known test-drug graph features.
- The graph feature gain is supported by the current 5-fold runs, but should still be validated once on the user's standard 8-GPU pipeline before final publication claims.
