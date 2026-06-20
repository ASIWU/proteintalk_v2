# 2026-06-02 13:29 HKT Single-Model Unseen-Cell Follow-Up Review

## Scope

- Reviewed corrected unseen-cell requirement: `cell_5fold_fold0..4` single-model mean AUPRC >= `0.85`.
- Excluded the earlier fold0 probability ensemble from valid results.
- Checked fold2 as the main bottleneck.

## Code Review Notes

- `model/fast_delta_model.py`
  - Keeps frozen text embeddings as input features; no language-model weights are trained.
  - Adds optional control-drug interaction and observed perturb-expression branches, both default off.
  - Adds Cell-index text embedding support through the same frozen feature pathway.
- `model/fast_lightning.py`
  - Adds optional focal BCE and hard-negative/hard-positive ranking without changing default losses.
- `train.py`, `infer.py`, `scripts/ptv3_experiment_common.sh`, and `scripts/report_cell_drug_time_eval.py`
  - Propagate new arguments and write them into manifests for config binding.

## Experimental Findings

- Historical best valid single-checkpoint fold2 AUPRC remains `0.650627`.
- New valid fold2 trials did not improve it:
  - train-only KNN prior + frozen LLM hybrid: `0.646335`;
  - focal BCE + hard-negative ranking: `0.640760`;
  - frozen Cell-index SapBERT text embedding: `0.633792`;
  - no Cell/cell_type categorical covariates + global ranking: `0.627146`;
  - prior + focal + hard ranking: `0.621385`;
  - observed perturb-expression delta branch: `0.601775`.
- Offline probes indicate the current input information does not support a straightforward `0.85` 5-fold mean:
  - tabular non-ensemble probe mean `0.7400`;
  - expanded train-only prior mean about `0.7534`;
  - post-hoc perturb-expression/delta probe mean `0.7357`;
  - old `new_features_tanh` probe mean `0.6364`.

## Validation

- Python compile passed for modified Python files.
- Shell syntax passed for the common unseen-cell launcher.
- GPU dry-runs passed for graph, dose, frozen text embedding, Cell-index embedding, focal/hard-ranking, and observed perturb-expression branches.

## Conclusion

The corrected single-model 5-fold AUPRC target was not reached. The current blocker is fold2; valid single-model variants remain around `0.60-0.65` AUPRC on that fold.
