# 2026-06-02 11:55 HKT LLM Cell-Type Conditioned Unseen-Cell Review

## Scope

- Reviewed the fast `cell_5fold` unseen-cell training path, existing LLM cell-type embedding implementation, and prior exp03 results.
- Focused on satisfying the current constraints: frozen LLM cell-type embeddings, graph features present, dose covariates present, and GPU training available.

## Findings

- Previous frozen LLM support only concatenated `cell_type_features` into the covariate encoder.
- Existing `cell_5fold` best test AUPRC in local manifests is close to but below `0.85`: fold0 `0.844330`; fold3 `0.843532`; fold2 remains the hardest fold at about `0.65`.
- The strongest prior unseen-cell configurations keep graph features and dose covariates enabled; `lr=1e-4`, `mse_weight=0.25`, and split-aware covariate dropout are the nearest fold0 baseline.

## Implementation Notes

- Added trainable cell-type conditioning around frozen LLM features in `model/fast_delta_model.py`.
- New `cell_type_llm_fusion_mode` options:
  - `covariate`: preserves the previous default behavior.
  - `piece`: appends a projected cell-type hidden piece to fusion.
  - `film`: uses projected cell-type hidden to FiLM-condition drug-pair and target hidden states.
  - `interaction`: adds a cell-drug-target interaction piece.
  - `hybrid`: combines piece, FiLM, and interaction.
- The raw LLM feature tensor is detached inside `_encode_cell_type_features`; only the projector/gates/heads are trainable.
- Added optional `cell_type_llm_condition_scale`, `cell_type_llm_logit_scale`, and `cell_type_llm_dropout`.
- Added `scripts/run_unseen_cell_llm_condition_search.sh` for bounded unseen-cell parameter screening with `MAX_RUN_SECONDS=86400` by default.
- Added residual scaling for cell-type pieces/interactions and zero-initialized the cell-type interaction output layer after early high-scale variants regressed.
- Added `scripts/ensemble_prediction_probs.py` to average prediction probabilities for the final two-member ensemble.
- Updated `scripts/report_cell_drug_time_eval.py` to forward the new LLM fusion/scale/dropout arguments during materialized inference.

## Corrected Result

The earlier fold0 two-member probability ensemble is invalid for the corrected objective because the requirement is single-model 5-fold mean AUPRC. It should not be counted as success.

Single-model follow-up focused on the fold2 bottleneck. Historical best fold2 remains about `0.650627`, and the added valid single-model variants did not improve it:

- frozen LLM hybrid + train-only KNN prior: `0.646335`;
- focal BCE + hard-negative ranking: `0.640760`;
- frozen Cell-index SapBERT text embedding: `0.633792`;
- no Cell/cell_type categorical covariates + global ranking: `0.627146`;
- prior + focal + hard ranking: `0.621385`;
- observed perturb-expression delta branch: `0.601775`.

The corrected single-model 5-fold mean target was not reached.

## Validation

- GPU dry run passed on `cell_5fold_fold0` with graph, dose, target-expression, and `CELL_TYPE_LLM_MODE=frozen` hybrid conditioning:
  - `dry_run batch=0 expression=(16, 10982) response_logits=(16, 1) synergy_logits=(16, 1)`
- Python compile passed:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py dataset/training_ready_fast_dataset.py model/fast_lightning.py scripts/ensemble_prediction_probs.py scripts/report_cell_drug_time_eval.py`
- Shell syntax passed:
  - `bash -n scripts/run_unseen_cell_llm_condition_search.sh scripts/ptv3_experiment_common.sh scripts/exp_03_single_cell_5fold.sh`
