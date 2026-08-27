# 2026-06-02 Single-Model Unseen-Cell Follow-Up

## Corrected Objective

- Task: `ptv3_main_singledrug`, unseen-cell split `cell_5fold_fold0..4`.
- Valid target: single-model 5-fold mean AUPRC >= `0.85`.
- Constraints kept in valid runs:
  - frozen LLM/text cell embedding input;
  - graph features enabled;
  - dose covariates enabled;
  - no probability averaging across checkpoints/models.

## Invalidated Earlier Claim

The earlier fold0 two-checkpoint probability average is not a valid result for this objective:

- it only evaluated `cell_5fold_fold0`;
- it used a two-member probability ensemble;
- therefore it does not satisfy the single-model 5-fold mean requirement.

The ensemble script remains in the worktree as an experimental utility, but it is not used for the corrected result claim.

## Single-Model Fold2 Bottleneck

Historical best single-checkpoint fold2 AUPRC remains about `0.6506`:

- `checkpoints/20260528_doseps_v1_stage2_covunk_cell_exp03_single_cell_5fold_single_cell_fold2`
- test AUPRC `0.650627`

Additional valid single-model fold2 trials after the correction:

| Run | Key Change | Fold2 AUPRC |
|---|---|---:|
| `20260602_1315_single_model_prior_v1` | frozen LLM hybrid + train-only KNN drug prior | `0.646335` |
| `20260602_1320_single_model_focal_hardrank_v1` | focal BCE + hard-negative ranking | `0.640760` |
| `20260602_1350_single_model_cellline_llm_v1` | frozen Cell-index SapBERT text embedding | `0.633792` |
| `20260602_1308_single_model_nocellcov_llm_rankglobal_v1` | no Cell/cell_type categorical covariates + global ranking | `0.627146` |
| `20260602_1325_single_model_prior_focal_hardrank_v1` | prior + focal + hard ranking | `0.621385` |
| `20260602_1405_single_model_observed_delta_v1` | observed perturb-expression delta branch | `0.601775` |

## Diagnostic Probes

- Non-ensemble tabular probe with control expression PCA, Morgan drug, graph, dose/time, frozen cell-type LLM PCA, and train-only KNN prior: 5-fold mean AUPRC `0.7400`; fold2 `0.6105`.
- Expanded train-only prior features: best prior-only mean about `0.7534`; fold2 about `0.6308`.
- Post-hoc true perturb-expression/delta probe did not solve the split: 5-fold mean AUPRC `0.7357`; fold2 `0.6306`.
- Legacy `new_features_tanh` probe from the old repo also did not solve it: 5-fold mean AUPRC `0.6364`; fold2 `0.5725`.

## Code Changes Kept

- `FastDeltaDrugResponseModel`
  - frozen LLM/text embedding conditioning modes;
  - control-drug interaction branch;
  - optional observed perturb-expression branch, default off.
- `FastProteinTalkLightning`
  - optional focal BCE;
  - optional hard-positive/hard-negative ranking.
- `train.py` / `infer.py` / experiment scripts
  - propagated new model/loss arguments;
  - added `--cell-type-llm-index-column` so frozen text embeddings can be keyed by `cell_type_index` or `Cell_index`.
- Generated artifact:
  - `data/training_ready/ptv3/derived/cell_line_llm_embedding_sapbert_768.npz`
  - frozen SapBERT text embeddings aligned to `Cell_index`.

## Current Status

The corrected single-model 5-fold target has not been reached. The main blocker is fold2: multiple single-model architectures and non-ensemble probes remain near `0.60-0.65` AUPRC on that fold.

## Validation

- `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py scripts/report_cell_drug_time_eval.py`
- `bash -n scripts/ptv3_experiment_common.sh scripts/exp_03_single_cell_5fold.sh`
- GPU dry-runs passed for frozen text embedding, graph, dose, hard-ranking/focal, Cell-index embedding, and observed-perturb branches.
