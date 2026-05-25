# 2026-05-25 11:39 HKT Unseen Cell Plan Implementation Review

## Scope

- Implemented the post-research unseen-cell experiment plan in the fast model path.
- Did not edit data files or data-processing code.
- Kept all new mechanisms default-off unless explicitly enabled through training arguments or script environment variables.

## Code Changes Reviewed

- `dataset/training_ready_fast_dataset.py`
  - Adds raw covariates alongside UNK-mapped covariates.
  - Adds optional row-level prior features.
- `model/fast_delta_model.py`
  - Adds optional cell-pair FiLM, prior-feature encoder/logit adapters, and pair-logit gate.
  - Keeps the baseline path unchanged when the new scales/modes are zero/off.
- `model/fast_lightning.py`
  - Adds optional supervised contrastive covariate loss, ranking loss, and gene-weighted MSE.
  - Uses raw covariates for auxiliary targets to avoid training on artificial UNK labels.
- `train.py`
  - Adds train-only KNN drug prior feature construction.
  - Adds train-only MSE gene weighting.
  - Records all new settings in `run_manifest.json`.
- `scripts/ptv3_experiment_common.sh`
  - Passes all new switches from environment variables into `train.py`.

## Experiment Summary

All runs below used `single_cell_5fold`, 1 GPU, batch size 256, full covariate UNK dropout `0.15`, unless noted.

| Setting | Folds | Avg AUROC | Avg AUPRC | Decision |
| --- | --- | ---: | ---: | --- |
| Baseline `MSE_WEIGHT=0.075` | 0-4 | 0.934443 | 0.767008 | Keep |
| `MSE_WEIGHT=0.05` | 0-4 | 0.931399 | 0.762206 | Reject |
| `PAIR_LOGIT_SCALE=1.0` | 0-4 | 0.929817 | 0.760380 | Reject |
| `PAIR_LOGIT_SCALE=2.0` | 0-4 | 0.931444 | 0.763142 | Reject |
| KNN prior learned logit | 2,4 | - | 0.680320 | Reject |
| KNN prior feature fusion | 2,4 | - | 0.653771 | Reject |
| Eval-only fixed prior logit | 2,4 | - | 0.670128 | Reject |
| Aux contrastive loss | 2,4 | - | 0.672500 | Reject |
| Cell-pair FiLM + ranking | 2,4 | - | 0.676239 | Reject |
| Variance-weighted MSE | 2,4 | - | 0.692514 | Reject |
| Larger hidden dim | 2,4 | - | 0.669263 | Reject |

Cell-type follow-up:

| Setting | Folds | Avg AUROC | Avg AUPRC | Decision |
| --- | --- | ---: | ---: | --- |
| Current branch cell-type baseline `MSE_WEIGHT=0.075` | 0-4 | 0.941852 | 0.810250 | Keep as current branch reference |
| `MSE_WEIGHT=0.10 + PAIR_LOGIT_SCALE=2.0` | 0,1 | 0.935692 | 0.745054 | Stop early |

## Reliability Notes

- KNN drug prior is built from train labels only. For train rows, same-cell priors are excluded when alternative train cells exist.
- Raw covariates are only used for auxiliary losses; main covariate embeddings still use the original UNK mapping path.
- The new modules are default-off, so standard baseline scripts are not changed unless their environment variables enable these switches.
- No tested lightweight change reached the requested `0.85` unseen-cell AUPRC target.

## Current Recommendation

- For unseen cell, keep full covariate UNK dropout `0.15` with `MSE_WEIGHT=0.075`.
- Keep the new switches for controlled ablations, but do not use them as default claims because the observed gains are fold-local and not 5-fold reliable.
