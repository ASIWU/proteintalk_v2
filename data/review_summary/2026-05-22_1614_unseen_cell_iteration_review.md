# 2026-05-22 16:14 HKT Unseen Cell/Cell-Type Iteration Review

## Scope

- Reviewed `train.py`, `infer.py`, `dataset/training_ready_fast_dataset.py`, `model/fast_delta_model.py`, and `scripts/ptv3_experiment_common.sh` while iterating on unseen cell and unseen cell-type AUPRC.
- Focused on split-aware covariate encoding, low-cost fusion switches, class-imbalance sampling, and fast inference compatibility.

## Findings

- The original categorical covariate path is weak for unseen cell/cell-type splits because many test categories are absent from train and therefore have untrained embeddings.
- Full-field split-aware covariate UNK with train-time UNK dropout is the only consistently positive change in this iteration.
- Removing MSE, stronger regularization, positive-row sampling, smaller model capacity, slim covariate lists, field-specific UNK, and auxiliary modality logit heads did not exceed the full-field covariate UNK setting.
- A train-only drug response prior diagnostic showed only tiny post-hoc gains when mixed with model predictions, so target-encoding prior features were not added to the model path.
- Fast `infer.py` needed checkpoint-aware covariate UNK support; otherwise covariate UNK checkpoints could load with embedding size mismatch or evaluate unknown covariates incorrectly.

## Verification

- `python -m py_compile train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py model/fast_lightning.py scripts/check_wandb_auth.py`
- Smoke inference for a covariate UNK checkpoint:
  `infer.py --model-type fast_delta --task-name ptv3_main_singledrug --split-strategy cell_type_5fold_fold0 --split-name test --limit-batches 1`

## Residual Risk

- The best AUPRC remains below 0.85: cell type `0.814780`, cell `0.761629` for full-field covariate UNK dropout `0.15` with `MSE_WEIGHT=0.1`.
- Further gains likely require a larger architecture/data change, not another small parameter tweak.
