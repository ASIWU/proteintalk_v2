# 2026-05-08 17:36 HKT Loss Contract / Legacy Trainer Recheck

## Scope

Rechecked the old trainer code after the user clarified the intended loss names:

- `loss1`: expression MSE loss.
- `loss2`: one active task-label BCE loss.
  - single-drug: PRISM response label.
  - double-drug: synergy label.

## Old Code Checked

Old single-drug trainer:

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/trainer.py`
- `training_step` returns `mse_loss + bce_loss`.
- It logs `train/mse_loss`, `train/bce_loss`, and `train/total_loss`.

Old double-drug trainer:

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model/trainer_dd.py`
- It supports expression MSE plus two BCE heads:
  - `effective_key1`, usually response / PRISM.
  - `effective_key2`, synergy.
- For the current main task expectation, the active label is one task-specific label, not two losses named `loss1` and `loss2`.

## Issue Found

The current port had confusing semantics:

- It logged `bce_loss1` and `bce_loss2` as response and synergy heads.
- For `ptv3_main_doubledrug`, PRISM labels are absent, so `bce_loss1` was always masked to zero.
- For `ptv3_main_singledrug`, synergy labels are absent, so `bce_loss2` was always masked to zero.
- This made the prior explanation misleading because it treated `loss1/loss2` as two classification heads instead of the user-defined MSE/BCE losses.

## Code Changes

Updated `model/training_ready_lightning.py`:

- Added active task-loss config:
  - `task_head`: `response` or `synergy`.
  - `task_label_key`.
  - `task_mask_key`.
- Training loss is now:
  - `total_loss = mse_weight * loss1 + bce_weight * loss2` when MSE is enabled.
  - `loss1` is MSE.
  - `loss2` is active task BCE.
- TensorBoard now logs:
  - `train/loss1`, `train/loss2`, `train/total_loss`.
  - `val/loss1`, `val/loss2`, `val/total_loss`.
  - aliases `mse_loss` and `bce_loss`.
- Active classification metrics are logged as `val/auroc`, `val/auprc`, `val/acc`.
- Explicit head metrics are also logged as `response_auroc` and `synergy_auroc` to avoid ambiguity.

Updated `train.py`:

- Added `--task-head {auto,response,synergy}`.
- Added `--task-label-key` and `--task-mask-key`.
- Added active `--bce-weight`.
- Auto task behavior:
  - task names containing `doubledrug` use `task_head=synergy`, label `synergy`, mask `synergy_label_mask`.
  - other tasks use `task_head=response`, label from `infer_label_key`, mask `sensitive_label_mask`.
- Run manifests now record the active task loss config.

Updated `infer.py`:

- Added the same active task config.
- Inference metrics now include an explicit `task` block, plus `response` and `synergy` blocks.
- Prediction output includes `pred_task_prob` and `task_label`.

## Verification

Compile and smoke:

- `py_compile` passed for `model/training_ready_lightning.py`, `train.py`, `infer.py`, dataset/model files, and `scripts/0507_training_stack_smoke.py`.
- `scripts/0507_training_stack_smoke.py` passed.

Corrected 8-epoch bounded GPU runs:

### `ptv3_main_doubledrug`

- experiment: `20260508_loss_contract_ptv3_dd_8ep_50b`
- manifest task config:
  - `task_head`: `synergy`
  - `task_label_key`: `synergy`
  - `task_mask_key`: `synergy_label_mask`
- best checkpoint: `/tmp/proteintalk_v2_loss_contract_checkpoints/20260508_loss_contract_ptv3_dd_8ep_50b/epoch=7.ckpt`
- train `loss1`: `37.513 -> 1.024`
- train `loss2`: `0.758 -> 0.725`
- val `loss1`: `2.548 -> 0.244`
- val `loss2`: `1.069 -> 0.633`
- val active AUROC: best `0.619`, final `0.475`
- val `pcc_all`: best `0.800`, final `0.726`

Full validation inference:

- rows: `142`
- active task AUROC: `0.489`
- active task AUPRC: `0.318`
- active task ACC: `0.662`
- `pred_task_prob` mean `0.4088`, std `0.00063`, all below `0.5`

### `ptv3_main_singledrug`

- experiment: `20260508_loss_contract_ptv3_sd_8ep_50b`
- manifest task config:
  - `task_head`: `response`
  - `task_label_key`: `PRISM1st_label_total`
  - `task_mask_key`: `sensitive_label_mask`
- best checkpoint: `/tmp/proteintalk_v2_loss_contract_checkpoints/20260508_loss_contract_ptv3_sd_8ep_50b/epoch=7.ckpt`
- train `loss1`: `39.552 -> 1.520`
- train `loss2`: `0.490 -> 0.317`
- val `loss1`: `2.385 -> 0.877`
- val `loss2`: `0.947 -> 0.677`
- val active AUROC: best `0.586`, final `0.533`
- val `pcc_all`: `0.370 -> 0.796`

Full validation inference:

- rows: `1438`
- active task AUROC: `0.423`
- active task AUPRC: `0.100`
- active task ACC: `0.883`
- `pred_task_prob` mean `0.0768`, std `0.00017`, all below `0.5`

## Conclusion

The earlier explanation was wrong in naming and framing. The new trainer now matches the clarified loss contract: `loss1` is expression MSE and `loss2` is the active task-label BCE. The real data pipeline and GPU training path are not fake: they read the training-ready feature tables, split files, expression matrices, embeddings, and PDI matrix, and they complete real forward/backward/checkpoint runs on H200.

However, the classification model quality is still not good enough. Expression learning is clear, but active task probabilities remain nearly constant on full validation. Classification metrics, especially accuracy, should not be trusted yet without imbalance handling or a classifier-specific training adjustment.
