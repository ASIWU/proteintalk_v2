# 2026-05-08 20:22 HKT DDP 8-GPU Training and Hyperparameter Review

## Scope

- Reviewed the current training architecture from `docs/`, `data/review_summary/`, `train.py`, `dataset/training_ready_dataset.py`, `model/training_ready_models.py`, `model/training_ready_lightning.py`, and `infer.py`.
- Validated the active Stage-4 training path for `ptv3_main_doubledrug` with `attention_v10_hetero_cls_ee`, active `task_head=synergy`, and Lightning DDP on the 8 available H200 GPUs.
- Ran short hyperparameter smokes for DDP precision/model options, optimizers, schedulers, loss switches, fusion modes, task heads, and the `baseline_emb_v3` model.

## Architecture Notes

- `loss1` is the expression reconstruction MSE over finite perturbation-expression targets.
- `loss2` is the active task classification loss selected by `--task-head`; for `ptv3_main_doubledrug` the default/checked active task is synergy.
- The current double-drug train split includes train-only merged single-drug rows, so many train batches contribute `loss1` but have masked synergy labels. This makes train `loss2` sparse/noisy, while full validation `loss2` and active metrics are the more useful classification signals.
- Multi-GPU training is routed through Lightning with `--accelerator gpu --devices N`; for graph models and multi-device `strategy=auto`, `train.py` resolves to `ddp_find_unused_parameters_true`.

## Issues Found and Fixed

- DDP validation/test logging failed under NCCL because epoch-end CPU tensors were logged with `sync_dist=True`. Loss tensors are now moved to the module device before distributed logging.
- DDP validation/test metrics were rank-local and vulnerable to distributed sampler padding. Eval tensors are now gathered across ranks, de-duplicated by `row_index`, and metrics are computed on the full validation/test slice.
- Expression metrics produced NaN when expression targets contained NaNs. `compute_validation_metrics` now computes expression metrics with finite prediction/target pairs and sanitizes non-finite control-expression values.
- BF16 DDP metric gathering failed when gathered tensors were converted directly to NumPy. Floating gathered eval tensors are now cast to float32 before CPU/NumPy conversion.
- `infer.py --save-expression-pred` now passes raw expression targets into the shared metric function so inference and Lightning metric semantics stay aligned.

## 8-GPU Verification

Final bounded run:

```text
experiment: 20260508_codex_ddp8_dd_8ep_50b_final
devices: 8 H200 GPUs
precision: 32-true
strategy: ddp_find_unused_parameters_true
batch_size: 2
max_epochs: 8
limit_train_batches: 50
limit_val_batches: 1.0
skip_test: true
checkpoint_dir: checkpoints/20260508_codex_ddp8_dd_8ep_50b_final
log_dir: logs/20260508_codex_ddp8_dd_8ep_50b_final/version_0
best_checkpoint: epoch=7.ckpt
best_model_score: 1.0077422857284546
```

Observed trends:

- `train/loss1_epoch`: `39.6103 -> 1.2408`
- `train/total_loss_epoch`: `39.6820 -> 1.3307`
- `val/loss1`: `2.7357 -> 0.2471`
- `val/total_loss`: `3.5367 -> 1.0077`
- `val/mse_all`: `1.8237 -> 0.1646`
- `val/pcc_all`: `0.9820 -> 0.9983`
- `val/auroc`: `0.5552 -> 0.6891`
- `val/auprc`: `0.4286 -> 0.5417`
- `val/loss2`: improved early (`0.8010 -> 0.6366` by epoch 4, `0.6419` at epoch 6) but rose to `0.7607` at epoch 7.

Conclusion: the 8-GPU DDP training loop runs end-to-end after the metric/logging fixes. Expression loss and expression metrics improve strongly. Active synergy metrics improve overall, but classification learning is still weaker than expression learning and should be monitored with full-validation inference for longer production runs.

## Hyperparameter Smokes

Successful short checks:

- `20260508_codex_hp_arch_ddp2_bf16_fix`: 2-GPU DDP, `bf16-mixed`, `hidden_dim=128`, `num_heads=4`, `num_layers=2`, `dropout=0.2`, `perturb_fusion_mode=mlp`, `graph_dropout`, `use_target`, gate target-protein fusion, short batch covariate list.
- `20260508_codex_hp_loss_optim`: no expression loss, focal loss, active BCE weight, `adamw_fused`, cosine scheduler, custom learning rate and gradient clipping.
- `20260508_codex_hp_fusion_sgd`: `fusion_mode=add`, `perturb_fusion_mode=concat`, SGD, step scheduler, `mse_weight=0.5`, `bce_weight2=2.0`, `drop_last`, and `epoch_len`.
- `20260508_codex_hp_baseline`: `baseline_emb_v3`, Adam, plateau scheduler.
- `20260508_codex_hp_response_warmup_fix`: response task head, `adamw_fused_0.5`, cosine warmup scheduler, positive/weighted response BCE.

Expected/diagnostic failure:

- `--pdi-input-orientation protein_by_drug` failed with the default PDI matrix because the default artifact shape is drug-by-protein `(6113, 11345)` and the protein-by-drug mode correctly expects `(11345, 6113)`. This option requires a transposed matching PDI artifact rather than the default path.

## Verification Commands

- `py_compile` passed for `train.py`, `dataset/training_ready_dataset.py`, `model/training_ready_models.py`, `model/training_ready_lightning.py`, and `infer.py`.
- TensorBoard scalars were parsed from the run logs listed above.
- Final `nvidia-smi` process query showed no active GPU compute processes.

## Notes

- The project conda activation currently leaves `/usr/bin` ahead of the env on `PATH`; use `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python` explicitly for deterministic command execution.
- The pre-existing `spot.py` GPU reservation process was stopped with approval before training so all 8 H200 GPUs were available.
