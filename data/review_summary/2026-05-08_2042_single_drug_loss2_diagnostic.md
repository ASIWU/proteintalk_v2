# 2026-05-08 20:42 HKT Single-Drug Loss2 Diagnostic

## Scope

- Rechecked the active-task `loss2` label/mask path after concern that double-drug training may mix PRISM response labels and synergy labels.
- Ran a bounded 8-GPU single-drug response training job to isolate whether `loss2` learns when only PRISM response labels are present.
- Ran single-GPU inference on epoch 0 and epoch 7 checkpoints to inspect probability spread.

## Code Path Finding

- `train.py` resolves `task_head=synergy` for double-drug tasks and `task_head=response` for single-drug tasks when `--task-head auto`; this diagnostic explicitly used `--task-head response`.
- `ProteinTalkLightning._losses` selects one active task logit via `_select_task_logits`; `loss2` is computed only from `task_label_key` and `task_mask_key`.
- Current double-drug default `loss2` therefore uses `synergy` with `synergy_label_mask`; PRISM labels may exist in the batch for merged single-drug rows, but they are not used by active double-drug `loss2`.

## Split Label Statistics

`ptv3_main_singledrug / pert_id_5fold_fold0`:

- Train: `12588` rows, all primary, all have PRISM response labels, no synergy labels.
- Valid: `1807` rows, PRISM labels `1453` non-responsive / `354` sensitive, no synergy labels.
- Test: `3591` rows, PRISM labels `3091` non-responsive / `500` sensitive, no synergy labels.

`ptv3_main_doubledrug / pert_id_5fold_fold0`:

- Train: `19275` rows = `17986` merged single-drug + `1289` native double-drug.
- Train PRISM valid labels: `17986`; train synergy valid labels: `1289`.
- Valid/test are native double-drug only and have synergy labels.

## Single-Drug Run

```text
experiment: 20260508_codex_singledrug_ddp8_response_8ep_50b
task: ptv3_main_singledrug
split: pert_id_5fold_fold0
task_head: response
devices: 8 H200 GPUs
strategy: ddp_find_unused_parameters_true
batch_size: 2
max_epochs: 8
limit_train_batches: 50
limit_val_batches: 1.0
checkpoint_dir: checkpoints/20260508_codex_singledrug_ddp8_response_8ep_50b
log_dir: logs/20260508_codex_singledrug_ddp8_response_8ep_50b/version_0
```

Scalar trends:

- `train/loss1_epoch`: `39.5642 -> 1.3403`
- `train/loss2_epoch`: `0.4428 -> 0.3815`, best `0.2882` at epoch 4
- `val/loss1`: `2.4013 -> 0.4402`
- `val/loss2`: `0.5285 -> 0.5743`, best `0.4973` at epoch 3
- `val/auroc`: `0.6699 -> 0.3929`
- `val/auprc`: `0.3184 -> 0.1547`
- `val/acc`: constant `0.8041`, equal to the negative-class majority baseline

Inference probability spread:

- Epoch 0 valid: AUROC `0.6700`, AUPRC `0.3160`, probability mean `0.107139`, std `0.0000035`, min `0.107130`, max `0.107147`; all predictions below `0.5`.
- Epoch 7 valid: AUROC `0.3929`, AUPRC `0.1547`, probability mean `0.072572`, std `0.000128`, min `0.072094`, max `0.072877`; all predictions below `0.5`.

## Conclusion

- The suspected PRISM/synergy mixing is not supported by the current loss path: double-drug active `loss2` uses synergy only, and merged single-drug rows are masked for synergy.
- `loss2` is not working well even in the isolated single-drug response setting. The model collapses to near-constant low response probabilities and majority-negative accuracy, while expression `loss1` learns strongly.
- This does not justify a data pipeline change yet. The next targeted fix should be in training/loss configuration or sampling, such as positive-class weighting, balanced label sampling, stronger classifier supervision, or temporarily training/evaluating the classifier without the expression loss dominating the shared representation.
