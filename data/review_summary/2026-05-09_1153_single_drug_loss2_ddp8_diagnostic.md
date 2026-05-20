# 2026-05-09 11:53 HKT Single-Drug Loss2 DDP8 Diagnostic

## Scope

- Focused only on `ptv3_main_singledrug`.
- Used 8 GPUs after user permission.
- No code changes were made during this diagnostic.

## Run

Command:

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -u train.py \
  --dataset-group ptv3 \
  --task-name ptv3_main_singledrug \
  --split-strategy pert_id_5fold_fold0 \
  --model-type attention_v10_hetero_cls_ee \
  --task-head response \
  --experiment-name 20260509_single_response_ddp8_bs16_fullaxis_batch_target_10ep \
  --batch-size 16 \
  --max-epochs 10 \
  --limit-train-batches 1.0 \
  --limit-val-batches 1.0 \
  --limit-test-batches 1.0 \
  --accelerator gpu \
  --devices 8 \
  --strategy ddp_find_unused_parameters_true \
  --precision 32-true \
  --num-workers 0 \
  --save-every-n-epochs 1
```

Effective run manifest:

- `task_head=response`
- `task_label_key=PRISM1st_label_total`
- `task_mask_key=sensitive_label_mask`
- `batch_cov_list=["machineID_new","Cell_plate","Cell","cell_type","batch","pert_time"]`
- `use_target=true`
- `positive_weight=null`
- checkpoint monitor: `val/total_loss`
- best checkpoint: `checkpoints/20260509_single_response_ddp8_bs16_fullaxis_batch_target_10ep/epoch=7.ckpt`

## Data Checks

- Single-drug rows: `18568`.
- `pert_id1 != pert_id2`: `0`.
- Fold0 label distribution:
  - train: `12588` valid labels, `1283` sensitive, positive rate `0.101922`.
  - valid: `1807` valid labels, `354` sensitive, positive rate `0.195905`.
  - test: `3591` valid labels, `500` sensitive, positive rate `0.139237`.

## Metrics

Train `loss2_epoch`:

`0.367291 -> 0.343022 -> 0.270965 -> 0.200627 -> 0.165101 -> 0.158578 -> 0.168295 -> 0.185455 -> 0.145249 -> 0.138862`

Validation:

- `val/loss2`: `0.555740 -> 0.483775 -> 0.559987 -> 0.520295 -> 0.398021 -> 0.420311 -> 0.981975 -> 0.335631 -> 0.379555 -> 0.344839`
- `val/auroc`: peaked at `0.888952`, final `0.878525`
- `val/auprc`: peaked at `0.729067`, final `0.725790`
- `val/acc`: final `0.866630`

Test from best `val/total_loss` checkpoint (`epoch=7.ckpt`):

- `test/loss2=0.311674`
- `test/auroc=0.857175`
- `test/auprc=0.572209`
- `test/acc=0.888889`

## Conclusion

Single-drug `loss2` is working in the current pipeline. The response head is trained with `PRISM1st_label_total` and `sensitive_label_mask`, train `loss2` decreases clearly, and validation/test AUROC/AUPRC are far above the positive-rate baselines.

The most likely reason earlier runs looked bad is not a dead loss head. It is the combination of imbalanced labels, unweighted BCE, checkpoint selection by `val/total_loss` rather than response-specific metrics, and fixed `0.5` threshold accuracy. The epoch 6 validation point had high BCE (`0.981975`) but strong AUROC/AUPRC (`0.867897`/`0.727821`), which indicates calibration/threshold instability rather than failed ranking.

Lightning warned that DDP test can duplicate samples with uneven dataloaders and that epoch-level test logs are not fully `sync_dist=True`. Use single-device evaluation or explicit synchronized metric aggregation for final reported test numbers.
