# 2026-05-08 18:46 HKT Training Pipeline and Time Recheck

## Scope

Rechecked the current training pipeline after fixing the
`ptv3_main_doubledrug` split so merged single-drug rows enter double-drug
training.

## Artifact Counts

- `ptv3_main_singledrug`, split `random`:
  - train: `12950` anchors, `6475` batches at batch size `2`
  - valid: `1438` anchors, `719` batches
  - expression width: `10982`
  - train labels: `12950` non-empty `PRISM1st_label_total`, `0` non-empty
    `synergy`
- `ptv3_main_doubledrug`, split `pert_id_5fold_fold0`:
  - train: `19275` anchors, `9638` batches at batch size `2`
  - valid: `142` anchors, `71` batches
  - expression width: `11092`
  - train membership: `17986` merged single-drug anchors + `1289` native
    double-drug anchors
  - train labels: `17986` non-empty `PRISM1st_label_total`, `1289` non-empty
    `synergy`

## Pipeline Check

- Stage 2 feature-table composition is now used correctly by Step 3 splits:
  merged single-drug rows are included in double-drug train indices.
- Double-drug valid/test remain native double-drug rows only, so synergy
  validation/test labels are not polluted by single-drug rows.
- Current `train.py --task-head auto` resolves:
  - single-drug: response head, `PRISM1st_label_total`,
    `sensitive_label_mask`
  - double-drug: synergy head, `synergy`, `synergy_label_mask`
- Because of that default, merged single-drug rows in double-drug training
  currently contribute expression `loss1`, but do not contribute `loss2`.
  Their PRISM labels are present in the batch data, but the active double-drug
  loss uses only the synergy head/mask.
- This was visible in the GPU timing run: many double-drug train batches logged
  `train/loss2_step=0.000` because most updated double-drug train batches are
  single-drug auxiliary rows with no synergy label.

## Timed H200 Benchmarks

Commands used the default `attention_v10_hetero_cls_ee`, batch size `2`,
one H200, `100` train batches, `20` validation batches, no test.

- Double-drug benchmark:
  - run: `20260508_time_recheck_dd_1ep_100b`
  - train step wall delta: `24.13 s` for logged steps `9 -> 99`
  - estimated train step time: `0.268 s/batch`
  - manifest train count: `19275`
  - manifest valid count: `142`
  - active task: synergy
- Single-drug benchmark:
  - run: `20260508_time_recheck_sd_1ep_100b`
  - train step wall delta: `23.73 s` for logged steps `9 -> 99`
  - estimated train step time: `0.264 s/batch`
  - manifest train count: `12950`
  - manifest valid count: `1438`
  - active task: response

## Time Estimates

Using the measured H200 step times and batch size `2`:

- One H200:
  - single-drug full epoch: about `29-30 min`
  - double-drug full epoch after split fix: about `43-44 min`
  - single-drug 8 epochs: about `3.9-4.0 h`
  - double-drug 8 epochs: about `5.8-5.9 h`
- Eight H200s with DDP and per-GPU batch size `2`:
  - ideal speedup: about `8x`, GPU-hours roughly unchanged
  - realistic speedup assumption: `5-7x`
  - single-drug 8 epochs: about `34-48 min` wall time, about `4.5-6.4`
    GPU-hours
  - double-drug 8 epochs: about `50-71 min` wall time, about `6.7-9.5`
    GPU-hours

## Remaining Risk

- If the intended double-drug training loss is row-level mixed BCE
  (`PRISM1st_label_total` for merged single-drug rows and `synergy` for native
  double-drug rows), the current trainer still needs one more change. It should
  compute `loss2` from the response head where `sensitive_label_mask == 0` and
  from the synergy head where `synergy_label_mask == 0`.
- Current multi-GPU validation metrics are also a risk: AUROC/AUPRC are computed
  from per-process collected predictions and then logged with `sync_dist=True`,
  which averages rank-local metrics rather than computing exact global metrics.
  Final reported metrics should either be produced by single-GPU inference or
  the trainer should gather predictions across DDP ranks before metric
  computation.
