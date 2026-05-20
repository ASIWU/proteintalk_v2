# 2026-05-08 18:00 HKT Double-Drug Train Split Single-Drug Merge Review

## Question

The double-drug training run was much shorter than the single-drug run. The
suspected issue was whether `ptv3_main_doubledrug` training includes all
`ptv3_main_singledrug` data as required.

## Findings

- Stage 2 already merges single-drug rows into the double-drug feature table:
  - `ptv3_main_doubledrug` feature rows: `20764`
  - `feature_membership="primary"` native double rows: `2196`
  - `feature_membership="merged_single_drug"` single-drug rows: `18568`
- The Step 3 split builder was the source of the training-count gap. Its anchor
  rule only accepted `feature_membership="primary"`, so merged single-drug rows
  were present in the feature table but absent from double-drug train indices.
- Before the fix, `pert_id_5fold_fold0` train contained only `1289` native
  double-drug anchors.

## Changes

- Updated `utils/09_build_data_splits.py` so `ptv3_main_doubledrug` pairing
  metadata includes both `primary` and `merged_single_drug` anchors.
- Native double-drug rows still define the pert-pair folds and valid/test rows.
- All merged `ptv3_main_singledrug` anchors are appended to every double-drug
  train split only. They are excluded from double-drug valid/test splits because
  they do not have `synergy` labels.
- Updated `docs/Data_Process_2.md` with the Stage-2 double-drug feature-table
  merge requirement and regeneration commands.
- Updated `docs/Data_Process_3.md` with the train-only single-drug auxiliary
  split rule and split regeneration command.

## Verification

- `python -m py_compile utils/09_build_data_splits.py` passed in `flow_v2`.
- Regenerated the full split set so `split_build_manifest.json` remains complete:
  `conda run -n flow_v2 python utils/09_build_data_splits.py --dataset-group all`
- Global split build manifest now lists `13` tasks.
- New `ptv3_main_doubledrug/pert_id_5fold_fold0` split:
  - train: `19275`
  - valid: `142`
  - test: `360`
  - train membership: `17986` merged single-drug anchors + `1289` native
    double-drug anchors
  - train labels: `17986` non-empty `PRISM1st_label_total`, `1289` non-empty
    `synergy`
- `train.py` dry run on the regenerated split passed with
  `attention_v10_hetero_cls_ee`, batch size `2`, hidden dim `8`, one dry-run
  batch:
  - expression output: `(2, 11092)`
  - response logits: `(2, 1)`
  - synergy logits: `(2, 1)`

## Caveat

The data split now includes all single-drug anchors in double-drug training. The
current default double-drug training config still uses `task_head=synergy`, so
those merged single-drug rows contribute expression `loss1`; their PRISM labels
are present in the batch data but are not consumed by double-drug `loss2` unless
the trainer is configured/extended for mixed row-level response+synergy BCE.
