# PTV1 exp_13 From exp_11 Graph-On Official Rerun

Generated: `2026-06-08T17:25:09+08:00`

This is an exp_13-only official rerun. It is not a cross-candidate leaderboard.
The run is pinned to the exp_11 best candidate `mse050_drop010` with graph features enabled.

## Source

- source exp_11: `20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split`
- source checkpoint: `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/checkpoints/20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split/epoch=5-step=168.ckpt`
- selected epoch: `5`
- all_train max epochs: `6`
- output prefix: `20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010`

## Results

| setting | AUPRC | AUROC | nAUPRC | rows | source checkpoint | selected epoch | applied max epochs |
|---|---:|---:|---:|---:|---|---:|---:|
| exp_11_source | 0.926324 | 0.963021 | 3.162962 | 799 | `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/checkpoints/20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split/epoch=5-step=168.ckpt` | 5 |  |
| exp_13_direct_from_exp11 | 0.561480 | 0.580062 | 1.092882 | 218 | `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/checkpoints/20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split/epoch=5-step=168.ckpt` |  |  |
| exp_13_all_train_from_exp11 | 0.579478 | 0.603900 | 1.127913 | 218 | `checkpoints/20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010_all_ptv1_for_extra_from_exp11/last.ckpt` | 5 | 6 |

## Audit

- status: `PASS`
- expected extra rows: `218`
- graph policy: `GRAPH_FEATURE_MODE=real`, `GRAPH_STRUCTURAL_RP=1`, `GRAPH_DRUG_CONCAT=1`
- direct policy: inference checkpoint must be the source exp_11 best checkpoint
- all_train policy: train all PTV1 with exp_11 parameters for selected epoch plus one, then infer from `last.ckpt`
- graph-off candidates are excluded from official selection and are not result rows in this report

## Artifacts

- source manifest: `checkpoints/20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split/run_manifest.json`
- all_train checkpoint manifest: `checkpoints/20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010_all_ptv1_for_extra_from_exp11/run_manifest.json`
- exp_13_direct_from_exp11 inference manifest: `outputs/2026-06/2026-06-08/20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010_extra_direct_from_exp11/ptv1_extra_singledrug/run_manifest.json`
- exp_13_all_train_from_exp11 inference manifest: `outputs/2026-06/2026-06-08/20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010_all_ptv1_for_extra_from_exp11/ptv1_extra_singledrug/run_manifest.json`
