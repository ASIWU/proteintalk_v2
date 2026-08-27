# exp_21-exp_26 PRISM2 Main-Label Results

Date: 2026-06-25 HKT

## Scope

本轮实验把原 exp_01-exp_06 的 PTV3 主单药标签列从 `PRISM1st_label_total` 另起一支切到 `PRISM2nd_label_total`，不覆盖原数据和原实验，新增任务和实验编号如下：

| new exp | source logic | task | label/head | split |
|---|---|---|---|---|
| exp21 | exp01 | `ptv3_main_singledrug_prism2` | `PRISM2nd_label_total` / response | `pert_stratified_5fold` |
| exp22 | exp02 | `ptv3_main_singledrug_prism2` | `PRISM2nd_label_total` / response | `cell_type_5fold` |
| exp23 | exp03 | `ptv3_main_singledrug_prism2` | `PRISM2nd_label_total` / response | `cell_5fold` |
| exp24 | exp04 | `ptv3_main_singledrug_prism2` | `PRISM2nd_label_total` / response | `pert_stratified_5fold`, `--no-mse-loss` |
| exp25 | exp05 | `ptv3_main_singledrug_prism2` | `PRISM2nd_label_total` / response | `pert_stratified_5fold`, `--graph-feature-mode zero` |
| exp26 | exp06 | `ptv3_main_doubledrug_prism2aux` | `synergy` / synergy | `pert_id_5fold`; train includes PRISM2-filtered single auxiliary rows |

Note: old `exp_05_single_no_pdi_5fold.sh` 的实际 ablation 是 `--graph-feature-mode zero`，所以 exp25 也按原逻辑实现为 no graph-feature，而不是只关闭 PDI。

## Data Build

New data root:

```text
data/training_ready_prism2_main
```

PRISM2 single-drug task summary:

| field | value |
|---|---:|
| feature rows | 8157 |
| control rows | 424 |
| non-control rows | 7733 |
| PRISM2 non-empty rows | 7733 |
| PRISM2-only rows | 165 |
| PRISM1-only rows | 0 |
| PRISM2 non-responsive | 5727 |
| PRISM2 sensitive | 2006 |

Validation passed with:

```text
utils/03_validate_training_ready_outputs.py --output-root data/training_ready_prism2_main
```

## DDI/DPI/PPI Reuse Check

DDI/DPI/PPI did not need to be rebuilt. The new PRISM2 main-label data keeps the same global drug and protein index as the existing PTV3 training-ready root:

| check | value |
|---|---:|
| same `pert_index` | true |
| same `protein_index` | true |
| pert index size | 6131 |
| protein index size | 11345 |
| DDI shape | 6131 x 6131 |
| PDI shape | 6131 x 11345 |
| PPI shape | 11345 x 11345 |
| drug embedding shape | 6131 x 2048 |
| protein embedding shape | 11345 x 1280 |

The new root reuses:

```text
data/training_ready_prism2_main/ptv3/derived -> ../../training_ready/ptv3/derived
```

A graph rebuild helper was still added for future mismatch cases:

```text
scripts/exp21_26_rebuild_prism2_graphs.sh
```

It was not run because the index hashes and matrix dimensions matched.

## Training Run

Training was run in tmux session `gpu2` with conda env `flow_v2`.

Runtime summary:

```text
logs/20260625_1807_prism2_exp21_26_nowandb_runtime_summary.tsv
```

All 30 train/test runs completed with status `0`.

## Aggregate Metrics

Metrics are test-set metrics from `run_manifest.json`. Values are mean plus sample std over 5 folds.

| exp | setting | task | split | folds | test count | AUPRC | AUROC | nAUPRC | baseline AUPRC | ACC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp21 | single PRISM2 pert_stratified | `ptv3_main_singledrug_prism2` | `pert_stratified_5fold` | 5 | 7627 | 0.6587 +/- 0.0618 | 0.8102 +/- 0.0297 | 2.6187 +/- 0.4155 | 0.2580 | 0.8103 |
| exp22 | single PRISM2 cell_type | `ptv3_main_singledrug_prism2` | `cell_type_5fold` | 5 | 7627 | 0.8571 +/- 0.0606 | 0.9186 +/- 0.0126 | 2.8513 +/- 0.5777 | 0.3148 | 0.8399 |
| exp23 | single PRISM2 cell | `ptv3_main_singledrug_prism2` | `cell_5fold` | 5 | 7627 | 0.8511 +/- 0.0451 | 0.9134 +/- 0.0269 | 3.0416 +/- 0.6745 | 0.2930 | 0.8690 |
| exp24 | single PRISM2 no MSE | `ptv3_main_singledrug_prism2` | `pert_stratified_5fold` | 5 | 7627 | 0.6772 +/- 0.0512 | 0.8042 +/- 0.0337 | 2.7132 +/- 0.5551 | 0.2580 | 0.8070 |
| exp25 | single PRISM2 no graph feature | `ptv3_main_singledrug_prism2` | `pert_stratified_5fold` | 5 | 7627 | 0.6252 +/- 0.0951 | 0.7819 +/- 0.0542 | 2.4674 +/- 0.3444 | 0.2580 | 0.8074 |
| exp26 | double PRISM2-aux pert_pair | `ptv3_main_doubledrug_prism2aux` | `pert_id_5fold` | 5 | 1791 | 0.6684 +/- 0.0483 | 0.7628 +/- 0.0302 | 1.6571 +/- 0.1317 | 0.4043 | 0.7067 |

## Fold Metrics

### exp21: Single PRISM2 Pert-Stratified

| fold | split | count | AUPRC | AUROC | nAUPRC | baseline | ACC | best val AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `pert_stratified_5fold_fold0` | 1544 | 0.7419 | 0.8210 | 2.4267 | 0.3057 | 0.7902 | 0.8195 |
| 1 | `pert_stratified_5fold_fold1` | 1554 | 0.6364 | 0.8185 | 2.8585 | 0.2227 | 0.8423 | 0.8807 |
| 2 | `pert_stratified_5fold_fold2` | 1524 | 0.6969 | 0.8024 | 2.4084 | 0.2894 | 0.7907 | 0.8388 |
| 3 | `pert_stratified_5fold_fold3` | 1611 | 0.6363 | 0.7645 | 2.1809 | 0.2917 | 0.7610 | 0.8866 |
| 4 | `pert_stratified_5fold_fold4` | 1394 | 0.5819 | 0.8447 | 3.2191 | 0.1808 | 0.8673 | 0.8756 |

### exp22: Single PRISM2 Cell-Type

| fold | split | count | AUPRC | AUROC | nAUPRC | baseline | ACC | best val AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `cell_type_5fold_fold0` | 1917 | 0.8714 | 0.9405 | 3.3015 | 0.2640 | 0.8419 | 0.9808 |
| 1 | `cell_type_5fold_fold1` | 1723 | 0.8591 | 0.9137 | 3.3640 | 0.2554 | 0.8746 | 0.8911 |
| 2 | `cell_type_5fold_fold2` | 3691 | 0.7568 | 0.9149 | 3.0497 | 0.2482 | 0.8610 | 0.8746 |
| 3 | `cell_type_5fold_fold3` | 151 | 0.8782 | 0.9154 | 2.5502 | 0.3444 | 0.8079 | 0.8982 |
| 4 | `cell_type_5fold_fold4` | 145 | 0.9202 | 0.9082 | 1.9914 | 0.4621 | 0.8138 | 0.8997 |

### exp23: Single PRISM2 Cell

| fold | split | count | AUPRC | AUROC | nAUPRC | baseline | ACC | best val AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `cell_5fold_fold0` | 2081 | 0.8238 | 0.9441 | 3.8013 | 0.2167 | 0.8991 | 0.8971 |
| 1 | `cell_5fold_fold1` | 916 | 0.8624 | 0.8811 | 2.3581 | 0.3657 | 0.8308 | 0.9460 |
| 2 | `cell_5fold_fold2` | 837 | 0.8984 | 0.9238 | 2.5233 | 0.3560 | 0.8602 | 0.9497 |
| 3 | `cell_5fold_fold3` | 2886 | 0.7878 | 0.8895 | 3.7150 | 0.2121 | 0.8794 | 0.9601 |
| 4 | `cell_5fold_fold4` | 907 | 0.8831 | 0.9285 | 2.8105 | 0.3142 | 0.8754 | 0.9506 |

### exp24: Single PRISM2 No MSE

| fold | split | count | AUPRC | AUROC | nAUPRC | baseline | ACC | best val AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `pert_stratified_5fold_fold0` | 1544 | 0.7578 | 0.8192 | 2.4789 | 0.3057 | 0.8148 | 0.8226 |
| 1 | `pert_stratified_5fold_fold1` | 1554 | 0.6861 | 0.8253 | 3.0815 | 0.2227 | 0.8391 | 0.8904 |
| 2 | `pert_stratified_5fold_fold2` | 1524 | 0.6762 | 0.7843 | 2.3367 | 0.2894 | 0.7907 | 0.8378 |
| 3 | `pert_stratified_5fold_fold3` | 1611 | 0.6344 | 0.7553 | 2.1747 | 0.2917 | 0.7641 | 0.8911 |
| 4 | `pert_stratified_5fold_fold4` | 1394 | 0.6317 | 0.8371 | 3.4943 | 0.1808 | 0.8264 | 0.8890 |

### exp25: Single PRISM2 No Graph Feature

| fold | split | count | AUPRC | AUROC | nAUPRC | baseline | ACC | best val AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `pert_stratified_5fold_fold0` | 1544 | 0.7765 | 0.8597 | 2.5402 | 0.3057 | 0.8180 | 0.7808 |
| 1 | `pert_stratified_5fold_fold1` | 1554 | 0.5598 | 0.7627 | 2.5143 | 0.2227 | 0.8327 | 0.8081 |
| 2 | `pert_stratified_5fold_fold2` | 1524 | 0.5997 | 0.7111 | 2.0725 | 0.2894 | 0.7664 | 0.8382 |
| 3 | `pert_stratified_5fold_fold3` | 1611 | 0.6522 | 0.7774 | 2.2354 | 0.2917 | 0.7784 | 0.8162 |
| 4 | `pert_stratified_5fold_fold4` | 1394 | 0.5377 | 0.7986 | 2.9745 | 0.1808 | 0.8415 | 0.8312 |

### exp26: Double PRISM2-Aux Pert-Pair

| fold | split | count | AUPRC | AUROC | nAUPRC | baseline | ACC | best val AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `pert_id_5fold_fold0` | 357 | 0.7292 | 0.7842 | 1.6794 | 0.4342 | 0.7171 | 0.7810 |
| 1 | `pert_id_5fold_fold1` | 380 | 0.6401 | 0.7712 | 1.7627 | 0.3632 | 0.7632 | 0.7437 |
| 2 | `pert_id_5fold_fold2` | 358 | 0.7044 | 0.7672 | 1.7511 | 0.4022 | 0.7011 | 0.7987 |
| 3 | `pert_id_5fold_fold3` | 352 | 0.6586 | 0.7812 | 1.6559 | 0.3977 | 0.7244 | 0.7531 |
| 4 | `pert_id_5fold_fold4` | 344 | 0.6095 | 0.7103 | 1.4361 | 0.4244 | 0.6279 | 0.7556 |

## Interpretation

1. The PRISM2 main-label branch is working end to end. The new task names, PRISM2 label selection, split generation, validation, training, checkpointing, and testing all completed.
2. DDI/DPI/PPI references are consistent with the new data root. Because `pert_index` and `protein_index` are identical to the existing PTV3 root, reusing the old derived graph matrices is correct for this branch.
3. For the pert-stratified single-drug setting, PRISM2 gives mean test AUPRC 0.6587. Removing MSE is slightly higher in this run at 0.6772, while removing graph features is lower at 0.6252.
4. Cell-type and cell splits are much easier in these results, with mean AUPRC around 0.85. The cell-type split has very small test folds for fold3/fold4, so those folds should be interpreted with the test counts shown above.
5. The double-drug PRISM2-aux setting trains and tests correctly on synergy labels. Its mean test AUPRC is 0.6684 over 1791 held-out double-drug test rows.

## Key Artifacts

| artifact | path |
|---|---|
| data root | `data/training_ready_prism2_main` |
| graph reuse check | `data/training_ready_prism2_main/ptv3/derived_graph_reuse_check.json` |
| PRISM2 data summary | `data/training_ready_prism2_main/ptv3/exp21_26_prism2_data_summary.json` |
| runtime summary | `logs/20260625_1807_prism2_exp21_26_nowandb_runtime_summary.tsv` |
| checkpoints | `checkpoints/20260625_1807_prism2_exp21_26_nowandb_*` |
| data build script | `scripts/build_exp21_26_prism2_data.sh` |
| training runner | `scripts/run_exp_21_26_prism2_main.sh` |
| graph rebuild helper | `scripts/exp21_26_rebuild_prism2_graphs.sh` |
