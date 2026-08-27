# 2026-06-10 PTV01-08 PosWeight Combo Tuning Report

This report consolidates the completed experiment requested by `docs/2026-06-10_ptv01_08_posweight_combo_tuning_plan.md`. The audit found detailed raw reports under `logs/` and `outputs/`, but no long-form `docs/2026-06-10_ptv01_08_posweight_combo_tuning_report.md` existed before this update; this file is the durable report generated from the recorded artifacts. No checkpoint rerun or GPU retraining was required.

## Source Artifacts

| kind | path |
| --- | --- |
| screen Markdown | logs/20260610_ptv01_08_posweight_combo_v1_param_search_report.md |
| screen TSV | outputs/2026-06/2026-06-10/20260610_ptv01_08_posweight_combo_v1_param_search_report.tsv |
| full Markdown | logs/20260610_ptv01_08_posweight_combo_v1_full_param_search_report.md |
| full TSV | outputs/2026-06/2026-06-10/20260610_ptv01_08_posweight_combo_v1_full_param_search_report.tsv |
| final Markdown | logs/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_dose_time_eval.md |
| final CSV | outputs/2026-06/2026-06-10/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_dose_time_eval.csv |
| final JSON | outputs/2026-06/2026-06-10/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_dose_time_eval.json |
| GPU job summary | logs/20260610_ptv01_08_posweight_combo_v1_gpu_job_summary.tsv |
| final runtime summary | logs/20260610_ptv01_08_posweight_combo_selected_v1_runtime_summary.tsv |
| previous selected baseline CSV | outputs/2026-06/2026-06-04/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.csv |

## Completion Audit

- Run span: `2026-06-10 16:03:16 HKT` to `2026-06-10 18:15:46 HKT` (2.21 wall-clock hours from recorded job timestamps).
- GPU job summary: `666` fold jobs, statuses `{'0': 666}`.
- Final selected runtime summary: `41` train/infer rows, statuses `{'0': 41}`.
- Final selected manifests: `32` total (`exp01`-`exp06` five folds each, plus one all-data run each for `exp07` and `exp08`).
- GPU distribution: `{'0': 81, '1': 84, '2': 82, '3': 84, '4': 84, '5': 82, '6': 85, '7': 84}`; GPUs `0-7` were all used.

| stage | GPU job rows | gpu_id:count |
| --- | --- | --- |
| stage1 | 504 | 0:61, 1:63, 2:63, 3:62, 4:64, 5:63, 6:64, 7:64 |
| stage2 | 36 | 0:5, 1:4, 2:4, 3:6, 4:5, 5:4, 6:5, 7:3 |
| stage3 | 36 | 0:4, 1:5, 2:5, 3:4, 4:4, 5:4, 6:5, 7:5 |
| stage4 | 60 | 0:8, 1:8, 2:7, 3:8, 4:7, 5:7, 6:7, 7:8 |
| selected | 30 | 0:3, 1:4, 2:3, 3:4, 4:4, 5:4, 6:4, 7:4 |

### Final Manifest Summary

| exp | manifest_count | run_status | test_status | LR | dropout | MSE weight | positive weight | graph mode | MSE target | Cell LLM | cell-type LLM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | 5 | fit_completed | test_completed | 5e-05 | 0.15 | 0.5 | None | real | all | frozen | frozen |
| exp02 | 5 | fit_completed | test_completed | 0.0002 | 0.15 | 0.25 | None | real | all | frozen | frozen |
| exp03 | 5 | fit_completed | test_completed | 5e-05 | 0.15 | 0.5 | None | real | all | frozen | frozen |
| exp04 | 5 | fit_completed | test_completed | 5e-05 | 0.15 | 0.5 | None | real | all | frozen | frozen |
| exp05 | 5 | fit_completed | test_completed | 5e-05 | 0.15 | 0.5 | None | zero | all | frozen | frozen |
| exp06 | 5 | fit_completed | test_completed | 5e-05 | 0.15 | 0.5 | None | real | all | frozen | frozen |
| exp07 | 1 | fit_completed | skipped | 5e-05 | 0.15 | 0.5 | None | real | all | frozen | frozen |
| exp08 | 1 | fit_completed | skipped | 5e-05 | 0.15 | 0.5 | None | real | all | frozen | frozen |

Notes: `exp07` and `exp08` use all-train extra-data inference, so their manifest `test_status=skipped` is expected; their inference outputs are recorded in the final CSV/JSON and runtime summary. `exp05` is the planned no-graph diagnostic and is the only final selected experiment with `graph_feature_mode=zero`.

## Selected Configs

All final selected configs came from the full 5-fold report, not directly from the 3-fold screen. The positive-weight candidates were evaluated, but none won a promoted full stage; the final selected suite uses unweighted BCE (`positive_weight=None`) with lower learning rate for stage1/stage2/stage4.

| stage | final use | selected config | LR | dropout | MSE weight | positive weight | full AUPRC | full n-AUPRC | full AUROC | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage1 | exp01, exp04, exp05, exp07 | mse050_lr5e5 | 5e-5 | 0.15 | 0.50 | none | 0.6774 | 5.7157 | 0.9091 | MSE target all; exp05 graph zero only |
| stage2 | exp03 | mse050_lr5e5 | 5e-5 | 0.15 | 0.50 | none | 0.7869 | 6.6056 | 0.9348 | unseen-cell stage |
| stage3 | exp02 | covdrop010 | 2e-4 | 0.15 | 0.25 | none | 0.8160 | 6.2013 | 0.9458 | COVARIATE_UNK_DROPOUT=0.10 |
| stage4 | exp06, exp08 | mse050_lr5e5 | 5e-5 | 0.15 | 0.50 | none | 0.7952 | 1.9749 | 0.8467 | dual pair fusion; pair type features; DDI; graph_pair_add_scale=0.5 |

### Best Positive-Weight Screen Candidates

| stage | screen rank | best posweight config | AUPRC | score | AUROC |
| --- | --- | --- | --- | --- | --- |
| stage1 | 7 | posw100_drop010 | 0.5565 | 0.6275 | 0.8652 |
| stage4 | 4 | posw100 | 0.7527 | 0.7527 | 0.8068 |

## Reference Epochs

The all-data extra runs used the required mean + nearest reference-epoch policy from selected full folds and then inferred from `last.ckpt`.

| exp | reference folds | raw mean epoch | selected epoch | applied max_epochs | fold epochs | checkpoint policy |
| --- | --- | --- | --- | --- | --- | --- |
| exp07 | checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp01_single_pert_stratified_5fold | 5.4000 | 5 | 6 | 4, 4, 3, 13, 3 | fixed_reference_epoch_last_ckpt |
| exp08 | checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp06_double_pert_pair_5fold | 2.2000 | 2 | 3 | 0, 1, 1, 8, 1 | fixed_reference_epoch_last_ckpt |

## Full 5-Fold Winners

| stage | rank | config | task | folds | complete | valid_cell_llm | auprc | nauprc | auroc | exp04_auprc | exp05_auprc | gap_no_mse_auprc | gap_no_graph_auprc | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage1 | 1 | mse050_lr5e5 | exp01_exp04_exp05 | 5 | True | True | 0.6774 | 5.7157 | 0.9091 | 0.6699 | 0.6099 | 0.0076 | 0.0675 | 0.7150 |
| stage1 | 2 | mse050 | exp01_exp04_exp05 | 5 | True | True | 0.6685 | 5.6496 | 0.8970 | 0.6536 | 0.6040 | 0.0149 | 0.0645 | 0.7081 |
| stage1 | 3 | mse050_lr3e4 | exp01_exp04_exp05 | 5 | True | True | 0.6544 | 5.5232 | 0.8986 | 0.6430 | 0.5991 | 0.0114 | 0.0553 | 0.6877 |
| stage2 | 1 | mse050_lr5e5 | exp03_unseen_cell | 5 | True | True | 0.7869 | 6.6056 | 0.9348 |  |  |  |  | 0.7869 |
| stage2 | 2 | covdrop010_lr5e5 | exp03_unseen_cell | 5 | True | True | 0.7862 | 6.5952 | 0.9312 |  |  |  |  | 0.7862 |
| stage2 | 3 | drop020 | exp03_unseen_cell | 5 | True | True | 0.7635 | 6.3980 | 0.9318 |  |  |  |  | 0.7635 |
| stage3 | 1 | covdrop010 | exp02_unseen_cell_type | 5 | True | True | 0.8160 | 6.2013 | 0.9458 |  |  |  |  | 0.8160 |
| stage3 | 2 | covdrop010_lr5e5 | exp02_unseen_cell_type | 5 | True | True | 0.8150 | 6.2164 | 0.9460 |  |  |  |  | 0.8150 |
| stage3 | 3 | mse050_lr5e5 | exp02_unseen_cell_type | 5 | True | True | 0.8136 | 6.1946 | 0.9438 |  |  |  |  | 0.8136 |
| stage4 | 1 | mse050_lr5e5 | exp06_double_unseen_drug | 5 | True | True | 0.7952 | 1.9749 | 0.8467 |  |  |  |  | 0.7952 |
| stage4 | 2 | rank005 | exp06_double_unseen_drug | 5 | True | True | 0.7572 | 1.8787 | 0.8252 |  |  |  |  | 0.7572 |
| stage4 | 3 | base | exp06_double_unseen_drug | 5 | True | True | 0.7430 | 1.8404 | 0.8186 |  |  |  |  | 0.7430 |

## Final Mean Results

This table includes all requested aggregation modes: `original`, `cell-drug-dose-bylasttime`, and `cell-drug-dose-avgtime`.

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | single unseen drug | mean5 | original | 0.9090 | 0.6791 | 0.1188 | 5.7299 | 17986 | 2137 | 15849 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-bylasttime | 0.9101 | 0.6804 | 0.1185 | 5.7644 | 9032 | 1070 | 7962 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-avgtime | 0.9104 | 0.6798 | 0.1185 | 5.7585 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | original | 0.9458 | 0.8159 | 0.1740 | 6.1988 | 17986 | 2137 | 15849 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-bylasttime | 0.9454 | 0.8157 | 0.1736 | 6.2134 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-avgtime | 0.9460 | 0.8179 | 0.1736 | 6.2214 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | original | 0.9348 | 0.7887 | 0.1549 | 6.6210 | 17986 | 2137 | 15849 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-bylasttime | 0.9344 | 0.7903 | 0.1548 | 6.6397 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-avgtime | 0.9359 | 0.7928 | 0.1548 | 6.6534 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | original | 0.9031 | 0.6712 | 0.1188 | 5.6631 | 17986 | 2137 | 15849 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-bylasttime | 0.9038 | 0.6732 | 0.1185 | 5.7022 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-avgtime | 0.9044 | 0.6736 | 0.1185 | 5.7056 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | original | 0.8621 | 0.6118 | 0.1188 | 5.1611 | 17986 | 2137 | 15849 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-bylasttime | 0.8622 | 0.6111 | 0.1185 | 5.1729 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-avgtime | 0.8623 | 0.6149 | 0.1185 | 5.2052 | 9032 | 1070 | 7962 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | original | 0.8477 | 0.8053 | 0.4043 | 2.0002 | 1791 | 723 | 1068 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-bylasttime | 0.8475 | 0.8070 | 0.4047 | 2.0031 | 896 | 362 | 534 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-avgtime | 0.8497 | 0.8076 | 0.4047 | 2.0047 | 896 | 362 | 534 | 0 | 0 |
| exp07 | extra single | mean_extra | original | 0.7767 | 0.5859 | 0.2090 | 2.7969 | 92671 | 19345 | 73326 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.7770 | 0.5897 | 0.2122 | 2.7703 | 86450 | 18286 | 68164 | 28 | 86450 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.7770 | 0.5897 | 0.2122 | 2.7703 | 86450 | 18286 | 68164 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.6579 | 0.0996 | 0.0476 | 2.2420 | 88970 | 3696 | 85274 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.6559 | 0.1005 | 0.0478 | 2.2568 | 66275 | 2926 | 63349 | 15 | 66275 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.6559 | 0.1005 | 0.0478 | 2.2568 | 66275 | 2926 | 63349 | 15 | 0 |

## Delta vs Previous Selected Baseline

Baseline prefix: `20260604_cell_llm_clip10_tuned_selected_v1`. Deltas below use the `original` method for `exp01`-`exp08`; `exp01`-`exp06` use `mean5`, and `exp07`/`exp08` use `mean_extra`.

| exp | task | split | prev AUPRC | new AUPRC | delta AUPRC | prev AUROC | new AUROC | delta AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | single unseen drug | mean5 | 0.6720 | 0.6791 | 0.0070 | 0.8977 | 0.9090 | 0.0113 |
| exp02 | single unseen cell type | mean5 | 0.8229 | 0.8159 | -0.0070 | 0.9456 | 0.9458 | 0.0002 |
| exp03 | single unseen cell | mean5 | 0.7779 | 0.7887 | 0.0108 | 0.9340 | 0.9348 | 0.0008 |
| exp04 | single w/o MSE | mean5 | 0.6627 | 0.6712 | 0.0085 | 0.8957 | 0.9031 | 0.0074 |
| exp05 | single w/o graph | mean5 | 0.6016 | 0.6118 | 0.0102 | 0.8370 | 0.8621 | 0.0251 |
| exp06 | double unseen drug pair | mean5 | 0.7891 | 0.8053 | 0.0161 | 0.8334 | 0.8477 | 0.0143 |
| exp07 | extra single | mean_extra | 0.5610 | 0.5859 | 0.0249 | 0.7646 | 0.7767 | 0.0121 |
| exp08 | extra double | mean_extra | 0.0919 | 0.0996 | 0.0078 | 0.6384 | 0.6579 | 0.0196 |

### Extra Double Source Delta

Source-level deltas use `exp08`, `split=combined`, `method=original`.

| source | prev AUPRC | new AUPRC | delta AUPRC | prev AUROC | new AUROC | delta AUROC | count | pos | neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guomics | 0.1055 | 0.1144 | 0.0088 | 0.6958 | 0.7178 | 0.0220 | 5633 | 195 | 5438 |
| nature | 0.0576 | 0.0606 | 0.0030 | 0.6360 | 0.6470 | 0.0110 | 68182 | 2394 | 65788 |
| nc | 0.1124 | 0.1238 | 0.0114 | 0.5833 | 0.6091 | 0.0258 | 15155 | 1107 | 14048 |

## Main Reading

- The best full stage1 config was `mse050_lr5e5` with original AUPRC `0.6774`, ahead of the next full config `mse050` by `0.0089` AUPRC.
- Stage2 was effectively tied between `mse050_lr5e5` and `covdrop010_lr5e5` (`0.7869` vs `0.7862` AUPRC), but `mse050_lr5e5` remained rank 1.
- Stage3 selected `covdrop010`; it slightly beat `covdrop010_lr5e5` in full AUPRC (`0.8160` vs `0.8150`).
- Stage4 selected `mse050_lr5e5` with full AUPRC `0.7952`, clearly ahead of `rank005` and `base`.
- Against `20260604_cell_llm_clip10_tuned_selected_v1`, the final run improves original AUPRC on `exp01`, `exp03`, `exp04`, `exp05`, `exp06`, `exp07`, and `exp08`; `exp02` is the main regression in original AUPRC.
- Extra single mean improves from `0.5610` to `0.5859` AUPRC. Extra double mean improves from `0.0919` to `0.0996` AUPRC.

## Screen Report

The full screen table is included because positive-weight candidates were a central part of this run. Stage1 rank uses `exp01_AUPRC + 0.5 * gap_no_MSE + 0.5 * gap_no_graph`; stages2-4 rank by stage task AUPRC.

| stage | rank | config | task | folds | complete | valid_cell_llm | auprc | nauprc | auroc | exp04_auprc | exp05_auprc | gap_no_mse_auprc | gap_no_graph_auprc | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage1 | 1 | mse050_lr5e5 | exp01_exp04_exp05 | 3 | True | True | 0.6164 | 5.1200 | 0.8974 | 0.6094 | 0.5443 | 0.0070 | 0.0721 | 0.6560 |
| stage1 | 2 | mse050 | exp01_exp04_exp05 | 3 | True | True | 0.6058 | 5.0412 | 0.8800 | 0.5857 | 0.5353 | 0.0201 | 0.0705 | 0.6511 |
| stage1 | 3 | mse050_lr3e4 | exp01_exp04_exp05 | 3 | True | True | 0.6006 | 4.9861 | 0.8835 | 0.5825 | 0.5178 | 0.0181 | 0.0828 | 0.6510 |
| stage1 | 4 | base | exp01_exp04_exp05 | 3 | True | True | 0.6003 | 4.9927 | 0.8823 | 0.5857 | 0.5229 | 0.0146 | 0.0775 | 0.6464 |
| stage1 | 5 | mse075_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.6037 | 5.0178 | 0.8765 | 0.6233 | 0.5068 | -0.0197 | 0.0968 | 0.6422 |
| stage1 | 6 | mse050_lr1e4 | exp01_exp04_exp05 | 3 | True | True | 0.6020 | 4.9856 | 0.8830 | 0.5991 | 0.5447 | 0.0030 | 0.0574 | 0.6322 |
| stage1 | 7 | posw100_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5565 | 4.6266 | 0.8652 | 0.5295 | 0.4413 | 0.0270 | 0.1151 | 0.6275 |
| stage1 | 8 | posw100_lr2e4_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5565 | 4.6266 | 0.8652 | 0.5295 | 0.4413 | 0.0270 | 0.1151 | 0.6275 |
| stage1 | 9 | mse050_drop050 | exp01_exp04_exp05 | 3 | True | True | 0.5901 | 4.8915 | 0.8789 | 0.5869 | 0.5224 | 0.0032 | 0.0677 | 0.6255 |
| stage1 | 10 | posw_negpos_lr5e5_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5736 | 4.7646 | 0.8865 | 0.5364 | 0.5113 | 0.0372 | 0.0623 | 0.6234 |
| stage1 | 11 | mse050_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5995 | 4.9829 | 0.8819 | 0.6233 | 0.5495 | -0.0238 | 0.0500 | 0.6126 |
| stage1 | 12 | posw10 | exp01_exp04_exp05 | 3 | True | True | 0.5589 | 4.6420 | 0.8751 | 0.5135 | 0.5010 | 0.0454 | 0.0578 | 0.6105 |
| stage1 | 13 | posw10_lr2e4_drop015 | exp01_exp04_exp05 | 3 | True | True | 0.5589 | 4.6420 | 0.8751 | 0.5135 | 0.5010 | 0.0454 | 0.0578 | 0.6105 |
| stage1 | 14 | posw10_lr5e5_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5741 | 4.7695 | 0.8847 | 0.5645 | 0.5154 | 0.0096 | 0.0587 | 0.6082 |
| stage1 | 15 | posw100 | exp01_exp04_exp05 | 3 | True | True | 0.5288 | 4.3915 | 0.8560 | 0.4941 | 0.4107 | 0.0347 | 0.1181 | 0.6052 |
| stage1 | 16 | posw100_lr2e4_drop015 | exp01_exp04_exp05 | 3 | True | True | 0.5288 | 4.3915 | 0.8560 | 0.4941 | 0.4107 | 0.0347 | 0.1181 | 0.6052 |
| stage1 | 17 | posw_negpos_drop020 | exp01_exp04_exp05 | 3 | True | True | 0.5632 | 4.6743 | 0.8748 | 0.5358 | 0.5067 | 0.0274 | 0.0565 | 0.6051 |
| stage1 | 18 | posw10_lr5e5 | exp01_exp04_exp05 | 3 | True | True | 0.5745 | 4.7732 | 0.8855 | 0.5647 | 0.5233 | 0.0099 | 0.0513 | 0.6051 |
| stage1 | 19 | posw10_lr5e5_drop015 | exp01_exp04_exp05 | 3 | True | True | 0.5745 | 4.7732 | 0.8855 | 0.5647 | 0.5233 | 0.0099 | 0.0513 | 0.6051 |
| stage1 | 20 | posw10_drop020 | exp01_exp04_exp05 | 3 | True | True | 0.5703 | 4.7385 | 0.8673 | 0.5729 | 0.5077 | -0.0026 | 0.0625 | 0.6002 |
| stage1 | 21 | posw_negpos_lr3e4 | exp01_exp04_exp05 | 3 | True | True | 0.5577 | 4.6363 | 0.8559 | 0.5592 | 0.4714 | -0.0015 | 0.0863 | 0.6000 |
| stage1 | 22 | posw_negpos_lr5e5 | exp01_exp04_exp05 | 3 | True | True | 0.5735 | 4.7556 | 0.8830 | 0.5649 | 0.5294 | 0.0085 | 0.0441 | 0.5998 |
| stage1 | 23 | posw_negpos_lr5e5_drop015 | exp01_exp04_exp05 | 3 | True | True | 0.5735 | 4.7556 | 0.8830 | 0.5649 | 0.5294 | 0.0085 | 0.0441 | 0.5998 |
| stage1 | 24 | posw100_lr5e5_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5410 | 4.4939 | 0.8749 | 0.5275 | 0.4512 | 0.0135 | 0.0898 | 0.5927 |
| stage1 | 25 | posw500_lr1e4 | exp01_exp04_exp05 | 3 | True | True | 0.5145 | 4.2580 | 0.8735 | 0.4658 | 0.4094 | 0.0487 | 0.1052 | 0.5915 |
| stage1 | 26 | posw_negpos_lr1e4 | exp01_exp04_exp05 | 3 | True | True | 0.5599 | 4.6435 | 0.8685 | 0.5709 | 0.4914 | -0.0110 | 0.0685 | 0.5886 |
| stage1 | 27 | posw_negpos_drop050 | exp01_exp04_exp05 | 3 | True | True | 0.5462 | 4.5429 | 0.8557 | 0.5371 | 0.4717 | 0.0091 | 0.0746 | 0.5881 |
| stage1 | 28 | mse050_drop020 | exp01_exp04_exp05 | 3 | True | True | 0.5740 | 4.7674 | 0.8667 | 0.5888 | 0.5364 | -0.0148 | 0.0376 | 0.5854 |
| stage1 | 29 | posw10_lr1e4 | exp01_exp04_exp05 | 3 | True | True | 0.5537 | 4.5843 | 0.8677 | 0.5556 | 0.4889 | -0.0019 | 0.0648 | 0.5851 |
| stage1 | 30 | posw100_lr1e4 | exp01_exp04_exp05 | 3 | True | True | 0.5374 | 4.4648 | 0.8680 | 0.5446 | 0.4423 | -0.0072 | 0.0951 | 0.5813 |
| stage1 | 31 | posw100_lr3e4 | exp01_exp04_exp05 | 3 | True | True | 0.5321 | 4.4173 | 0.8575 | 0.5596 | 0.4297 | -0.0275 | 0.1025 | 0.5696 |
| stage1 | 32 | posw10_lr3e4 | exp01_exp04_exp05 | 3 | True | True | 0.5541 | 4.5984 | 0.8745 | 0.5801 | 0.5020 | -0.0260 | 0.0521 | 0.5671 |
| stage1 | 33 | posw100_drop020 | exp01_exp04_exp05 | 3 | True | True | 0.5271 | 4.3654 | 0.8684 | 0.5339 | 0.4473 | -0.0068 | 0.0799 | 0.5637 |
| stage1 | 34 | posw500_drop050 | exp01_exp04_exp05 | 3 | True | True | 0.5031 | 4.1608 | 0.8696 | 0.4833 | 0.4043 | 0.0197 | 0.0988 | 0.5623 |
| stage1 | 35 | posw500 | exp01_exp04_exp05 | 3 | True | True | 0.5073 | 4.2102 | 0.8621 | 0.5015 | 0.4047 | 0.0058 | 0.1026 | 0.5616 |
| stage1 | 36 | posw_negpos_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5459 | 4.5207 | 0.8583 | 0.5570 | 0.5126 | -0.0111 | 0.0334 | 0.5571 |
| stage1 | 37 | posw_negpos_lr2e4_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5459 | 4.5207 | 0.8583 | 0.5570 | 0.5126 | -0.0111 | 0.0334 | 0.5571 |
| stage1 | 38 | posw_negpos | exp01_exp04_exp05 | 3 | True | True | 0.5371 | 4.4541 | 0.8667 | 0.5499 | 0.4877 | -0.0128 | 0.0494 | 0.5554 |
| stage1 | 39 | posw_negpos_lr2e4_drop015 | exp01_exp04_exp05 | 3 | True | True | 0.5371 | 4.4541 | 0.8667 | 0.5499 | 0.4877 | -0.0128 | 0.0494 | 0.5554 |
| stage1 | 40 | posw100_lr5e5 | exp01_exp04_exp05 | 3 | True | True | 0.5188 | 4.3203 | 0.8718 | 0.5166 | 0.4555 | 0.0022 | 0.0634 | 0.5516 |
| stage1 | 41 | posw100_lr5e5_drop015 | exp01_exp04_exp05 | 3 | True | True | 0.5188 | 4.3203 | 0.8718 | 0.5166 | 0.4555 | 0.0022 | 0.0634 | 0.5516 |
| stage1 | 42 | posw200 | exp01_exp04_exp05 | 3 | True | True | 0.5114 | 4.2381 | 0.8558 | 0.5126 | 0.4304 | -0.0012 | 0.0810 | 0.5513 |
| stage1 | 43 | posw10_drop050 | exp01_exp04_exp05 | 3 | True | True | 0.5325 | 4.4148 | 0.8555 | 0.5475 | 0.4836 | -0.0150 | 0.0490 | 0.5495 |
| stage1 | 44 | posw500_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5033 | 4.1708 | 0.8605 | 0.5017 | 0.4192 | 0.0016 | 0.0841 | 0.5462 |
| stage1 | 45 | posw50 | exp01_exp04_exp05 | 3 | True | True | 0.5229 | 4.3384 | 0.8646 | 0.5476 | 0.4552 | -0.0247 | 0.0678 | 0.5445 |
| stage1 | 46 | posw500_lr3e4 | exp01_exp04_exp05 | 3 | True | True | 0.4940 | 4.1013 | 0.8704 | 0.4856 | 0.4014 | 0.0084 | 0.0925 | 0.5444 |
| stage1 | 47 | posw500_drop020 | exp01_exp04_exp05 | 3 | True | True | 0.4943 | 4.0961 | 0.8601 | 0.4911 | 0.4064 | 0.0032 | 0.0879 | 0.5398 |
| stage1 | 48 | posw500_lr5e5 | exp01_exp04_exp05 | 3 | True | True | 0.4991 | 4.1446 | 0.8656 | 0.4841 | 0.4368 | 0.0151 | 0.0623 | 0.5378 |
| stage1 | 49 | posw100_drop050 | exp01_exp04_exp05 | 3 | True | True | 0.4999 | 4.1281 | 0.8660 | 0.5247 | 0.4424 | -0.0249 | 0.0575 | 0.5162 |
| stage1 | 50 | posw10_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5101 | 4.2314 | 0.8511 | 0.5536 | 0.4953 | -0.0436 | 0.0148 | 0.4957 |
| stage1 | 51 | posw10_lr2e4_drop010 | exp01_exp04_exp05 | 3 | True | True | 0.5101 | 4.2314 | 0.8511 | 0.5536 | 0.4953 | -0.0436 | 0.0148 | 0.4957 |
| stage2 | 1 | mse050_lr5e5 | exp03_unseen_cell | 3 | True | True | 0.7743 | 6.0639 | 0.9258 |  |  |  |  | 0.7743 |
| stage2 | 2 | covdrop010_lr5e5 | exp03_unseen_cell | 3 | True | True | 0.7711 | 6.0296 | 0.9203 |  |  |  |  | 0.7711 |
| stage2 | 3 | drop020 | exp03_unseen_cell | 3 | True | True | 0.7519 | 5.8706 | 0.9243 |  |  |  |  | 0.7519 |
| stage2 | 4 | covdrop010 | exp03_unseen_cell | 3 | True | True | 0.7491 | 5.7718 | 0.9207 |  |  |  |  | 0.7491 |
| stage2 | 5 | base | exp03_unseen_cell | 3 | True | True | 0.7484 | 5.8421 | 0.9169 |  |  |  |  | 0.7484 |
| stage2 | 6 | mse050 | exp03_unseen_cell | 3 | True | True | 0.7456 | 5.7689 | 0.9243 |  |  |  |  | 0.7456 |
| stage2 | 7 | covdrop010_drop050 | exp03_unseen_cell | 3 | True | True | 0.7386 | 5.6560 | 0.9203 |  |  |  |  | 0.7386 |
| stage3 | 1 | covdrop010 | exp02_unseen_cell_type | 3 | True | True | 0.8304 | 7.2226 | 0.9558 |  |  |  |  | 0.8304 |
| stage3 | 2 | covdrop010_lr5e5 | exp02_unseen_cell_type | 3 | True | True | 0.8283 | 7.2509 | 0.9564 |  |  |  |  | 0.8283 |
| stage3 | 3 | mse050_lr5e5 | exp02_unseen_cell_type | 3 | True | True | 0.8233 | 7.2018 | 0.9532 |  |  |  |  | 0.8233 |
| stage3 | 4 | mse050 | exp02_unseen_cell_type | 3 | True | True | 0.8195 | 7.1242 | 0.9549 |  |  |  |  | 0.8195 |
| stage3 | 5 | base | exp02_unseen_cell_type | 3 | True | True | 0.8131 | 7.0855 | 0.9546 |  |  |  |  | 0.8131 |
| stage3 | 6 | covdrop010_drop050 | exp02_unseen_cell_type | 3 | True | True | 0.8073 | 7.0277 | 0.9490 |  |  |  |  | 0.8073 |
| stage3 | 7 | covunk_celltype | exp02_unseen_cell_type | 3 | True | True | 0.8058 | 7.0804 | 0.9422 |  |  |  |  | 0.8058 |
| stage4 | 1 | mse050_lr5e5 | exp06_double_unseen_drug | 3 | True | True | 0.8004 | 1.9062 | 0.8463 |  |  |  |  | 0.8004 |
| stage4 | 2 | rank005 | exp06_double_unseen_drug | 3 | True | True | 0.7750 | 1.8447 | 0.8213 |  |  |  |  | 0.7750 |
| stage4 | 3 | base | exp06_double_unseen_drug | 3 | True | True | 0.7628 | 1.8147 | 0.8189 |  |  |  |  | 0.7628 |
| stage4 | 4 | posw100 | exp06_double_unseen_drug | 3 | True | True | 0.7527 | 1.7903 | 0.8068 |  |  |  |  | 0.7527 |
| stage4 | 5 | dbl_mse050 | exp06_double_unseen_drug | 3 | True | True | 0.7518 | 1.7890 | 0.8136 |  |  |  |  | 0.7518 |
| stage4 | 6 | posw10_lr5e5 | exp06_double_unseen_drug | 3 | True | True | 0.7509 | 1.7865 | 0.8130 |  |  |  |  | 0.7509 |
| stage4 | 7 | mse050 | exp06_double_unseen_drug | 3 | True | True | 0.7508 | 1.7886 | 0.8131 |  |  |  |  | 0.7508 |
| stage4 | 8 | posw100_lr5e5 | exp06_double_unseen_drug | 3 | True | True | 0.7409 | 1.7638 | 0.8082 |  |  |  |  | 0.7409 |
| stage4 | 9 | posw_negpos | exp06_double_unseen_drug | 3 | True | True | 0.7408 | 1.7598 | 0.8167 |  |  |  |  | 0.7408 |
| stage4 | 10 | posw50_lr5e5 | exp06_double_unseen_drug | 3 | True | True | 0.7377 | 1.7553 | 0.8119 |  |  |  |  | 0.7377 |
| stage4 | 11 | posw10_drop050 | exp06_double_unseen_drug | 3 | True | True | 0.7270 | 1.7323 | 0.8048 |  |  |  |  | 0.7270 |
| stage4 | 12 | posw10 | exp06_double_unseen_drug | 3 | True | True | 0.7104 | 1.6900 | 0.7869 |  |  |  |  | 0.7104 |
| stage4 | 13 | posw100_drop050 | exp06_double_unseen_drug | 3 | True | True | 0.7089 | 1.6856 | 0.7912 |  |  |  |  | 0.7089 |
| stage4 | 14 | posw50 | exp06_double_unseen_drug | 3 | True | True | 0.6996 | 1.6649 | 0.7790 |  |  |  |  | 0.6996 |
| stage4 | 15 | posw50_drop050 | exp06_double_unseen_drug | 3 | True | True | 0.6715 | 1.5990 | 0.7669 |  |  |  |  | 0.6715 |

## Final Fold Summary

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | single unseen drug | mean5 | original | 0.9090 | 0.6791 | 0.1188 | 5.7299 | 17986 | 2137 | 15849 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-bylasttime | 0.9101 | 0.6804 | 0.1185 | 5.7644 | 9032 | 1070 | 7962 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-avgtime | 0.9104 | 0.6798 | 0.1185 | 5.7585 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | original | 0.9458 | 0.8159 | 0.1740 | 6.1988 | 17986 | 2137 | 15849 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-bylasttime | 0.9454 | 0.8157 | 0.1736 | 6.2134 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-avgtime | 0.9460 | 0.8179 | 0.1736 | 6.2214 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | original | 0.9348 | 0.7887 | 0.1549 | 6.6210 | 17986 | 2137 | 15849 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-bylasttime | 0.9344 | 0.7903 | 0.1548 | 6.6397 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-avgtime | 0.9359 | 0.7928 | 0.1548 | 6.6534 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | original | 0.9031 | 0.6712 | 0.1188 | 5.6631 | 17986 | 2137 | 15849 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-bylasttime | 0.9038 | 0.6732 | 0.1185 | 5.7022 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-avgtime | 0.9044 | 0.6736 | 0.1185 | 5.7056 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | original | 0.8621 | 0.6118 | 0.1188 | 5.1611 | 17986 | 2137 | 15849 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-bylasttime | 0.8622 | 0.6111 | 0.1185 | 5.1729 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-avgtime | 0.8623 | 0.6149 | 0.1185 | 5.2052 | 9032 | 1070 | 7962 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | original | 0.8477 | 0.8053 | 0.4043 | 2.0002 | 1791 | 723 | 1068 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-bylasttime | 0.8475 | 0.8070 | 0.4047 | 2.0031 | 896 | 362 | 534 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-avgtime | 0.8497 | 0.8076 | 0.4047 | 2.0047 | 896 | 362 | 534 | 0 | 0 |

## Final Extra Mean

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp07 | extra single | mean_extra | original | 0.7767 | 0.5859 | 0.2090 | 2.7969 | 92671 | 19345 | 73326 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.7770 | 0.5897 | 0.2122 | 2.7703 | 86450 | 18286 | 68164 | 28 | 86450 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.7770 | 0.5897 | 0.2122 | 2.7703 | 86450 | 18286 | 68164 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.6579 | 0.0996 | 0.0476 | 2.2420 | 88970 | 3696 | 85274 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.6559 | 0.1005 | 0.0478 | 2.2568 | 66275 | 2926 | 63349 | 15 | 66275 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.6559 | 0.1005 | 0.0478 | 2.2568 | 66275 | 2926 | 63349 | 15 | 0 |

## Final Extra Subset Summary

### exp07 Extra Single MAT/Assay Subsets

Rows are grouped by MAT/assay subset. Within each subset, `original` is the row-level metric, followed by the two cell-drug-dose aggregation views.

| subset | method | AUROC | AUPRC | n-AUPRC | baseline | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mat1_480_faims | original | 0.7303 | 0.4944 | 2.4364 | 0.2029 | 17140 | 3478 | 13662 | 0 | 0 |
| mat1_480_faims | cell-drug-dose-bylasttime | 0.7281 | 0.4930 | 2.4023 | 0.2052 | 16445 | 3375 | 13070 | 0 | 16445 |
| mat1_480_faims | cell-drug-dose-avgtime | 0.7281 | 0.4930 | 2.4023 | 0.2052 | 16445 | 3375 | 13070 | 0 | 0 |
| mat1_qe | original | 0.7311 | 0.4940 | 2.4343 | 0.2029 | 17140 | 3478 | 13662 | 0 | 0 |
| mat1_qe | cell-drug-dose-bylasttime | 0.7286 | 0.4925 | 2.3997 | 0.2052 | 16445 | 3375 | 13070 | 0 | 16445 |
| mat1_qe | cell-drug-dose-avgtime | 0.7286 | 0.4925 | 2.3997 | 0.2052 | 16445 | 3375 | 13070 | 0 | 0 |
| mat2_480_faims | original | 0.8241 | 0.6819 | 3.1595 | 0.2158 | 13882 | 2996 | 10886 | 0 | 0 |
| mat2_480_faims | cell-drug-dose-bylasttime | 0.8262 | 0.6911 | 3.1139 | 0.2219 | 12125 | 2691 | 9434 | 9 | 12125 |
| mat2_480_faims | cell-drug-dose-avgtime | 0.8262 | 0.6911 | 3.1139 | 0.2219 | 12125 | 2691 | 9434 | 9 | 0 |
| mat2_qe | original | 0.8271 | 0.6894 | 3.1945 | 0.2158 | 13882 | 2996 | 10886 | 0 | 0 |
| mat2_qe | cell-drug-dose-bylasttime | 0.8290 | 0.6978 | 3.1441 | 0.2219 | 12125 | 2691 | 9434 | 9 | 12125 |
| mat2_qe | cell-drug-dose-avgtime | 0.8290 | 0.6978 | 3.1441 | 0.2219 | 12125 | 2691 | 9434 | 9 | 0 |
| mat3_qe | original | 0.7306 | 0.4992 | 2.3676 | 0.2108 | 18332 | 3865 | 14467 | 0 | 0 |
| mat3_qe | cell-drug-dose-bylasttime | 0.7305 | 0.4995 | 2.3638 | 0.2113 | 18287 | 3864 | 14423 | 0 | 18287 |
| mat3_qe | cell-drug-dose-avgtime | 0.7305 | 0.4995 | 2.3638 | 0.2113 | 18287 | 3864 | 14423 | 0 | 0 |
| mat4_qe | original | 0.8170 | 0.6567 | 3.1890 | 0.2059 | 12295 | 2532 | 9763 | 0 | 0 |
| mat4_qe | cell-drug-dose-bylasttime | 0.8194 | 0.6644 | 3.1982 | 0.2077 | 11023 | 2290 | 8733 | 10 | 11023 |
| mat4_qe | cell-drug-dose-avgtime | 0.8194 | 0.6644 | 3.1982 | 0.2077 | 11023 | 2290 | 8733 | 10 | 0 |

### exp08 Extra Double Source Subsets

Rows are grouped by source. Within each source, split order is `unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`, then `combined`; each split keeps the three metric methods together.

#### exp08 guomics

| split | method | AUROC | AUPRC | n-AUPRC | baseline | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseenCell_seenDrugCombo | original | 1.0000 | 1.0000 | 21.0000 | 0.0476 | 42 | 2 | 40 | 0 | 0 |
| unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 1.0000 | 1.0000 | 21.0000 | 0.0476 | 42 | 2 | 40 | 0 | 42 |
| unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 1.0000 | 1.0000 | 21.0000 | 0.0476 | 42 | 2 | 40 | 0 | 0 |
| unseenCell_unseenDrugCombo | original | 0.7157 | 0.1013 | 2.9355 | 0.0345 | 5591 | 193 | 5398 | 0 | 0 |
| unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.7157 | 0.1013 | 2.9355 | 0.0345 | 5591 | 193 | 5398 | 0 | 5591 |
| unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.7157 | 0.1013 | 2.9355 | 0.0345 | 5591 | 193 | 5398 | 0 | 0 |
| combined | original | 0.7178 | 0.1144 | 3.3042 | 0.0346 | 5633 | 195 | 5438 | 0 | 0 |
| combined | cell-drug-dose-bylasttime | 0.7178 | 0.1144 | 3.3042 | 0.0346 | 5633 | 195 | 5438 | 0 | 5633 |
| combined | cell-drug-dose-avgtime | 0.7178 | 0.1144 | 3.3042 | 0.0346 | 5633 | 195 | 5438 | 0 | 0 |

#### exp08 nature

| split | method | AUROC | AUPRC | n-AUPRC | baseline | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseenCell_seenDrugCombo | original | 0.6746 | 0.1078 | 1.8065 | 0.0597 | 16652 | 994 | 15658 | 0 | 0 |
| unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.6482 | 0.1112 | 1.7606 | 0.0632 | 10576 | 668 | 9908 | 4 | 10576 |
| unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.6482 | 0.1112 | 1.7606 | 0.0632 | 10576 | 668 | 9908 | 4 | 0 |
| unseenCell_unseenDrugCombo | original | 0.6109 | 0.0365 | 1.3423 | 0.0272 | 51530 | 1400 | 50130 | 0 | 0 |
| unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.6175 | 0.0382 | 1.3963 | 0.0274 | 34911 | 956 | 33955 | 11 | 34911 |
| unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.6175 | 0.0382 | 1.3963 | 0.0274 | 34911 | 956 | 33955 | 11 | 0 |
| combined | original | 0.6470 | 0.0606 | 1.7270 | 0.0351 | 68182 | 2394 | 65788 | 0 | 0 |
| combined | cell-drug-dose-bylasttime | 0.6408 | 0.0632 | 1.7714 | 0.0357 | 45487 | 1624 | 43863 | 15 | 45487 |
| combined | cell-drug-dose-avgtime | 0.6408 | 0.0632 | 1.7714 | 0.0357 | 45487 | 1624 | 43863 | 15 | 0 |

#### exp08 nc

| split | method | AUROC | AUPRC | n-AUPRC | baseline | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseenCell_seenDrugCombo | original | 0.6879 | 0.2180 | 2.3725 | 0.0919 | 2057 | 189 | 1868 | 0 | 0 |
| unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.6879 | 0.2180 | 2.3725 | 0.0919 | 2057 | 189 | 1868 | 0 | 2057 |
| unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.6879 | 0.2180 | 2.3725 | 0.0919 | 2057 | 189 | 1868 | 0 | 0 |
| unseenCell_unseenDrugCombo | original | 0.5908 | 0.1014 | 1.4467 | 0.0701 | 13098 | 918 | 12180 | 0 | 0 |
| unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.5908 | 0.1014 | 1.4467 | 0.0701 | 13098 | 918 | 12180 | 0 | 13098 |
| unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.5908 | 0.1014 | 1.4467 | 0.0701 | 13098 | 918 | 12180 | 0 | 0 |
| combined | original | 0.6091 | 0.1238 | 1.6948 | 0.0730 | 15155 | 1107 | 14048 | 0 | 0 |
| combined | cell-drug-dose-bylasttime | 0.6091 | 0.1238 | 1.6948 | 0.0730 | 15155 | 1107 | 14048 | 0 | 15155 |
| combined | cell-drug-dose-avgtime | 0.6091 | 0.1238 | 1.6948 | 0.0730 | 15155 | 1107 | 14048 | 0 | 0 |

## Final Fold Detail

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | single unseen drug | fold0 | original | 0.8736 | 0.5678 | 0.1209 | 4.6958 | 3606 | 436 | 3170 | 0 | 0 |
| exp01 | single unseen drug | fold0 | cell-drug-dose-bylasttime | 0.8731 | 0.5683 | 0.1215 | 4.6781 | 1811 | 220 | 1591 | 0 | 0 |
| exp01 | single unseen drug | fold0 | cell-drug-dose-avgtime | 0.8725 | 0.5664 | 0.1215 | 4.6625 | 1811 | 220 | 1591 | 0 | 0 |
| exp01 | single unseen drug | fold1 | original | 0.9491 | 0.8019 | 0.1200 | 6.6813 | 3591 | 431 | 3160 | 0 | 0 |
| exp01 | single unseen drug | fold1 | cell-drug-dose-bylasttime | 0.9492 | 0.8058 | 0.1203 | 6.6986 | 1804 | 217 | 1587 | 0 | 0 |
| exp01 | single unseen drug | fold1 | cell-drug-dose-avgtime | 0.9497 | 0.8018 | 0.1203 | 6.6659 | 1804 | 217 | 1587 | 0 | 0 |
| exp01 | single unseen drug | fold2 | original | 0.8998 | 0.5889 | 0.1085 | 5.4288 | 3614 | 392 | 3222 | 0 | 0 |
| exp01 | single unseen drug | fold2 | cell-drug-dose-bylasttime | 0.9021 | 0.5923 | 0.1060 | 5.5869 | 1811 | 192 | 1619 | 0 | 0 |
| exp01 | single unseen drug | fold2 | cell-drug-dose-avgtime | 0.9030 | 0.5905 | 0.1060 | 5.5701 | 1811 | 192 | 1619 | 0 | 0 |
| exp01 | single unseen drug | fold3 | original | 0.9039 | 0.7407 | 0.1126 | 6.5781 | 3597 | 405 | 3192 | 0 | 0 |
| exp01 | single unseen drug | fold3 | cell-drug-dose-bylasttime | 0.9073 | 0.7398 | 0.1115 | 6.6325 | 1811 | 202 | 1609 | 0 | 0 |
| exp01 | single unseen drug | fold3 | cell-drug-dose-avgtime | 0.9078 | 0.7413 | 0.1115 | 6.6458 | 1811 | 202 | 1609 | 0 | 0 |
| exp01 | single unseen drug | fold4 | original | 0.9186 | 0.6961 | 0.1322 | 5.2655 | 3578 | 473 | 3105 | 0 | 0 |
| exp01 | single unseen drug | fold4 | cell-drug-dose-bylasttime | 0.9186 | 0.6958 | 0.1331 | 5.2257 | 1795 | 239 | 1556 | 0 | 0 |
| exp01 | single unseen drug | fold4 | cell-drug-dose-avgtime | 0.9193 | 0.6988 | 0.1331 | 5.2481 | 1795 | 239 | 1556 | 0 | 0 |
| exp02 | single unseen cell type | fold0 | original | 0.9448 | 0.7619 | 0.1454 | 5.2391 | 7750 | 1127 | 6623 | 0 | 0 |
| exp02 | single unseen cell type | fold0 | cell-drug-dose-bylasttime | 0.9428 | 0.7533 | 0.1443 | 5.2217 | 3854 | 556 | 3298 | 0 | 0 |
| exp02 | single unseen cell type | fold0 | cell-drug-dose-avgtime | 0.9440 | 0.7605 | 0.1443 | 5.2716 | 3854 | 556 | 3298 | 0 | 0 |
| exp02 | single unseen cell type | fold1 | original | 0.9208 | 0.6995 | 0.1062 | 6.5880 | 5453 | 579 | 4874 | 0 | 0 |
| exp02 | single unseen cell type | fold1 | cell-drug-dose-bylasttime | 0.9195 | 0.7020 | 0.1073 | 6.5452 | 2741 | 294 | 2447 | 0 | 0 |
| exp02 | single unseen cell type | fold1 | cell-drug-dose-avgtime | 0.9212 | 0.7026 | 0.1073 | 6.5505 | 2741 | 294 | 2447 | 0 | 0 |
| exp02 | single unseen cell type | fold2 | original | 0.9618 | 0.8957 | 0.2246 | 3.9879 | 374 | 84 | 290 | 0 | 0 |
| exp02 | single unseen cell type | fold2 | cell-drug-dose-bylasttime | 0.9584 | 0.8868 | 0.2194 | 4.0420 | 196 | 43 | 153 | 0 | 0 |
| exp02 | single unseen cell type | fold2 | cell-drug-dose-avgtime | 0.9608 | 0.8936 | 0.2194 | 4.0733 | 196 | 43 | 153 | 0 | 0 |
| exp02 | single unseen cell type | fold3 | original | 0.9409 | 0.8846 | 0.3265 | 2.7090 | 196 | 64 | 132 | 0 | 0 |
| exp02 | single unseen cell type | fold3 | cell-drug-dose-bylasttime | 0.9435 | 0.8922 | 0.3300 | 2.7035 | 100 | 33 | 67 | 0 | 0 |
| exp02 | single unseen cell type | fold3 | cell-drug-dose-avgtime | 0.9435 | 0.8914 | 0.3300 | 2.7011 | 100 | 33 | 67 | 0 | 0 |
| exp02 | single unseen cell type | fold4 | original | 0.9606 | 0.8377 | 0.0672 | 12.4701 | 4213 | 283 | 3930 | 0 | 0 |
| exp02 | single unseen cell type | fold4 | cell-drug-dose-bylasttime | 0.9627 | 0.8444 | 0.0673 | 12.5548 | 2141 | 144 | 1997 | 0 | 0 |
| exp02 | single unseen cell type | fold4 | cell-drug-dose-avgtime | 0.9607 | 0.8414 | 0.0673 | 12.5103 | 2141 | 144 | 1997 | 0 | 0 |
| exp03 | single unseen cell | fold0 | original | 0.9177 | 0.8384 | 0.3039 | 2.7590 | 1369 | 416 | 953 | 0 | 0 |
| exp03 | single unseen cell | fold0 | cell-drug-dose-bylasttime | 0.9151 | 0.8358 | 0.3037 | 2.7517 | 698 | 212 | 486 | 0 | 0 |
| exp03 | single unseen cell | fold0 | cell-drug-dose-avgtime | 0.9170 | 0.8401 | 0.3037 | 2.7661 | 698 | 212 | 486 | 0 | 0 |
| exp03 | single unseen cell | fold1 | original | 0.9373 | 0.7799 | 0.1051 | 7.4180 | 5174 | 544 | 4630 | 0 | 0 |
| exp03 | single unseen cell | fold1 | cell-drug-dose-bylasttime | 0.9373 | 0.7797 | 0.1093 | 7.1330 | 2644 | 289 | 2355 | 0 | 0 |
| exp03 | single unseen cell | fold1 | cell-drug-dose-avgtime | 0.9377 | 0.7814 | 0.1093 | 7.1491 | 2644 | 289 | 2355 | 0 | 0 |
| exp03 | single unseen cell | fold2 | original | 0.8881 | 0.6541 | 0.1834 | 3.5663 | 1303 | 239 | 1064 | 0 | 0 |
| exp03 | single unseen cell | fold2 | cell-drug-dose-bylasttime | 0.8871 | 0.6529 | 0.1845 | 3.5398 | 656 | 121 | 535 | 0 | 0 |
| exp03 | single unseen cell | fold2 | cell-drug-dose-avgtime | 0.8893 | 0.6596 | 0.1845 | 3.5759 | 656 | 121 | 535 | 0 | 0 |
| exp03 | single unseen cell | fold3 | original | 0.9592 | 0.8361 | 0.1121 | 7.4614 | 5408 | 606 | 4802 | 0 | 0 |
| exp03 | single unseen cell | fold3 | cell-drug-dose-bylasttime | 0.9617 | 0.8469 | 0.1049 | 8.0743 | 2641 | 277 | 2364 | 0 | 0 |
| exp03 | single unseen cell | fold3 | cell-drug-dose-avgtime | 0.9647 | 0.8457 | 0.1049 | 8.0629 | 2641 | 277 | 2364 | 0 | 0 |
| exp03 | single unseen cell | fold4 | original | 0.9716 | 0.8349 | 0.0702 | 11.9002 | 4732 | 332 | 4400 | 0 | 0 |
| exp03 | single unseen cell | fold4 | cell-drug-dose-bylasttime | 0.9708 | 0.8360 | 0.0715 | 11.6997 | 2393 | 171 | 2222 | 0 | 0 |
| exp03 | single unseen cell | fold4 | cell-drug-dose-avgtime | 0.9707 | 0.8370 | 0.0715 | 11.7131 | 2393 | 171 | 2222 | 0 | 0 |
| exp04 | single w/o MSE | fold0 | original | 0.8646 | 0.5913 | 0.1209 | 4.8908 | 3606 | 436 | 3170 | 0 | 0 |
| exp04 | single w/o MSE | fold0 | cell-drug-dose-bylasttime | 0.8637 | 0.5905 | 0.1215 | 4.8605 | 1811 | 220 | 1591 | 0 | 0 |
| exp04 | single w/o MSE | fold0 | cell-drug-dose-avgtime | 0.8646 | 0.5921 | 0.1215 | 4.8737 | 1811 | 220 | 1591 | 0 | 0 |
| exp04 | single w/o MSE | fold1 | original | 0.9436 | 0.7632 | 0.1200 | 6.3589 | 3591 | 431 | 3160 | 0 | 0 |
| exp04 | single w/o MSE | fold1 | cell-drug-dose-bylasttime | 0.9437 | 0.7657 | 0.1203 | 6.3659 | 1804 | 217 | 1587 | 0 | 0 |
| exp04 | single w/o MSE | fold1 | cell-drug-dose-avgtime | 0.9441 | 0.7647 | 0.1203 | 6.3571 | 1804 | 217 | 1587 | 0 | 0 |
| exp04 | single w/o MSE | fold2 | original | 0.8838 | 0.5580 | 0.1085 | 5.1441 | 3614 | 392 | 3222 | 0 | 0 |
| exp04 | single w/o MSE | fold2 | cell-drug-dose-bylasttime | 0.8863 | 0.5645 | 0.1060 | 5.3244 | 1811 | 192 | 1619 | 0 | 0 |
| exp04 | single w/o MSE | fold2 | cell-drug-dose-avgtime | 0.8872 | 0.5608 | 0.1060 | 5.2897 | 1811 | 192 | 1619 | 0 | 0 |
| exp04 | single w/o MSE | fold3 | original | 0.9117 | 0.7610 | 0.1126 | 6.7589 | 3597 | 405 | 3192 | 0 | 0 |
| exp04 | single w/o MSE | fold3 | cell-drug-dose-bylasttime | 0.9122 | 0.7602 | 0.1115 | 6.8154 | 1811 | 202 | 1609 | 0 | 0 |
| exp04 | single w/o MSE | fold3 | cell-drug-dose-avgtime | 0.9133 | 0.7648 | 0.1115 | 6.8569 | 1811 | 202 | 1609 | 0 | 0 |
| exp04 | single w/o MSE | fold4 | original | 0.9118 | 0.6825 | 0.1322 | 5.1628 | 3578 | 473 | 3105 | 0 | 0 |
| exp04 | single w/o MSE | fold4 | cell-drug-dose-bylasttime | 0.9129 | 0.6850 | 0.1331 | 5.1448 | 1795 | 239 | 1556 | 0 | 0 |
| exp04 | single w/o MSE | fold4 | cell-drug-dose-avgtime | 0.9127 | 0.6858 | 0.1331 | 5.1507 | 1795 | 239 | 1556 | 0 | 0 |
| exp05 | single w/o graph | fold0 | original | 0.8420 | 0.5275 | 0.1209 | 4.3625 | 3606 | 436 | 3170 | 0 | 0 |
| exp05 | single w/o graph | fold0 | cell-drug-dose-bylasttime | 0.8415 | 0.5247 | 0.1215 | 4.3192 | 1811 | 220 | 1591 | 0 | 0 |
| exp05 | single w/o graph | fold0 | cell-drug-dose-avgtime | 0.8406 | 0.5244 | 0.1215 | 4.3167 | 1811 | 220 | 1591 | 0 | 0 |
| exp05 | single w/o graph | fold1 | original | 0.8897 | 0.7381 | 0.1200 | 6.1493 | 3591 | 431 | 3160 | 0 | 0 |
| exp05 | single w/o graph | fold1 | cell-drug-dose-bylasttime | 0.8883 | 0.7358 | 0.1203 | 6.1171 | 1804 | 217 | 1587 | 0 | 0 |
| exp05 | single w/o graph | fold1 | cell-drug-dose-avgtime | 0.8896 | 0.7414 | 0.1203 | 6.1638 | 1804 | 217 | 1587 | 0 | 0 |
| exp05 | single w/o graph | fold2 | original | 0.8446 | 0.4986 | 0.1085 | 4.5972 | 3614 | 392 | 3222 | 0 | 0 |
| exp05 | single w/o graph | fold2 | cell-drug-dose-bylasttime | 0.8438 | 0.4990 | 0.1060 | 4.7062 | 1811 | 192 | 1619 | 0 | 0 |
| exp05 | single w/o graph | fold2 | cell-drug-dose-avgtime | 0.8452 | 0.4996 | 0.1060 | 4.7127 | 1811 | 192 | 1619 | 0 | 0 |
| exp05 | single w/o graph | fold3 | original | 0.8832 | 0.6835 | 0.1126 | 6.0705 | 3597 | 405 | 3192 | 0 | 0 |
| exp05 | single w/o graph | fold3 | cell-drug-dose-bylasttime | 0.8829 | 0.6788 | 0.1115 | 6.0858 | 1811 | 202 | 1609 | 0 | 0 |
| exp05 | single w/o graph | fold3 | cell-drug-dose-avgtime | 0.8872 | 0.6892 | 0.1115 | 6.1794 | 1811 | 202 | 1609 | 0 | 0 |
| exp05 | single w/o graph | fold4 | original | 0.8509 | 0.6115 | 0.1322 | 4.6260 | 3578 | 473 | 3105 | 0 | 0 |
| exp05 | single w/o graph | fold4 | cell-drug-dose-bylasttime | 0.8544 | 0.6173 | 0.1331 | 4.6360 | 1795 | 239 | 1556 | 0 | 0 |
| exp05 | single w/o graph | fold4 | cell-drug-dose-avgtime | 0.8488 | 0.6196 | 0.1331 | 4.6535 | 1795 | 239 | 1556 | 0 | 0 |
| exp06 | double unseen drug pair | fold0 | original | 0.8410 | 0.8126 | 0.4342 | 1.8716 | 357 | 155 | 202 | 0 | 0 |
| exp06 | double unseen drug pair | fold0 | cell-drug-dose-bylasttime | 0.8401 | 0.8114 | 0.4358 | 1.8620 | 179 | 78 | 101 | 0 | 0 |
| exp06 | double unseen drug pair | fold0 | cell-drug-dose-avgtime | 0.8437 | 0.8147 | 0.4358 | 1.8697 | 179 | 78 | 101 | 0 | 0 |
| exp06 | double unseen drug pair | fold1 | original | 0.8536 | 0.8230 | 0.3632 | 2.2664 | 380 | 138 | 242 | 0 | 0 |
| exp06 | double unseen drug pair | fold1 | cell-drug-dose-bylasttime | 0.8517 | 0.8244 | 0.3632 | 2.2701 | 190 | 69 | 121 | 0 | 0 |
| exp06 | double unseen drug pair | fold1 | cell-drug-dose-avgtime | 0.8550 | 0.8268 | 0.3632 | 2.2766 | 190 | 69 | 121 | 0 | 0 |
| exp06 | double unseen drug pair | fold2 | original | 0.8711 | 0.8080 | 0.4022 | 2.0087 | 358 | 144 | 214 | 0 | 0 |
| exp06 | double unseen drug pair | fold2 | cell-drug-dose-bylasttime | 0.8721 | 0.8097 | 0.4022 | 2.0131 | 179 | 72 | 107 | 0 | 0 |
| exp06 | double unseen drug pair | fold2 | cell-drug-dose-avgtime | 0.8716 | 0.8087 | 0.4022 | 2.0106 | 179 | 72 | 107 | 0 | 0 |
| exp06 | double unseen drug pair | fold3 | original | 0.8459 | 0.7923 | 0.3977 | 1.9921 | 352 | 140 | 212 | 0 | 0 |
| exp06 | double unseen drug pair | fold3 | cell-drug-dose-bylasttime | 0.8437 | 0.7922 | 0.3977 | 1.9919 | 176 | 70 | 106 | 0 | 0 |
| exp06 | double unseen drug pair | fold3 | cell-drug-dose-avgtime | 0.8500 | 0.7959 | 0.3977 | 2.0011 | 176 | 70 | 106 | 0 | 0 |
| exp06 | double unseen drug pair | fold4 | original | 0.8268 | 0.7905 | 0.4244 | 1.8624 | 344 | 146 | 198 | 0 | 0 |
| exp06 | double unseen drug pair | fold4 | cell-drug-dose-bylasttime | 0.8299 | 0.7972 | 0.4244 | 1.8784 | 172 | 73 | 99 | 0 | 0 |
| exp06 | double unseen drug pair | fold4 | cell-drug-dose-avgtime | 0.8281 | 0.7917 | 0.4244 | 1.8653 | 172 | 73 | 99 | 0 | 0 |

## Reproducibility Notes

- Environment expected by repository instructions: `conda activate flow_v2`.
- Runner: `scripts/run_ptv01_08_posweight_combo_tune.sh`.
- Report generator for final metrics: `scripts/report_cell_drug_time_eval.py`.
- Tuning reporter wrapper: `scripts/report_cell_celltype_llm_clip10_param_search.py`.
- The final selected run already materialized fold predictions under `outputs/2026-06/2026-06-10/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_fold_predictions/`; rerunning was not necessary for this report.
- All displayed metric values are rounded to 4 decimals, matching the tuning plan. Raw precision remains in the CSV/JSON source artifacts.
