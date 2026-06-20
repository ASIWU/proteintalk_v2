# 2026-06-09 Cell + Cell-type LLM Clip10 Tuned Results

## Run Scope

This is the completed tuning run for the graph + frozen Cell LLM + frozen cell-type LLM architecture.

- Screen prefix: `20260608_cell_celltype_llm_clip10_tune_v1`
- Full promoted prefix: `20260608_cell_celltype_llm_clip10_tune_v1_full`
- Final selected prefix: `20260608_cell_celltype_llm_clip10_tuned_selected_v1`
- Full report: `logs/20260608_cell_celltype_llm_clip10_tune_v1_full_param_search_report.md`
- Final selected Markdown: `logs/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.md`
- Final selected CSV: `outputs/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.csv`
- Final selected JSON: `outputs/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.json`
- Launch logs: `logs/20260609_cell_celltype_llm_clip10_tune_v1_full_launch.log`, `logs/20260609_cell_celltype_llm_clip10_tuned_selected_v1_auto_launch.log`

## Completion Audit

- Screen phase completed on 2026-06-09 before the full launch.
- Full promoted 5-fold phase completed and produced the full ranking report.
- Final selected exp01-exp08 phase completed and produced CSV/JSON/Markdown evaluation outputs.
- Final selected manifest audit: `32/32` manifests present, `audit_errors=0`.
- Final exp01-exp06 have five completed folds each; exp07 and exp08 have one all-data training run each plus extra-data inference.
- All final manifests use `cell_llm_mode=frozen`, Cell embedding rows `74`, feature dim `4096`.
- All final manifests use `cell_type_llm_mode=frozen`, cell-type embedding rows `14`, feature dim `4096`.
- exp05 is the diagnostic no-graph ablation and records `graph_feature_mode=zero`; exp01/02/03/04/06/07/08 record `graph_feature_mode=real`.
- Extra prediction artifacts: exp07 has six single-drug extra task prediction files; exp08 has three double-drug extra task prediction files.

## Screen To Full Promotion

| stage | rank1 screen config | screen AUPRC | screen score | promoted |
| --- | --- | --- | --- | --- |
| stage1 | mse050 | 0.6058 | 0.6511 | mse050, mse050_warmdecay, mse050_lr3e4 |
| stage2 | covdrop010_lr1e4 | 0.7674 | 0.7674 | covdrop010_lr1e4, lr1e4, drop020 |
| stage3 | covdrop010 | 0.8304 | 0.8304 | covdrop010, covdrop010_lr1e4, covdrop010_drop020, covdrop010_drop010 |
| stage4 | rank005 | 0.7750 | 0.7750 | rank005, base, dbl_mse050 |

Stage1 score is `exp01_AUPRC + 0.5000*(exp01_AUPRC-exp04_AUPRC) + 0.5000*(exp01_AUPRC-exp05_AUPRC)`. Other stages rank by AUPRC with n-AUPRC/AUROC tie-breaks.

## Full 5-fold Winners

| stage | selected config | final use | AUPRC | n-AUPRC | AUROC | notes |
| --- | --- | --- | --- | --- | --- | --- |
| stage1 | mse050 | exp01/exp04/exp05/exp07 | 0.6685 | 5.6496 | 0.8970 | tied with mse050_warmdecay; earlier rank kept |
| stage2 | covdrop010_lr1e4 | exp03 | 0.7823 | 6.5148 | 0.9322 | unseen cell |
| stage3 | covdrop010 | exp02 | 0.8160 | 6.2013 | 0.9458 | unseen cell type |
| stage4 | rank005 | exp06/exp08 | 0.7572 | 1.8787 | 0.8252 | double-drug fixed settings |

Final selected mapping:

- exp01, exp04, exp05, exp07 use stage1 config `mse050`.
- exp03 uses stage2 config `covdrop010_lr1e4`.
- exp02 uses stage3 config `covdrop010`.
- exp06 and exp08 use stage4 config `rank005` with double-drug fixed settings.

Reference epoch policy:

| exp | source folds | selected epoch | all-data max_epochs | checkpoint |
|---|---|---:|---:|---|
| exp07 | selected exp01 5-fold | 11 | 12 | `last.ckpt` |
| exp08 | selected exp06 5-fold | 4 | 5 | `last.ckpt` |

## Final Mean Results, Original Row Level

| exp | task | split | AUROC | AUPRC | n-AUPRC | baseline | count | pos | neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | single unseen drug | mean5 | 0.8969 | 0.6694 | 5.6579 | 0.1188 | 17986 | 2137 | 15849 |
| exp02 | single unseen cell type | mean5 | 0.9458 | 0.8159 | 6.1988 | 0.1740 | 17986 | 2137 | 15849 |
| exp03 | single unseen cell | mean5 | 0.9323 | 0.7841 | 6.5310 | 0.1549 | 17986 | 2137 | 15849 |
| exp04 | single w/o MSE | mean5 | 0.9065 | 0.6571 | 5.5424 | 0.1188 | 17986 | 2137 | 15849 |
| exp05 | single w/o graph | mean5 | 0.8425 | 0.6066 | 5.1123 | 0.1188 | 17986 | 2137 | 15849 |
| exp06 | double unseen drug pair | mean5 | 0.8258 | 0.7677 | 1.9047 | 0.4043 | 1791 | 723 | 1068 |
| exp07 | extra single | mean_extra | 0.7604 | 0.5610 | 2.6771 | 0.2090 | 92671 | 19345 | 73326 |
| exp08 | extra double | mean_extra | 0.6503 | 0.0964 | 2.1467 | 0.0476 | 88970 | 3696 | 85274 |

## Final Mean Results, All Grouping Methods

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | baseline | count | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | single unseen drug | mean5 | original | 0.8969 | 0.6694 | 5.6579 | 0.1188 | 17986 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-bylasttime | 0.8983 | 0.6692 | 5.6818 | 0.1185 | 9032 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-avgtime | 0.8979 | 0.6727 | 5.7080 | 0.1185 | 9032 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | original | 0.9458 | 0.8159 | 6.1988 | 0.1740 | 17986 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-bylasttime | 0.9454 | 0.8157 | 6.2134 | 0.1736 | 9032 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-avgtime | 0.9460 | 0.8179 | 6.2214 | 0.1736 | 9032 | 0 | 0 |
| exp03 | single unseen cell | mean5 | original | 0.9323 | 0.7841 | 6.5310 | 0.1549 | 17986 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-bylasttime | 0.9323 | 0.7868 | 6.5616 | 0.1548 | 9032 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-avgtime | 0.9330 | 0.7873 | 6.5589 | 0.1548 | 9032 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | original | 0.9065 | 0.6571 | 5.5424 | 0.1188 | 17986 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-bylasttime | 0.9066 | 0.6573 | 5.5662 | 0.1185 | 9032 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-avgtime | 0.9079 | 0.6599 | 5.5878 | 0.1185 | 9032 | 0 | 0 |
| exp05 | single w/o graph | mean5 | original | 0.8425 | 0.6066 | 5.1123 | 0.1188 | 17986 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-bylasttime | 0.8430 | 0.6064 | 5.1275 | 0.1185 | 9032 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-avgtime | 0.8432 | 0.6110 | 5.1672 | 0.1185 | 9032 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | original | 0.8258 | 0.7677 | 1.9047 | 0.4043 | 1791 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-bylasttime | 0.8262 | 0.7686 | 1.9050 | 0.4047 | 896 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-avgtime | 0.8258 | 0.7685 | 1.9053 | 0.4047 | 896 | 0 | 0 |
| exp07 | extra single | mean_extra | original | 0.7604 | 0.5610 | 2.6771 | 0.2090 | 92671 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.7597 | 0.5633 | 2.6452 | 0.2122 | 86450 | 28 | 86450 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.7597 | 0.5633 | 2.6452 | 0.2122 | 86450 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.6503 | 0.0964 | 2.1467 | 0.0476 | 88970 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.6472 | 0.0959 | 2.1231 | 0.0478 | 66275 | 15 | 66275 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.6472 | 0.0959 | 2.1231 | 0.0478 | 66275 | 15 | 0 |

## Extra Split Results, Original Row Level

These tables expand the exp07/exp08 extra-data rows instead of reporting only `mean_extra`.

### exp07 Extra Single by MAT

| matrix | assay | AUROC | AUPRC | n-AUPRC | baseline | count | pos | neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mat1 | 480_faims | 0.7094 | 0.4706 | 2.3192 | 0.2029 | 17140 | 3478 | 13662 |
| mat1 | qe | 0.7118 | 0.4585 | 2.2597 | 0.2029 | 17140 | 3478 | 13662 |
| mat2 | 480_faims | 0.8107 | 0.6642 | 3.0776 | 0.2158 | 13882 | 2996 | 10886 |
| mat2 | qe | 0.8145 | 0.6693 | 3.1010 | 0.2158 | 13882 | 2996 | 10886 |
| mat3 | qe | 0.7120 | 0.4666 | 2.2130 | 0.2108 | 18332 | 3865 | 14467 |
| mat4 | qe | 0.8042 | 0.6368 | 3.0922 | 0.2059 | 12295 | 2532 | 9763 |

### exp08 Extra Double by Source

| source | split | AUROC | AUPRC | n-AUPRC | baseline | count | pos | neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guomics | combined | 0.6795 | 0.0924 | 2.6690 | 0.0346 | 5633 | 195 | 5438 |
| guomics | unseenCell_seenDrugCombo | 1.0000 | 1.0000 | 21.0000 | 0.0476 | 42 | 2 | 40 |
| guomics | unseenCell_unseenDrugCombo | 0.6772 | 0.0859 | 2.4883 | 0.0345 | 5591 | 193 | 5398 |
| nc | combined | 0.6041 | 0.1237 | 1.6941 | 0.0730 | 15155 | 1107 | 14048 |
| nc | unseenCell_seenDrugCombo | 0.6838 | 0.2231 | 2.4278 | 0.0919 | 2057 | 189 | 1868 |
| nc | unseenCell_unseenDrugCombo | 0.5853 | 0.1002 | 1.4300 | 0.0701 | 13098 | 918 | 12180 |
| nature | combined | 0.6672 | 0.0729 | 2.0771 | 0.0351 | 68182 | 2394 | 65788 |
| nature | unseenCell_seenDrugCombo | 0.6769 | 0.1190 | 1.9930 | 0.0597 | 16652 | 994 | 15658 |
| nature | unseenCell_unseenDrugCombo | 0.6350 | 0.0444 | 1.6348 | 0.0272 | 51530 | 1400 | 50130 |

## Delta vs Untuned Cell + Cell-type Selected v1

Baseline prefix: `20260608_cell_celltype_llm_clip10_selected_v1`.

| exp | task | split | prev AUPRC | tuned AUPRC | delta AUPRC | prev AUROC | tuned AUROC | delta AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | single unseen drug | mean5 | 0.6644 | 0.6694 | 0.0050 | 0.8959 | 0.8969 | 0.0010 |
| exp02 | single unseen cell type | mean5 | 0.8146 | 0.8159 | 0.0012 | 0.9453 | 0.9458 | 0.0005 |
| exp03 | single unseen cell | mean5 | 0.7683 | 0.7841 | 0.0159 | 0.9318 | 0.9323 | 0.0004 |
| exp04 | single w/o MSE | mean5 | 0.6571 | 0.6571 | 0.0000 | 0.9065 | 0.9065 | 0.0000 |
| exp05 | single w/o graph | mean5 | 0.6126 | 0.6066 | -0.0060 | 0.8438 | 0.8425 | -0.0013 |
| exp06 | double unseen drug pair | mean5 | 0.7566 | 0.7677 | 0.0112 | 0.8166 | 0.8258 | 0.0092 |
| exp07 | extra single | mean_extra | 0.5555 | 0.5610 | 0.0055 | 0.7641 | 0.7604 | -0.0036 |
| exp08 | extra double | mean_extra | 0.1016 | 0.0964 | -0.0053 | 0.6433 | 0.6503 | 0.0070 |

## Interpretation

- Tuning improved the main target exp01 from AUPRC `0.6644` to `0.6694`, while keeping a gap over exp04/no-MSE (`+0.0123`) and exp05/no-graph (`+0.0628`).
- The strongest gains versus the untuned cell+cell-type selected run are exp03 AUPRC `+0.0159` and exp06 AUPRC `+0.0112`.
- exp05 no-graph drops by AUPRC `-0.0060`, which is acceptable for the diagnostic ablation and increases the graph-enabled gap relative to exp01.
- exp07 extra single improves by AUPRC `+0.0055`; exp08 extra double loses AUPRC `-0.0053` but improves AUROC `+0.0070`.
- Full detail, including all folds and extra subsets, is retained in `logs/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.md` and `outputs/20260608_cell_celltype_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.csv`.
