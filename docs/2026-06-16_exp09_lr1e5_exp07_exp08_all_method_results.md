# 2026-06-16 Exp09 LR1e-5 Exp07/Exp08 All-Method Results

This document expands the exp09 `LEARNING_RATE=1e-5` external evaluation into the complete requested breakdown: exp07 mat1-4 single-drug results and exp08 guomics/nature/nc double-drug results, each under all three evaluation methods.

## Source Artifacts

- Exp09 run prefix: `20260615_2010_exp09_lr1e5_v1`
- Checkpoint directory: `checkpoints/20260615_2010_exp09_lr1e5_v1_unified_all_single_double_for_extra`
- Source CSV: `outputs/2026-06/2026-06-15/20260615_2010_exp09_lr1e5_v1_unified_all_single_double_for_extra_valid_oracle_eval.csv`
- Source JSON: `outputs/2026-06/2026-06-15/20260615_2010_exp09_lr1e5_v1_unified_all_single_double_for_extra_valid_oracle_eval.json`
- Previous detailed report: `logs/20260615_2138_exp09_lr1e5_exp07_exp08_detailed_results.md`

## Result Semantics

- `valid`: selected-reference epoch evaluation. exp07 uses epoch `5`, derived from exp01 selected 5-fold mean epoch; exp08 uses epoch `2`, derived from exp06 selected 5-fold mean epoch.
- `oracle`: all 50 exp09 checkpoints were evaluated and the best epoch was selected by mean-extra original AUPRC for each external target. exp07 selected epoch `6`; exp08 selected epoch `18`.
- Evaluation methods: `original`, `cell-drug-dose-avgtime`, and `cell-drug-dose-bylasttime`. In this output, `avgtime` and `bylasttime` have identical scores whenever all grouped rows lack usable time ordering; `missing_time` records that condition.
- Metrics: `baseline` is the positive-label prevalence for that row group; `n-AUPRC = AUPRC / baseline`.

## Mean Extra Summary

| view | epoch | task/dataset | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 2 | mean_extra_double | mean_extra | original | 0.624993 | 0.066003 | 0.047592 | 1.491196 | 88970 | 3696 | 85274 | 0 | 0 |
| valid | 2 | mean_extra_double | mean_extra | cell-drug-dose-avgtime | 0.622186 | 0.065540 | 0.047788 | 1.469636 | 66275 | 2926 | 63349 | 15 | 0 |
| valid | 2 | mean_extra_double | mean_extra | cell-drug-dose-bylasttime | 0.622186 | 0.065540 | 0.047788 | 1.469636 | 66275 | 2926 | 63349 | 15 | 66275 |
| valid | 5 | mean_extra_single | mean_extra | original | 0.775946 | 0.589831 | 0.209041 | 2.815143 | 92671 | 19345 | 73326 | 0 | 0 |
| valid | 5 | mean_extra_single | mean_extra | cell-drug-dose-avgtime | 0.776064 | 0.593307 | 0.212230 | 2.786729 | 86450 | 18286 | 68164 | 28 | 0 |
| valid | 5 | mean_extra_single | mean_extra | cell-drug-dose-bylasttime | 0.776064 | 0.593307 | 0.212230 | 2.786729 | 86450 | 18286 | 68164 | 28 | 86450 |
| oracle | 6 | mean_extra_single | mean_extra | original | 0.776271 | 0.592182 | 0.209041 | 2.826417 | 92671 | 19345 | 73326 | 0 | 0 |
| oracle | 6 | mean_extra_single | mean_extra | cell-drug-dose-avgtime | 0.776753 | 0.595954 | 0.212230 | 2.799220 | 86450 | 18286 | 68164 | 28 | 0 |
| oracle | 6 | mean_extra_single | mean_extra | cell-drug-dose-bylasttime | 0.776753 | 0.595954 | 0.212230 | 2.799220 | 86450 | 18286 | 68164 | 28 | 86450 |
| oracle | 18 | mean_extra_double | mean_extra | original | 0.653835 | 0.091182 | 0.047592 | 2.149049 | 88970 | 3696 | 85274 | 0 | 0 |
| oracle | 18 | mean_extra_double | mean_extra | cell-drug-dose-avgtime | 0.647377 | 0.089464 | 0.047788 | 2.091074 | 66275 | 2926 | 63349 | 15 | 0 |
| oracle | 18 | mean_extra_double | mean_extra | cell-drug-dose-bylasttime | 0.647377 | 0.089464 | 0.047788 | 2.091074 | 66275 | 2926 | 63349 | 15 | 66275 |

## Exp07 Single-Drug Mat1-4

Rows below include every mat-level exp07 dataset available in the source summary. `mat1` and `mat2` each have both `480_faims` and `qe` variants; `mat3` and `mat4` have `qe` variants.

### Exp07 Valid

| view | epoch | task/dataset | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 5 | mat1_480_faims | extra | original | 0.721544 | 0.492334 | 0.202917 | 2.426281 | 17140 | 3478 | 13662 | 0 | 0 |
| valid | 5 | mat1_480_faims | extra | cell-drug-dose-avgtime | 0.719647 | 0.490957 | 0.205230 | 2.392233 | 16445 | 3375 | 13070 | 0 | 0 |
| valid | 5 | mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.719647 | 0.490957 | 0.205230 | 2.392233 | 16445 | 3375 | 13070 | 0 | 16445 |
| valid | 5 | mat1_qe | extra | original | 0.723494 | 0.493534 | 0.202917 | 2.432195 | 17140 | 3478 | 13662 | 0 | 0 |
| valid | 5 | mat1_qe | extra | cell-drug-dose-avgtime | 0.721503 | 0.492166 | 0.205230 | 2.398122 | 16445 | 3375 | 13070 | 0 | 0 |
| valid | 5 | mat1_qe | extra | cell-drug-dose-bylasttime | 0.721503 | 0.492166 | 0.205230 | 2.398122 | 16445 | 3375 | 13070 | 0 | 16445 |
| valid | 5 | mat2_480_faims | extra | original | 0.830276 | 0.695674 | 0.215819 | 3.223412 | 13882 | 2996 | 10886 | 0 | 0 |
| valid | 5 | mat2_480_faims | extra | cell-drug-dose-avgtime | 0.831740 | 0.703136 | 0.221938 | 3.168161 | 12125 | 2691 | 9434 | 9 | 0 |
| valid | 5 | mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.831740 | 0.703136 | 0.221938 | 3.168161 | 12125 | 2691 | 9434 | 9 | 12125 |
| valid | 5 | mat2_qe | extra | original | 0.831779 | 0.697071 | 0.215819 | 3.229886 | 13882 | 2996 | 10886 | 0 | 0 |
| valid | 5 | mat2_qe | extra | cell-drug-dose-avgtime | 0.832979 | 0.704632 | 0.221938 | 3.174904 | 12125 | 2691 | 9434 | 9 | 0 |
| valid | 5 | mat2_qe | extra | cell-drug-dose-bylasttime | 0.832979 | 0.704632 | 0.221938 | 3.174904 | 12125 | 2691 | 9434 | 9 | 12125 |
| valid | 5 | mat3_qe | extra | original | 0.723654 | 0.492259 | 0.210834 | 2.334821 | 18332 | 3865 | 14467 | 0 | 0 |
| valid | 5 | mat3_qe | extra | cell-drug-dose-avgtime | 0.723584 | 0.492559 | 0.211298 | 2.331115 | 18287 | 3864 | 14423 | 0 | 0 |
| valid | 5 | mat3_qe | extra | cell-drug-dose-bylasttime | 0.723584 | 0.492559 | 0.211298 | 2.331115 | 18287 | 3864 | 14423 | 0 | 18287 |
| valid | 5 | mat4_qe | extra | original | 0.824929 | 0.668115 | 0.205937 | 3.244261 | 12295 | 2532 | 9763 | 0 | 0 |
| valid | 5 | mat4_qe | extra | cell-drug-dose-avgtime | 0.826931 | 0.676392 | 0.207747 | 3.255837 | 11023 | 2290 | 8733 | 10 | 0 |
| valid | 5 | mat4_qe | extra | cell-drug-dose-bylasttime | 0.826931 | 0.676392 | 0.207747 | 3.255837 | 11023 | 2290 | 8733 | 10 | 11023 |

### Exp07 Oracle

| view | epoch | task/dataset | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 6 | mat1_480_faims | extra | original | 0.721431 | 0.494596 | 0.202917 | 2.437428 | 17140 | 3478 | 13662 | 0 | 0 |
| oracle | 6 | mat1_480_faims | extra | cell-drug-dose-avgtime | 0.719548 | 0.493196 | 0.205230 | 2.403141 | 16445 | 3375 | 13070 | 0 | 0 |
| oracle | 6 | mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.719548 | 0.493196 | 0.205230 | 2.403141 | 16445 | 3375 | 13070 | 0 | 16445 |
| oracle | 6 | mat1_qe | extra | original | 0.723467 | 0.495666 | 0.202917 | 2.442703 | 17140 | 3478 | 13662 | 0 | 0 |
| oracle | 6 | mat1_qe | extra | cell-drug-dose-avgtime | 0.721525 | 0.494307 | 0.205230 | 2.408557 | 16445 | 3375 | 13070 | 0 | 0 |
| oracle | 6 | mat1_qe | extra | cell-drug-dose-bylasttime | 0.721525 | 0.494307 | 0.205230 | 2.408557 | 16445 | 3375 | 13070 | 0 | 16445 |
| oracle | 6 | mat2_480_faims | extra | original | 0.830957 | 0.697127 | 0.215819 | 3.230144 | 13882 | 2996 | 10886 | 0 | 0 |
| oracle | 6 | mat2_480_faims | extra | cell-drug-dose-avgtime | 0.833085 | 0.705154 | 0.221938 | 3.177254 | 12125 | 2691 | 9434 | 9 | 0 |
| oracle | 6 | mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.833085 | 0.705154 | 0.221938 | 3.177254 | 12125 | 2691 | 9434 | 9 | 12125 |
| oracle | 6 | mat2_qe | extra | original | 0.832551 | 0.699310 | 0.215819 | 3.240261 | 13882 | 2996 | 10886 | 0 | 0 |
| oracle | 6 | mat2_qe | extra | cell-drug-dose-avgtime | 0.834519 | 0.707499 | 0.221938 | 3.187819 | 12125 | 2691 | 9434 | 9 | 0 |
| oracle | 6 | mat2_qe | extra | cell-drug-dose-bylasttime | 0.834519 | 0.707499 | 0.221938 | 3.187819 | 12125 | 2691 | 9434 | 9 | 12125 |
| oracle | 6 | mat3_qe | extra | original | 0.724052 | 0.495379 | 0.210834 | 2.349623 | 18332 | 3865 | 14467 | 0 | 0 |
| oracle | 6 | mat3_qe | extra | cell-drug-dose-avgtime | 0.724001 | 0.495706 | 0.211298 | 2.346010 | 18287 | 3864 | 14423 | 0 | 0 |
| oracle | 6 | mat3_qe | extra | cell-drug-dose-bylasttime | 0.724001 | 0.495706 | 0.211298 | 2.346010 | 18287 | 3864 | 14423 | 0 | 18287 |
| oracle | 6 | mat4_qe | extra | original | 0.825167 | 0.671014 | 0.205937 | 3.258341 | 12295 | 2532 | 9763 | 0 | 0 |
| oracle | 6 | mat4_qe | extra | cell-drug-dose-avgtime | 0.827836 | 0.679861 | 0.207747 | 3.272536 | 11023 | 2290 | 8733 | 10 | 0 |
| oracle | 6 | mat4_qe | extra | cell-drug-dose-bylasttime | 0.827836 | 0.679861 | 0.207747 | 3.272536 | 11023 | 2290 | 8733 | 10 | 11023 |

## Exp08 Double-Drug Guomics/Nature/NC

Rows below include `combined`, `unseenCell_seenDrugCombo`, and `unseenCell_unseenDrugCombo` for each exp08 dataset, under all three evaluation methods.

### Exp08 Valid

| view | epoch | task/dataset | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 2 | guomics | combined | original | 0.704782 | 0.062658 | 0.034617 | 1.810006 | 5633 | 195 | 5438 | 0 | 0 |
| valid | 2 | guomics | combined | cell-drug-dose-avgtime | 0.704782 | 0.062658 | 0.034617 | 1.810006 | 5633 | 195 | 5438 | 0 | 0 |
| valid | 2 | guomics | combined | cell-drug-dose-bylasttime | 0.704782 | 0.062658 | 0.034617 | 1.810006 | 5633 | 195 | 5438 | 0 | 5633 |
| valid | 2 | guomics | unseenCell_seenDrugCombo | original | 0.900000 | 0.266667 | 0.047619 | 5.600000 | 42 | 2 | 40 | 0 | 0 |
| valid | 2 | guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.900000 | 0.266667 | 0.047619 | 5.600000 | 42 | 2 | 40 | 0 | 0 |
| valid | 2 | guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.900000 | 0.266667 | 0.047619 | 5.600000 | 42 | 2 | 40 | 0 | 42 |
| valid | 2 | guomics | unseenCell_unseenDrugCombo | original | 0.703604 | 0.061007 | 0.034520 | 1.767314 | 5591 | 193 | 5398 | 0 | 0 |
| valid | 2 | guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.703604 | 0.061007 | 0.034520 | 1.767314 | 5591 | 193 | 5398 | 0 | 0 |
| valid | 2 | guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.703604 | 0.061007 | 0.034520 | 1.767314 | 5591 | 193 | 5398 | 0 | 5591 |
| valid | 2 | nature | combined | original | 0.622611 | 0.054807 | 0.035112 | 1.560931 | 68182 | 2394 | 65788 | 0 | 0 |
| valid | 2 | nature | combined | cell-drug-dose-avgtime | 0.614190 | 0.053420 | 0.035703 | 1.496252 | 45487 | 1624 | 43863 | 15 | 0 |
| valid | 2 | nature | combined | cell-drug-dose-bylasttime | 0.614190 | 0.053420 | 0.035703 | 1.496252 | 45487 | 1624 | 43863 | 15 | 45487 |
| valid | 2 | nature | unseenCell_seenDrugCombo | original | 0.626647 | 0.087605 | 0.059693 | 1.467605 | 16652 | 994 | 15658 | 0 | 0 |
| valid | 2 | nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.605370 | 0.087883 | 0.063162 | 1.391393 | 10576 | 668 | 9908 | 4 | 0 |
| valid | 2 | nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.605370 | 0.087883 | 0.063162 | 1.391393 | 10576 | 668 | 9908 | 4 | 10576 |
| valid | 2 | nature | unseenCell_unseenDrugCombo | original | 0.595711 | 0.040206 | 0.027169 | 1.479885 | 51530 | 1400 | 50130 | 0 | 0 |
| valid | 2 | nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.595261 | 0.038902 | 0.027384 | 1.420632 | 34911 | 956 | 33955 | 11 | 0 |
| valid | 2 | nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.595261 | 0.038902 | 0.027384 | 1.420632 | 34911 | 956 | 33955 | 11 | 34911 |
| valid | 2 | nc | combined | original | 0.547585 | 0.080543 | 0.073045 | 1.102650 | 15155 | 1107 | 14048 | 0 | 0 |
| valid | 2 | nc | combined | cell-drug-dose-avgtime | 0.547585 | 0.080543 | 0.073045 | 1.102650 | 15155 | 1107 | 14048 | 0 | 0 |
| valid | 2 | nc | combined | cell-drug-dose-bylasttime | 0.547585 | 0.080543 | 0.073045 | 1.102650 | 15155 | 1107 | 14048 | 0 | 15155 |
| valid | 2 | nc | unseenCell_seenDrugCombo | original | 0.582390 | 0.124422 | 0.091881 | 1.354159 | 2057 | 189 | 1868 | 0 | 0 |
| valid | 2 | nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.582390 | 0.124422 | 0.091881 | 1.354159 | 2057 | 189 | 1868 | 0 | 0 |
| valid | 2 | nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.582390 | 0.124422 | 0.091881 | 1.354159 | 2057 | 189 | 1868 | 0 | 2057 |
| valid | 2 | nc | unseenCell_unseenDrugCombo | original | 0.538798 | 0.074003 | 0.070087 | 1.055868 | 13098 | 918 | 12180 | 0 | 0 |
| valid | 2 | nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.538798 | 0.074003 | 0.070087 | 1.055868 | 13098 | 918 | 12180 | 0 | 0 |
| valid | 2 | nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.538798 | 0.074003 | 0.070087 | 1.055868 | 13098 | 918 | 12180 | 0 | 13098 |

### Exp08 Oracle

| view | epoch | task/dataset | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 18 | guomics | combined | original | 0.745047 | 0.116683 | 0.034617 | 3.370648 | 5633 | 195 | 5438 | 0 | 0 |
| oracle | 18 | guomics | combined | cell-drug-dose-avgtime | 0.745047 | 0.116683 | 0.034617 | 3.370648 | 5633 | 195 | 5438 | 0 | 0 |
| oracle | 18 | guomics | combined | cell-drug-dose-bylasttime | 0.745047 | 0.116683 | 0.034617 | 3.370648 | 5633 | 195 | 5438 | 0 | 5633 |
| oracle | 18 | guomics | unseenCell_seenDrugCombo | original | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| oracle | 18 | guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| oracle | 18 | guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 42 |
| oracle | 18 | guomics | unseenCell_unseenDrugCombo | original | 0.743567 | 0.104003 | 0.034520 | 3.012842 | 5591 | 193 | 5398 | 0 | 0 |
| oracle | 18 | guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.743567 | 0.104003 | 0.034520 | 3.012842 | 5591 | 193 | 5398 | 0 | 0 |
| oracle | 18 | guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.743567 | 0.104003 | 0.034520 | 3.012842 | 5591 | 193 | 5398 | 0 | 5591 |
| oracle | 18 | nature | combined | original | 0.636358 | 0.062815 | 0.035112 | 1.788981 | 68182 | 2394 | 65788 | 0 | 0 |
| oracle | 18 | nature | combined | cell-drug-dose-avgtime | 0.616984 | 0.057662 | 0.035703 | 1.615057 | 45487 | 1624 | 43863 | 15 | 0 |
| oracle | 18 | nature | combined | cell-drug-dose-bylasttime | 0.616984 | 0.057662 | 0.035703 | 1.615057 | 45487 | 1624 | 43863 | 15 | 45487 |
| oracle | 18 | nature | unseenCell_seenDrugCombo | original | 0.656823 | 0.101821 | 0.059693 | 1.705764 | 16652 | 994 | 15658 | 0 | 0 |
| oracle | 18 | nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.622507 | 0.095214 | 0.063162 | 1.507459 | 10576 | 668 | 9908 | 4 | 0 |
| oracle | 18 | nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.622507 | 0.095214 | 0.063162 | 1.507459 | 10576 | 668 | 9908 | 4 | 10576 |
| oracle | 18 | nature | unseenCell_unseenDrugCombo | original | 0.605121 | 0.041082 | 0.027169 | 1.512122 | 51530 | 1400 | 50130 | 0 | 0 |
| oracle | 18 | nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.595328 | 0.038708 | 0.027384 | 1.413529 | 34911 | 956 | 33955 | 11 | 0 |
| oracle | 18 | nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.595328 | 0.038708 | 0.027384 | 1.413529 | 34911 | 956 | 33955 | 11 | 34911 |
| oracle | 18 | nc | combined | original | 0.580101 | 0.094047 | 0.073045 | 1.287518 | 15155 | 1107 | 14048 | 0 | 0 |
| oracle | 18 | nc | combined | cell-drug-dose-avgtime | 0.580101 | 0.094047 | 0.073045 | 1.287518 | 15155 | 1107 | 14048 | 0 | 0 |
| oracle | 18 | nc | combined | cell-drug-dose-bylasttime | 0.580101 | 0.094047 | 0.073045 | 1.287518 | 15155 | 1107 | 14048 | 0 | 15155 |
| oracle | 18 | nc | unseenCell_seenDrugCombo | original | 0.630502 | 0.147123 | 0.091881 | 1.601226 | 2057 | 189 | 1868 | 0 | 0 |
| oracle | 18 | nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.630502 | 0.147123 | 0.091881 | 1.601226 | 2057 | 189 | 1868 | 0 | 0 |
| oracle | 18 | nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.630502 | 0.147123 | 0.091881 | 1.601226 | 2057 | 189 | 1868 | 0 | 2057 |
| oracle | 18 | nc | unseenCell_unseenDrugCombo | original | 0.566540 | 0.083786 | 0.070087 | 1.195456 | 13098 | 918 | 12180 | 0 | 0 |
| oracle | 18 | nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.566540 | 0.083786 | 0.070087 | 1.195456 | 13098 | 918 | 12180 | 0 | 0 |
| oracle | 18 | nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.566540 | 0.083786 | 0.070087 | 1.195456 | 13098 | 918 | 12180 | 0 | 13098 |

## Key Observations

- exp07 oracle mean-extra original reaches AUROC `0.776271`, AUPRC `0.592182`, n-AUPRC `2.826417` at epoch `6`; valid epoch `5` is very close at AUPRC `0.589831`.
- exp07 mat2 is the strongest mat block: oracle original AUPRC is `0.697127` for `mat2_480_faims` and `0.699310` for `mat2_qe`; grouped dose/time methods further raise AUPRC to `0.705154` and `0.707499`.
- exp08 oracle mean-extra original reaches AUROC `0.653835`, AUPRC `0.091182`, n-AUPRC `2.149049` at epoch `18`; valid epoch `2` is lower at AUPRC `0.066003`.
- exp08 guomics is strongest under oracle, especially `combined` AUPRC `0.116683` and `unseenCell_unseenDrugCombo` AUPRC `0.104003`. The `unseenCell_seenDrugCombo` guomics split has only `42` rows and `2` positives, so its perfect score should be treated as a small-sample result.
- For exp08 nature, grouped dose/time evaluation reduces row count from `68182` original combined rows to `45487` grouped rows because of dose/time grouping and conflicts; its oracle combined original AUPRC is `0.062815`, grouped AUPRC is `0.057662`.
- exp08 nc is unchanged across the three methods in this summary because the grouped calculations preserve the same rows and scores for this dataset/split set.
