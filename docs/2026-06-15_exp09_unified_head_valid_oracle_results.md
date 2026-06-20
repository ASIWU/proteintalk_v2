# 2026-06-15 Exp09 Unified Head Valid/Oracle Results

Exp09 follows exp08's all-data training setup, but uses `task_head=unified`: single-drug sensitivity (`response_label`) and double-drug synergy (`synergy_label`) are mapped into one `task_label` with policy `unified_synergy_first_else_response`, and one head/loss predicts both tasks.

## Artifacts

- Prefix: `20260615_1101_exp09_selectedref_v1`
- Training checkpoint dir: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra`
- Valid output root: `outputs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid`
- Oracle output roots: `outputs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_oracle_epoch*`
- Runtime summary: `logs/20260615_1101_exp09_selectedref_v1_runtime_summary.tsv`
- Detailed markdown report: `logs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.md`
- Detailed CSV: `outputs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.csv`
- Detailed JSON: `outputs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.json`

## Training Config

| key | value |
|---|---|
| `task_head` | `unified` |
| `task_label_policy` | `unified_synergy_first_else_response` |
| `dataset_group` | `ptv3` |
| `task_name` | `ptv3_main_doubledrug` |
| `split_strategy` | `all_train_subset_test` |
| `model_type` | `fast_delta` |
| `max_epochs` | `50` |
| `learning_rate` | `5e-05` |
| `mse_weight` | `0.5` |
| `mse_target_mode` | `all` |
| `use_ddi` | `True` |
| `pair_fusion_mode` | `dual` |
| `pair_type_features` | `True` |
| `cell_llm_mode` | `frozen` |
| `cell_llm_embedding_path` | `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz` |
| `cell_type_llm_mode` | `frozen` |
| `cell_type_llm_embedding_path` | `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz` |

## Completion Check

- Training reached `max_epochs=50`; checkpoints `epoch=0.ckpt` through `epoch=49.ckpt` were used for oracle.
- Runtime summary rows: `460` total = `1` train + `9` valid infer + `450` oracle infer; nonzero statuses: `0`.
- Valid manifests: `9` tasks.
- Oracle candidate roots: `50` epoch roots, `450` task manifests.
- Report rows: `102` metric rows after selecting oracle epochs.
- The first report subprocess was stopped after the model outputs had completed because the old Python per-group collapsed metric loop did not finish after about 50 CPU-minutes. The report was regenerated from the same prediction files after vectorizing `collapsed_frame`; no training or inference output was rerun.

## Reference Epoch Policy

`valid` uses reference epochs from the selected exp01/exp06 5-fold runs; `oracle` evaluates every exp09 checkpoint on the same exp07/exp08 extra datasets and selects the best epoch per exp by mean-extra original AUPRC, with AUROC and n-AUPRC as tie-break fields in the reporter.

| target | reference | raw mean epoch | selected epoch | folds |
|---|---|---:|---:|---|
| exp07 extra single valid | exp01 selected 5-fold | 5.400 | 5 | fold0=4, fold1=4, fold2=3, fold3=13, fold4=3 |
| exp08 extra double valid | exp06 selected 5-fold | 2.200 | 2 | fold0=0, fold1=1, fold2=1, fold3=8, fold4=1 |

Oracle selected epochs:

| exp | best epoch | AUROC | AUPRC | baseline | n-AUPRC | count | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| exp07 | 2 | 0.780482 | 0.596656 | 0.209041 | 2.848712 | 92671 | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt` |
| exp08 | 8 | 0.655564 | 0.101437 | 0.047592 | 2.423031 | 88970 | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=8.ckpt` |

## Test Structure

`valid` and `oracle` use the same exp07/exp08 external datasets. The selected oracle structure table below lists only the winning exp07 and exp08 epochs; the full oracle scan covered epochs `0..49` with the same 9 tasks at each epoch.

| view | task | rows | checkpoint epoch | task head | label policy |
|---|---|---:|---:|---|---|
| valid | `ptv3_extra_doubledrug_guomics` | 5633 | 2 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_doubledrug_nature` | 68182 | 2 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_doubledrug_nc` | 15155 | 2 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_singledrug_mat1_480_faims` | 17140 | 5 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_singledrug_mat1_qe` | 17140 | 5 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_singledrug_mat2_480_faims` | 13882 | 5 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_singledrug_mat2_qe` | 13882 | 5 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_singledrug_mat3_qe` | 18332 | 5 | `unified` | `unified_synergy_first_else_response` |
| valid | `ptv3_extra_singledrug_mat4_qe` | 12295 | 5 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp07 | `ptv3_extra_singledrug_mat1_480_faims` | 17140 | 2 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp07 | `ptv3_extra_singledrug_mat1_qe` | 17140 | 2 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp07 | `ptv3_extra_singledrug_mat2_480_faims` | 13882 | 2 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp07 | `ptv3_extra_singledrug_mat2_qe` | 13882 | 2 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp07 | `ptv3_extra_singledrug_mat3_qe` | 18332 | 2 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp07 | `ptv3_extra_singledrug_mat4_qe` | 12295 | 2 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp08 | `ptv3_extra_doubledrug_guomics` | 5633 | 8 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp08 | `ptv3_extra_doubledrug_nature` | 68182 | 8 | `unified` | `unified_synergy_first_else_response` |
| oracle_exp08 | `ptv3_extra_doubledrug_nc` | 15155 | 8 | `unified` | `unified_synergy_first_else_response` |

Prediction columns:

`feature_row_index`, `sample_id`, `control`, `Cell`, `cell_type`, `pert_id1`, `pert_id2`, `pert_index1`, `pert_index2`, `pred_task_prob`, `pred_response_prob`, `pred_synergy_prob`, `task_label`, `response_label`, `synergy_label`, `pert_time`, `pert_time_norm`, `pert_dose1`, `pert_dose2`, `pert_dose1_norm`, `pert_dose2_norm`, `batch`, `test`, `test_label`

## Detailed Metrics

The following tables are regenerated from `scripts/report_exp09_valid_oracle.py` and match the CSV/JSON artifacts listed above.

## Oracle Selection
| exp | epoch | AUROC | AUPRC | baseline | n-AUPRC | count | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| exp07 | 2 | 0.780482 | 0.596656 | 0.209041 | 2.848712 | 92671 | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt` |
| exp08 | 8 | 0.655564 | 0.101437 | 0.047592 | 2.423031 | 88970 | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=8.ckpt` |

## Mean Summary
| view | epoch | exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | 2 | exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.781223 | 0.599214 | 0.212230 | 2.815909 | 86450 | 18286 | 68164 | 28 | 0 |
| oracle | 2 | exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.781223 | 0.599214 | 0.212230 | 2.815909 | 86450 | 18286 | 68164 | 28 | 86450 |
| oracle | 2 | exp07 | extra single | mean_extra | original | 0.780482 | 0.596656 | 0.209041 | 2.848712 | 92671 | 19345 | 73326 | 0 | 0 |
| oracle | 8 | exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.650482 | 0.099730 | 0.047788 | 2.363663 | 66275 | 2926 | 63349 | 15 | 0 |
| oracle | 8 | exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.650482 | 0.099730 | 0.047788 | 2.363663 | 66275 | 2926 | 63349 | 15 | 66275 |
| oracle | 8 | exp08 | extra double | mean_extra | original | 0.655564 | 0.101437 | 0.047592 | 2.423031 | 88970 | 3696 | 85274 | 0 | 0 |
| valid | 5 | exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.783927 | 0.596785 | 0.212230 | 2.803797 | 86450 | 18286 | 68164 | 28 | 0 |
| valid | 5 | exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.783927 | 0.596785 | 0.212230 | 2.803797 | 86450 | 18286 | 68164 | 28 | 86450 |
| valid | 5 | exp07 | extra single | mean_extra | original | 0.783579 | 0.592539 | 0.209041 | 2.828677 | 92671 | 19345 | 73326 | 0 | 0 |
| valid | 2 | exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.644968 | 0.084808 | 0.047788 | 1.955118 | 66275 | 2926 | 63349 | 15 | 0 |
| valid | 2 | exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.644968 | 0.084808 | 0.047788 | 1.955118 | 66275 | 2926 | 63349 | 15 | 66275 |
| valid | 2 | exp08 | extra double | mean_extra | original | 0.648752 | 0.085506 | 0.047592 | 1.985561 | 88970 | 3696 | 85274 | 0 | 0 |

## Subset Detail
| view | epoch | exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-avgtime | 0.732821 | 0.509097 | 0.205230 | 2.480621 | 16445 | 3375 | 13070 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.732821 | 0.509097 | 0.205230 | 2.480621 | 16445 | 3375 | 13070 | 0 | 16445 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | original | 0.734584 | 0.510744 | 0.202917 | 2.517005 | 17140 | 3478 | 13662 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-avgtime | 0.733559 | 0.508347 | 0.205230 | 2.476967 | 16445 | 3375 | 13070 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-bylasttime | 0.733559 | 0.508347 | 0.205230 | 2.476967 | 16445 | 3375 | 13070 | 0 | 16445 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat1_qe | extra | original | 0.735535 | 0.510156 | 0.202917 | 2.514111 | 17140 | 3478 | 13662 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-avgtime | 0.830050 | 0.695120 | 0.221938 | 3.132043 | 12125 | 2691 | 9434 | 9 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.830050 | 0.695120 | 0.221938 | 3.132043 | 12125 | 2691 | 9434 | 9 | 12125 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | original | 0.827275 | 0.688985 | 0.215819 | 3.192422 | 13882 | 2996 | 10886 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-avgtime | 0.831362 | 0.699001 | 0.221938 | 3.149530 | 12125 | 2691 | 9434 | 9 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-bylasttime | 0.831362 | 0.699001 | 0.221938 | 3.149530 | 12125 | 2691 | 9434 | 9 | 12125 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat2_qe | extra | original | 0.828739 | 0.692599 | 0.215819 | 3.209167 | 13882 | 2996 | 10886 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-avgtime | 0.735850 | 0.514328 | 0.211298 | 2.434142 | 18287 | 3864 | 14423 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-bylasttime | 0.735850 | 0.514328 | 0.211298 | 2.434142 | 18287 | 3864 | 14423 | 0 | 18287 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat3_qe | extra | original | 0.735866 | 0.513986 | 0.210834 | 2.437877 | 18332 | 3865 | 14467 | 0 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-avgtime | 0.823694 | 0.669393 | 0.207747 | 3.222149 | 11023 | 2290 | 8733 | 10 | 0 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-bylasttime | 0.823694 | 0.669393 | 0.207747 | 3.222149 | 11023 | 2290 | 8733 | 10 | 11023 |
| oracle | 2 | exp07 | ptv3_extra_singledrug_mat4_qe | extra | original | 0.820893 | 0.663466 | 0.205937 | 3.221687 | 12295 | 2532 | 9763 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-avgtime | 0.696793 | 0.132494 | 0.034617 | 3.827387 | 5633 | 195 | 5438 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-bylasttime | 0.696793 | 0.132494 | 0.034617 | 3.827387 | 5633 | 195 | 5438 | 0 | 5633 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | combined | original | 0.696793 | 0.132494 | 0.034617 | 3.827387 | 5633 | 195 | 5438 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 42 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | original | 1.000000 | 1.000000 | 0.047619 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.694409 | 0.118312 | 0.034520 | 3.427376 | 5591 | 193 | 5398 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.694409 | 0.118312 | 0.034520 | 3.427376 | 5591 | 193 | 5398 | 0 | 5591 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | original | 0.694409 | 0.118312 | 0.034520 | 3.427376 | 5591 | 193 | 5398 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-avgtime | 0.666581 | 0.068546 | 0.035703 | 1.919917 | 45487 | 1624 | 43863 | 15 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-bylasttime | 0.666581 | 0.068546 | 0.035703 | 1.919917 | 45487 | 1624 | 43863 | 15 | 45487 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | combined | original | 0.681829 | 0.073666 | 0.035112 | 2.098021 | 68182 | 2394 | 65788 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.648048 | 0.098098 | 0.063162 | 1.553116 | 10576 | 668 | 9908 | 4 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.648048 | 0.098098 | 0.063162 | 1.553116 | 10576 | 668 | 9908 | 4 | 10576 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | original | 0.678532 | 0.104476 | 0.059693 | 1.750237 | 16652 | 994 | 15658 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.649851 | 0.050229 | 0.027384 | 1.834253 | 34911 | 956 | 33955 | 11 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.649851 | 0.050229 | 0.027384 | 1.834253 | 34911 | 956 | 33955 | 11 | 34911 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | original | 0.657962 | 0.053062 | 0.027169 | 1.953064 | 51530 | 1400 | 50130 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-avgtime | 0.588071 | 0.098150 | 0.073045 | 1.343684 | 15155 | 1107 | 14048 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-bylasttime | 0.588071 | 0.098150 | 0.073045 | 1.343684 | 15155 | 1107 | 14048 | 0 | 15155 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | combined | original | 0.588071 | 0.098150 | 0.073045 | 1.343684 | 15155 | 1107 | 14048 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.642335 | 0.173448 | 0.091881 | 1.887734 | 2057 | 189 | 1868 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.642335 | 0.173448 | 0.091881 | 1.887734 | 2057 | 189 | 1868 | 0 | 2057 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | original | 0.642335 | 0.173448 | 0.091881 | 1.887734 | 2057 | 189 | 1868 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.574796 | 0.085460 | 0.070087 | 1.219336 | 13098 | 918 | 12180 | 0 | 0 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.574796 | 0.085460 | 0.070087 | 1.219336 | 13098 | 918 | 12180 | 0 | 13098 |
| oracle | 8 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | original | 0.574796 | 0.085460 | 0.070087 | 1.219336 | 13098 | 918 | 12180 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-avgtime | 0.738712 | 0.505159 | 0.205230 | 2.461433 | 16445 | 3375 | 13070 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.738712 | 0.505159 | 0.205230 | 2.461433 | 16445 | 3375 | 13070 | 0 | 16445 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | original | 0.740760 | 0.506824 | 0.202917 | 2.497687 | 17140 | 3478 | 13662 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-avgtime | 0.737481 | 0.499608 | 0.205230 | 2.434385 | 16445 | 3375 | 13070 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-bylasttime | 0.737481 | 0.499608 | 0.205230 | 2.434385 | 16445 | 3375 | 13070 | 0 | 16445 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat1_qe | extra | original | 0.739778 | 0.501427 | 0.202917 | 2.471092 | 17140 | 3478 | 13662 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-avgtime | 0.830910 | 0.699385 | 0.221938 | 3.151260 | 12125 | 2691 | 9434 | 9 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.830910 | 0.699385 | 0.221938 | 3.151260 | 12125 | 2691 | 9434 | 9 | 12125 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | original | 0.828677 | 0.689903 | 0.215819 | 3.196674 | 13882 | 2996 | 10886 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-avgtime | 0.832200 | 0.702240 | 0.221938 | 3.164124 | 12125 | 2691 | 9434 | 9 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-bylasttime | 0.832200 | 0.702240 | 0.221938 | 3.164124 | 12125 | 2691 | 9434 | 9 | 12125 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat2_qe | extra | original | 0.830126 | 0.692186 | 0.215819 | 3.207253 | 13882 | 2996 | 10886 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-avgtime | 0.739993 | 0.507356 | 0.211298 | 2.401141 | 18287 | 3864 | 14423 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-bylasttime | 0.739993 | 0.507356 | 0.211298 | 2.401141 | 18287 | 3864 | 14423 | 0 | 18287 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat3_qe | extra | original | 0.740101 | 0.507081 | 0.210834 | 2.405124 | 18332 | 3865 | 14467 | 0 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-avgtime | 0.824264 | 0.666961 | 0.207747 | 3.210443 | 11023 | 2290 | 8733 | 10 | 0 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-bylasttime | 0.824264 | 0.666961 | 0.207747 | 3.210443 | 11023 | 2290 | 8733 | 10 | 11023 |
| valid | 5 | exp07 | ptv3_extra_singledrug_mat4_qe | extra | original | 0.822031 | 0.657812 | 0.205937 | 3.194235 | 12295 | 2532 | 9763 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-avgtime | 0.716144 | 0.093393 | 0.034617 | 2.697860 | 5633 | 195 | 5438 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-bylasttime | 0.716144 | 0.093393 | 0.034617 | 2.697860 | 5633 | 195 | 5438 | 0 | 5633 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | combined | original | 0.716144 | 0.093393 | 0.034617 | 2.697860 | 5633 | 195 | 5438 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.975000 | 0.750000 | 0.047619 | 15.750000 | 42 | 2 | 40 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.975000 | 0.750000 | 0.047619 | 15.750000 | 42 | 2 | 40 | 0 | 42 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | original | 0.975000 | 0.750000 | 0.047619 | 15.750000 | 42 | 2 | 40 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.715036 | 0.084821 | 0.034520 | 2.457162 | 5591 | 193 | 5398 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.715036 | 0.084821 | 0.034520 | 2.457162 | 5591 | 193 | 5398 | 0 | 5591 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | original | 0.715036 | 0.084821 | 0.034520 | 2.457162 | 5591 | 193 | 5398 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-avgtime | 0.644255 | 0.067249 | 0.035703 | 1.883599 | 45487 | 1624 | 43863 | 15 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-bylasttime | 0.644255 | 0.067249 | 0.035703 | 1.883599 | 45487 | 1624 | 43863 | 15 | 45487 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | combined | original | 0.655606 | 0.069343 | 0.035112 | 1.974927 | 68182 | 2394 | 65788 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.633832 | 0.106548 | 0.063162 | 1.686904 | 10576 | 668 | 9908 | 4 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.633832 | 0.106548 | 0.063162 | 1.686904 | 10576 | 668 | 9908 | 4 | 10576 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | original | 0.661646 | 0.107198 | 0.059693 | 1.795834 | 16652 | 994 | 15658 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.625668 | 0.045399 | 0.027384 | 1.657887 | 34911 | 956 | 33955 | 11 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.625668 | 0.045399 | 0.027384 | 1.657887 | 34911 | 956 | 33955 | 11 | 34911 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | original | 0.627137 | 0.047022 | 0.027169 | 1.730742 | 51530 | 1400 | 50130 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-avgtime | 0.574506 | 0.093782 | 0.073045 | 1.283895 | 15155 | 1107 | 14048 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-bylasttime | 0.574506 | 0.093782 | 0.073045 | 1.283895 | 15155 | 1107 | 14048 | 0 | 15155 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | combined | original | 0.574506 | 0.093782 | 0.073045 | 1.283895 | 15155 | 1107 | 14048 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.652680 | 0.171501 | 0.091881 | 1.866544 | 2057 | 189 | 1868 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.652680 | 0.171501 | 0.091881 | 1.866544 | 2057 | 189 | 1868 | 0 | 2057 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | original | 0.652680 | 0.171501 | 0.091881 | 1.866544 | 2057 | 189 | 1868 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.556850 | 0.080477 | 0.070087 | 1.148238 | 13098 | 918 | 12180 | 0 | 0 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.556850 | 0.080477 | 0.070087 | 1.148238 | 13098 | 918 | 12180 | 0 | 13098 |
| valid | 2 | exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | original | 0.556850 | 0.080477 | 0.070087 | 1.148238 | 13098 | 918 | 12180 | 0 | 0 |
