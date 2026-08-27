# 2026-06-02 LLM Cell Embedding + Dose Experiment

## 2026-06-05 HKT Corrected Clip10 Tuned Selected v1 Completion

The corrected Cell LLM clip10 parameter search, promoted 5-fold validation, and final selected exp01-exp08 suite completed under the required final prefix:

- Screen prefix: `20260604_cell_llm_clip10_tune_v1`
- Promoted full prefix: `20260604_cell_llm_clip10_tune_v1_full`
- Final selected prefix: `20260604_cell_llm_clip10_tuned_selected_v1`
- Final report: `logs/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.md`
- Final CSV: `outputs/2026-06/2026-06-04/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.csv`
- Final JSON: `outputs/2026-06/2026-06-04/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.json`

Completion and validation:

- Screen manifests: stage1 `135/135`, stage2 `42/42`, stage3 `42/42`, stage4 `42/42`.
- Full promoted manifests: `100/100`.
- Final selected manifests: `32/32` (`exp01`-`exp06` five folds each, plus one all-data run each for `exp07` and `exp08`).
- Validation errors: `0` for screen, full, and selected.
- All final selected manifests use `cell_llm_mode=frozen` and `cell_llm_summary.embedding_rows=74`, with no old `cell_type_llm` manifest key.
- Final `exp05` uses `graph_feature_mode=zero`; final `exp01`/`exp02`/`exp03`/`exp04`/`exp06`/`exp07`/`exp08` use `graph_feature_mode=real`.

Selected configs from the promoted 5-fold report:

| stage | final use | config | key settings |
|---|---|---|---|
| stage1 | exp01, exp04, exp05, exp07 | `mse050_target_pdi` | LR `2e-4`, batch `256`, dropout `0.15`, MSE weight `0.50`, `MSE_TARGET_MODE=pdi` |
| stage2 | exp03 | `covdrop010_drop010` | LR `2e-4`, batch `256`, dropout `0.10`, `COVARIATE_UNK_DROPOUT=0.10` |
| stage3 | exp02 | `covdrop010_lr1e4` | LR `1e-4`, batch `256`, dropout `0.15`, `COVARIATE_UNK_DROPOUT=0.10` |
| stage4 | exp06, exp08 | `drop020_mseinactive010` | LR `2e-4`, batch `256`, dropout `0.20`, `MSE_INACTIVE_LABEL_WEIGHT=0.10`, double fixed settings |

Promotion summary:

- Stage1 promoted `mse050_target_pdi`, `mse100_drop010`, `mse075`.
- Stage2 promoted `covdrop010`, `covdrop010_drop010`, `covdrop010_drop020`, `covunk_cell`, `covdrop010_lr1e4` because rank 3 through rank 5 were within the `0.002` margin.
- Stage3 promoted `covdrop010_lr1e4`, `covdrop010_drop010`, `covdrop010_lr3e4`.
- Stage4 promoted `drop020_lr1e4`, `drop020_mseinactive010`, `drop020`.

Full 5-fold selected winners:

| stage | selected config | full 5-fold AUPRC | n-AUPRC | AUROC |
|---|---|---:|---:|---:|
| stage1 | `mse050_target_pdi` | 0.669226 | 5.639285 | 0.897766 |
| stage2 | `covdrop010_drop010` | 0.776425 | 6.484912 | 0.934077 |
| stage3 | `covdrop010_lr1e4` | 0.822909 | 6.258591 | 0.945678 |
| stage4 | `drop020_mseinactive010` | 0.779638 | 1.933931 | 0.831522 |

Reference epoch policy:

| exp | reference folds | raw mean epoch | selected epoch | all-data max_epochs | checkpoint policy |
|---|---|---:|---:|---:|---|
| exp07 | selected exp01 5-fold | 10.4 | 10 | 11 | fixed reference epoch, `last.ckpt` |
| exp08 | selected exp06 5-fold | 8.8 | 9 | 10 | fixed reference epoch, `last.ckpt` |

### Tuned Clip10 Final Mean Results

Original row/timepoint-level mean metrics:

| exp | task | split | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp01 | single unseen drug | mean5 | 0.897739 | 0.672035 | 0.118838 | 5.664013 | 17986 | 2137 | 15849 |
| exp02 | single unseen cell type | mean5 | 0.945617 | 0.822881 | 0.173980 | 6.257368 | 17986 | 2137 | 15849 |
| exp03 | single unseen cell | mean5 | 0.934035 | 0.777877 | 0.154930 | 6.494667 | 17986 | 2137 | 15849 |
| exp04 | single w/o MSE | mean5 | 0.895744 | 0.662743 | 0.118838 | 5.591658 | 17986 | 2137 | 15849 |
| exp05 | single w/o graph | mean5 | 0.837038 | 0.601639 | 0.118838 | 5.065338 | 17986 | 2137 | 15849 |
| exp06 | double unseen drug pair | mean5 | 0.833379 | 0.789136 | 0.404342 | 1.957651 | 1791 | 723 | 1068 |
| exp07 | extra single | mean_extra | 0.764647 | 0.561044 | 0.209041 | 2.677275 | 92671 | 19345 | 73326 |
| exp08 | extra double | mean_extra | 0.638376 | 0.091853 | 0.047592 | 2.076231 | 88970 | 3696 | 85274 |

Compared with the previous corrected, untuned prefix `20260604_cell_llm_dose_clip10_selected_v1`, the tuned run changes are modest:

| exp | previous AUPRC | tuned AUPRC | delta | previous AUROC | tuned AUROC | delta |
|---|---:|---:|---:|---:|---:|---:|
| exp01 | 0.661912 | 0.672035 | +0.010123 | 0.891681 | 0.897739 | +0.006058 |
| exp02 | 0.811791 | 0.822881 | +0.011091 | 0.945987 | 0.945617 | -0.000369 |
| exp03 | 0.776145 | 0.777877 | +0.001733 | 0.930525 | 0.934035 | +0.003511 |
| exp04 | 0.662743 | 0.662743 | +0.000000 | 0.895744 | 0.895744 | +0.000000 |
| exp05 | 0.608214 | 0.601639 | -0.006575 | 0.844887 | 0.837038 | -0.007849 |
| exp06 | 0.782355 | 0.789136 | +0.006781 | 0.834738 | 0.833379 | -0.001359 |
| exp07 | 0.569953 | 0.561044 | -0.008909 | 0.770714 | 0.764647 | -0.006067 |
| exp08 | 0.094090 | 0.091853 | -0.002237 | 0.639328 | 0.638376 | -0.000951 |

Notes:

- `exp04` is unchanged because it uses the no-MSE path; the tuned stage1 winner changes `MSE_TARGET_MODE` to `pdi`, which has no effective impact when MSE is disabled.
- `exp07`/`exp08` changed because their all-data reference epochs were recomputed from the final selected exp01/exp06 folds.
- The older result tables below are retained as previous corrected baselines and historical context, not as the final tuned result.

The final CSV/JSON contain the full cell-drug-dose grouping table. At the mean level, the strongest grouped metrics were:

- `exp01`: avg-time AUPRC `0.673188`.
- `exp02`: by-last-time AUPRC `0.824977`.
- `exp03`: by-last-time AUPRC `0.784919`.
- `exp06`: avg-time AUPRC `0.792831`.
- `exp07`: grouped extra AUPRC `0.563027`.
- `exp08`: grouped extra AUPRC `0.092616`.

## 2026-06-04 Correction: Old Cell-Type LLM Results Are Invalid

The `20260602_llm_dose_graphallowed_selected_v1`, `20260603_llm_dose_exp01params_wograph_v1`, and `20260603_update0527_clip10` results below are retained only as invalid historical baselines. They used the tissue-level `cell_type` LLM embedding artifact, but the intended LLM input for this experiment is the cell-line `Cell` value aligned by `Cell_index`.

Corrected Cell LLM setup:

- Embedding artifact: `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz`
- Sidecar: `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.json`
- Alignment: `Cell_index`, 74 rows, row 0 `no` zero vector, 4096 dimensions.
- Public interface: `--cell-llm-*` CLI flags and `CELL_LLM_*` environment variables. The old `--cell-type-llm-*` interface was removed.
- Corrected runner: `scripts/run_cell_llm_clip10_selected_suite.sh`
- Corrected prefix: `20260604_cell_llm_dose_clip10_selected_v1`

Corrected exp01-exp08 training completed on 2026-06-04. exp01-exp06 each have 5 folds; exp07 and exp08 each have one all-data training run plus extra-data inference.

Report outputs:

- Markdown: `logs/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.md`
- CSV: `outputs/2026-06/2026-06-04/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.csv`
- JSON: `outputs/2026-06/2026-06-04/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.json`
- Runtime summary: `logs/20260604_cell_llm_dose_clip10_selected_v1_runtime_summary.tsv`

Validation summary:

- Training manifests: 32 total; exp01-exp06 have 5 folds each, exp07/exp08 have 1 all-data run each.
- All corrected manifests use `cell_llm_mode=frozen` and `cell_llm_summary.embedding_rows=74`.
- No corrected manifest contains old `cell_type_llm` naming.
- exp05 manifests use `graph_feature_mode=zero`; all other corrected experiments use `graph_feature_mode=real`.
- Corrected embedding artifact has shape `(74, 4096)`, row 0 zero, finite nonzero rows, and no old cell-type sidecar keys.

Corrected reference epochs:

| exp | source folds | raw mean epoch | selected epoch | all-data max_epochs | fold epochs |
|---|---|---:|---:|---:|---|
| exp07 | corrected exp01 5-fold | 10.8 | 11 | 12 | 11, 14, 9, 9, 11 |
| exp08 | corrected exp06 5-fold | 5.6 | 6 | 7 | 1, 2, 7, 17, 1 |

### Previous Corrected Cell LLM Fold Mean Results

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp01 | single unseen drug | mean5 | original | 0.891681 | 0.661912 | 0.118838 | 5.588333 | 17986 | 2137 | 15849 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-bylasttime | 0.893318 | 0.661510 | 0.118495 | 5.605313 | 9032 | 1070 | 7962 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-avgtime | 0.892772 | 0.664745 | 0.118495 | 5.633421 | 9032 | 1070 | 7962 |
| exp02 | single unseen cell type | mean5 | original | 0.945987 | 0.811791 | 0.173980 | 6.099584 | 17986 | 2137 | 15849 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-bylasttime | 0.946116 | 0.812399 | 0.173634 | 6.123506 | 9032 | 1070 | 7962 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-avgtime | 0.946638 | 0.814642 | 0.173634 | 6.140953 | 9032 | 1070 | 7962 |
| exp03 | single unseen cell | mean5 | original | 0.930525 | 0.776145 | 0.154930 | 6.475614 | 17986 | 2137 | 15849 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-bylasttime | 0.930792 | 0.781390 | 0.154765 | 6.510187 | 9032 | 1070 | 7962 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-avgtime | 0.931654 | 0.783743 | 0.154765 | 6.543263 | 9032 | 1070 | 7962 |
| exp04 | single w/o MSE | mean5 | original | 0.895744 | 0.662743 | 0.118838 | 5.591658 | 17986 | 2137 | 15849 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-bylasttime | 0.895877 | 0.661093 | 0.118495 | 5.599254 | 9032 | 1070 | 7962 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-avgtime | 0.896755 | 0.662937 | 0.118495 | 5.614298 | 9032 | 1070 | 7962 |
| exp05 | single w/o graph | mean5 | original | 0.844887 | 0.608214 | 0.118838 | 5.127502 | 17986 | 2137 | 15849 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-bylasttime | 0.847890 | 0.607898 | 0.118495 | 5.141581 | 9032 | 1070 | 7962 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-avgtime | 0.845515 | 0.610225 | 0.118495 | 5.161525 | 9032 | 1070 | 7962 |
| exp06 | double unseen drug pair | mean5 | original | 0.834738 | 0.782355 | 0.404342 | 1.941130 | 1791 | 723 | 1068 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-bylasttime | 0.831333 | 0.782816 | 0.404659 | 1.940891 | 896 | 362 | 534 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-avgtime | 0.837509 | 0.788619 | 0.404659 | 1.955379 | 896 | 362 | 534 |

### Previous Corrected Cell LLM Extra Mean Results

| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | extra single | mean_extra | original | 0.770714 | 0.569953 | 0.209041 | 2.720985 | 92671 | 19345 | 73326 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.770137 | 0.572016 | 0.212230 | 2.687594 | 86450 | 18286 | 68164 | 28 | 86450 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.770137 | 0.572016 | 0.212230 | 2.687594 | 86450 | 18286 | 68164 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.639328 | 0.094090 | 0.047592 | 2.134296 | 88970 | 3696 | 85274 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.637417 | 0.094491 | 0.047788 | 2.135325 | 66275 | 2926 | 63349 | 15 | 66275 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.637417 | 0.094491 | 0.047788 | 2.135325 | 66275 | 2926 | 63349 | 15 | 0 |

### Corrected Extra Subset Highlights

The full subset table, including all cell-drug-dose grouping modes, is in the Markdown report above. Original row/timepoint-level subset metrics are:

| exp | task | split | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | 0.725465 | 0.494621 | 0.202917 | 2.437549 | 17140 | 3478 | 13662 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | 0.728655 | 0.478140 | 0.202917 | 2.356331 | 17140 | 3478 | 13662 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | 0.817774 | 0.664276 | 0.215819 | 3.077932 | 13882 | 2996 | 10886 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | 0.821920 | 0.665957 | 0.215819 | 3.085717 | 13882 | 2996 | 10886 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | 0.722604 | 0.481093 | 0.210834 | 2.281860 | 18332 | 3865 | 14467 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | 0.807864 | 0.635630 | 0.205937 | 3.086522 | 12295 | 2532 | 9763 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | 0.686824 | 0.103788 | 0.034617 | 2.998149 | 5633 | 195 | 5438 |
| exp08 | ptv3_extra_doubledrug_nature | combined | 0.656060 | 0.064996 | 0.035112 | 1.851122 | 68182 | 2394 | 65788 |
| exp08 | ptv3_extra_doubledrug_nc | combined | 0.575099 | 0.113484 | 0.073045 | 1.553617 | 15155 | 1107 | 14048 |

## 2026-06-04 16:41 HKT Corrected Clip10 Parameter Search Tooling

Added corrected Cell LLM clip10 tuning tooling for the next selected run. No new long training jobs or `20260604_cell_llm_clip10_tuned_selected_v1` final artifacts were launched by this documentation update.

- Runner: `scripts/run_cell_llm_clip10_param_search.sh`
- Tuning report: `scripts/report_cell_llm_clip10_param_search.py`
- Default screen prefix: `20260604_cell_llm_clip10_tune_v1`
- Default promoted full prefix: `20260604_cell_llm_clip10_tune_v1_full`
- Default final selected prefix: `20260604_cell_llm_clip10_tuned_selected_v1`
- Screen folds: `0 2 4`; promoted/full/final folds: `0 1 2 3 4`
- Shared corrected settings: `CELL_LLM_MODE=frozen`, `CELL_LLM_EMBEDDING_PATH=data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz`, dose covariates, `fast_delta`, hidden `512`, expression latent `768`, covariate dim `96`, `bf16-mixed`, graph features enabled except exp05.
- Runner fail-fast checks validate the `(74,4096)` Cell embedding, row 0 zero vector, no old sidecar keys, `Cell_index < 74`, and dose range `0..10`.

Search and selection behavior:

- Stage1 runs each candidate across exp01/exp04/exp05, using the no-MSE script for exp04 and graph-zero script for exp05.
- Stage1 score is `exp01_AUPRC + 0.5*(exp01_AUPRC-exp04_AUPRC) + 0.5*(exp01_AUPRC-exp05_AUPRC)`.
- Stage1 tie-breaks are exp01 AUPRC, exp01-exp05 gap, then exp01-exp04 gap.
- Stages 2, 3, and 4 rank exp03, exp02, and exp06 respectively by mean original fold AUPRC, with n-AUPRC then AUROC tie-breaks.
- Promotion is top 3 by default, extended up to top 5 when the third and following candidate differ by less than `0.002`.
- Final selected mode uses stage1 parameters for exp01/04/05 and exp07, stage2 for exp03, stage3 for exp02, and stage4 for exp06 and exp08. exp07/exp08 use mean-nearest reference epochs from selected exp01/exp06 folds and infer extra data from `last.ckpt`.

## Setup

- Primary graph-enabled selected prefix: `20260602_llm_dose_graphallowed_selected_v1`
- Fresh w/o graph retrain prefix: `20260603_llm_dose_exp01params_wograph_v1`
- update_0527/clip10 exp07-exp08 rerun prefix: `20260603_update0527_clip10`
- Cell-type embedding artifact: `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.npz`
- Cell-type LLM mode: `frozen`
- Dose covariate: enabled with `pert_dose1 pert_dose2`
- Model type: `fast_delta`
- Hidden dim: `512` for all selected and fresh w/o graph runs; hidden size was not tuned.
- Expression latent dim: `768`
- Covariate embedding dim: `96`
- Precision: `bf16-mixed`
- Main graph mode: `GRAPH_FEATURE_MODE=real`
- Fresh exp05 w/o graph mode: `GRAPH_FEATURE_MODE=zero`

The final graph-enabled selected suite contains exp01, exp02, exp03, exp04, exp06, exp07, and exp08. exp05 was rerun separately under the fresh w/o graph prefix using the current exp01/exp04 parameters.

## Selected Parameters

| exp | role | config | LR | batch | dropout | weight decay | MSE weight | graph mode | additional settings |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| exp01 | single unseen drug | `mse050` | 2e-4 | 256 | 0.15 | 1e-4 | 0.50 | real | selected by exp01 AUPRC and exp04 gap |
| exp04 | single w/o MSE | `mse050` | 2e-4 | 256 | 0.15 | 1e-4 | 0.50 | real | graph-enabled no-MSE comparison |
| exp05 | single w/o graph | `mse050` fresh retrain | 2e-4 | 256 | 0.15 | 1e-4 | 0.50 | zero | fresh prefix `20260603_llm_dose_exp01params_wograph_v1` |
| exp02 | single unseen cell type | `covdrop010` | 2e-4 | 256 | 0.15 | 1e-4 | 0.25 | real | `COVARIATE_UNK_DROPOUT=0.10` |
| exp03 | single unseen cell | `covdrop010` | 2e-4 | 256 | 0.15 | 1e-4 | 0.25 | real | `COVARIATE_UNK_DROPOUT=0.10` |
| exp06 | double unseen drug pair | `drop020` | 2e-4 | 256 | 0.20 | 1e-4 | 0.25 | real | `PAIR_FUSION_MODE=dual`, `PAIR_TYPE_FEATURES=1`, `USE_DDI=1`, `GRAPH_PAIR_ADD_SCALE=0.5` |
| exp07 | extra single | exp01 params | 2e-4 | 256 | 0.15 | 1e-4 | 0.50 | real | all-data train to reference epoch 8 |
| exp08 | extra double | exp06 params | 2e-4 | 256 | 0.20 | 1e-4 | 0.25 | real | all-data train to reference epoch 3 |

## 2026-06-03 update_0527 Clip10 Rerun

This rerun updates only exp07 and exp08 on the rebuilt `update_0527` extra data with dose clipped at 10 during standardization. The older `20260602_llm_dose_graphallowed_selected_v1` exp07/exp08 tables below remain the historical selected-suite baseline.

- Base prefix: `20260603_update0527_clip10`
- exp07 run: `20260603_update0527_clip10_exp07_extra_single_all_train_infer_all_single_for_extra`
- exp08 run: `20260603_update0527_clip10_exp08_extra_double_all_train_infer_all_single_double_for_extra`
- exp07 policy: exp01 selected reference epoch 8, applied `max_epochs=9`, checkpoint `last.ckpt`.
- exp08 policy: exp06 selected reference epoch 3, applied `max_epochs=4`, checkpoint `last.ckpt`.
- Shared settings: `GRAPH_FEATURE_MODE=real`, `CELL_TYPE_LLM_MODE=frozen`, `USE_DOSE_COVARIATE=1`, `DOSE_COVARIATE_FIELDS="pert_dose1 pert_dose2"`, logger disabled.
- Dose validation: inference feature rows used by reporting have numeric dose max 10.0; cell-drug-dose grouping uses clipped training-ready dose from `feature_table`.

### update_0527 Extra Mean Results

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | extra single | mean_extra | original | 0.773245 | 0.575615 | 2.747902 | 92671 | 19345 | 73326 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.772859 | 0.577147 | 2.711722 | 86450 | 18286 | 68164 | 28 | 86450 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.772859 | 0.577147 | 2.711722 | 86450 | 18286 | 68164 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.650254 | 0.099832 | 2.287323 | 88970 | 3696 | 85274 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.649842 | 0.100581 | 2.298787 | 66275 | 2926 | 63349 | 15 | 66275 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.649842 | 0.100581 | 2.298787 | 66275 | 2926 | 63349 | 15 | 0 |

### update_0527 Extra Subset Results

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | original | 0.730373 | 0.497622 | 2.452339 | 17140 | 3478 | 13662 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.727614 | 0.495961 | 2.416615 | 16445 | 3375 | 13070 | 0 | 16445 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-avgtime | 0.727614 | 0.495961 | 2.416615 | 16445 | 3375 | 13070 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | original | 0.728081 | 0.478615 | 2.358674 | 17140 | 3478 | 13662 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-bylasttime | 0.725577 | 0.476847 | 2.323480 | 16445 | 3375 | 13070 | 0 | 16445 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-avgtime | 0.725577 | 0.476847 | 2.323480 | 16445 | 3375 | 13070 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | original | 0.818638 | 0.669544 | 3.102339 | 13882 | 2996 | 10886 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.819504 | 0.675959 | 3.045708 | 12125 | 2691 | 9434 | 9 | 12125 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-avgtime | 0.819504 | 0.675959 | 3.045708 | 12125 | 2691 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | original | 0.822672 | 0.671628 | 3.111995 | 13882 | 2996 | 10886 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-bylasttime | 0.823368 | 0.675546 | 3.043848 | 12125 | 2691 | 9434 | 9 | 12125 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-avgtime | 0.823368 | 0.675546 | 3.043848 | 12125 | 2691 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | original | 0.728605 | 0.492527 | 2.336092 | 18332 | 3865 | 14467 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-bylasttime | 0.728574 | 0.492884 | 2.332654 | 18287 | 3864 | 14423 | 0 | 18287 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-avgtime | 0.728574 | 0.492884 | 2.332654 | 18287 | 3864 | 14423 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | original | 0.811102 | 0.643754 | 3.125971 | 12295 | 2532 | 9763 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-bylasttime | 0.812516 | 0.645684 | 3.108025 | 11023 | 2290 | 8733 | 10 | 11023 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-avgtime | 0.812516 | 0.645684 | 3.108025 | 11023 | 2290 | 8733 | 10 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | original | 1.000000 | 1.000000 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 1.000000 | 1.000000 | 21.000000 | 42 | 2 | 40 | 0 | 42 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 1.000000 | 1.000000 | 21.000000 | 42 | 2 | 40 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | original | 0.707527 | 0.108675 | 3.148186 | 5591 | 193 | 5398 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.707527 | 0.108675 | 3.148186 | 5591 | 193 | 5398 | 0 | 5591 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.707527 | 0.108675 | 3.148186 | 5591 | 193 | 5398 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | original | 0.709723 | 0.122690 | 3.544165 | 5633 | 195 | 5438 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-bylasttime | 0.709723 | 0.122690 | 3.544165 | 5633 | 195 | 5438 | 0 | 5633 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-avgtime | 0.709723 | 0.122690 | 3.544165 | 5633 | 195 | 5438 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | original | 0.666253 | 0.101574 | 1.701613 | 16652 | 994 | 15658 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.645071 | 0.103980 | 1.646251 | 10576 | 668 | 9908 | 4 | 10576 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.645071 | 0.103980 | 1.646251 | 10576 | 668 | 9908 | 4 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | original | 0.607788 | 0.039429 | 1.451269 | 51530 | 1400 | 50130 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.619871 | 0.042174 | 1.540090 | 34911 | 956 | 33955 | 11 | 34911 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.619871 | 0.042174 | 1.540090 | 34911 | 956 | 33955 | 11 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | original | 0.644916 | 0.060669 | 1.727862 | 68182 | 2394 | 65788 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-bylasttime | 0.643680 | 0.062917 | 1.762254 | 45487 | 1624 | 43863 | 15 | 45487 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-avgtime | 0.643680 | 0.062917 | 1.762254 | 45487 | 1624 | 43863 | 15 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | original | 0.676034 | 0.217424 | 2.366360 | 2057 | 189 | 1868 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.676034 | 0.217424 | 2.366360 | 2057 | 189 | 1868 | 0 | 2057 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.676034 | 0.217424 | 2.366360 | 2057 | 189 | 1868 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | original | 0.578522 | 0.095542 | 1.363193 | 13098 | 918 | 12180 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.578522 | 0.095542 | 1.363193 | 13098 | 918 | 12180 | 0 | 13098 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.578522 | 0.095542 | 1.363193 | 13098 | 918 | 12180 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | original | 0.596124 | 0.116138 | 1.589941 | 15155 | 1107 | 14048 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-bylasttime | 0.596124 | 0.116138 | 1.589941 | 15155 | 1107 | 14048 | 0 | 15155 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-avgtime | 0.596124 | 0.116138 | 1.589941 | 15155 | 1107 | 14048 | 0 | 0 |

### update_0527 Output Files

- Markdown report: `logs/20260603_update0527_clip10_cell_drug_dose_time_eval.md`
- CSV report: `outputs/2026-06/2026-06-03/20260603_update0527_clip10_cell_drug_dose_time_eval.csv`
- JSON report: `outputs/2026-06/2026-06-03/20260603_update0527_clip10_cell_drug_dose_time_eval.json`
- Runtime summary: `logs/20260603_update0527_clip10_runtime_summary.tsv`
- exp08 test-label report: `outputs/2026-06/2026-06-03/20260603_update0527_clip10_exp08_extra_double_all_train_infer_all_single_double_for_extra/extra_doubledrug_test_label_auprc.csv`
- exp07 checkpoint: `checkpoints/20260603_update0527_clip10_exp07_extra_single_all_train_infer_all_single_for_extra/last.ckpt`
- exp08 checkpoint: `checkpoints/20260603_update0527_clip10_exp08_extra_double_all_train_infer_all_single_double_for_extra/last.ckpt`

## Tuning Summary

The tuning screens used 3 folds unless otherwise noted. The final selected parameters were then rerun as full 5-fold runs. exp07 and exp08 used reference epochs derived from the corresponding exp01 and exp06 5-fold selected runs.

### exp01/exp04 Screen

The primary selection uses exp01 AUPRC and the exp01-exp04 no-MSE gap. `mse050` and `mse050_warmdecay` tied in the screen; `mse050` was kept because it is the simpler constant schedule.

| rank | config | exp01 AUPRC | exp01 n-AUPRC | exp01 AUROC | exp04 AUPRC | gap | score | folds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mse050_warmdecay` | 0.601019 | 4.992900 | 0.875339 | 0.566716 | 0.034303 | 0.610322 | 3 |
| 2 | `mse050` | 0.601019 | 4.992900 | 0.875339 | 0.566716 | 0.034303 | 0.610322 | 3 |
| 3 | `base` | 0.594017 | 4.929110 | 0.878612 | 0.566716 | 0.027301 | 0.596318 | 3 |
| 4 | `mse075_drop010` | 0.600358 | 4.982489 | 0.879950 | 0.582329 | 0.018029 | 0.593387 | 3 |
| 5 | `mse075` | 0.587894 | 4.870044 | 0.867560 | 0.566716 | 0.021178 | 0.584071 | 3 |
| 6 | `mse050_drop020` | 0.588496 | 4.886119 | 0.880126 | 0.571043 | 0.017453 | 0.580949 | 3 |
| 7 | `ctrl_drop010` | 0.587651 | 4.878177 | 0.873154 | 0.580816 | 0.006836 | 0.569487 | 3 |
| 8 | `mse050_drop010` | 0.584346 | 4.860214 | 0.866933 | 0.582329 | 0.002017 | 0.561364 | 3 |
| 9 | `mse050_gl3` | 0.575474 | 4.774400 | 0.865450 | 0.567828 | 0.007646 | 0.558120 | 3 |
| 10 | `mse050_lr1e4` | 0.585587 | 4.853891 | 0.885943 | 0.595592 | -0.010006 | 0.550581 | 3 |
| 11 | `mse050_lr3e4` | 0.572183 | 4.749219 | 0.868875 | 0.596662 | -0.024478 | 0.522705 | 3 |

### exp03 Screen

| rank | config | AUPRC | n-AUPRC | AUROC | folds |
|---:|---|---:|---:|---:|---:|
| 1 | `covdrop010` | 0.772444 | 6.023262 | 0.929304 | 3 |
| 2 | `drop020` | 0.760337 | 5.910394 | 0.924586 | 3 |
| 3 | `drop010` | 0.756555 | 5.860148 | 0.922641 | 3 |
| 4 | `covunk_cell` | 0.754699 | 5.810662 | 0.926418 | 3 |
| 5 | `base` | 0.754609 | 5.873117 | 0.918624 | 3 |
| 6 | `lr1e4` | 0.750654 | 5.864899 | 0.917976 | 3 |
| 7 | `bs128` | 0.748216 | 5.843115 | 0.917640 | 3 |
| 8 | `ctrl_drop010` | 0.728460 | 5.479557 | 0.913763 | 3 |
| 9 | `lr3e4` | 0.714594 | 5.320460 | 0.913094 | 3 |

### exp02 Screen

| rank | config | AUPRC | n-AUPRC | AUROC | folds |
|---:|---|---:|---:|---:|---:|
| 1 | `covdrop010` | 0.833036 | 7.194466 | 0.958323 | 3 |
| 2 | `bs128` | 0.819702 | 7.108959 | 0.950119 | 3 |
| 3 | `covunk_celltype` | 0.809943 | 7.147847 | 0.945169 | 3 |
| 4 | `lr3e4` | 0.797122 | 6.979784 | 0.944597 | 3 |
| 5 | `drop010` | 0.795928 | 7.010898 | 0.940428 | 3 |
| 6 | `base` | 0.794897 | 6.962629 | 0.941101 | 3 |
| 7 | `ctrl_drop010` | 0.794284 | 6.984704 | 0.940420 | 3 |
| 8 | `lr1e4` | 0.789214 | 6.973119 | 0.939187 | 3 |
| 9 | `drop020` | 0.777142 | 6.850687 | 0.935158 | 3 |

### exp06 Screen

| rank | config | AUPRC | n-AUPRC | AUROC | folds |
|---:|---|---:|---:|---:|---:|
| 1 | `drop020` | 0.777211 | 1.847418 | 0.817691 | 3 |
| 2 | `dbl_mse050` | 0.772527 | 1.836270 | 0.816162 | 3 |
| 3 | `lr1e4` | 0.753583 | 1.792662 | 0.816311 | 3 |
| 4 | `dbl_mse010` | 0.742186 | 1.765341 | 0.798652 | 3 |
| 5 | `base` | 0.740761 | 1.759180 | 0.788747 | 3 |
| 6 | `lr3e4` | 0.737163 | 1.755273 | 0.788816 | 3 |
| 7 | `rank005` | 0.711897 | 1.695254 | 0.795543 | 3 |
| 8 | `drop010` | 0.701600 | 1.670604 | 0.776007 | 3 |
| 9 | `bs128` | 0.688482 | 1.640539 | 0.772111 | 3 |

## Reference Epochs for exp07/exp08

| exp | source folds | aggregation | raw epoch | selected epoch | fold epochs |
|---|---|---|---:|---:|---|
| exp07 | exp01 5-fold | mean, nearest rounding | 8.000 | 8 | 7, 7, 12, 3, 11 |
| exp08 | exp06 5-fold | mean, nearest rounding | 2.800 | 3 | 1, 2, 3, 7, 1 |

## Fold Mean Results

Metrics are reported under three evaluation modes:

- `original`: original row/timepoint-level metric.
- `cell-drug-dose-bylasttime`: one datapoint per cell-drug-dose group, selecting the max/last available timepoint.
- `cell-drug-dose-avgtime`: one datapoint per cell-drug-dose group, averaging over timepoints.

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp01 | single unseen drug | mean5 | original | 0.896777 | 0.668358 | 5.644481 | 17986 | 2137 | 15849 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-bylasttime | 0.896356 | 0.666578 | 5.652189 | 9032 | 1070 | 7962 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-avgtime | 0.897524 | 0.670102 | 5.681576 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | original | 0.944767 | 0.813731 | 6.152131 | 17986 | 2137 | 15849 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-bylasttime | 0.944626 | 0.816383 | 6.189566 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-avgtime | 0.945199 | 0.816138 | 6.189919 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | original | 0.934788 | 0.781348 | 6.535777 | 17986 | 2137 | 15849 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-bylasttime | 0.935238 | 0.784053 | 6.545866 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-avgtime | 0.935575 | 0.786530 | 6.582468 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | original | 0.889031 | 0.647067 | 5.464223 | 17986 | 2137 | 15849 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-bylasttime | 0.890058 | 0.647762 | 5.490830 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-avgtime | 0.890330 | 0.648585 | 5.498221 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | original | 0.848554 | 0.592835 | 4.992117 | 17986 | 2137 | 15849 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-bylasttime | 0.850929 | 0.591692 | 5.000361 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-avgtime | 0.849799 | 0.596132 | 5.034897 | 9032 | 1070 | 7962 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | original | 0.829343 | 0.779302 | 1.933956 | 1791 | 723 | 1068 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-bylasttime | 0.828391 | 0.779412 | 1.932101 | 896 | 362 | 534 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-avgtime | 0.831392 | 0.782794 | 1.941174 | 896 | 362 | 534 | 0 | 0 |

## Fresh exp05 W/o Graph Fold Detail

This exp05 result is a fresh retrain with current exp01 parameters, not a reused old result.

| fold | method | AUROC | AUPRC | n-AUPRC | count | pos | neg |
|---|---|---:|---:|---:|---:|---:|---:|
| fold0 | original | 0.812175 | 0.517841 | 4.282876 | 3606 | 436 | 3170 |
| fold0 | cell-drug-dose-bylasttime | 0.811485 | 0.519825 | 4.279101 | 1811 | 220 | 1591 |
| fold0 | cell-drug-dose-avgtime | 0.808694 | 0.515012 | 4.239485 | 1811 | 220 | 1591 |
| fold1 | original | 0.874514 | 0.715863 | 5.964414 | 3591 | 431 | 3160 |
| fold1 | cell-drug-dose-bylasttime | 0.879304 | 0.721448 | 5.997660 | 1804 | 217 | 1587 |
| fold1 | cell-drug-dose-avgtime | 0.878349 | 0.717878 | 5.967982 | 1804 | 217 | 1587 |
| fold2 | original | 0.814533 | 0.449185 | 4.141212 | 3614 | 392 | 3222 |
| fold2 | cell-drug-dose-bylasttime | 0.816791 | 0.448512 | 4.230495 | 1811 | 192 | 1619 |
| fold2 | cell-drug-dose-avgtime | 0.814713 | 0.449933 | 4.243895 | 1811 | 192 | 1619 |
| fold3 | original | 0.883797 | 0.668056 | 5.933325 | 3597 | 405 | 3192 |
| fold3 | cell-drug-dose-bylasttime | 0.886006 | 0.664104 | 5.953925 | 1811 | 202 | 1609 |
| fold3 | cell-drug-dose-avgtime | 0.889034 | 0.670681 | 6.012885 | 1811 | 202 | 1609 |
| fold4 | original | 0.857751 | 0.613228 | 4.638756 | 3578 | 473 | 3105 |
| fold4 | cell-drug-dose-bylasttime | 0.861056 | 0.604573 | 4.540623 | 1795 | 239 | 1556 |
| fold4 | cell-drug-dose-avgtime | 0.858203 | 0.627157 | 4.710237 | 1795 | 239 | 1556 |

## Gap Analysis

| comparison | AUPRC difference | interpretation |
|---|---:|---|
| exp01 graph-enabled minus exp04 no-MSE | 0.021291 | MSE improves the selected graph-enabled single unseen-drug run modestly. |
| exp01 graph-enabled minus fresh exp05 w/o graph | 0.075523 | Graph features provide a larger gain than the MSE/no-MSE gap in this setting. |
| exp04 no-MSE minus fresh exp05 w/o graph | 0.054232 | Even without MSE, graph-enabled is materially above w/o graph. |

## Extra Mean Results

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | extra single | mean_extra | original | 0.768817 | 0.568332 | 2.682490 | 84625 | 17837 | 66788 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.768989 | 0.568828 | 2.683925 | 84474 | 17809 | 66665 | 28 | 84474 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.768989 | 0.568828 | 2.683925 | 84474 | 17809 | 66665 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.606803 | 0.076974 | 1.835971 | 41218 | 2025 | 39193 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.579388 | 0.063464 | 1.630458 | 32568 | 1459 | 31109 | 474 | 32568 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.579388 | 0.063464 | 1.630458 | 32568 | 1459 | 31109 | 474 | 0 |

## Extra Subset Results

exp07 covers the extra single-drug mat subsets. exp08 covers the extra double-drug guomics, nature, and nc subsets.

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | original | 0.722916 | 0.482056 | 2.350022 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.722916 | 0.482056 | 2.350022 | 15834 | 3248 | 12586 | 0 | 15834 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-avgtime | 0.722916 | 0.482056 | 2.350022 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | original | 0.720166 | 0.462153 | 2.252994 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-bylasttime | 0.720166 | 0.462153 | 2.252994 | 15834 | 3248 | 12586 | 0 | 15834 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-avgtime | 0.720166 | 0.462153 | 2.252994 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | original | 0.817154 | 0.672239 | 3.065228 | 12138 | 2662 | 9476 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.817132 | 0.672942 | 3.065907 | 12087 | 2653 | 9434 | 9 | 12087 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-avgtime | 0.817132 | 0.672942 | 3.065907 | 12087 | 2653 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | original | 0.821225 | 0.672864 | 3.068076 | 12138 | 2662 | 9476 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-bylasttime | 0.821658 | 0.673894 | 3.070242 | 12087 | 2653 | 9434 | 9 | 12087 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-avgtime | 0.821658 | 0.673894 | 3.070242 | 12087 | 2653 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | original | 0.720323 | 0.473668 | 2.243965 | 17609 | 3717 | 13892 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-bylasttime | 0.720323 | 0.473668 | 2.243965 | 17609 | 3717 | 13892 | 0 | 17609 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-avgtime | 0.720323 | 0.473668 | 2.243965 | 17609 | 3717 | 13892 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | original | 0.811120 | 0.647011 | 3.114656 | 11072 | 2300 | 8772 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-bylasttime | 0.811743 | 0.648259 | 3.120419 | 11023 | 2290 | 8733 | 10 | 11023 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-avgtime | 0.811743 | 0.648259 | 3.120419 | 11023 | 2290 | 8733 | 10 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | original | 0.663019 | 0.084151 | 2.563320 | 3107 | 102 | 3005 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-bylasttime | 0.663019 | 0.084151 | 2.563320 | 3107 | 102 | 3005 | 0 | 3107 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-avgtime | 0.663019 | 0.084151 | 2.563320 | 3107 | 102 | 3005 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | original | 1.000000 | 1.000000 | 23.000000 | 23 | 1 | 22 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 1.000000 | 1.000000 | 23.000000 | 23 | 1 | 22 | 0 | 23 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 1.000000 | 1.000000 | 23.000000 | 23 | 1 | 22 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | original | 0.660173 | 0.071038 | 2.169111 | 3084 | 101 | 2983 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.660173 | 0.071038 | 2.169111 | 3084 | 101 | 2983 | 0 | 3084 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.660173 | 0.071038 | 2.169111 | 3084 | 101 | 2983 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | original | 0.635654 | 0.064762 | 1.821898 | 22956 | 816 | 22140 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-bylasttime | 0.554990 | 0.026766 | 1.206555 | 15011 | 333 | 14678 | 399 | 15011 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-avgtime | 0.554990 | 0.026766 | 1.206555 | 15011 | 333 | 14678 | 399 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | original | 0.636115 | 0.107282 | 1.710425 | 5341 | 335 | 5006 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.557636 | 0.040006 | 1.168100 | 2949 | 101 | 2848 | 185 | 2949 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.557636 | 0.040006 | 1.168100 | 2949 | 101 | 2848 | 185 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | original | 0.616476 | 0.041475 | 1.518888 | 17615 | 481 | 17134 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.545094 | 0.023859 | 1.240481 | 12062 | 232 | 11830 | 214 | 12062 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.545094 | 0.023859 | 1.240481 | 12062 | 232 | 11830 | 214 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | original | 0.521736 | 0.082008 | 1.122696 | 15155 | 1107 | 14048 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-bylasttime | 0.520154 | 0.079475 | 1.121500 | 14450 | 1024 | 13426 | 75 | 14450 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-avgtime | 0.520154 | 0.079475 | 1.121500 | 14450 | 1024 | 13426 | 75 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | original | 0.611060 | 0.134071 | 1.459174 | 2057 | 189 | 1868 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.630034 | 0.136728 | 1.592527 | 1817 | 156 | 1661 | 28 | 1817 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.630034 | 0.136728 | 1.592527 | 1817 | 156 | 1661 | 28 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | original | 0.499915 | 0.072004 | 1.027353 | 13098 | 918 | 12180 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.498405 | 0.070292 | 1.023045 | 12633 | 868 | 11765 | 47 | 12633 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.498405 | 0.070292 | 1.023045 | 12633 | 868 | 11765 | 47 | 0 |

## Delta vs 20260601 LLM-Only Full Suite

The table compares original-method AUPRC and n-AUPRC against `20260601_llm_celltype_full_v1`. exp01, exp04, and exp05 are unchanged because the selected exp01/04/05 parameters are the same `mse050` setup; exp02, exp03, exp06, exp07, and exp08 reflect the dose-oriented parameter selection.

| exp | task | metric | llm-only | llm+dose selected | delta |
|---|---|---:|---:|---:|---:|
| exp01 | single unseen drug | AUPRC | 0.668358 | 0.668358 | +0.000000 |
| exp01 | single unseen drug | n-AUPRC | 5.644481 | 5.644481 | +0.000000 |
| exp02 | single unseen cell type | AUPRC | 0.802386 | 0.813731 | +0.011345 |
| exp02 | single unseen cell type | n-AUPRC | 6.088709 | 6.152131 | +0.063422 |
| exp03 | single unseen cell | AUPRC | 0.777466 | 0.781348 | +0.003883 |
| exp03 | single unseen cell | n-AUPRC | 6.511693 | 6.535777 | +0.024084 |
| exp04 | single w/o MSE | AUPRC | 0.647067 | 0.647067 | +0.000000 |
| exp04 | single w/o MSE | n-AUPRC | 5.464223 | 5.464223 | +0.000000 |
| exp05 | single w/o graph | AUPRC | 0.592835 | 0.592835 | +0.000000 |
| exp05 | single w/o graph | n-AUPRC | 4.992117 | 4.992117 | +0.000000 |
| exp06 | double unseen drug pair | AUPRC | 0.730924 | 0.779302 | +0.048378 |
| exp06 | double unseen drug pair | n-AUPRC | 1.811493 | 1.933956 | +0.122464 |
| exp07 | extra single | AUPRC | 0.539938 | 0.568332 | +0.028393 |
| exp07 | extra single | n-AUPRC | 2.548555 | 2.682490 | +0.133935 |
| exp08 | extra double | AUPRC | 0.068313 | 0.076974 | +0.008660 |
| exp08 | extra double | n-AUPRC | 1.571625 | 1.835971 | +0.264346 |

## Readout

- exp02 and exp03 benefited from the `covdrop010` setting. exp02 improved by 0.011345 AUPRC over the earlier LLM-only full suite, and exp03 improved by 0.003883 AUPRC.
- exp06 improved the most among validation tasks: AUPRC increased from 0.730924 to 0.779302 with the selected `drop020` double-drug setup.
- exp07 extra single improved from AUPRC 0.539938 to 0.568332.
- exp08 extra double remains the weakest external result in absolute AUPRC, but n-AUPRC improved from 1.571625 to 1.835971.
- The fresh w/o graph exp05 run confirms the graph-enabled exp01 is 0.075523 AUPRC above w/o graph under the same current exp01 parameter setting.

## Output Files

Primary selected suite:

- Markdown report: `logs/20260602_llm_dose_graphallowed_selected_v1_cell_drug_dose_time_eval.md`
- CSV report: `outputs/2026-06/2026-06-02/20260602_llm_dose_graphallowed_selected_v1_cell_drug_dose_time_eval.csv`
- JSON report: `outputs/2026-06/2026-06-02/20260602_llm_dose_graphallowed_selected_v1_cell_drug_dose_time_eval.json`
- Runtime summary: `logs/20260602_llm_dose_graphallowed_selected_v1_runtime_summary.tsv`
- exp07 reference epoch summary: `logs/20260602_llm_dose_graphallowed_selected_v1_exp07_extra_single_all_train_infer_all_single_for_extra_reference_epoch_summary.json`
- exp08 reference epoch summary: `logs/20260602_llm_dose_graphallowed_selected_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra_reference_epoch_summary.json`

Fresh w/o graph retrain:

- Markdown report: `logs/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.md`
- CSV report: `outputs/2026-06/2026-06-03/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.csv`
- JSON report: `outputs/2026-06/2026-06-03/20260603_llm_dose_exp01params_wograph_v1_cell_drug_dose_time_eval.json`
- Runtime summary: `logs/20260603_llm_dose_exp01params_wograph_v1_runtime_summary.tsv`

## Validation

- `python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py`
- Primary selected suite completed with 27 training manifests: exp01, exp02, exp03, exp04, exp06, exp07, and exp08.
- All 27 primary selected manifests report `graph_feature_mode=real`, `cell_type_llm_mode=frozen`, and `use_dose_covariate=True`.
- Fresh exp05 w/o graph retrain completed all 5 folds under `20260603_llm_dose_exp01params_wograph_v1`.
- All 5 fresh exp05 manifests report `graph_feature_mode=zero`, `cell_type_llm_mode=frozen`, and `use_dose_covariate=True`.
- Fresh exp05 manifests match current exp01 `mse050` hyperparameters: LR 2e-4, batch size 256, dropout 0.15, weight decay 1e-4, MSE weight 0.50, hidden dim 512.
- Fresh exp05 report materialized fold predictions under `outputs/2026-06/2026-06-03/20260603_llm_dose_exp01params_wograph_v1_cell_drug_fold_predictions/exp05/fold*`.
