# 2026-06-01 LLM Cell-Type Embedding Experiment

## Setup

- Run prefix: `20260601_llm_celltype_full_v1`
- Cell-type description model: `gpt-5.4`
- Cell-type embedding model: `Qwen/Qwen3-Embedding-8B`
- Embedding artifact: `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.npz`
- Sidecar metadata: `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.json`
- Embedding shape: `14 x 4096`; row `0` (`no`) is a zero vector; the remaining 13 cell-type embeddings are L2-normalized.
- API/proxy check: `proxy_on2` was enabled; `ALL_PROXY` was unset for the OpenAI-compatible Python client because the `flow_v2` env lacks SOCKS support. `gpt-5.4` chat and `Qwen/Qwen3-Embedding-8B` embeddings both returned successfully.

## Code Changes

- Added `utils/11_build_cell_type_llm_embeddings.py` to generate frozen cell-type summaries and embeddings from `.env` `openai_api_key` / `openai_base_url`.
- Added `--cell-type-llm-mode {off,frozen}` and `--cell-type-llm-embedding-path` to `train.py` and `infer.py`.
- `FastProteinTalkDataset` now carries a row-level `cell_type_features` tensor when frozen LLM features are enabled.
- `FastDeltaDrugResponseModel` now concatenates frozen LLM cell-type features with categorical covariate embeddings before the existing covariate projection.
- Experiment launcher common args now propagate the LLM cell-type settings through training, inference, and report materialization.

## exp01 Tuning

The exp01 tuning run used folds `0 2 4` with `CELL_TYPE_LLM_MODE=frozen`. The selected configuration for the full run was `mse050`; `mse050_warmdecay` tied it exactly in this screen, and the simpler constant schedule was kept.

| stage | config | task | AUPRC | nAUPRC | AUROC | count |
|---|---|---|---:|---:|---:|---:|
| stage1 | base | exp01_gap | 0.594017 | 4.929110 | 0.878612 | 10798 |
| stage1 | ctrl_drop010 | exp01_gap | 0.587651 | 4.878177 | 0.873154 | 10798 |
| stage1 | mse050 | exp01_gap | 0.601019 | 4.992900 | 0.875339 | 10798 |
| stage1 | mse050_drop010 | exp01_gap | 0.584346 | 4.860214 | 0.866933 | 10798 |
| stage1 | mse050_drop020 | exp01_gap | 0.588496 | 4.886119 | 0.880126 | 10798 |
| stage1 | mse050_gl3 | exp01_gap | 0.575474 | 4.774400 | 0.865450 | 10798 |
| stage1 | mse050_lr1e4 | exp01_gap | 0.585587 | 4.853891 | 0.885943 | 10798 |
| stage1 | mse050_lr3e4 | exp01_gap | 0.572183 | 4.749219 | 0.868875 | 10798 |
| stage1 | mse050_warmdecay | exp01_gap | 0.601019 | 4.992900 | 0.875339 | 10798 |
| stage1 | mse075 | exp01_gap | 0.587894 | 4.870044 | 0.867560 | 10798 |
| stage1 | mse075_drop010 | exp01_gap | 0.600358 | 4.982489 | 0.879950 | 10798 |

Full tuning report: `logs/20260601_llm_celltype_exp01_v1_param_search_report.md`

## Fold Mean Results

Metrics are reported under all three requested evaluation modes:

- `original`: the original row/timepoint-level metric.
- `cell-drug-dose-bylasttime`: one datapoint per cell-drug-dose group, selecting the max/last available timepoint.
- `cell-drug-dose-avgtime`: one datapoint per cell-drug-dose group, averaging over timepoints.

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp01 | single unseen drug | mean5 | original | 0.896777 | 0.668358 | 5.644481 | 17986 | 2137 | 15849 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-bylasttime | 0.896356 | 0.666578 | 5.652189 | 9032 | 1070 | 7962 | 0 | 0 |
| exp01 | single unseen drug | mean5 | cell-drug-dose-avgtime | 0.897524 | 0.670102 | 5.681576 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | original | 0.938004 | 0.802386 | 6.088709 | 17986 | 2137 | 15849 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-bylasttime | 0.938108 | 0.802598 | 6.103713 | 9032 | 1070 | 7962 | 0 | 0 |
| exp02 | single unseen cell type | mean5 | cell-drug-dose-avgtime | 0.939295 | 0.804917 | 6.117233 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | original | 0.930120 | 0.777466 | 6.511693 | 17986 | 2137 | 15849 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-bylasttime | 0.929635 | 0.776444 | 6.511910 | 9032 | 1070 | 7962 | 0 | 0 |
| exp03 | single unseen cell | mean5 | cell-drug-dose-avgtime | 0.931440 | 0.782651 | 6.560937 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | original | 0.889031 | 0.647067 | 5.464223 | 17986 | 2137 | 15849 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-bylasttime | 0.890058 | 0.647762 | 5.490830 | 9032 | 1070 | 7962 | 0 | 0 |
| exp04 | single w/o MSE | mean5 | cell-drug-dose-avgtime | 0.890330 | 0.648585 | 5.498221 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | original | 0.848554 | 0.592835 | 4.992117 | 17986 | 2137 | 15849 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-bylasttime | 0.850929 | 0.591692 | 5.000361 | 9032 | 1070 | 7962 | 0 | 0 |
| exp05 | single w/o graph | mean5 | cell-drug-dose-avgtime | 0.849799 | 0.596132 | 5.034897 | 9032 | 1070 | 7962 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | original | 0.805866 | 0.730924 | 1.811493 | 1791 | 723 | 1068 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-bylasttime | 0.806231 | 0.731150 | 1.810433 | 896 | 362 | 534 | 0 | 0 |
| exp06 | double unseen drug pair | mean5 | cell-drug-dose-avgtime | 0.808670 | 0.738014 | 1.827928 | 896 | 362 | 534 | 0 | 0 |

## Extra Mean Results

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | extra single | mean_extra | original | 0.754697 | 0.539938 | 2.548555 | 84625 | 17837 | 66788 | 0 | 0 |
| exp07 | extra single | mean_extra | cell-drug-dose-bylasttime | 0.754989 | 0.540448 | 2.550102 | 84474 | 17809 | 66665 | 28 | 84474 |
| exp07 | extra single | mean_extra | cell-drug-dose-avgtime | 0.754989 | 0.540448 | 2.550102 | 84474 | 17809 | 66665 | 28 | 0 |
| exp08 | extra double | mean_extra | original | 0.582704 | 0.068313 | 1.571625 | 41218 | 2025 | 39193 | 0 | 0 |
| exp08 | extra double | mean_extra | cell-drug-dose-bylasttime | 0.581057 | 0.060513 | 1.559995 | 32568 | 1459 | 31109 | 474 | 32568 |
| exp08 | extra double | mean_extra | cell-drug-dose-avgtime | 0.581057 | 0.060513 | 1.559995 | 32568 | 1459 | 31109 | 474 | 0 |

## Extra Subset Results

exp07 produced the available extra single-drug mat subsets. guomics/nature/nc are extra double-drug tasks in this data contract and are reported under exp08.

| exp | task | split | method | AUROC | AUPRC | n-AUPRC | count | pos | neg | conflicts | missing_time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | original | 0.703960 | 0.456922 | 2.227496 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-bylasttime | 0.703960 | 0.456922 | 2.227496 | 15834 | 3248 | 12586 | 0 | 15834 |
| exp07 | ptv3_extra_singledrug_mat1_480_faims | extra | cell-drug-dose-avgtime | 0.703960 | 0.456922 | 2.227496 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | original | 0.703557 | 0.434658 | 2.118958 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-bylasttime | 0.703557 | 0.434658 | 2.118958 | 15834 | 3248 | 12586 | 0 | 15834 |
| exp07 | ptv3_extra_singledrug_mat1_qe | extra | cell-drug-dose-avgtime | 0.703557 | 0.434658 | 2.118958 | 15834 | 3248 | 12586 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | original | 0.811849 | 0.655448 | 2.988667 | 12138 | 2662 | 9476 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-bylasttime | 0.811997 | 0.656221 | 2.989726 | 12087 | 2653 | 9434 | 9 | 12087 |
| exp07 | ptv3_extra_singledrug_mat2_480_faims | extra | cell-drug-dose-avgtime | 0.811997 | 0.656221 | 2.989726 | 12087 | 2653 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | original | 0.807704 | 0.616521 | 2.811169 | 12138 | 2662 | 9476 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-bylasttime | 0.808275 | 0.617466 | 2.813161 | 12087 | 2653 | 9434 | 9 | 12087 |
| exp07 | ptv3_extra_singledrug_mat2_qe | extra | cell-drug-dose-avgtime | 0.808275 | 0.617466 | 2.813161 | 12087 | 2653 | 9434 | 9 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | original | 0.705308 | 0.459083 | 2.174869 | 17609 | 3717 | 13892 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-bylasttime | 0.705308 | 0.459083 | 2.174869 | 17609 | 3717 | 13892 | 0 | 17609 |
| exp07 | ptv3_extra_singledrug_mat3_qe | extra | cell-drug-dose-avgtime | 0.705308 | 0.459083 | 2.174869 | 17609 | 3717 | 13892 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | original | 0.795803 | 0.616998 | 2.970174 | 11072 | 2300 | 8772 | 0 | 0 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-bylasttime | 0.796834 | 0.618340 | 2.976401 | 11023 | 2290 | 8733 | 10 | 11023 |
| exp07 | ptv3_extra_singledrug_mat4_qe | extra | cell-drug-dose-avgtime | 0.796834 | 0.618340 | 2.976401 | 11023 | 2290 | 8733 | 10 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | original | 0.954545 | 0.500000 | 11.500000 | 23 | 1 | 22 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.954545 | 0.500000 | 11.500000 | 23 | 1 | 22 | 0 | 23 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.954545 | 0.500000 | 11.500000 | 23 | 1 | 22 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | original | 0.618843 | 0.063517 | 1.939472 | 3084 | 101 | 2983 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.618843 | 0.063517 | 1.939472 | 3084 | 101 | 2983 | 0 | 3084 |
| exp08 | ptv3_extra_doubledrug_guomics | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.618843 | 0.063517 | 1.939472 | 3084 | 101 | 2983 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | original | 0.622466 | 0.065518 | 1.995731 | 3107 | 102 | 3005 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-bylasttime | 0.622466 | 0.065518 | 1.995731 | 3107 | 102 | 3005 | 0 | 3107 |
| exp08 | ptv3_extra_doubledrug_guomics | combined | cell-drug-dose-avgtime | 0.622466 | 0.065518 | 1.995731 | 3107 | 102 | 3005 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | original | 0.608597 | 0.099830 | 1.591611 | 5341 | 335 | 5006 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.585738 | 0.060865 | 1.777126 | 2949 | 101 | 2848 | 185 | 2949 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.585738 | 0.060865 | 1.777126 | 2949 | 101 | 2848 | 185 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | original | 0.585425 | 0.036583 | 1.339726 | 17615 | 481 | 17134 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.600542 | 0.026211 | 1.362756 | 12062 | 232 | 11830 | 214 | 12062 |
| exp08 | ptv3_extra_doubledrug_nature | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.600542 | 0.026211 | 1.362756 | 12062 | 232 | 11830 | 214 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | original | 0.601144 | 0.056115 | 1.578660 | 22956 | 816 | 22140 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-bylasttime | 0.596383 | 0.033812 | 1.524192 | 15011 | 333 | 14678 | 399 | 15011 |
| exp08 | ptv3_extra_doubledrug_nature | combined | cell-drug-dose-avgtime | 0.596383 | 0.033812 | 1.524192 | 15011 | 333 | 14678 | 399 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | original | 0.610277 | 0.139622 | 1.519585 | 2057 | 189 | 1868 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-bylasttime | 0.639995 | 0.147053 | 1.712789 | 1817 | 156 | 1661 | 28 | 1817 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_seenDrugCombo | cell-drug-dose-avgtime | 0.639995 | 0.147053 | 1.712789 | 1817 | 156 | 1661 | 28 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | original | 0.502707 | 0.071983 | 1.027054 | 13098 | 918 | 12180 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-bylasttime | 0.499701 | 0.070588 | 1.027341 | 12633 | 868 | 11765 | 47 | 12633 |
| exp08 | ptv3_extra_doubledrug_nc | unseenCell_unseenDrugCombo | cell-drug-dose-avgtime | 0.499701 | 0.070588 | 1.027341 | 12633 | 868 | 11765 | 47 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | original | 0.524503 | 0.083307 | 1.140485 | 15155 | 1107 | 14048 | 0 | 0 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-bylasttime | 0.524321 | 0.082208 | 1.160060 | 14450 | 1024 | 13426 | 75 | 14450 |
| exp08 | ptv3_extra_doubledrug_nc | combined | cell-drug-dose-avgtime | 0.524321 | 0.082208 | 1.160060 | 14450 | 1024 | 13426 | 75 | 0 |

## Output Files

- Full markdown report: `logs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.md`
- CSV: `outputs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.csv`
- JSON: `outputs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.json`
- Runtime summary: `logs/20260601_llm_celltype_full_v1_runtime_summary.tsv`

## Validation

- `python -m py_compile utils/11_build_cell_type_llm_embeddings.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py train.py infer.py scripts/report_cell_drug_time_eval.py`
- `bash -n scripts/ptv3_experiment_common.sh scripts/run_dose_param_search.sh scripts/exp_01_single_pert_stratified_5fold.sh scripts/exp_07_extra_single_all_train_infer.sh scripts/exp_08_extra_double_all_train_infer.sh`
- Smoke training dry-run with `--cell-type-llm-mode frozen` completed before the full suite.
