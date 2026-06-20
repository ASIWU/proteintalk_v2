# Data Standardization Session Summary

## 2026-06-19 08:16 HKT PTV1 README Retrain Full Cell/Cell-Type Completion

- Completed the README/script-aligned full retrain for the available PTV1 `cell_5fold` and `cell_type_5fold` tasks under `baseline/ptv2_benchmark_260514/PTV1_model_20260428`.
- Ran in `flow_v2` through tmux `gpu2:brainctl` with `--max-parallel 3`, `total_epoch=8000`, README/PTV1 script optimizer/model/SWAG settings, and only the two datasets present in this repository copy.
- Output root: `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720/`.
- Completed 10/10 folds with `summary.csv` status `ok` for all rows and no current task-log error markers.
- Result: `cell_5fold` and `cell_type_5fold` both reached or exceeded the provided checkpoint means across AUROC/AUPRC/AP; strict per-fold/per-metric tolerance still misses `cell_5fold` fold4 AUPRC/AP and `cell_type_5fold` fold3 AUROC.
- Stable report added at `docs/2026-06-19_ptv1_flow_v2_cell_celltype_readme_retrain_report.md`.

## 2026-06-17 22:30 HKT PTV1 README Retrain Parallel Runner Support

- Added `--max-parallel` to `baseline/ptv2_benchmark_260514/PTV1_model_20260428/run_cell_celltype_5fold_retrain_eval.py` so multiple folds can be trained concurrently inside one CUDA-visible GPU shell.
- Added per-fold subprocess logging via `--task-log-dir`; when `--max-parallel > 1`, logs default to `<output-root>/task_logs/` and the parent process remains the only writer of `summary.csv` and `report.md`.
- Preserved the default serial behavior with `--max-parallel 1`; existing README-parameter retrain commands continue to run unchanged unless parallelism is requested explicitly.
- Validation passed:
  - `python -m py_compile baseline/ptv2_benchmark_260514/PTV1_model_20260428/run_cell_celltype_5fold_retrain_eval.py`.

## 2026-06-17 17:18 HKT PTV1 Flow V2 Epoch-Matched Cell/Cell-Type Retrain

- Added and ran an epoch-matched retrain workflow for the available PTV1 `cell_5fold` and `cell_type_5fold` tasks under `baseline/ptv2_benchmark_260514/PTV1_model_20260428`.
- Used the provided checkpoint filenames to cap each fold at `reference_epoch + 1`, while retaining README-compatible batch/model/optimizer/SWAG/seed settings.
- Optimized PTV1 dataset construction to avoid repeated full-frame pandas filters during `L1000Dataset` initialization.
- Added controls to skip train-time test-set evaluation, skip per-epoch checkpoint files, and configure training log interval; final test metrics are still computed by reloading the best checkpoint and running prediction on the test split.
- Completed 10/10 available folds in `gpu2` with `flow_v2`; metrics and generated report are under `baseline/ptv2_benchmark_260514/retrain_runs/flow_v2_cell_celltype_5fold_retrain_epochmatched_20260617_1617/`.
- Stable report added at `docs/2026-06-17_ptv1_flow_v2_cell_celltype_retrain_report.md`.
- Result: dataset-level means nearly match the provided checkpoints, but strict per-fold 0.01 tolerance is not met for `cell_5fold` fold3, `cell_5fold` fold4 AUPRC/AP, and `cell_type_5fold` fold3 AUROC.

## 2026-06-17 15:31 HKT PTV1 Flow V2 Cell/Cell-Type Benchmark Completion

- Removed the unused top-level `torcheval.metrics.functional` import from `baseline/ptv2_benchmark_260514/PTV1_model_20260428/metrics.py`; `flow_v2` has `torch==2.7.1+cu126` but `torchaudio==2.7.1+cu128`, and importing `torcheval.metrics` triggers the incompatible audio extension before prediction starts.
- Re-ran the available PTV1 benchmark folds in `gpu2` with `conda activate flow_v2`, H200 CUDA, and the existing checkpoint files under `baseline/ptv2_benchmark_260514/checkpoints/`.
- Completed all available `cell_5fold` and `cell_type_5fold` folds, 10/10 metrics files, with results in `baseline/ptv2_benchmark_260514/eval_runs/flow_v2_cell_celltype_5fold_eval_20260617_1535_fixed/`.
- Added the stable report `docs/2026-06-17_ptv1_flow_v2_cell_celltype_benchmark_report.md`.
- Confirmed that the current `PTV1_model_20260428` benchmark data/checkpoint tree has `cell_5fold` and `cell_type_5fold` only; no runnable `pert_stratified_5fold` data/checkpoint directory is present in this copy.

## 2026-06-17 15:15 HKT PTV1 Baseline Cell/Cell-Type Evaluation Runner

- Updated `baseline/ptv2_benchmark_260514/PTV1_model_20260428/main.py` prediction checkpoint loading to pass `weights_only=False`, which is required for the provided full training checkpoint dictionaries under current PyTorch defaults.
- Added `baseline/ptv2_benchmark_260514/PTV1_model_20260428/run_cell_celltype_5fold_eval.py` to run the available `cell_5fold` and `cell_type_5fold` benchmark folds from current repository paths, invoke `plot_result.py`, and write `summary.csv` plus `report.md`.
- Scope intentionally limited to the two datasets present in the current `baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified/` tree.

## 2026-06-05 03:37 HKT Corrected Cell LLM Clip10 Tuned Selected Completion

Completed the corrected Cell LLM clip10 tuning goal from `docs/2026-06-04-goal.md` with final prefix `20260604_cell_llm_clip10_tuned_selected_v1`.

- Screen prefix `20260604_cell_llm_clip10_tune_v1` completed all required candidates: stage1 `135/135`, stage2 `42/42`, stage3 `42/42`, stage4 `42/42`.
- Promoted full prefix `20260604_cell_llm_clip10_tune_v1_full` completed `100/100` manifests and wrote `logs/20260604_cell_llm_clip10_tune_v1_full_param_search_report.md`.
- Final selected prefix `20260604_cell_llm_clip10_tuned_selected_v1` completed `32/32` manifests and wrote:
  - `logs/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.md`
  - `outputs/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.csv`
  - `outputs/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.json`
- Final selected configs: stage1 `mse050_target_pdi`, stage2 `covdrop010_drop010`, stage3 `covdrop010_lr1e4`, stage4 `drop020_mseinactive010`.
- Final manifest audit passed: `cell_llm_mode=frozen`, `cell_llm_summary.embedding_rows=74`, no `cell_type_llm` manifest key, `exp05` graph mode `zero`, other final experiments graph mode `real`.
- exp07 used selected exp01 mean-nearest reference epoch `10` and `last.ckpt`; exp08 used selected exp06 mean-nearest reference epoch `9` and `last.ckpt`.
- Static checks passed:
  - `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_llm_clip10_param_search.py scripts/cell_llm_clip10_goal_status.py`
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh`
- Results documentation updated in `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`.

## 2026-05-11 12:49 HKT Existing-Training Extra Re-Inference Script

Added `scripts/reinfer_extra_from_existing_training.sh` to reuse an already trained all-data checkpoint directory and rerun only extra inference.

- The script does not call `train.py`; it only resolves a checkpoint and calls `infer.py` for the extra datasets.
- Default mode is extra single with `SOURCE_EXP_NAME=20260510_extra_single_all_train_infer_all_single_for_extra`.
- Default checkpoint policy is `CHECKPOINT_POLICY=best`, which reads `best_model_path` from the source run's `run_manifest.json`.
- It also supports `CHECKPOINT_POLICY=last`, `CHECKPOINT_POLICY=reference`, and `CHECKPOINT_POLICY=explicit` with `CHECKPOINT_PATH`.
- `EXTRA_MODE=double` switches to the extra double task list and `synergy` head.
- Default output is a new folder named `${SOURCE_EXP_NAME}_${CHECKPOINT_POLICY}_reinfer`, so old outputs are not overwritten unless `OUTPUT_EXP_NAME` is explicitly reused.
- The script prints the result table with `scripts/show_extra_results.py` by default.

Validation:

- `bash -n scripts/reinfer_extra_from_existing_training.sh` passed.
- Bounded extra-single smoke reused `checkpoints/20260510_extra_single_all_train_infer_all_single_for_extra/epoch=49.ckpt` and wrote 6 one-row extra-single outputs to `outputs/20260511_reinfer_existing_single_smoke`.
- Bounded extra-double smoke reused an existing double smoke checkpoint and wrote 3 one-row extra-double outputs to `outputs/20260511_reinfer_existing_double_smoke`.

## 2026-05-11 12:32 HKT 0509 Wrapper Reference-Epoch Update

Updated `scripts/0509_1.sh` through `scripts/0509_4.sh` to align the production wrappers with the hardened reference-epoch policy:

- Added `set -euo pipefail` and shebangs to all four wrapper scripts.
- Converted hard-coded inline environment assignments to overridable defaults, so smoke runs can set alternate `EXP_PREFIX`, fold count, batch size, logger backend, and reference paths without editing the scripts.
- `scripts/0509_3.sh` now defaults `REFERENCE_5FOLD_CKPT_PATH=checkpoints/20260510_single_pert_stratified_5fold`, `REFERENCE_EPOCH_AGG=median`, `REFERENCE_EPOCH_MIN_COUNT=5`, `SAVE_LAST_CKPT=1`, and an empty `SCHEDULER_NAME`.
- `scripts/0509_4.sh` now defaults `REFERENCE_5FOLD_CKPT_PATH=checkpoints/20260510_double_pert_pair_5fold`, `REFERENCE_EPOCH_AGG=median`, `REFERENCE_EPOCH_MIN_COUNT=5`, `SAVE_LAST_CKPT=1`, and an empty `SCHEDULER_NAME`.
- `scripts/0509_1.sh` and `scripts/0509_2.sh` did not need reference-epoch logic themselves; they remain the source 5-fold runs used by the extra single/double wrappers.

Validation:

- `bash -n scripts/0509_1.sh scripts/0509_2.sh scripts/0509_3.sh scripts/0509_4.sh` passed.
- 8-GPU bounded wrapper smoke passed:
  - `20260511_wrapper_0509_1_smoke`: single-drug fold0 train/test passed.
  - `20260511_wrapper_0509_2_smoke`: double-drug fold0 train/test passed.
  - `20260511_wrapper_0509_3_smoke`: reference extra single branch passed and wrote 6 bounded extra-single outputs.
  - `20260511_wrapper_0509_4_smoke`: reference extra double branch passed and wrote 3 bounded extra-double outputs.
- Extra wrapper manifests recorded `reference_epoch_policy`, `fixed_reference_epoch_last_ckpt`, selected epoch, and selected `last.ckpt` path.

## 2026-05-11 11:13 HKT Reference-Epoch Extra Evaluation

新增 [scripts/select_reference_epoch.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/scripts/select_reference_epoch.py:1)，用于从 5-fold 训练产生的 `run_manifest.json` 中读取各 fold 的 `best_model_path` epoch，并按 `median` / `mean` / `min` / `max` 聚合出一个固定 epoch。

- [scripts/ptv3_experiment_common.sh](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/scripts/ptv3_experiment_common.sh:1) 新增 `REFERENCE_5FOLD_CKPT_PATH`、`REFERENCE_EPOCH_AGG`、`REFERENCE_EPOCH_ROUNDING`、`REFERENCE_EPOCH_MIN_COUNT` 配置，并将 reference epoch 工具加入 preflight。
- [scripts/exp_07_extra_single_all_train_infer.sh](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/scripts/exp_07_extra_single_all_train_infer.sh:1) 和 [scripts/exp_08_extra_double_all_train_infer.sh](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/scripts/exp_08_extra_double_all_train_infer.sh:1) 在提供 `REFERENCE_5FOLD_CKPT_PATH` 时，会使用对应主任务 5-fold 的聚合最佳 epoch，训练 all-data 模型到 `selected_epoch + 1`，设置 `--monitor none`，并用 `last.ckpt` 做 extra inference。
- reference 模式会在 all-data 训练的 `run_manifest.json` 中写入 `reference_epoch_policy`，记录 reference path、reference task、selected epoch、aggregation、rounding、min_count 和实际使用的 `last.ckpt`。
- 默认未提供 reference path 时保持旧行为：训练 all-data 模型后读取该 run 的 `best_model_path` 进行 extra inference。
- 这样 extra single / extra double 测试不再通过 extra test labels 选择 checkpoint；checkpoint epoch 由独立 5-fold reference run 预先决定。
- 验证：`bash -n`、`py_compile`、`git diff --check` 均通过；使用 8 张 H200 做 bounded smoke，覆盖 single fold、double fold、reference extra single 和 reference extra double。最终 `20260511_ref_policy_extra_single_smoke2` 写出 6 个 extra single inference 输出，`20260511_ref_policy_extra_double_smoke2` 写出 3 个 extra double inference 输出，且 all-data manifest 均记录 `reference_epoch_policy`。

## 2026-05-11 12:16 HKT Reference-Epoch Adversarial Hardening

对 reference-epoch extra evaluation 策略做 adversarial review 后发现并修复多个真实 loophole：

- `scripts/select_reference_epoch.py` 现在默认要求 reference fold checkpoint 文件真实存在、`best_model_score` 非空、`run_status=fit_completed`，并可要求 `test_status=test_completed`。
- selector 新增 task head、model type、dataset group、split strategy regex 校验，避免把 single/double、不同模型或错误 split 混进 reference epoch。
- selector 默认拒绝重复 `split_strategy` 和混合 reference 配置；会检查 model/loss/monitor/optimizer/key hyperparameter 等字段，避免 broad path 把不同实验混到一起。
- `REFERENCE_EPOCH_MIN_COUNT` 默认从 `1` 改成 `5`，使 production reference policy 默认要求完整 5-fold；smoke/debug 可显式覆盖为 `1`。
- scripts 07/08 在 reference policy 下若 `SAVE_LAST_CKPT=0` 会训练前 fail-fast，因为最终 extra inference 必须使用固定 epoch训练结束后的 `last.ckpt`。
- scripts 07/08 在 reference policy 下拒绝 `SCHEDULER_NAME=plateau`，避免 all-data validation split 通过 validation-driven LR scheduler 影响最终模型权重。
- all-data manifest 的 `reference_epoch_policy` 现在嵌入 selector 生成的 `reference_summary`，包括每个 reference fold 的 manifest path、split、checkpoint、epoch、score 和 monitor。

验证：

- 正向 selector：现有 `20260510_single_pert_stratified_5fold` 通过严格校验，median epoch 仍为 `52`。
- 负向 selector：对整个 `checkpoints/` 做 broad reference 输入会失败，原因包括重复 fold0 和混合 `best_ckpt_metric` / `max_epochs`。
- 负向 script：`SAVE_LAST_CKPT=0` + reference policy 会在训练前失败。
- `bash -n`、`py_compile`、`git diff --check` 均通过。
- 使用 8 张 H200 做最终 bounded smoke：
  - `20260511_ref_policy_single_smoke3`: single-drug fold0 training/test passed。
  - `20260511_ref_policy_double_smoke3`: double-drug fold0 training/test passed。
  - `20260511_ref_policy_extra_single_smoke4`: reference extra single all-data training passed after final policy guards，并写出 6 个 bounded extra single outputs。
  - `20260511_ref_policy_extra_double_smoke4`: reference extra double all-data training passed after final policy guards，并写出 3 个 bounded extra double outputs。

## 2026-05-11 10:33 HKT Extra Inference Result Summary Script

新增 [scripts/show_extra_results.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/scripts/show_extra_results.py:1)，用于汇总 `infer.py` 生成的 extra single / extra double 输出目录中的 `metrics.json`。

- 默认读取 `outputs/20260510_extra_single_all_train_infer_all_single_for_extra`，直接打印 extra single 各数据集的 AUROC / AUPRC / ACC / valid / positive / negative / prediction rows。
- 默认根据 `run_manifest.json` 中的 `task_head` 选择指标头，因此 extra single 使用 `response`，extra double 可直接使用 `synergy`。
- 支持传入任意 output 目录或单个 `metrics.json`，并可通过 `--all-heads`、`--format markdown|csv`、`--csv-out`、`--include-paths` 做更完整检查。

## 背景

本轮工作基于 [docs/Data_Process.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process.md:1) 的第一阶段要求，目标是对 `data/rawdata/` 下的原始数据做逐文件梳理、标准化，并产出可复现的 task-level 输入与全局 meta。

本轮已经完成第一版可运行实现，并产出了标准化结果到 `data/standardized/`。

## 2026-04-24 Embedding / Graph Builder Compliance Review

本次根据 `docs/Data_Process.md` 和 `docs/Data_Process_2.md` 重新检查 embedding / graph 生成代码是否满足需求。

### 本次已落实的修改

0. [utils/06_export_uniprot_ids_from_global_meta.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/06_export_uniprot_ids_from_global_meta.py:1)
   新增从 stage-2 `global_meta.json` 导出 UniProt accession txt 的工具：
   - 默认按 `protein_index_to_id` / `protein_index` 顺序输出。
   - 默认排除 `control` 和 `no`。
   - 默认校验 UniProt accession 格式，避免生成不能直接用于 UniProt FASTA 下载的列表。
   - 支持可选 audit JSON 记录导出数量、特殊值、非法 ID 和重复 ID。
1. [utils/05_build_graph_matrices_from_global_meta.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/05_build_graph_matrices_from_global_meta.py:1)
   历史记录：当时曾修正 PPI reference filter；该 reference filter 后续已在 2026-04-27 PPI No-Threshold Update 中移除。

### 检查结论

1. drug embedding builder 使用 stage-2 `global_meta.json["pert_index"]` 排序，并输出 `.pkl` payload。
2. protein embedding builder 使用 stage-2 `global_meta.json["protein_index"]` 排序，并输出 `.pkl` payload；实际运行仍需要 FASTA 和 transformers model / local model path。
3. PPI / DDI / PDI builders 都按 `global_meta.json` index 分配矩阵轴；PTV1 和 PTV3 通过传入各自 `data/training_ready/<dataset>/global_meta.json` 分别生成。
4. PPI / PDI 仍依赖外部 edge / mapping 数据；当前代码提供 mapping 参数和 online mapping fallback，但本次未执行真实外部大文件生成。

### 校验结果

```bash
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile utils/04_build_embeddings_from_global_meta.py utils/05_build_graph_matrices_from_global_meta.py utils/06_export_uniprot_ids_from_global_meta.py
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/04_build_embeddings_from_global_meta.py --help
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/05_build_graph_matrices_from_global_meta.py --help
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/06_export_uniprot_ids_from_global_meta.py --help
```

结果：语法检查和 CLI entrypoints 均通过；另用最小 `combined_score`-only PPI edge table 做 smoke test，确认输出矩阵 shape 与 `protein_index` 长度一致，且 edge weight 可按 index 正确写入。已导出当前 UniProt txt：

- `data/training_ready/ptv3/derived/uniprot_ids.txt`: `11343` lines
- `data/training_ready/ptv1/derived/uniprot_ids.txt`: `5576` lines

## 2026-04-24 Extra Double-Drug Nature / NC Rawdata Rerun

本次根据用户更新后的 `data/rawdata/extra_doubeldrug/` 中 Nature 与 NC rawdata 重新检查并运行 stage-1 / stage-2。

### 本次已落实的修改

1. [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)
   `ptv3_extra_doubledrug_nc` 与 `ptv3_extra_doubledrug_nature` 已切换并兼容新文件：
   - NC: `data/rawdata/extra_doubeldrug/260424nc_drugComb_info_unique_with_smiles.csv`
   - Nature: `data/rawdata/extra_doubeldrug/260424nature_drugComb_info_unique_with_smiles.csv`
   - 旧 `20260411...csv` 路径保留为 fallback。
2. 新文件中的 chiral / no-chiral SMILES 已按 `Smiles*_with_chiral > smiles* > Smiles*_no_chiral` 进入标准化 smiles，并额外保留 raw audit columns。
3. NC 额外保留 `anchor_lib`、`group`、`group1`、`Cell2` 原始列；Nature 额外保留 `Tissue`、`Cancer.Type`、`Anchor.Pathway`、`Library.Pathway`、`Synergy?` 原始列。
4. 这两个 raw 文件仍只包含 metadata / label / smiles / target 字段，不包含 perturbation proteome matrix；因此 stage-1 `protein_count` 仍为 `0`，stage-2 只从 matched controls 追加有真实 proteome 的 control rows。

### 本次重新生成后的关键尺寸

- stage-1:
  - `ptv3_extra_doubledrug_nc`: `16394 x 0`
  - `ptv3_extra_doubledrug_nature`: `23389 x 0`
- stage-2:
  - `ptv3_extra_doubledrug_nc`: processed / feature `16412 x 11343`，其中 `18` 行为 matched control proteome rows
  - `ptv3_extra_doubledrug_nature`: processed / feature `23415 x 11343`，其中 `26` 行为 matched control proteome rows

### 校验结果

使用完整环境路径执行：

```bash
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile utils/00_standardize_rawdata.py utils/01_validate_standardized_outputs.py utils/02_build_training_ready_data.py utils/03_validate_training_ready_outputs.py
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/00_standardize_rawdata.py
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/01_validate_standardized_outputs.py
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/02_build_training_ready_data.py
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/03_validate_training_ready_outputs.py
```

结果：stage-1 和 stage-2 均 `Validation passed.`

## 2026-04-24 Double-Drug Rawdata Rerun

本次根据用户更新后的 `data/rawdata/doubledrug/` 与 `data/rawdata/extra_doubeldrug/` 重新检查并运行了 stage-1 / stage-2。

### 本次已落实的修改

1. [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)
   主双药任务已切换并兼容新文件：
   - info: `data/rawdata/doubledrug/20260422ptv3_J_3496_sampinfo_final.csv`
   - expression: `data/rawdata/doubledrug/20260422ptv3_J_3496samp_9202prot_final_edit.csv`
   - expression protein columns 按直接 `UniProt ID` 解析
2. 新双药 info 中 raw `control` 为空、但 `pert_id1 == control` 且 `pert_id2 == control` 的真实 control 行，已标准化为 `control == sample_id`。本次共识别 `405` 行。
3. 主双药 smiles 解析现在优先使用主单药 `pert_id -> smiles` registry，并用 raw `Smiles1/2_with_chiral`、`Smiles1/2_no_chiral` 作为 fallback。
4. Guomics extra double task 已切换到：
   - `data/rawdata/extra_doubeldrug/260423ptv3_Guomics_drug_combo_unique_with_smlies.csv`
   - 新文件的 `Library_Primary.Pathway` 保留为审计列，不当作 target protein 直接映射。

### 本次重新生成后的关键尺寸

- stage-1:
  - `ptv3_main_doubledrug`: `3496 x 9202`
  - `ptv3_extra_doubledrug_guomics`: `9001 x 0`
- stage-2:
  - `ptv3_main_doubledrug`: processed `2196 x 9202`; feature `20764 x 11092`
  - `ptv3_extra_doubledrug_guomics`: processed / feature `9004 x 11092`

### 校验结果

本机短名 `conda activate flow_v2` 未注册到当前 shell，实际使用完整环境路径激活：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
python utils/00_standardize_rawdata.py
python utils/01_validate_standardized_outputs.py
python utils/02_build_training_ready_data.py
python utils/03_validate_training_ready_outputs.py
```

结果：stage-1 和 stage-2 均 `Validation passed.`

## 2026-04-21 PTV1 Workflow Update

本次根据 [docs/Data_Process_ptv1.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process_ptv1.md:1) 对 `PTV1` 分支做了专项更新，并同步修改了 stage-1 / stage-2 文档与代码。

### 本次已落实的修改

1. [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)
   `ptv1_aivc` 不再只按单药字段简化处理，而是：
   - 使用 `Library_dose -> pert_dose1`
   - 使用 `Anchor_dose -> pert_dose2`
   - 使用 `pert_id -> pert_id1`
   - 使用 `Anchor_id -> pert_id2`
   - 从 `ptv1.csv` 补齐 smiles / target，并在双侧都存在时合并
2. 新增 `ptv1_extra_singledrug` stage-1 task：
   - 输入目录：`data/rawdata/ptv1_extra_singledrug/`
   - `E115_id -> pert_id1`
   - `cell -> Cell / Cell_plate`
   - control 从 `ptv1_aivc` 的 control 行按 `cell -> protein_plate` 匹配
   - smiles / target 从 `PTV3` stage-1 global meta 读取
   - 样本先按 unique `(cell, E115_id)` 去重
   - `ground_truth` 固定取 `ppODE_swa1` 对应行
3. [utils/02_build_training_ready_data.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/02_build_training_ready_data.py:1)
   已将 `ptv1_extra_singledrug` 纳入 stage-2 构建，不再额外过滤其 non-control 行，只在 feature / processed 输出中追加匹配到的 control 行。
4. [utils/03_validate_training_ready_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/03_validate_training_ready_outputs.py:1)
   已同步支持 `ptv1_extra_singledrug` 的 row-filter 校验。
5. [docs/Data_Process.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process.md:1)、
   [docs/Data_Process_2.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process_2.md:1)、
   [docs/data_process_summary_01.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/data_process_summary_01.md:1)、
   [docs/data_process_summary_02.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/data_process_summary_02.md:1)
   已同步更新。

### 本次验证结果

在临时输出目录下做了 focused ptv1 verification：

- stage-1:
  - `ptv1_aivc`: `15002 x 5576`
  - `ptv1_extra_singledrug`: `182 x 0`
- stage-2:
  - `ptv1_aivc`: processed / feature 都是 `15002 x 5576`
  - `ptv1_extra_singledrug`: processed / feature 都是 `186 x 5576`

已重新执行：

```bash
source ~/.bashrc
conda activate flow_v2
python utils/01_validate_standardized_outputs.py --output-root /tmp/<ptv1_verify>/standardized
python utils/03_validate_training_ready_outputs.py --output-root /tmp/<ptv1_verify>/training_ready
```

结果：`Validation passed.`

### 已记录的原始数据问题与当前处理

1. `ptv1_aivc`
   同一个 `(BioRep, protein_plate)` 下经常存在多个 `pert_time == 0` 的合法 control 行。当前代码会记录 candidate 数量，并选取一个确定性代表 control sample id；被选中的 control 行始终要求 `pert_time == 0`。
2. `ptv1_extra_singledrug`
   `test12091214_sample_predictions_E115id.csv` 中不同 model 行的 `ground_truth` 仍然存在冲突，但当前规则已经固定为：按 unique `(cell, E115_id)` 去重，并使用 `ppODE_swa1` 对应的标签；其余冲突只记录到审计信息。

## 2026-04-21 Stage-2 Dose Index Update

本次对 stage-2 训练数据构建规则和文档做了同步更新，重点是 `pert_dose` 的 index 规则。

### 本次已落实的修改

1. [utils/02_build_training_ready_data.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/02_build_training_ready_data.py:1)
   将 `pert_dose1` / `pert_dose2` 的共享映射从“按唯一值顺序编号”改为“`ceil(dose)` 后再写成 string index”。
2. `pert_dose` 缺失值继续保留特殊 token `"no"`，其 index 改为 `string(max_numeric_index + 1)`。
3. [utils/03_validate_training_ready_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/03_validate_training_ready_outputs.py:1)
   已同步调整，支持 string 型 dose index，并校验 `"no" == max + 1`。
4. [docs/Data_Process_2.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process_2.md:1)
   已更新 `pert_dose` 的规则说明。
5. [docs/data_process_summary_02.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/data_process_summary_02.md:1)
   已补充 embedding / PPI / DDI / PDI 的运行说明，以及 dataloader `__getitem__` 读取示例。

## 2026-04-20 Rerun Update

本次基于用户修改后的 raw data 重新运行并更新了 [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)。

### 本次已落实的修改

1. 主双药任务改为兼容新的 raw 文件名与 schema：
   - info: `data/rawdata/doubledrug/20260417ptv3_J_3509sampinfo.csv`
   - expression: `data/rawdata/doubledrug/20260417ptv3_J_3509samp_9112prot_finall_edit.csv`
   - expression sample key 从旧的 `samp_id` 兼容到新的 `sample_id`
   - 蛋白列从旧的带前缀列名兼容到当前直接使用 `UniProt ID`

2. 主单药 `control` 脏值问题已随 raw 修复消失：
   - 当前标准化结果中 `control_status` 只有 `ok` 和 `missing`
   - `unresolved_control_reference` 数量为 `0`

3. extra baseline 中缺失原始 info 的 3 个样本已按上次约定删除：
   - `S1_B`
   - `S1_CAC`
   - `S1_O`

4. `ptv3` 的 `pert_id -> smiles` 现在按 first-seen wins 统一回写到各 task `info.csv`：
   - `global_meta.json` 中当前 `pert_mapping_conflicts` 数量为 `0`

5. Guomics extra double task 已切换到新的 raw 文件：
   - `data/rawdata/extra_doubeldrug/260417ptv3_Guomics_drug_combo_unique_with_smlies.csv`

### 本次重新生成后的关键尺寸

- `ptv3_main_singledrug`: `28602 x 10982`
- `ptv3_main_doubledrug`: `3509 x 9112`
- `ptv3_extra_baseline`: `75 x 10169`
- `ptv3_extra_doubledrug_guomics`: `9009 x 0`
- `ptv1_aivc`: `15002 x 5576`

### 校验结果

已重新执行：

```bash
source ~/.bashrc
conda activate flow_v2
python utils/00_standardize_rawdata.py
python utils/01_validate_standardized_outputs.py
```

结果：`Validation passed.`

## 2026-04-15 本轮新增代码

### 1. 主标准化脚本

文件: [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)

作用:

- 按 task 处理主单药、主双药、extra baseline、extra 单药、extra 双药、PTV1
- 输出每个 task 的:
  - `info.csv`
  - `expression_matrix.npy`
  - `protein_order.json`
  - `sample_ids.json`
  - `sample_id_to_row_index.json`
  - 小任务时额外输出 `expression_dict.pkl`
- 生成:
  - `data/standardized/ptv3/global_meta.json`
  - `data/standardized/ptv1/global_meta.json`
  - `data/standardized/file_audit.json`

### 2. 校验脚本

文件: [utils/01_validate_standardized_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/01_validate_standardized_outputs.py:1)

作用:

- 校验每个 task 的 `info.csv`、矩阵形状、蛋白顺序、sample 索引是否一致
- 校验 materialized `expression_dict.pkl` 的样本数是否匹配

## 本轮产物

输出目录:

- [data/standardized](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized:1)
- [data/standardized/file_audit.json](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized/file_audit.json:1)
- [data/standardized/ptv3/global_meta.json](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized/ptv3/global_meta.json:1)
- [data/standardized/ptv1/global_meta.json](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/data/standardized/ptv1/global_meta.json:1)

验证结果:

- `ptv3_main_singledrug`: `28602 x 10982`
- `ptv3_main_doubledrug`: `3549 x 9205`
- `ptv3_extra_baseline`: `78 x 10169`
- `ptv1_aivc`: `15002 x 5576`
- extra 单药 / extra 双药: 当前输出为 metadata + 空表达结构

校验脚本已通过。

## 当前实现的重要规则

### 1. 主单药 / 主双药 / extra baseline / PTV1

- 有真实蛋白表达矩阵
- 蛋白列统一为 `UniProt ID`
- 按 `sample_id` 与样本表对齐

### 2. extra 单药 / extra 双药

当前仓库中没有对应的原始 perturbation proteome 矩阵，因此第一版实现采用统一接口输出:

- `info.csv`
- 空的 `expression_matrix.npy`
- 空的 `protein_order.json`
- 空的 `expression_dict.pkl`

这里“空表达结构”的含义是:

- 矩阵 shape 为 `(样本数, 0)`
- 不是数值填 0
- 而是“只有样本，没有蛋白表达列”

### 3. control 匹配

extra 数据的 control 候选池来自:

- 主单药中 `control == sample_id` 的 control 样本
- extra baseline

匹配优先级:

1. `Cell` 硬匹配
2. `Machine Match`
3. `Type Match`
4. `Batch Match`
5. `Plate Match`

匹配结果会记录审计列:

- `control_match_level`
- `control_match_source_task`
- `control_match_pool_kind`
- `control_match_score`
- `control_match_machine`
- `control_match_type`
- `control_match_batch`
- `control_match_plate`

### 4. PTV1 隔离

- `ptv1` 独立输出到 `data/standardized/ptv1`
- 不与 `ptv3` 共用索引空间

## 本轮遇到的问题

### 1. 主单药 info 文件编码问题

文件:

- `data/rawdata/singledrug/20260403_ptv3_v2_bind_bio_sampleID_machineID_details.csv`

问题:

- 该文件并非稳定 `utf-8`
- 实际需要 `latin1` fallback 才能完整读取

处理:

- 在主脚本里对该文件启用了多编码 fallback

### 2. 挂载盘不支持直接 `numpy.memmap`

问题:

- 运行主标准化脚本时，`np.lib.format.open_memmap(...)` 在挂载目录下报错:
  - `OSError: [Errno 19] No such device`

处理:

- 改为先把大矩阵写到 `/tmp`
- 完成后再 `move` 回 `data/standardized/...`
- 校验脚本不再使用 `mmap_mode='r'`，改为只读 `.npy` header 获取 shape

### 3. 主单药 `control` 字段有脏值

问题:

- 有 112 行 `control` 不是合法 `sample_id`
- 这些脏值其实是 SMILES，不是 sample family 引用

唯一 4 个异常值:

- `C=CCn1c(=O)c2cnc(Nc3ccc(N4CCN(C)CC4)cc3)nc2n1-c1cccc(C(C)(C)O)n1`
- `Cc1cc(NC(=O)NCCN2CCC(O)(Cc3ccccc3)CC2)c2ccccc2n1.Cl`
- `Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=NC(CC(=O)Nc1ccc(O)cc1)c1nnc(C)n1-2`
- `Cn1cc(CC(N)C(=O)O)c2ccccc21`

这些值对应的典型药物:

- `pert_id=115`, `drugname=Adavosertib`
- `pert_id=L9200_2016`
- `pert_id=L9200_2703`
- `pert_id=L9200_2896`

第一版处理:

- 这类值不会强行当作 `sample_id`
- 标准化结果中:
  - `control` 被清空
  - `control_raw` 保留原值
  - `control_status=unresolved_non_sample_id`

### 4. extra baseline 中有 3 条表达样本缺失原始 info

缺失样本:

- `S1_B`
- `S1_CAC`
- `S1_O`

第一版处理:

- 保留表达矩阵中的这 3 行
- 用占位 metadata 保留下来
- `raw_record_issue=info_missing_in_raw_file`

后续核查结果:

- 这 3 个样本没有被其他 extra task 用作 control
- 只在 `ptv3_extra_baseline` 自己里面存在

### 5. `pert_id -> smiles` 冲突较多

问题:

- `ptv3/global_meta.json` 中记录到 `937` 条冲突
- 大多是不同手性写法、盐形式、去手性写法、或不同但相关的 SMILES 规范化结果

第一版处理:

- `global_meta.json` 中的 `pertid_to_smiles` 实际已经是 first-seen wins
- 同时把后续冲突额外记在 `pert_mapping_conflicts`

## 用户在讨论阶段提出的修改边界

用户原始回复要点:

1. 为什么叫做表达结构是空矩阵？
2. 请告诉我不合法的 `sample_id` 是什么样的，我好判断如何修改
3. 缺乏原始 info 的 `extra_baseline` 可以直接删除
4. 冲突的 `pert_id -> smiles` 以第一次出现的为准，用 `pert_id` 对齐，因为手性不重要

## 我当时给出的回答摘要

### 对第 1 点的回答

我解释了“空矩阵”的含义:

- shape 是 `(样本数, 0)`
- 表示没有蛋白表达列
- 不是值全为 0

同时我提出建议:

- 如果不喜欢这种接口，可以改成 `info-only` 模式
- 即 metadata-only 的 extra task 不再生成空矩阵 / 空 dict

这个点用户尚未最终拍板。

### 对第 2 点的回答

我说明:

- 不合法的不是 `sample_id` 列本身
- 而是主单药 `control` 字段里出现了 4 个 SMILES 字符串

并把这 4 个具体值列了出来，供后续判断如何修。

### 对第 3 点的回答

我核查后确认:

- `S1_B`
- `S1_CAC`
- `S1_O`

这 3 个 extra baseline 样本没有被其他 task 使用为 control，因此直接删除是安全的。

### 对第 4 点的回答

我同意该规则，并进一步建议:

- 不只是 `global_meta.json` 里用 first-seen wins
- 各 task 的 `info.csv` 里也要把同一个 `pert_id` 的 `smiles` 统一替换成 canonical first-seen value

这样才是彻底按 `pert_id` 对齐，而不是只有 meta 对齐、task 内部仍各写各的。

## 下次继续时应执行的修改

### 已明确可以改的

1. 从 `ptv3_extra_baseline` 中删除缺失 info 的 3 个样本:
   - `S1_B`
   - `S1_CAC`
   - `S1_O`

2. 对 `ptv3` 全局 `pert_id -> smiles`:
   - 以第一次出现为准
   - 按 `pert_id` 统一回写到各 task 的 `info.csv`
   - 手性冲突不再保留为阻断问题

### 尚未最终拍板的

1. metadata-only 的 extra tasks 是否继续保留“空表达结构”

备选方案:

- 方案 A: 保留当前 `(n, 0)` 空矩阵输出
- 方案 B: 改为 `info-only`，不再生成空矩阵 / 空 dict

用户暂时离开前，尚未对这个点做最终确认。

## 建议的下次操作顺序

1. 修改 `utils/00_standardize_rawdata.py`
   - 删除 extra baseline 中缺失 info 的 3 个样本
   - 增加 canonical `pert_id -> smiles` 回写逻辑

2. 根据用户最终确认，决定:
   - 保留空表达结构
   - 或切换为 `info-only`

3. 重新运行:

```bash
source ~/.bashrc
conda activate flow_v2
python utils/00_standardize_rawdata.py
python utils/01_validate_standardized_outputs.py
```

4. 更新:

- `data/standardized/file_audit.json`
- `data/standardized/ptv3/global_meta.json`
- 本文档

## 2026-04-27 14:47 HKT Protein Embedding Fallback Update

- 修改 `utils/04_build_embeddings_from_global_meta.py` 的 protein embedding 逻辑。
- 对于 `global_meta.json["protein_index"]` 中无法从 FASTA 映射到 sequence 的 protein，以及 `control` / `no`，不再跳过 embedding 生成。
- 这些条目现在使用空字符串 `""` 作为 sequence 输入 ESM，从而保证每个 `protein_index` 条目都会经过模型并写入 embedding matrix。
- 输出 payload 中保留 `unresolved_items`，但当所有行都经过模型时该字段为空；新增 `sequence_fallback_items` 记录哪些 protein 使用了 empty-sequence fallback。
- 使用 fake tokenizer/model smoke test 验证：缺失 sequence、`control`、`no` 都生成非零 embedding row，且 matrix shape 保持和 `protein_index` 长度一致。

## 2026-04-27 14:51 HKT Protein Embedding Count Checker

- 新增 `utils/07_check_protein_embedding_count.py`，用于检查 protein embedding pickle 中 `embedding_matrix` 的行数是否等于 `global_meta.json["protein_index"]` 的 entity 数量。
- 脚本默认检查 `data/training_ready/ptv3/global_meta.json` 和 `data/training_ready/ptv3/derived/protein_embedding_esm.pkl`，也支持通过 `--global-meta` 和 `--embedding-pkl` 指定其他路径。
- 当前 PTV3 结果检查通过：`protein_index` count 为 `11345`，`embedding_matrix` row count 为 `11345`。

## 2026-04-27 15:03 HKT Protein Embedding Dimension Clarification

- 复查代码中关于 protein embedding shape 的假设，没有发现运行时代码把 protein embedding feature dimension 固定为 `1024`。
- 当前命令中的 `--max-length 1024` 是 ESM tokenizer 的 protein sequence 输入长度上限，不是 embedding feature dimension。
- 当前使用的 `facebook/esm2_t33_650M_UR50D` 模型 hidden size 为 `1280`，因此 `protein_embedding_esm.pkl` 中 `embedding_matrix` shape 为 `(11345, 1280)` 是合理的。
- 更新 `utils/04_build_embeddings_from_global_meta.py`，在新生成的 protein embedding payload 中写入 `embedding_dim` 和 `max_length`，并在 CLI help 中说明 `--max-length` 不是 embedding dimension。
- 更新 `utils/07_check_protein_embedding_count.py`，输出完整 matrix shape 和 feature dimension，并在 payload 包含 `embedding_dim` 时校验该值与矩阵列数一致。

## 2026-04-27 15:15 HKT Drug / Graph Coverage Fix

- 修改 `utils/04_build_embeddings_from_global_meta.py` 的 drug embedding 逻辑：对于缺失 SMILES、无效 SMILES、以及 `no`，使用空 SMILES `""` 生成 RDKit fallback molecule，不再跳过这些 `pert_index` 条目。
- drug embedding payload 中 `unresolved_items` 现在在所有行都生成时为空；新增 `smiles_fallback_items` 记录使用 fallback 的 perturbation。
- 将 drug embedding 的 Morgan fingerprint 生成切换到 RDKit `GetMorganGenerator`，避免旧 API 的批量 deprecation warning。
- 修改 `utils/05_build_graph_matrices_from_global_meta.py` 的 DDI 逻辑：为每个 `pert_index` 条目生成 fingerprint，包括 fallback empty-SMILES fingerprint；fallback-vs-fallback 的非对角相似度强制为 `0.0`，避免缺失 SMILES 之间产生虚假相似性，diagonal 保持 `1.0`。
- PPI builder 现在在外部 edge 无法映射到 metadata protein space 时，仍写出完整 shape 的 zero matrix 和 `.meta.json` warning，而不是直接报错不产出文件。
- PDI builder 的 `.meta.json` 增加 `pert_count`、`protein_count`、mapped counts、和 `matched_link_count`，用于核对矩阵覆盖范围。
- 已重新生成 `data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl`，检查通过：`pert_index` count 为 `6113`，embedding matrix shape 为 `(6113, 2048)`。
- 已生成 `data/training_ready/ptv3/derived/ddi_matrix.npy` 和 `ddi_matrix.meta.json`，检查通过：DDI matrix shape 为 `(6113, 6113)`，fingerprint count 为 `6113`。

## 2026-04-27 16:12 HKT Self-Contained PDI Builder Update

- 修改 `utils/05_build_graph_matrices_from_global_meta.py` 的 PDI subcommand，使其可以直接从 `--stitch-db-dir` 解析 STITCH 资源。
- 默认目录为 `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/stitch_db`，会自动查找 `protein_chemical.links.detailed.v5.0.parquet`、`chemicals.inchikeys.v5.0.tsv`、以及 `uniprot_to_string.db`。
- 新增 parquet row-group streaming，避免读取 28G STITCH links parquet 到内存；仍保留 delimited TSV/CSV 输入兼容。
- 新增 SQLite mapping loader，从 `mapping(alias, string_protein_id)` 表直接生成 UniProt -> STRING protein ID 映射，不再必须预先导出 `protein-node-mapping-json`。
- `--links-tsv` 改为兼容参数；新增 `--links-path`、`--stitch-db-dir`、`--protein-mapping-db`。
- README 的 PDI 运行说明已更新为单目录 STITCH 资源流程。
- 验证：`py_compile` 通过，`pdi --help` 正常；使用最小 synthetic STITCH 目录 smoke test 验证 PDI 输出 shape、score 写入、和 `.meta.json` matched count 正确。

## 2026-04-27 16:40 HKT Graph Matrix Distribution Visualizer

- 新增 `utils/08_visualize_graph_matrix_distributions.py`，用于可视化 PPI、PDI、DDI matrix value distribution。
- 脚本逐个加载矩阵，分别绘制 all-value sample 和 nonzero-value sample 的 histogram，并额外输出 nonzero overlay comparison。
- 输出 summary JSON / CSV，包含 shape、zero fraction、nonzero count、nonzero row/column count、exact min/max/mean/std，以及 sampled percentiles。
- 默认读取 `data/training_ready/ptv3/derived/{ppi,pdi,ddi}_matrix.npy`，默认输出到 `data/training_ready/ptv3/derived/graph_value_distributions/`。
- 已在当前 PTV3 artifacts 上执行并生成图和统计：
  - PPI zero fraction `0.991346`，nonzero p99 `0.999000`
  - PDI zero fraction `0.994454`，nonzero p99 `0.958000`
  - DDI zero fraction `0.008558`，nonzero p99 `0.224490`

## 2026-04-27 17:03 HKT PPI No-Threshold Update

- 根据用户要求，删除 `utils/05_build_graph_matrices_from_global_meta.py` 中继承自旧脚本的 PPI confidence/reference filter。
- 现在 PPI 不再应用 `fused >= 0.30`、`combined_score >= 400`、或 low-textmining-only edge drop。
- PPI 仍然只保留可以映射到 `global_meta.json["protein_index"]` 的 edge，并丢弃 self-loop；这是矩阵对齐和图定义所需，不是 score threshold。
- `--topk` 仍作为显式参数保留，用于每个 protein 只保留最强 K 个邻居；默认值改为 `0`，表示不做 top-k pruning。
- 新生成的 `ppi_matrix.meta.json` 会写入 `score_filter.applied: false`，明确说明没有 PPI score threshold/filter。
- 同步更新 `README.md` 和 `scripts/0427_1.sh`，默认 PPI 命令不再传入 `--topk 100`。
- 验证：`py_compile` 通过；synthetic PPI smoke test 使用 `combined_score=100` 的低分 edge，输出矩阵保留该 edge 并写入 `0.1`。

## 2026-04-27 17:07 HKT Graph Distribution Visualizer README Tutorial

- 在 `README.md` 中新增 `utils/08_visualize_graph_matrix_distributions.py` 使用教程。
- 教程包含默认 PTV3 运行命令、输出文件列表、只绘制部分矩阵、调整 sample size、以及覆盖单个 matrix path 的示例。
- 补充说明 all-value histogram 会被 sparse matrix 的 zero 值主导，实际选择 edge-weight transformation 时应重点查看 nonzero histogram 和 summary CSV。

## 2026-04-27 17:12 HKT PPI Source Score Floor Review

- 复查重新生成后的 `ppi_matrix.meta.json`，确认 `score_filter.applied: false` 且 `topk: 0`。
- 复查 raw PPI 输入文件 `/root/beam_wuhao/H100/vcc_data/westlake/20250410_6508308PPI_protein_links_detailed_v12_both_prot1&2_.csv`，发现 `combined_score` 原始最小值为 `150`。
- 在 metadata protein space 过滤后的 raw PPI row 中，`combined_score` 最小值仍为 `150`。
- 因为 PPI builder 按 STRING-like score `/ 1000` 写入矩阵，所以 source data 的 `150` 会变成 nonzero matrix floor `0.15`；这不是代码中的 threshold。
- 修改 PPI 输出 metadata：未来重新生成的 `ppi_matrix.meta.json` 会包含 `score_summary`，记录 raw / scaled `combined_score` min/max 和 fused score min/max，避免再次混淆 source floor 与代码过滤。
- 验证：synthetic PPI smoke test 使用 `combined_score=50`，矩阵保留该 edge 并写入 `0.05`。

## 2026-04-27 20:56 HKT Process 3/4 Split, Training, and Inference Implementation

- 新增 `utils/09_build_data_splits.py`，基于 `docs/Data_Process_3.md` 为 training-ready PTV1/PTV3 tasks 生成 train/valid/test split artifacts。
- split row index 均为 task `feature_table` 行号；anchor rule 为 `not is_control and source_row_role == self and feature_membership == primary`，control row 通过 `set_info.pkl` 关联。
- 已生成 `data/training_ready/split_build_manifest.json` 及每个 task 的 `split_manifest.json`；覆盖 random、cell、cell_type、pert_stratified、5-fold、all_train_subset_test、fixed PTV1、test_only 等策略。
- 新增 `dataset/training_ready_dataset.py`、`model/training_ready_models.py`、`model/training_ready_lightning.py`、`train.py`、`infer.py`，用于 Step 4 training/inference。
- Step 4 模型注册仅保留要求的 6 个 model name，并统一支持 double-drug 输入；graph 模型只支持 PDI，PDI matrix 方向为 `[drug, protein]`。
- 训练代码支持 training-ready artifacts、Step 3 split artifacts、protein/drug embeddings、PDI graph matrix、dry-run batch check、checkpoint manifest、以及 finite-mask expression/BCE losses。
- 推理代码支持 extra single/double drug style tasks，输出 `predictions.parquet`、`metrics.json`、可选 `expression_pred.npy`、和 `run_manifest.json`。
- 已通过 `py_compile`、`train.py --help`、`infer.py --help`、3 个 model dry run、1-batch training smoke test、以及 `ptv3_extra_doubledrug_guomics` full inference smoke test。
- 详细实现说明、验证命令、以及未明确决策点已记录在 `docs/data_process_summary_04.md`。

## 2026-04-27 21:06 HKT Process 3/4 Compliance Review and Process 3 Summary

- 新增 `docs/data_process_summary_03.md`，单独记录 Process 3 split builder 的实现细节、输出文件、split policy、label coverage、skipped anchor、验证命令、和需要用户确认的点。
- 复查 `docs/Data_Process_3.md` 与 `utils/09_build_data_splits.py` 的对应关系：主要 split policy 已实现并可运行，但 label coverage 当前是 audit 不是 hard error。
- 复查 split manifests：当前被检查的 valid anchors label 都不缺失；但 `ptv1_aivc` 有 923 个 candidate anchors 因 missing control 被跳过，`ptv3_main_singledrug` 有 158 个 candidate anchors 因 missing control 被跳过。
- 记录重要假设：double-drug `pert_id` 5-fold 当前按 ordered `pert_id1 + pert_id2` pair 划分；PTV3 extra singledrug 当前按 Step 4 inference 需求作为 `test_only`。
- 复查 `docs/Data_Process_4.md` 与 Step 4 代码：训练/推理入口、6 个模型名、PDI-only graph、double-drug 输入、去除 `embedding_methods` 和 `inverse_machine_id` 已实现；但模型是 compact compatible implementation，不是逐行复刻 legacy model architecture。
- 新增 review summary：`data/review_summary/2026-04-27_2106_process3_process4_compliance_review.md`。

## 2026-05-07 15:20 HKT User Decision Follow-up

- 根据用户确认，missing-control anchors 继续跳过，不修改 Process 2 强制补 control。
- 根据用户确认，double-drug `pert_id` 5-fold 继续使用 ordered `pert_id1 + pert_id2` combination 作为 unseen unit。
- 根据用户确认，所有 extra tasks，包括 single-drug 和 double-drug，继续保持 `test_only`。
- 根据用户确认，target UniProt IDs 缺失于 `protein_index` 时继续直接丢弃。
- 修改 `utils/09_build_data_splits.py`：PTV1 `fixed_experiment_type` split 现在在 Step 3 直接解析 `data/rawdata/ptv1/experiment_type_list`，不再依赖 training-ready `data_split` column；同时为 `ptv1_aivc` 增加 `random` split。
- 重新生成 `data/training_ready/split_build_manifest.json` 和所有 task 的 split manifests；`ptv1_aivc` 当前策略包括 `fixed_experiment_type`、`random`、`pert_id_5fold_fold0..4`、`all_train_subset_test`。
- 修改 `train.py`：训练继续用 validation set 做 checkpoint selection，fit 结束后默认用 best validation checkpoint 执行 test；新增 `--skip-test` 和 `--limit-test-batches`。
- smoke test 验证 best-checkpoint 流程：`ptv3_main_doubledrug` 1 train batch / 1 val batch / 1 test batch 成功从 `/tmp/proteintalk_v2_checkpoints/smoke_best_ckpt_test/epoch=0.ckpt` restore 后测试。

## 2026-05-07 16:22 HKT Legacy Behavior Gap Review

- 根据用户确认，split label coverage 继续只记录部分缺失；但当某个 task 的 checked label column 整列缺失或全空时，`utils/09_build_data_splits.py` 现在会发出 `RuntimeWarning`，并在 `split_manifest.json` 中记录 `all_labels_missing`、`missing_anchor_fraction` 和 warning message。
- 已重新生成所有 split manifests；当前 checked label columns 均为 `ok`，没有整列全空 label。
- 复查 current dataset/dataloader、trainer/metrics、`train.py`、`infer.py` 与 legacy single/double 版本的差异，详细记录到 `data/review_summary/2026-05-07_1622_legacy_behavior_gap_review.md`。
- 复查结论：当前 training-ready dataset sampling contract 接近 legacy，但模型和 metrics 仍不是 original-behavior preserving。
- 关键 blocker：`model/training_ready_models.py` 当前使用 compact shared `DoubleDrugContextModel`，不等价于 legacy Transformer token/CLS 架构、target-token/protein-gate 架构、legacy `baseline_emb_v3`，也不等价于 original PyG PDI graph network。
- 已确认 `flow_v2` 环境包含 `torch_geometric 2.7.0`，后续可以按 legacy `create_pdi_only_graph` + `PDIOnlyProteinDrugNet` 方向移植；当前 `[drug, protein]` PDI artifact 需要默认 transpose 成 legacy `[protein, drug]` 后构图。

## 2026-05-07 16:43 HKT Legacy Model Port and Entrypoint Review

- 修改 `model/training_ready_models.py`：移除 compact shared `DoubleDrugContextModel`，改为 legacy-style token/CLS Transformer internals；恢复 legacy `ValueEmbedding`、stacked batch-cov embedding、target protein token path、gene/protein gate path、以及 double-drug response/synergy heads。
- 修改 graph model path：恢复 legacy `create_pdi_only_graph` + `PDIOnlyProteinDrugNet`；当前 `[drug, protein]` PDI artifact 默认按 `reverse_pdi=True` transpose 成 legacy `[protein, drug]` 后构图。
- 修改 `baseline_emb_v3`：不再使用 compact Transformer 替代实现，改为 legacy baseline embedding dataset + fusion MLP 结构，并为 current double-drug trainer 增加 synergy head；该模型要求 gene embedding `.npy` 的 row order 与当前 feature table 对齐。
- 修改 `train.py` 和 `infer.py`：新增 legacy model 所需 CLI 参数，包括 `--fusion-mode`、`--num-heads`、`--num-layers`、`--cls-type`、`--graph-dropout`、`--target-protein-fusion-model`、`--gate-weight`、`--pdi-input-orientation`、`--emb-dataset-path`、`--gene-emb-dim`；`--hidden-dim` 默认改回 legacy `256`。
- 重新复查 dataset class、dataloader construction、`train.py`、`infer.py` 与 legacy 差异，记录到 `data/review_summary/2026-05-07_1643_legacy_model_port_and_entrypoint_review.md`。
- 验证：`py_compile` 通过；六个 selected model 在 small synthetic double-drug batch 上 forward 通过；真实 PTV3 single/double dataloader batch 可读取并可构建对应模型。
- 仍需注意：current trainer metrics 仍比 legacy metrics 少；current dataset 会先把 control expression NaN 转成 0；legacy attention 对当前 PTV3 全量约 11k protein tokens 做 full attention，实际训练可能需要 top-k protein 或更大 GPU memory。

## 2026-05-07 17:09 HKT Training Metrics, NaN Semantics, and Forward Patch

- 修改 `dataset/training_ready_dataset.py`：移除 control expression 的 `nan_to_num(..., 0.0)` 预处理；control 和 perturb expression 现在都按 training-ready matrix 原值进入 batch，使 legacy-style `ValueEmbedding.nan_embedding` 负责 NaN 输入。
- 修改 `model/training_ready_lightning.py`：恢复 legacy `utils/eval_dd.compute_validation_metrics` 风格的 validation/test metric suite，包含 task1/task2 AUROC、AUPRC、ACC，以及 expression `mse_all`、`mae_all`、`pcc_all`、`r2_all`、top50、delta、MMD、energy distance 等指标。
- 同步调整 Lightning loss path，更接近 legacy double-drug trainer：expression MSE 只按 perturb label NaN mask 过滤，BCE 使用 label mask 权重 `1 - mask` 后求和除以有效权重。
- 修改 `model/training_ready_models.py`：修正 `attention_v10_hetero_cls_ee_no_target` graph double-drug forward 中 perturb token 逻辑；按 legacy double model，非 `mlp` 模式保留两个 graph drug tokens，`mlp` 模式才逐 drug 过 `pert_proj`。
- 修改 target-model `fusion_mode=add` 分支：非 graph target models 不再把 target token mean 加入 add context，保持 legacy target / target_proemb forward 行为；graph gate model 仍保留 target mean add context。
- 新增复查记录：`data/review_summary/2026-05-07_1709_training_metrics_nan_forward_review.md`。
- 验证：`py_compile` 通过；dataset NaN preservation smoke 通过；`ValueEmbedding` NaN input smoke 通过；legacy-style metric smoke 通过；graph no-target double-drug forward smoke 通过；`train.py --help` 和 `infer.py --help` 通过。
- 仍需注意：legacy selected target/gate models在旧代码中没有完整 double-drug class；当前实现保持单药 legacy forward 主体，并把 perturb input 适配到当前统一 double-drug contract。若需要逐模型完全冻结行为，下一步应按 model name 拆成更显式的 per-model legacy adapter。

## 2026-05-07 17:33 HKT Trainer Options, Model Consolidation, and Inference Metrics

- 根据用户新确认，保留 dataset storage divergence 和 pre-tokenized covariates/drug ids；不再拆 per-model legacy adapters；不实现 top-k protein selection。
- 修改 `model/training_ready_lightning.py`：新增 legacy `FocalLossWithAlpha`、task-specific positive weights、`adamw_fused` / `adamw_fused_<factor>` optimizer path、`step` scheduler、`cosine_warmup` scheduler、以及 `UnfreezeCallback`。
- 修改 `train.py`：新增 `--focal-loss`、`--positive-weight`、`--positive-weight1`、`--positive-weight2`、动态 `--optimizer-name`、`--scheduler-name {cosine,step,plateau,cosine_warmup}`、`--unfreeze-at-epoch`、`--unfreeze-layer-name`、`--gradient-clip-val`；run manifest 记录这些训练参数。
- 修改 `model/training_ready_models.py`：active model registry 现在只保留 `attention_v10_hetero_cls_ee` 和 `baseline_emb_v3`；`attention_v10_hetero_cls_ee` 是 consolidated PDI hetero graph model，`--use-target` 控制 target protein tokens，`--target-protein-fusion-model gate` 控制 gene/protein gate。
- 修改 `infer.py`：model choices 同步缩减为两个；新增 `--use-target`；推理 loop 中直接收集 labels/masks，避免推理后重复读取 dataset；当 `--save-expression-pred` 启用时，额外写入 full `legacy_validation_metrics`。
- 新增复查记录：`data/review_summary/2026-05-07_1733_legacy_trainer_options_model_consolidation_infer_metrics.md`。
- 验证：`py_compile` 通过；`train.py --help` 和 `infer.py --help` 通过；trainer optimizer/scheduler/focal smoke 通过；consolidated PDI graph no-target、target、target+gate forward smoke 通过。

## 2026-05-07 19:06 HKT Training Stack Bug Fixes

- 修改 `dataset/training_ready_dataset.py`：`encode_binary_label` 现在正确识别 numeric `0.0` / `1.0` 和 string `"0.0"` / `"1.0"`；修复 `ptv1_extra_singledrug` 的 float label 被当作 missing mask 的问题。
- 修改 `dataset/training_ready_dataset.py`：eval/test/inference mode 不再随机选择 control row，改为稳定选择排序后的第一个 control row；train mode 仍保留随机 sampling。
- 修改 `train.py`：修复 valid indices fallback 到 test indices 时仍加载 valid `set_info` 的问题；run manifest 新增 `valid_source`。
- 修改 `train.py`：checkpoint 初始化默认 strict load；新增 `--allow-partial-checkpoint-load` 作为显式 opt-in。
- 修改 `train.py`：新增 `--strategy`；当 graph model 使用 multi-device 且 strategy 为 `auto` 时自动使用 `ddp_find_unused_parameters_true`，避免 unused modules 导致 DDP failure。
- 修改 `infer.py`：checkpoint load 默认 strict；新增 `--allow-partial-checkpoint-load`；新增 checkpoint `run_manifest.json` config validation，默认阻止 model/config mismatch，`--allow-checkpoint-config-mismatch` 仅用于显式迁移/调试。
- 修改 `model/training_ready_models.py`：`baseline_emb_v3` 不再 clamp feature row index 到 embedding row 范围；embedding row count 不覆盖当前 feature index 时直接报错。
- 新增 `scripts/0507_training_stack_smoke.py`，覆盖 numeric label encoding、deterministic eval dataset、active model registry、以及 consolidated PDI graph no-target / target / target+gate forward。
- 验证：`py_compile` 通过；`train.py --help` 和 `infer.py --help` 通过；`scripts/0507_training_stack_smoke.py` 通过。

## 2026-05-07 20:25 HKT Remove Group-Size Tensor Dimension

- 修改 `dataset/training_ready_dataset.py`：彻底移除 `group_size` 参数和 per-item group sampling；每个 dataset item 现在只返回一个 control row 和一个 perturb row。DataLoader 后 expression shape 为 `(batch_size, n_genes)`，graph `pert_id` shape 为 `(batch_size, 2)`，embedding `pert_id` shape 为 `(batch_size, 2, drug_embedding_dim)`。
- 修改 `model/training_ready_models.py`：移除 model/build_model 中的 `group_size` 参数和所有 `bs * gs` reshape；consolidated PDI Transformer 现在直接处理 `(batch_size, n_tokens, hidden_dim)` token sequence，输出 expression `(batch_size, n_genes)`、response logits `(batch_size, 1)`、synergy logits `(batch_size, 1)`。
- 修改 `model/training_ready_lightning.py`：loss、validation/test collection、BCE mask handling、和 metric aggregation 不再索引旧 group axis；mask shape mismatch 现在直接报错。
- 修改 `train.py` 和 `infer.py`：删除 `--group-size` CLI、manifest/config plumbing、dataset/model constructor 参数、以及 inference 中的旧 `[:, 0]` group-axis slicing。
- 修改 `scripts/0507_training_stack_smoke.py`：更新为 no-group shape regression，并新增 tiny Lightning trainer fit，覆盖 train/validation loss和 metric aggregation。
- 更新 `docs/data_process_summary_04.md`：Step 4 summary 现在记录 no-group dataset/model contract 和新的实测 dry-run shapes。
- 新增复查记录：`data/review_summary/2026-05-07_2025_group_size_dimension_removal_review.md`。
- 验证：代码中 `group_size` / `--group-size` 搜索已无命中；`py_compile` 通过；`train.py --help` 和 `infer.py --help` 通过且无 `--group-size`；synthetic smoke trainer fit 通过；真实 `ptv3_main_doubledrug` dry-run 输出 `(2, 11092)` / `(2, 1)` / `(2, 1)`；真实 `ptv3_main_singledrug` dry-run 输出 `(2, 10982)` / `(2, 1)` / `(2, 1)`。
- 注意：真实 `ptv3_main_doubledrug` one-batch CPU fit 已跑过 sanity validation 和 metric aggregation，但在第一步 training backward 被系统 kill，exit 137；这与本地 CPU memory/backward through full PDI graph 更一致，不是 shape mismatch。

## 2026-05-08 16:52 HKT Train Pipeline / Dimension / GPU Review

- 复查 `docs/` 下所有流程文档，并对 stage-1 / stage-2 validator、split CLI、`train.py`、`dataset/training_ready_dataset.py`、`model/training_ready_models.py`、`model/training_ready_lightning.py` 做训练链路检查。
- 修改 `model/training_ready_lightning.py`：`safe_mean` 现在在输入全为非有限值时直接返回 `NaN`，避免 one-sample / one-class validation 中出现 `RuntimeWarning: Mean of empty slice`。
- 验证 `utils/01_validate_standardized_outputs.py` 和 `utils/03_validate_training_ready_outputs.py` 均通过；当前 PTV3 关键维度为 protein embedding `(11345, 1280)`、drug embedding `(6113, 2048)`、PDI `(6113, 11345)`、PPI `(11345, 11345)`、DDI `(6113, 6113)`。
- 验证 `attention_v10_hetero_cls_ee` 默认配置在 H200 GPU 上可完成 `ptv3_main_doubledrug` batch size `1` 的 one-train-batch + one-validation-batch smoke run；输出维度和 metric aggregation 正常。
- 验证 `baseline_emb_v3` 在当前默认 external Geneformer embedding path 下可完成 batch size `1` 的 one-batch smoke run。
- 新增复查记录：`data/review_summary/2026-05-08_1652_train_pipeline_dimension_gpu_review.md`。

## 2026-05-08 17:36 HKT Loss Contract / Legacy Trainer Recheck

- 根据用户澄清重新复查旧 trainer：旧 single-drug trainer 是 expression MSE + 一个 label BCE；旧 double-drug trainer 支持 expression MSE + response/synergy 两个 BCE head，但当前 main task 需求应按一个 active task label 来解释 `loss2`。
- 修正 `model/training_ready_lightning.py`：新增 active task loss config，`loss1` 现在明确是 expression MSE，`loss2` 是 active task BCE；`val/auroc` / `val/auprc` / `val/acc` 现在指 active task metric，同时保留 `response_auroc` / `synergy_auroc` 显式 head metric。
- 修正 `train.py`：新增 `--task-head {auto,response,synergy}`、`--task-label-key`、`--task-mask-key`、active `--bce-weight`，并在 run manifest 记录 active task loss config；默认 double-drug 使用 synergy，single-drug 使用 PRISM response。
- 修正 `infer.py`：使用同一 active task config，输出 `task` metrics、`pred_task_prob` 和 `task_label`，同时保留 response/synergy metrics block。
- 验证：`py_compile` 通过；`scripts/0507_training_stack_smoke.py` 通过；在 H200 上重新跑 corrected 8-epoch bounded real-data training。
- corrected double-drug run：`task_head=synergy`，train `loss1 37.513 -> 1.024`，val `loss1 2.548 -> 0.244`，val `loss2 1.069 -> 0.633`，full-valid active AUROC `0.489`，`pred_task_prob` 仍近似常数。
- corrected single-drug run：`task_head=response`，train `loss1 39.552 -> 1.520`，val `loss1 2.385 -> 0.877`，val `loss2 0.947 -> 0.677`，full-valid active AUROC `0.423`，`pred_task_prob` 仍近似常数。
- 新增复查记录：`data/review_summary/2026-05-08_1736_loss_contract_legacy_recheck.md`。

## 2026-05-08 18:00 HKT Double-Drug Train Split Single-Drug Merge Fix

- 复查 double-drug / single-drug 训练时间差异：Stage-2 已经把 `ptv3_main_singledrug` processed rows 合并进 `ptv3_main_doubledrug` feature table，当前 double feature table 为 `20764` rows，其中 `18568` rows 是 `feature_membership="merged_single_drug"`。
- 问题在 Step3 split：原 anchor rule 只使用 `feature_membership="primary"`，所以 double-drug train indices 只有 native double-drug anchors；`pert_id_5fold_fold0` 之前 train 只有 `1289`。
- 修改 `utils/09_build_data_splits.py`：对 `ptv3_main_doubledrug`，pairing metadata 同时纳入 `primary` 和 `merged_single_drug` anchors；native double-drug rows 仍决定 fold / valid / test，merged single-drug anchors 只追加到 train。
- 更新 `docs/Data_Process_2.md` 和 `docs/Data_Process_3.md`，记录 double-drug feature-table merge requirement、train-only single-drug split rule、以及重建命令。
- 已用 `utils/09_build_data_splits.py --dataset-group all` 重新生成完整 split set，保证 `split_build_manifest.json` 仍包含全部 `13` 个 tasks；`ptv3_main_doubledrug/pert_id_5fold_fold0` train `19275` = `17986` merged single-drug anchors + `1289` native double-drug anchors；valid/test 仍只包含 native double-drug anchors (`142` / `360`)。
- 验证：`py_compile` 通过；split manifest label coverage 只检查 native double-drug synergy anchors，状态为 `ok`；`train.py` dry run 在更新后的 split 上通过，输出 expression `(2, 11092)`、response logits `(2, 1)`、synergy logits `(2, 1)`。
- 注意：当前默认 double-drug `task_head=synergy` 下，merged single-drug rows 进入 train 后会贡献 expression `loss1`；其 `PRISM1st_label_total` 存在于 batch 中，但不会进入 double-drug `loss2`，除非后续启用/实现 mixed row-level response+synergy BCE。
- 新增复查记录：`data/review_summary/2026-05-08_1800_double_train_single_merge_review.md`。

## 2026-05-08 20:22 HKT DDP 8-GPU Training and Hyperparameter Verification

- 复查当前 Stage-4 training architecture：`train.py` 负责 dataset/model/Lightning trainer wiring；`dataset/training_ready_dataset.py` 输出 no-group batch contract；`model/training_ready_models.py` 当前 active models 为 `attention_v10_hetero_cls_ee` 和 `baseline_emb_v3`；`model/training_ready_lightning.py` 负责 `loss1` expression MSE、active-task `loss2` BCE/focal BCE、validation/test metrics；`infer.py` 使用同一 active-task metric contract。
- 修正 `model/training_ready_lightning.py`：validation/test epoch loss logging 现在把 CPU loss tensors 移动到 module device 后再 `sync_dist=True`，避免 NCCL 对 CPU tensor all-reduce 报错。
- 修正 `model/training_ready_lightning.py`：validation/test metric aggregation 现在按字段跨 DDP ranks `all_gather`，再用 `row_index` 去重，避免 rank-local metrics 和 DistributedSampler padding 影响 full-validation metric。
- 修正 `model/training_ready_lightning.py`：`compute_validation_metrics` 现在只用 finite expression prediction/target pairs，并把 non-finite control expression 清理为 `0.0`；BF16 eval gather 时 floating tensors 会先 cast 到 float32 再转 CPU/NumPy。
- 修正 `infer.py`：`--save-expression-pred` 路径不再提前把 expression target NaN 填 0，而是把 raw target 交给 shared `compute_validation_metrics`，保持 inference 与 Lightning metrics 语义一致。
- 完成 8 H200 DDP bounded run：`20260508_codex_ddp8_dd_8ep_50b_final`，`--devices 8`，`--strategy ddp_find_unused_parameters_true`，`batch_size=2`，`max_epochs=8`，`limit_train_batches=50`，full validation，best checkpoint 为 `checkpoints/20260508_codex_ddp8_dd_8ep_50b_final/epoch=7.ckpt`，best `val/total_loss=1.0077422857284546`。
- 8-GPU run 主要趋势：`train/loss1_epoch 39.6103 -> 1.2408`，`val/loss1 2.7357 -> 0.2471`，`val/total_loss 3.5367 -> 1.0077`，`val/mse_all 1.8237 -> 0.1646`，`val/pcc_all 0.9820 -> 0.9983`，active synergy `val/auroc 0.5552 -> 0.6891`，`val/auprc 0.4286 -> 0.5417`；`val/loss2` early improved to about `0.6366` / `0.6419` but final epoch rose to `0.7607`。
- Hyperparameter smoke 验证通过：2-GPU `bf16-mixed` + target gate + graph dropout；no-MSE focal BCE + `adamw_fused` + cosine；`fusion_mode=add` / `perturb_fusion_mode=concat` + SGD + step；`baseline_emb_v3` + Adam + plateau；response task head + `adamw_fused_0.5` + cosine warmup。
- 诊断性失败：`--pdi-input-orientation protein_by_drug` 与默认 PDI artifact shape `(6113, 11345)` 不匹配，代码按预期报错要求 `(11345, 6113)`；该参数需要配套转置后的 PDI matrix path。
- 验证：`py_compile` 通过；TensorBoard scalars 已从上述 logs 解析；最终 `nvidia-smi` process query 无 active GPU compute process。
- 新增复查记录：`data/review_summary/2026-05-08_2022_ddp8_multigpu_training_hyperparam_review.md`。

## 2026-05-08 21:24 HKT Double-Drug Loss2 Pipeline Correction

- 纠正上一轮错误修改：PTV3 training-ready expression matrix 必须继续按 `docs/Data_Process_2.md` 使用每个 task 的 source protein union/full ordered protein axis，不能使用 top-k / top2k protein axis。

## 2026-05-08 22:18 HKT full protein axis / batch covariate / single-drug perturb slot fix

- 修改 `utils/00_standardize_rawdata.py`：single-drug rows 的 `pert_id2`
  不再写空值，改为复制 `pert_id1`；`ptv1_aivc` 中原本为空的
  `pert_id2` 也复制 `pert_id1`。extra baseline control rows 写为
  `pert_id1 == pert_id2 == control`。
- 修改 `utils/02_build_training_ready_data.py`：Stage-2 在 index encoding
  前再次强制 single-drug rows 满足 `pert_id2 == pert_id1`，并把 `batch`
  加入 `DISCRETE_FIELDS` / `value_to_index` / `batch_index` 输出。
- 修改 `dataset/training_ready_dataset.py`、`train.py`、`infer.py`、
  `model/training_ready_models.py`：默认 batch covariates 现在包含 `batch`；
  `attention_v10_hetero_cls_ee` 默认启用 target protein tokens，可用
  `--no-use-target` 显式关闭。
- 更新 `docs/Data_Process.md`、`docs/Data_Process_2.md`、`docs/Data_Process_4.md`：
  记录 single-drug `pert_id2 == pert_id1`、`batch` covariate、full protein
  axis、`use_target` 默认开启，以及新 no-group contract 中永久忽略
  legacy `group_size` 维度。
- CPU-only 重新生成：
  - `utils/00_standardize_rawdata.py`
  - `utils/02_build_training_ready_data.py`
  - `utils/03_validate_training_ready_outputs.py`
  - `utils/09_build_data_splits.py --dataset-group all`
- 验证结果：
  - `ptv3_main_singledrug` feature matrix shape: `18568 x 10982`
  - `ptv3_main_doubledrug` feature matrix shape: `20764 x 11092`
  - `ptv1_aivc` feature matrix shape: `15002 x 5576`
  - `ptv1_extra_singledrug` feature matrix shape: `186 x 5576`
  - `batch_index` 已出现在检查的 regenerated feature tables 中。
  - `ptv3_main_singledrug`、`ptv3_extra_singledrug_mat1_480_faims`、
    `ptv1_extra_singledrug` 均无 `pert_id2 != pert_id1` mismatch；
    `ptv3_main_doubledrug` 中 merged single-drug rows 也无 mismatch。
  - PTV3 existing derived artifacts still match current meta:
    protein embedding `(11345, 1280)`、drug embedding `(6113, 2048)`、
    PPI `(11345, 11345)`、PDI `(6113, 11345)`、DDI `(6113, 6113)`。
    因为本次未改变 protein/pert index 的 size/order contract，PTV3
    embeddings/graph matrices 不需要重新生成。
  - `py_compile` 通过；`scripts/0507_training_stack_smoke.py` CPU smoke 通过，
    输出 `GPU available: False, used: False`。
- 纠正上一轮错误修改：training 只保留两个 loss，`loss1` 为 expression MSE，`loss2` 为当前 task 的单一 classification BCE；single-drug 的 `loss2` 使用 response/sensitivity，double-drug 的 `loss2` 使用 synergy，不再提供 dual-loss training path。
- 保留 double-drug data contract 修正：合并进 `ptv3_main_doubledrug` feature table 的 single-drug auxiliary rows 清空 active `synergy`，并用 `training_label_scope="single_drug_auxiliary_synergy_masked"` 标记；这些 rows 不参与 double-drug synergy `loss2`。

## 2026-05-09 12:44 HKT Experiment Support and Extra-Data Inference Smoke

- 修改 `train.py`：新增 `--pdi-mode {real,zero}`，用于 no-PDI ablation 在保持真实 PDI shape 的前提下将 PDI 置零；新增 checkpoint monitor 参数；run manifest 现在记录 split audit，包括 membership/source task 计数和 active label nonempty/empty 计数。
- 修改 `infer.py`：新增 `--pdi-mode` 和 `--limit-batches`；checkpoint config validation 只比较 architecture-affecting 字段，因此 extra validation 可以使用不同 label key 而不需要 `--allow-checkpoint-config-mismatch`；修复 `--limit-batches` 下 prediction metadata 仍使用全 split 导致的 DataFrame length mismatch，改为使用 batch 中实际处理的 `row_index`。
- 验证 all-single 数据路径：`ptv3_main_singledrug/all_train_subset_test` train `17986`，valid `1798`，test `3597`，train 全部为 nonempty `PRISM1st_label_total`。
- 验证 all-single+double 数据路径：`ptv3_main_doubledrug/all_train_subset_test` train `19777` = `17986` merged single-drug + `1791` native double-drug；active `synergy` 只在 `1791` native double rows 非空，merged single rows 为 empty/masked，不参与 double-drug `loss2`。
- 8 H200 DDP 2-epoch bounded smoke 全部通过：single `pert_stratified_5fold_fold0`、`cell_type_5fold_fold0`、`cell_5fold_fold0`、no-MSE、no-PDI、double `pert_id_5fold_fold0`、all-single for extra single、all-single+double for extra double。
- Extra inference smoke 全部通过：extra single `mat1_480_faims`、`mat1_qe`、`mat2_480_faims`、`mat2_qe`、`mat3_qe`、`mat4_qe`；extra double `nature`、`nc`、`guomics`。每个 inference smoke 使用 `--limit-batches 2` 并成功写出 prediction/metrics/run manifest。
- 验证：`py_compile` 通过；最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 新增复查记录：`data/review_summary/2026-05-09_1244_experiment_support_infer_smoke_review.md`。

## 2026-05-09 13:37 HKT Strategy Audit Guard Tightening

- 复查上一轮 strategy 后发现真实 loophole：`infer.py` checkpoint config validation 没有比较 `task_head`，因此 response checkpoint 可能被误用于 synergy inference。
- 修改 `infer.py`：checkpoint validation 现在比较 `task_head`、dataset/meta/embedding/PDI resolved artifact paths、`pdi_mode` 和 architecture-affecting args；错误 task head 默认 hard fail，除非显式使用 debug/migration 参数 `--allow-checkpoint-config-mismatch`。
- 修改 `infer.py`：对 non-graph/axis-fixed model 增加 `feature_ordered_protein_index.json` 精确一致性检查；对 `attention_v10_hetero_cls_ee` 这类 graph token model 允许 extra task 使用自己的 full protein axis，因为 protein tokens 由当前 task ordered protein index 和 shared protein embedding 重新构建。
- Audit 结论：extra single/double 的 full protein axes 与 main train task 不完全一致，但这对当前 graph attention strategy 是预期且可支持的；若改用 `baseline_emb_v3` 等 axis-fixed model，新的 axis guard 会阻止 silent mismatch。
- Deliberate negative test 通过：用 all-single response checkpoint 去跑 extra double synergy inference 时，`infer.py` 按预期报错 `task_head: current='synergy' checkpoint='response'`。
- Guarded extra inference 全部重跑通过：extra single `mat1_480_faims`、`mat1_qe`、`mat2_480_faims`、`mat2_qe`、`mat3_qe`、`mat4_qe`；extra double `nature`、`nc`、`guomics`。每个 output manifest 均为正确 `task_head` / label key，`limit_batches=2`，写出 4 prediction rows。
- 验证：`py_compile` 通过；strategy audit 通过；guarded inference output audit 通过；最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 新增复查记录：`data/review_summary/2026-05-09_1337_strategy_audit_guard_review.md`。

## 2026-05-09 14:11 HKT Split Validation Loophole Fix

- 复查 strategy 时发现第二个真实 loophole：`ptv3_main_singledrug/cell_type_5fold_fold*` 的 valid split 全部为 `0`，训练入口会 fallback 到 test split 做 validation，导致 cell-type 5-fold 实验存在 checkpoint selection leakage。
- 修改 `utils/09_build_data_splits.py`：新增 `validation_item_count()`，当 non-test groups/rows 数量足够时至少保留 1 个 validation group/row，同时保证 train 不被清空；`split_items()`、group 5-fold、stratified pert 5-fold、`all_train_subset_test` 使用同一 validation count helper。
- 更新 `docs/Data_Process_3.md`：记录正式训练 split 的 valid 不能为空，并明确 double-drug `pert_id_5fold_fold*` 当前是 ordered pair `pert_id1+pert_id2` 的 pair-level holdout。
- 已重新运行 `utils/09_build_data_splits.py --dataset-group all` 生成全部 split artifacts。
- 修复后 `ptv3_main_singledrug/cell_type_5fold_fold0..4` valid counts 分别为 `1177`、`1865`、`1865`、`1865`、`1865`，且 train/valid/test row overlap 均为 `0`。
- 新增 full strategy audit：检查 full protein axes 非 2000、expression matrix 与 ordered protein axis 一致、`batch_index` 存在、single-drug `pert_id2 == pert_id1`、double merged single rows 的 `synergy` active label 被 mask、extra single/double label coverage、所有目标 split family 的 train/valid/test 非空和 group leakage。
- 8-GPU smoke 验证修复后的 `cell_type_5fold_fold0`：`20260509_strategy_cell_type_valid_fix_smoke` 成功完成 1 epoch bounded run，run manifest 显示 `valid_source=valid`，counts `train=9059`、`valid=1177`、`test=7750`，overlap 全为 `0`。
- 验证：`py_compile` 通过；full strategy audit 通过；最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 重要定义：当前 double-drug `pert_id_5fold_fold*` 在代码和 manifest 中定义为 ordered pair `pert_id1+pert_id2` 的 pair-level holdout；不是 individual-drug-cold holdout。
- 新增复查记录：`data/review_summary/2026-05-09_1411_split_validation_loophole_review.md`。

## 2026-05-09 14:26 HKT Training Entrypoint Leakage Guard

- 再次 adversarial review 后发现第三个 loophole：即使 split artifacts 已修复，`train.py` 仍保留 empty-valid fallback 到 test 的运行时路径；如果未来传入错误/custom split，可能重新引入 test-as-validation checkpoint selection。
- 修改 `train.py`：formal training split 的 valid indices 为空时直接报错，不再 fallback 到 test；错误信息明确说明这是为了避免用 test split 选择 checkpoint。
- 发现另一个 reporting loophole：`all_train_subset_test` 的 valid/test 是 train anchors 的子集，不能作为 final internal test metric；如果用户忘记 `--skip-test`，`train.py` 可能输出 misleading internal test results。
- 修改 `train.py`：`--split-strategy all_train_subset_test` 必须同时使用 `--skip-test`，否则直接报错；最终 claims 必须用 `infer.py` 在 extra-data tasks 上评估。
- 验证：
  - `py_compile` 通过。
  - `cell_type_5fold_fold0` CPU dry run 通过，输出 expression `(1, 10982)`、response/synergy logits `(1, 1)`。
  - 人工 empty-valid split 触发 expected fail-fast error。
  - `all_train_subset_test` 不带 `--skip-test` 触发 expected fail-fast error；带 `--skip-test` 的 dry run 通过。
  - full strategy audit 通过；最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 新增复查记录：`data/review_summary/2026-05-09_1426_training_entrypoint_guard_review.md`。

## 2026-05-09 14:28 HKT Inference Manifest Strictness Guard

- 继续 adversarial review 后发现第四个 loophole：`infer.py` 遇到 checkpoint 目录缺少 `run_manifest.json` 时会跳过 config validation；如果 manifest 存在但缺少字段，也会跳过缺失字段。这可能让旧 checkpoint 或不完整 manifest 绕过 `task_head` / artifact / architecture 检查。
- 修改 `infer.py`：checkpoint 旁边默认必须存在 `run_manifest.json`；缺失 manifest 直接报错。只有显式使用 `--allow-missing-checkpoint-manifest` 时才允许 migration/debug 路径。
- 修改 `infer.py`：当前 inference config 中的 required keys 如果不在 checkpoint manifest 中，会计为 config mismatch；不再静默跳过。non-graph/axis-fixed model 还要求 manifest 中存在 `ordered_protein_index_path` 并做 axis 精确一致性检查。
- 验证：
  - `py_compile` 通过。
  - 缺失 manifest 的 inference command 触发 expected `FileNotFoundError`。
  - 不完整 manifest 触发 expected config mismatch，列出缺失 required keys。
  - manifest-backed extra single inference smoke 通过，成功写出 1 row prediction。
  - old-code semantic check 通过：旧版 `9_2_extrafindbs_dd_dataprocess.py` 把 double-drug `pert_id` 构造成 `pert_id1 + '+' + pert_id2`，旧版 `6_dd_datasplit.py` 再按该 `pert_id` 分组做 5-fold，因此当前 pair-level double-drug split 语义与旧版一致。
  - full strategy audit 通过；最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 新增复查记录：`data/review_summary/2026-05-09_1428_inference_manifest_strictness_review.md`。

## 2026-05-09 14:53 HKT Data/Training Pipeline Recheck and Full Experiment Script

- 按用户要求重新读取 `docs/Data_Process_1.md` 到 `docs/Data_Process_4.md`、`docs/Training_gudline.md`、本历史文件以及 `data/review_summary/` 中最新 review，确认当前 accepted constraints：full protein axis，禁止 top2000；single-drug `pert_id2 == pert_id1`；training 只有两个 loss；double-drug merged single rows 的 `synergy` label 必须 mask；`batch` 是默认 batch covariate；`use_target` 默认开启；`group_size` 永久忽略。
- 修改 `utils/03_validate_training_ready_outputs.py`，补强 training-ready validator：
  - `batch` 现在是 required discrete field，并要求 `global_meta.json` 中存在 `value_to_index["batch"]` 和 sentinel `no`。
  - 新增 full protein axis guard：feature/label axis 不能是 `2000`，且不得小于 standardized source 中可见 protein 数。
  - 新增 single-drug second-slot guard：single-drug task 中 non-control rows 必须 `pert_id2 == pert_id1` 且 `pert_index2 == pert_index1`；double-drug merged single rows 也执行同一检查。
- 修改 `docs/data_process_summary_04.md`，同步默认 `batch_cov_list`，加入 `batch`。
- 新增 `scripts/run_ptv3_training_experiments.sh`，覆盖当前计划中的 full experiment family：
  - single-drug `pert_stratified_5fold_fold*`、`cell_type_5fold_fold*`、`cell_5fold_fold*`；
  - single-drug no-MSE ablation (`--no-mse-loss`)；
  - single-drug no-PDI ablation (`--pdi-mode zero`)；
  - double-drug pair-level `pert_id_5fold_fold*`；
  - all-single training followed by extra single inference；
  - all-single+double training followed by extra double inference。
- 验证：
  - `py_compile` 通过；
  - stage-1 validator 通过；
  - strengthened training-ready validator 通过；
  - strategy audit 通过，确认 PTV3 global axes protein `11345` / drug `6113`，`batch_index` 存在，all target split families 非空且 formal split row overlap 为 `0`。
- 8 H200 DDP 2-epoch bounded smoke 通过所有 training strategy class：single pert stratified, single cell_type, single cell, single no-MSE, single no-PDI, double pair-level pert, all-single, all-single+double。smoke 使用 `batch_size=2`、`limit_train/val/test_batches=2`，目的是验证 code path/manifest/checkpoint，不作为最终模型质量结论。
- Extra-data inference smoke 通过全部外部任务：single `mat1_480_faims`、`mat1_qe`、`mat2_480_faims`、`mat2_qe`、`mat3_qe`、`mat4_qe`；double `nature`、`nc`、`guomics`。每个任务使用 `--limit-batches 2` 写出 4 row predictions 和 `run_manifest.json`。
- 验证 `scripts/run_ptv3_training_experiments.sh` 语法通过，且最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 新增复查记录：`data/review_summary/2026-05-09_1453_data_training_pipeline_recheck_review.md`。

## 2026-05-09 17:11 HKT Adversarial Strategy Loophole Review

- 对上一版 full experiment strategy 做 adversarial review，重点检查 split semantics、label masks、checkpoint/manifest binding、all-train extra inference、ablation flags、stale output reuse 和 script execution path。
- 发现并修复真实 loophole 1：double-drug `pert_id_5fold_fold*` 使用 ordered pair 时，reversed drug pairs 可能跨 train/valid/test。审计发现 29 个 unordered pair 有 reversed order，fold 内存在 unordered pair overlap。
  - 修改 `utils/09_build_data_splits.py`：double-drug fold key 改为 canonical unordered pair。
  - 修改 `docs/Data_Process_3.md`：明确 `pert_id_5fold_fold*` 使用 canonical unordered pair，避免 reversed-pair leakage；仍不是 individual-drug cold-start split。
  - 重新运行 `utils/09_build_data_splits.py --dataset-group all`，恢复 global `split_build_manifest.json` 包含全部 13 个 PTV1/PTV3 tasks。
- 发现并修复真实 loophole 2：experiment script 允许复用已有 checkpoint/log/output 目录，可能混入旧 checkpoint 或旧 predictions。
  - 修改 `scripts/run_ptv3_training_experiments.sh`：默认 `ALLOW_EXISTING_RUN=0`，非空 checkpoint/log/output 目录直接 fail；需要复用时必须显式设置 `ALLOW_EXISTING_RUN=1`。
  - 新增默认 `RUN_PREFLIGHT=1`，full script 开始前执行 py_compile、standardized validator 和 training-ready validator。
  - `best_checkpoint()` 现在要求 manifest 存在、`run_status == fit_completed` 且 checkpoint 文件存在。
- 发现并修复真实 loophole 3：checkpoint manifest 之前没有 completed 状态，failed run 的 incomplete manifest 可能和旧 checkpoint 组合被误用。
  - 修改 `train.py`：run manifest 初始写入 `run_status="fit_started"`，fit 后写入 `run_status="fit_completed"`、`fit_completed_at`，test 后/skip 后写入 `test_status`；artifact paths 记录为 resolved absolute paths。
  - 修改 `infer.py`：默认拒绝 `run_manifest.json` 未标记 `fit_completed` 的 checkpoint；仅 debug/migration 可显式使用 `--allow-incomplete-checkpoint-manifest`。
- 验证：
  - `py_compile` 通过；
  - `bash -n scripts/run_ptv3_training_experiments.sh` 通过；
  - stage-1 validator 和 strengthened training-ready validator 通过；
  - custom strategy audit 通过：PTV3 global protein `11345`、drug `6113`；所有 checked tasks 非 2000 axis；`batch_index` 存在；single second slot 正确；double auxiliary synergy mask 正确；all formal target splits 有 positive/negative labels；extra test_only coverage 正确。
  - double-drug unordered-pair leakage audit 通过：5 个 `pert_id_5fold_fold*` 的 train/valid/test unordered pair overlap 全为 `0`。
  - negative checks 通过：old/incomplete manifest checkpoint 被 `infer.py` 拒绝；已有 `EXP_PREFIX` 被 script 拒绝。
  - patched full script bounded smoke 通过：`EXP_PREFIX=20260509_confidence_smoke FOLDS=0 MAX_EPOCHS=2 LIMIT_*_BATCHES=2 INFER_LIMIT_BATCHES=2` 覆盖 single pert/cell_type/cell、no-MSE、no-PDI、double canonical pair、all-single、all-single+double 和全部 9 个 extra inference tasks。
  - 所有 confidence smoke training manifests 均为 `run_status=fit_completed`，best checkpoint 存在；9 个 extra inference manifests 均写出 4 predictions。
  - 最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 新增复查记录：`data/review_summary/2026-05-09_1711_adversarial_strategy_loophole_review.md`。

## 2026-05-09 17:23 HKT DDP Deduplicated Loss and Split-Contract Audit

- 继续执行 "Are you 100% confident?" adversarial loop 后发现新的真实 loophole：DDP validation/test metrics 已经按 `row_index` all-gather 并去重，但用于 `ModelCheckpoint(monitor="val/total_loss")` 的 epoch loss 仍可能来自每个 rank 的 batch scalar aggregation。Lightning DDP validation/test sampler 在样本数不能整除 world size 时会复制样本，这会让 checkpoint selection loss 和去重后的 reported metrics 不完全一致。
- 修改 `model/training_ready_lightning.py`：
  - 新增 per-sample MSE/BCE loss collection；
  - validation/test epoch end 先 all-gather predictions、labels、`row_index` 和 per-sample loss，再应用同一套 `row_index` 去重；
  - `val/total_loss`、`val/loss1`、`val/loss2`、`test/total_loss`、`test/loss1`、`test/loss2` 现在由去重后的全局样本重新计算，因此 checkpoint monitor 和 reported eval rows 使用同一语义。
- 修改 `utils/03_validate_training_ready_outputs.py`，把 split contract 纳入默认 training-ready validator：
  - 检查 global `split_build_manifest.json` 包含当前 full experiment 所需 PTV3 tasks；
  - 检查 single-drug `pert_stratified_5fold_fold*`、`cell_type_5fold_fold*`、`cell_5fold_fold*`、`all_train_subset_test` 均存在且 split 非空；
  - 检查 double-drug `pert_id_5fold_fold*`、`all_train_subset_test` 均存在；
  - 检查全部 extra single/double tasks 的 `test_only` split 覆盖 primary non-control anchors；
  - 检查 formal valid/test label 至少包含两个类别，double train 中缺失 synergy label 数必须等于 merged single auxiliary rows；
  - 检查 double-drug canonical unordered pair 在 train/valid/test 之间无 overlap。
- 重新验证 data composition：
  - `ptv3_main_singledrug`: 18,568 primary rows，expression shape `(18568, 10982)`，single `pert_id2 == pert_id1` bad count `0`；
  - `ptv3_main_doubledrug`: 20,764 rows，expression shape `(20764, 11092)`，其中 2,196 native double rows + 18,568 merged single auxiliary rows；auxiliary `synergy` non-empty count `0`，scope 全部为 `single_drug_auxiliary_synergy_masked`。
- 额外发现并修复一个 external validation traceability loophole：graph model 允许 extra task protein axis 与 training checkpoint axis 不同，但 `infer.py` 输出 manifest 之前没有记录 inference axis 和 checkpoint axis。
  - 修改 `infer.py`：inference `run_manifest.json` 现在记录 `ordered_protein_index_path`、`protein_axis_size`、checkpoint manifest path、checkpoint task/split、`checkpoint_ordered_protein_index_path`、`checkpoint_protein_axis_size` 和 `protein_axis_matches_checkpoint`。
  - 对 `ptv3_extra_doubledrug_nature` 做 CPU 1-batch probe，确认 manifest 明确记录 inference axis `11343`、checkpoint axis `11092`、`protein_axis_matches_checkpoint=false`，并写出 1 row prediction。
- 验证：
  - `py_compile` 通过；
  - `bash -n scripts/run_ptv3_training_experiments.sh` 通过；
  - strengthened `utils/03_validate_training_ready_outputs.py` 通过，包含新的 split-contract audit；
  - patched 8-GPU bounded smoke 通过：`EXP_PREFIX=20260509_confidence2_smoke FOLDS=0 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 INFER_LIMIT_BATCHES=1 BATCH_SIZE=2 INFER_BATCH_SIZE=2 bash scripts/run_ptv3_training_experiments.sh`；
  - smoke 覆盖 8 个 training jobs：single pert/cell_type/cell、single no-MSE、single no-PDI、double canonical pair、all-single、all-single+double；所有 checkpoint manifests 为 `run_status=fit_completed` 且 best checkpoint 存在；
  - smoke 覆盖全部 9 个 extra inference tasks；每个 bounded output manifest 写出 2 predictions；
  - 新 inference manifest axis probe 通过：`outputs/20260509_confidence2_manifest_axis_probe_double/run_manifest.json` 记录 external/checkpoint protein axis 差异；
  - 最终 `nvidia-smi --query-compute-apps` 无 active GPU compute process。
- 结论边界：当前 strategy 在 code path、data contract、split leakage、checkpoint binding、stale-output guard 和 extra inference execution 上没有已知 loophole。该结论不等价于保证长训练后的 biological/scientific metric 一定收敛。
- 新增复查记录：`data/review_summary/2026-05-09_1723_ddp_dedup_loss_split_contract_review.md`。

## 2026-05-09 17:34 HKT Required Experiment Runner and Runtime Summary

- 新增 `scripts/train_required_ptv3_experiments.sh`，作为用户实际启动 full experiment set 的简化入口：
  - 默认运行 5-fold single-drug `pert_stratified` / `cell_type` / `cell`；
  - 默认运行 5-fold single-drug ablation：no-MSE 和 no-PDI；
  - 默认运行 5-fold double-drug canonical pert-pair；
  - 默认运行 all-single -> extra single inference；
  - 默认运行 all-single+double -> extra double inference。
- 简化脚本默认使用 8 H200、`MAX_EPOCHS=50`、per-GPU `BATCH_SIZE=16`、`RUN_PREFLIGHT=1`、`RUN_INFERENCE=1`。可通过环境变量覆盖 `EXP_PREFIX`、`FOLDS`、`MAX_EPOCHS`、`BATCH_SIZE` 等。
- 修改 `scripts/run_ptv3_training_experiments.sh`：每个 train/infer job 现在记录 wall-clock runtime 到 `${LOG_DIR}/${EXP_PREFIX}_runtime_summary.tsv`，字段包含 job kind、experiment、task、split、status、start/end UTC、duration seconds 和 artifact path。
- 修改 `train.py`：training manifest 现在记录 `max_epochs`、`batch_size`、accelerator/devices、precision、num_workers、learning rate 和 checkpoint save interval，便于后续 runtime 和 hyperparameter audit。
- 修改 `infer.py`：inference manifest 现在记录 `started_at` 和 `completed_at`，便于计算 extra-data validation runtime。
- 验证：
  - `bash -n scripts/run_ptv3_training_experiments.sh` 通过；
  - `bash -n scripts/train_required_ptv3_experiments.sh` 通过；
  - `py_compile train.py infer.py` 通过。

## 2026-05-09 17:44 HKT Human-Readable Split Experiment Scripts

- 用户反馈 `scripts/run_ptv3_training_experiments.sh` 和 `scripts/train_required_ptv3_experiments.sh` 对人工阅读仍然过于复杂。
- 新增共享 helper：`scripts/ptv3_experiment_common.sh`，只存放环境激活、preflight、train/infer、checkpoint lookup 和 runtime summary 记录等重复逻辑。
- 新增 8 个小型 user-facing experiment scripts，每个脚本只表达一个实验族：
  - `scripts/exp_01_single_pert_stratified_5fold.sh`
  - `scripts/exp_02_single_cell_type_5fold.sh`
  - `scripts/exp_03_single_cell_5fold.sh`
  - `scripts/exp_04_single_no_mse_5fold.sh`
  - `scripts/exp_05_single_no_pdi_5fold.sh`
  - `scripts/exp_06_double_pert_pair_5fold.sh`
  - `scripts/exp_07_extra_single_all_train_infer.sh`
  - `scripts/exp_08_extra_double_all_train_infer.sh`
- 新增 `scripts/run_all_required_ptv3_experiments.sh`，内容只是按顺序调用 8 个小脚本；`scripts/train_required_ptv3_experiments.sh` 改为 backward-compatible thin wrapper。
- 新增 `scripts/README_ptv3_experiments.md`，列出每个脚本的用途、full run 命令、quick test 命令和常用环境变量。
- 在旧 all-in-one runner 顶部加入注释，明确它是 legacy runner，建议人工使用新的 `exp_*.sh` 脚本。
- 验证：
  - `bash -n` 通过所有新增 split scripts、helper、all launcher 和 compatibility launcher；
  - `git diff --check` 通过。

## 2026-05-09 17:52 HKT Wandb Logger and Easy Hyperparameter Overrides

- 用户询问当前 logging 是否支持 wandb，以及如何方便修改 learning rate、batch size、checkpoint save interval 等 hyperparameter。
- 对比旧版 `/mnt/shared-storage-user/beam/wuhao/H100/proteintalk/ProteinTalkv2/train.py` / `train_dd.py`，旧版使用 `WandbLogger(project=project_name, name=experiment_name, save_dir=log_dir, log_model=False)`，否则 fallback 到 TensorBoardLogger。
- 修改 `train.py`：
  - 新增 `--logger-backend tensorboard|wandb|both|none`，默认仍为 tensorboard；
  - 新增旧版兼容 flag `--log-to-wandb/--log_to_wandb`；
  - 新增 `--wandb-project`、`--wandb-entity`、`--wandb-group`、`--wandb-tags`、`--wandb-mode`、`--wandb-log-model`；
  - 新增 `--log-every-n-steps` 和 `--check-val-every-n-epoch`；
  - 新增 checkpoint 控制：`--save-every-n-train-steps`、`--save-top-k`、`--save-last-ckpt/--no-save-last-ckpt`、`--checkpoint-filename`；
  - 支持 `--monitor none`，用于 step-based unmonitored checkpoint saving；
  - 若用户要求 train-step checkpoint 但 monitor 是 `val/*`，直接 fail-fast，避免 step 保存时找不到 validation metric。
- 修改 `scripts/ptv3_experiment_common.sh`：
  - 暴露常用 hyperparameter 环境变量：`LEARNING_RATE`、`BATCH_SIZE`、`HIDDEN_DIM`、`NUM_HEADS`、`NUM_LAYERS`、`DROPOUT`、`GRADIENT_CLIP_VAL`、`OPTIMIZER_NAME`、`SCHEDULER_NAME`、`MSE_WEIGHT`、`BCE_WEIGHT`、`POSITIVE_WEIGHT`、`FOCAL_LOSS`；
  - 暴露 checkpoint 环境变量：`SAVE_EVERY_N_EPOCHS`、`SAVE_EVERY_N_TRAIN_STEPS`、`SAVE_TOP_K`、`SAVE_LAST_CKPT`、`CHECKPOINT_FILENAME`、`MONITOR`；
  - 暴露 wandb/logger 环境变量：`LOGGER_BACKEND`、`LOG_TO_WANDB`、`WANDB_PROJECT`、`WANDB_ENTITY`、`WANDB_GROUP`、`WANDB_TAGS`、`WANDB_MODE`、`WANDB_LOG_MODEL`。
- 修改 `scripts/README_ptv3_experiments.md`，加入 wandb、hyperparameter 和 checkpoint 示例命令。

## 2026-05-09 18:35 HKT Script and Hyperparameter GPU Smoke Test

- 按用户授权使用 8 H200 对 human-readable experiment scripts 和 hyperparameter surface 做 bounded smoke test。
- 验证 `bash -n` 通过 `scripts/ptv3_experiment_common.sh`、8 个 `scripts/exp_*.sh`、`scripts/run_all_required_ptv3_experiments.sh`、`scripts/train_required_ptv3_experiments.sh`。
- 验证 `py_compile` 通过 `train.py`、`infer.py`、`dataset/training_ready_dataset.py`、`model/training_ready_models.py`、`model/training_ready_lightning.py`。
- 8-GPU full script smoke 通过：`EXP_PREFIX=20260509_scripts_all_smoke FOLDS=0 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 INFER_LIMIT_BATCHES=1 BATCH_SIZE=2 INFER_BATCH_SIZE=2 RUN_PREFLIGHT=1 LOGGER_BACKEND=tensorboard bash scripts/run_all_required_ptv3_experiments.sh`。
  - 覆盖 8 个 training jobs：single pert/cell_type/cell、single no-MSE、single no-PDI、double canonical pair、all-single、all-single+double。
  - 覆盖全部 9 个 extra inference tasks：6 个 extra single、3 个 extra double。
  - runtime summary 写入 `logs/20260509_scripts_all_smoke_runtime_summary.tsv`。
- 8-GPU hyperparameter smoke 通过：`EXP_PREFIX=20260509_hparam_wandb_smoke ... LOGGER_BACKEND=wandb WANDB_MODE=disabled ... bash scripts/exp_01_single_pert_stratified_5fold.sh`。
  - 验证 lr、batch size、hidden dim、heads、layers、dropout、gradient clip、optimizer、cosine scheduler、MSE/BCE weights、positive weight、focal loss、wandb disabled mode、epoch checkpoint save。
- 发现并修复 `LOGGER_BACKEND=none` bug：`LearningRateMonitor` 不能在无 logger 的 Trainer 中使用。
  - 修改 `train.py`：先构建 logger；只有 logger 存在时才添加 `LearningRateMonitor`。
  - 复测 step checkpoint 通过：`EXP_PREFIX=20260509_ckpt_step_smoke2 ... SAVE_EVERY_N_TRAIN_STEPS=1 MONITOR=none LOGGER_BACKEND=none ... bash scripts/exp_06_double_pert_pair_5fold.sh`，生成 `epoch=0-step=1.ckpt` 和 `last.ckpt`，manifest 为 `fit_completed/test_completed`。
- 8-GPU `LOGGER_BACKEND=both` smoke 通过：`EXP_PREFIX=20260509_logger_both_smoke ... LOGGER_BACKEND=both WANDB_MODE=disabled ... bash scripts/exp_04_single_no_mse_5fold.sh`。
- compatibility wrapper 执行通过：`EXP_PREFIX=20260509_wrapper_smoke FOLDS=0 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 BATCH_SIZE=1 RUN_PREFLIGHT=0 RUN_INFERENCE=0 LOGGER_BACKEND=none bash scripts/train_required_ptv3_experiments.sh`；runtime summary 包含 8 个 training rows，status 全部为 `0`。
- 结论：当前 split scripts、main runner、compatibility runner、wandb/tensorboard/both/none logger paths、epoch checkpoint 和 step checkpoint hyperparameters 的 bounded execution path 均已通过。

## 2026-05-09 22:11 HKT Training Manifest Test Metrics

- 用户希望 5-fold training 后能直接在每个 fold 的 JSON 中查看 Lightning test 结果。
- 修改 `train.py`：
  - 新增 JSON-safe serializer，支持将 Tensor / NumPy scalar / NumPy array 转成普通 JSON 值；
  - 非有限 float metric 转成 `null`，避免在 manifest 中写入非标准 JSON `NaN` / `Infinity`；
  - `trainer.test(...)` 的返回值现在写入 checkpoint run manifest 的 `test_results`；
  - 新增 `test_result_detail`，记录 test metric 来源、tested checkpoint、task、split strategy、test split row count、`limit_test_batches` 和 metric name 列表；
  - test 开始前 manifest 先写 `test_status=running` 和 `test_started_at`，完成后写 `test_status=test_completed` 和 `test_completed_at`；
  - `--skip-test` 路径写入空 `test_results` 和 skipped reason。
- 验证：
  - `python -m py_compile train.py` 通过。

## 2026-05-09 22:23 HKT Wandb Train-Step Logging Default

- 用户发现 single-drug wandb train-loss step curve 每个 epoch 只有一个点，而 double-drug 有多个点。
- 复查本地运行配置后确认原因：
  - single-drug run 使用 `BATCH_SIZE=128`、`DEVICES=8`，fold0 train rows 为 `12297`，每个 epoch 约 `13` 个 optimizer steps；
  - 旧默认 `LOG_EVERY_N_STEPS=10`，因此每个 epoch 只会记录约一个 train-step logging event；
  - double-drug 对照 run 使用 `BATCH_SIZE=64`、`DEVICES=8`，fold0 train rows 为 `19272`，每个 epoch 约 `38` steps，所以同样的 logging interval 会显示多个点。
- 修改：
  - `train.py --log-every-n-steps` 默认值从 `10` 改为 `1`；
  - `scripts/ptv3_experiment_common.sh` 的 `LOG_EVERY_N_STEPS` 默认值从 `10` 改为 `1`；
  - `ptv3_print_settings` 现在打印 `LOG_EVERY_N_STEPS` 和 `CHECK_VAL_EVERY_N_EPOCH`；
  - `scripts/README_ptv3_experiments.md` 更新默认值，并说明大 batch 时应使用 `LOG_EVERY_N_STEPS=1` 才能在 wandb/TensorBoard 看到每个 optimizer step。
- 验证：
  - `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile train.py` 通过；
  - `bash -n scripts/ptv3_experiment_common.sh` 通过。

## 2026-05-11 01:57 HKT Training Launcher Scripts for Remaining Runs

- 按 `scripts/0509_1.sh` 的格式新增 3 个 thin launcher scripts：
  - `scripts/0509_2.sh`: double-drug canonical pert-pair 5-fold training, calls `scripts/exp_06_double_pert_pair_5fold.sh`.
  - `scripts/0509_3.sh`: train on all single-drug data and infer extra single-drug datasets, calls `scripts/exp_07_extra_single_all_train_infer.sh`.
  - `scripts/0509_4.sh`: train on all single+double data and infer extra double-drug datasets, calls `scripts/exp_08_extra_double_all_train_infer.sh`.
- 3 个脚本沿用本地 W&B endpoint/cache/API key export 和 inline env override 风格。
- `0509_2.sh` / `0509_4.sh` 使用 `BATCH_SIZE=64`，`0509_3.sh` 使用 `BATCH_SIZE=128`。

## 2026-05-11 02:08 HKT Configurable Best-Checkpoint Metric

- 新增 `train.py --best-ckpt-metric`，支持按 named validation metric 选择 best checkpoint：
  - `valid_auprc` / `auprc`: monitor `val/task_auprc`，mode `max`；
  - `valid_total_loss` / `total_loss`: monitor `val/total_loss`，mode `min`；
  - `valid_loss1` / `loss1`: monitor `val/loss1`，mode `min`；
  - `valid_loss2` / `loss2`: monitor `val/loss2`，mode `min`。
- 默认 checkpoint selection 从原来的 `val/total_loss` 改为 `valid_auprc`。
- 修改 `scripts/ptv3_experiment_common.sh`，新增 `BEST_CKPT_METRIC` 环境变量，默认 `valid_auprc`；保留 raw `MONITOR` / `MONITOR_MODE` override。
- 修改 `scripts/0509_1.sh`、`scripts/0509_2.sh`、`scripts/0509_3.sh`、`scripts/0509_4.sh`，显式写入 `BEST_CKPT_METRIC=valid_auprc`。
- 修改 `model/training_ready_lightning.py`，active-task validation metrics 不再使用 rank-zero-only logging，保证 `val/task_auprc` 可被 DDP checkpoint callback 读取。
- 更新 `scripts/README_ptv3_experiments.md` 的 checkpoint section，说明 4 种选择方法和示例命令。

## 2026-05-11 02:21 HKT Checkpoint Selection Adversarial Fixes

- 对新的 checkpoint selection strategy 做 adversarial review，发现并修复以下 loopholes：
  - legacy `scripts/run_ptv3_training_experiments.sh` 仍默认 `MONITOR=val/total_loss`，已改为默认 `BEST_CKPT_METRIC=valid_auprc`，并只在用户显式设置时传 raw `MONITOR` / `MONITOR_MODE`。
  - `train.py --log-every-n-steps` 直接 CLI 默认仍为 `5`，已改为 `1`，与 scripts 和 docs 一致。
  - `SCHEDULER_NAME=plateau` 仍固定 monitor `val/total_loss`，已改为跟随 checkpoint selection metric 和 mode。
  - `valid_auprc` 在 validation batch 只有一个类别时会变成 non-finite，已新增 monitor guard：checkpoint monitor missing / NaN / Inf 时默认 fail-fast；仅显式设置 `ALLOW_NONFINITE_MONITOR=1` / `--allow-nonfinite-monitor` 时允许继续。
- 修改 `scripts/ptv3_experiment_common.sh` 和 legacy runner，暴露 `ALLOW_NONFINITE_MONITOR`，默认 `0`。
- 更新 `scripts/README_ptv3_experiments.md`，说明 non-finite monitor fail-fast、tiny smoke test 可改用 `BEST_CKPT_METRIC=total_loss/loss1/loss2`，以及 plateau scheduler 会跟随 checkpoint metric。
- 检查当前 PTV3 full validation splits：single-drug / double-drug 的目标 experiment strategies 都包含 active label 两个类别，支持正常 full-validation 使用 `valid_auprc`。
- 验证：
  - `py_compile train.py model/training_ready_lightning.py` 通过；
  - 所有 top-level `scripts/*.sh` 的 `bash -n` 通过；
  - `train.py --help` 显示 `--best-ckpt-metric` 和 `--allow-nonfinite-monitor`；
  - metric mapping 检查通过：`valid_auprc -> val/task_auprc/max`、`total_loss -> val/total_loss/min`、`loss1 -> val/loss1/min`、`loss2 -> val/loss2/min`；
  - `utils/01_validate_standardized_outputs.py` 和 `utils/03_validate_training_ready_outputs.py` 均通过。

## 2026-05-13 20:26 HKT Remaining 5-Fold Launcher

- 新增 `scripts/0513_1.sh`，按 `scripts/0511_1.sh` 的环境初始化和 inline env override 风格顺序调度：
  `exp_04_single_no_mse_5fold`、`exp_02_single_cell_type_5fold`、`exp_03_single_cell_5fold`、`exp_06_double_pert_pair_5fold`、`exp_08_extra_double_all_train_infer`、`exp_05_single_no_pdi_5fold`。
- 默认 single-drug runs 使用 `BATCH_SIZE=128`，double-drug 5-fold 和 extra double all-train/infer 使用 `BATCH_SIZE=64`。
- `exp_08` 默认使用 `REFERENCE_5FOLD_CKPT_PATH=checkpoints/20260513_double_pert_pair_5fold`，匹配同脚本内先执行的 double 5-fold prefix。
- 验证：`bash -n scripts/0513_1.sh` 通过。

## 2026-05-13 20:29 HKT Loss2 Checkpoint Selection for Remaining Runs

- 修改 `scripts/0513_1.sh`，将 6 段实验调用的默认 `BEST_CKPT_METRIC` 从 `valid_auprc` 改为 `loss2`。
- `loss2` 是 `train.py` 支持的 alias，对应 `valid_loss2`，checkpoint monitor 为 `val/loss2`，mode 为 `min`；当前语义为 active task BCE classification loss。

## 2026-05-14 15:27 HKT Extra Raw Control Mapping Export

- 新增 `utils/10_export_extra_raw_with_controls.py`，用于把 PTV3 extra single/double 原始 CSV 回填为可对外共享的 `sample_id -> control` 映射文件。
- 导出规则：
  - `sample_id` 从 stage-1 `data/standardized/ptv3/tasks/<task>/info.csv` 的 `source_row_index` 对齐回 raw row；
  - `control` 只从 stage-2 `data/training_ready/ptv3/tasks/<task>/processed.csv` 中保留下来的 `source_row_role == self` rows 回填；
  - 被 stage-2 label/filter 过滤掉的 raw rows 保留生成的 `sample_id`，但 `control` 置空。
- 输出路径：`data/standardized/ptv3/extra_raw_with_controls/`。
  - 每个 raw CSV 写出一份 `*_with_sample_id_control.csv`；
  - 额外写出 `extra_sample_id_control_mapping.csv`、`extra_sample_id_control_mapping_nonempty.csv` 和 `export_manifest.json`。
- 本次导出覆盖 9 个 PTV3 extra task：总 raw rows `398726`，其中 `133409` 行有非空 control 映射。
- 验证：
  - `conda run -n flow_v2 python -m py_compile utils/10_export_extra_raw_with_controls.py` 通过；
  - `conda run -n flow_v2 python utils/10_export_extra_raw_with_controls.py --help` 通过；
  - 实际导出成功，manifest 检查 `alignment_issues == []`。

## 2026-05-14 15:41 HKT Extra Raw Control Export Source Restriction

- 用户检查导出 CSV 后指出 `control` 列不应出现主双药 control ids（例如 `control_ 1` / `control_ 114`），导出应使用原始 single-drug 或 extra-baseline 中的 sample id。
- 原因确认：
  - 旧版导出直接复用了 stage-2 `processed.csv` 的 `control`；
  - stage-1 control pool 过去包含 `ptv3_main_doubledrug` self-control rows，因此部分 extra rows 被匹配到主双药的 `control_*` sample ids；
  - 这些 `control_*` 不是导出脚本新生成的，但不符合本次对外共享语义。
- 修改 `utils/10_export_extra_raw_with_controls.py`：
  - stage-2 `processed.csv` 现在只用于判断 raw row 是否保留；
  - 导出用 `control` 会重新从允许来源池匹配，允许来源只包括 `ptv3_main_singledrug` self-controls 和 `ptv3_extra_baseline` sample ids；
  - 匹配排序仍使用原 stage-1 规则：同 Cell 下优先 machine / type / batch / plate，然后按 sample id 稳定排序。
- 已重新生成 `data/standardized/ptv3/extra_raw_with_controls/`。
- 验证：
  - `python -m py_compile utils/10_export_extra_raw_with_controls.py` 通过；
  - 重新导出的 `control_match_source_task` 只包含 `ptv3_main_singledrug`、`ptv3_extra_baseline` 和空值；
  - `control` 列包含 `control` 字符串的行数为 `0`；
  - 9 个 extra task 的 kept rows 仍全部有非空 control，`rows_kept_without_control == 0`；
  - 固定随机种子 `20260514` 抽查每个 extra task 1 条非空 control 和 6 条 filtered blank control，结果 `FAILURES 0`。

## 2026-05-14 15:47 HKT Extra Raw Filter Status Export Columns

- 用户指出导出脚本现在重新匹配 allowed control，需要明确标志哪些 raw rows 是之前被过滤掉的。
- 修改 `utils/10_export_extra_raw_with_controls.py`，在每个 annotated raw CSV 和汇总 mapping 中新增：
  - `stage2_filter_status`: `kept_after_stage2_filter` 或 `filtered_out_before_stage2_processed`；
  - `control_export_status`: `matched_allowed_control`、`filtered_out_blank_control` 或 `kept_but_no_allowed_control_match`。
- 当前重新导出结果：
  - `filtered_out_before_stage2_processed / filtered_out_blank_control`: `265317` rows；
  - `kept_after_stage2_filter / matched_allowed_control`: `133409` rows；
  - `kept_but_no_allowed_control_match`: `0` rows。
- 这些状态列也写入 `export_manifest.json` 的 per-task counts。

## 2026-05-18 20:24 HKT 0513 Double Fold2 Resume Launcher

- 新增 `scripts/0518_1.sh`，用于在 `scripts/0513_1.sh` 停在 `20260513_double_pert_pair_5fold_double_pert_pair_fold2` 后继续执行。
- 当前确认 fold0/fold1 manifest 为 `fit_completed` 且 `test_completed`，fold2 manifest 停在 `fit_started`，runtime summary 中 fold2 exit status 为 `137`。
- `train.py --checkpoint-path` 只加载模型权重，不会恢复 Lightning trainer epoch/optimizer 状态；因此新脚本默认把未完成的 fold2 checkpoint/log/output 目录移动到 `_archived_failed_restarts/` 后，用同一实验名前缀从 fold2 重新训练，再继续 fold3/fold4。
- double 5-fold 完成后，脚本继续执行原 `0513_1.sh` 后续两段：`exp_08_extra_double_all_train_infer` 和 `exp_05_single_no_pdi_5fold`；默认超参数保持 `0513_1.sh` 的 `loss2` checkpoint selection、double batch size 64、single no-pdi batch size 128。

## 2026-05-21 11:58 HKT New Version GraphJump and PCEP Iteration

- 在 `new_version` 中新增两类可开关结构：
  - `PCEP`：低开销逐蛋白 expression/protein-embedding pooling，用于恢复一定逐蛋白可解释性；
  - multi-hop graph feature 与 selective jump fusion：在离线 graph cache 中加入 `PDI-PPI^2`、`DDI^2`、`DDI-PDI`、`DDI-PDI-PPI` 等多跳上下文，并支持 softmax/sparsemax gate 选择不同 graph blocks。
- 修改范围：
  - `new_version/graph_feature_utils.py`
  - `new_version/fast_delta_model.py`
  - `new_version/train.py`
  - `new_version/run_single_unseen_sweep.sh`
  - `new_version/run_single_unseen_5fold.sh`
- 验证：
  - `python -m py_compile` 通过；
  - 单 GPU dry run 通过；
  - 2 GPU DDP smoke fit/test 通过；
  - 完成两轮 2 GPU、50 epoch、5-fold real/zero sweep，共 40 个完整 runs。
- 结果：
  - baseline3 仍最佳：AUPRC `0.656369`，AUROC `0.900316`，graph gap `+0.057573`；
  - baseline3 + PCEP：AUPRC `0.651715`，AUROC `0.899814`，graph gap `+0.059653`；
  - selective multi-hop sparse jump：AUPRC `0.619492`；
  - selective multi-hop sparse jump + PCEP：AUPRC `0.623436`；
  - multihop concat：AUPRC `0.612670`。
- 结论：标准 baseline 暂不替换，继续使用 `graph128_struct_drugcat_logit2_no_pos`；PCEP 可作为解释性 ablation 保留；当前 multi-hop/selective jump 版本不建议升为主模型。

## 2026-05-21 14:46 HKT Baseline4 Single-GPU PCEP Ablation

- 将 `baseline3 + PCEP` 定义为 `baseline4`，并新增 `new_version/run_baseline4_1gpu_parallel.sh`。
- 新脚本按 `GPU_IDS=0,1` 启动两个独立单卡 worker，每个实验使用 `--devices 1`、`batch_size=256`，用于在两张 GPU 上并行跑不同 fold/ablation，而不是用 DDP 跑同一个 fold。
- 完成 `ptv3_main_singledrug`、`pert_stratified_5fold`、50 epoch 的完整 5-fold：
  - baseline4：AUPRC `0.666491`，AUROC `0.903489`，ACC `0.915167`，fit `41.8s/fold`；
  - baseline4 w/o graph feature：AUPRC `0.561250`，AUROC `0.835845`，ACC `0.912883`，fit `42.0s/fold`；
  - baseline4 w/o MSE loss：AUPRC `0.644541`，AUROC `0.888019`，ACC `0.913273`，fit `41.2s/fold`。
- Ablation 结论：
  - graph feature 贡献 `+0.105241` AUPRC、`+0.067644` AUROC；
  - MSE auxiliary loss 贡献 `+0.021950` AUPRC、`+0.015470` AUROC。
- 额外完成 `mse_weight` sweep：
  - `0.05`: AUPRC `0.649057`，AUROC `0.887431`；
  - `0.10`: AUPRC `0.656212`，AUROC `0.898468`；
  - `0.25`: AUPRC `0.666491`，AUROC `0.903489`；
  - `0.50`: AUPRC `0.658686`，AUROC `0.893493`。
- 结论：`mse_weight=0.25` 保持为 baseline4；baseline4 相比 baseline3 AUPRC `+0.010122`、AUROC `+0.003173`，同时支持一张 GPU 一个 fold 的高效并行。

## 2026-05-21 15:22 HKT Baseline4 Root Integration and 8-GPU Parallel Launcher

- 将 `new_version` 中的 baseline4 训练框架迁移到根路径训练/推理体系：
  - 新增 `dataset/training_ready_fast_dataset.py`；
  - 新增 `model/fast_delta_model.py`；
  - 新增 `model/fast_lightning.py`；
  - 新增 `model/graph_feature_utils.py`；
  - `train.py` 新增 `model_type=fast_delta` 分支，并将默认训练配置改为单卡、`batch_size=256`、bf16、baseline4 graph/PCEP 参数；
  - `infer.py` 新增 `fast_delta` 推理分支。
- 保留旧 `attention_v10_hetero_cls_ee` 分支以兼容历史 checkpoint，但 `scripts/exp_0[1-8].sh` 默认已切换到 `fast_delta`。
- 重写 `scripts/ptv3_experiment_common.sh` 默认配置：
  - `DEVICES=1`；
  - `BATCH_SIZE=256`；
  - `PRECISION=bf16-mixed`；
  - `LOGGER_BACKEND=wandb`、`LOG_TO_WANDB=1`；
  - baseline4 默认启用 PPI/PDI/DDI graph feature、structural RP、drug concat、`graph_logit_scale=2.0`、`PCEP`。
- `scripts/exp_05_single_no_pdi_5fold.sh` 改为 w/o graph feature ablation，实际传入 `--graph-feature-mode zero`。
- 新增 `scripts/0521_baseline4_8gpu_parallel.sh`：
  - 第一阶段把 exp01-06 的 fold 任务拆成单卡任务队列，默认在 `GPU_IDS=0,1,2,3,4,5,6,7` 上同时最多跑 8 个任务；
  - 第二阶段等待 reference 5-fold 完成后，并行启动 exp07/08 all-train + extra inference；
  - 默认使用 wandb，用户可用 `LOGGER_BACKEND=none LOG_TO_WANDB=0` 做 smoke/debug。
- `infer.py` fast 分支新增 checkpoint protein axis 对齐：extra task 的表达矩阵若与训练 checkpoint 的 protein axis 不同，会按 checkpoint axis 重排，缺失蛋白列补 NaN，MSE 相关计算按 mask 忽略，分类推理继续可用。
- 2 GPU smoke 测试通过：
  - 命令使用 `GPU_IDS=0,1 FOLDS=0 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 INFER_LIMIT_BATCHES=1 LOGGER_BACKEND=none LOG_TO_WANDB=0`；
  - exp01-06 fold0 均 `fit_completed/test_completed`；
  - exp07/08 all-train 均 `fit_completed`；
  - 6 个 extra single + 3 个 extra double inference 均写出 `predictions.parquet`；
  - runtime summary: `logs/20260521_root_baseline4_smoke2_runtime_summary.tsv`。

## 2026-05-21 15:25 HKT Baseline4 Extra Inference Reference Epoch Mean

- 用户确认 extra inference 应使用 unseen drug 5-fold 的 average best epoch。
- 将 `scripts/ptv3_experiment_common.sh` 中 `REFERENCE_EPOCH_AGG` 默认值从 `median` 改为 `mean`。
- 当前 exp07/exp08 reference epoch 逻辑为：
  - single extra：从 `ptv3_main_singledrug` 的 `pert_stratified_5fold_fold*` reference runs 读取每个 fold 的 `best_model_path` epoch，取 mean 后按 `REFERENCE_EPOCH_ROUNDING` 得到 `selected_epoch`；
  - double extra：从 `ptv3_main_doubledrug` 的 `pert_id_5fold_fold*` reference runs 读取每个 fold 的 `best_model_path` epoch，取 mean 后按 `REFERENCE_EPOCH_ROUNDING` 得到 `selected_epoch`；
  - all-train 使用 `max_epochs=selected_epoch+1` 和 `--monitor none`，最后用 `last.ckpt` 对 extra task 做 inference。

## 2026-05-21 15:54 HKT Baseline4 Graph Cache Prebuild and Launcher Hardening

- 新增 `scripts/prebuild_graph_cache.py`，用于在并行训练前单进程预构建 baseline4 的 PPI/PDI/DDI compressed graph feature cache，并写出 graph cache summary。
- `scripts/0521_baseline4_8gpu_parallel.sh` 在启动 8 个 fold worker 前会先调用 graph cache prebuild；`GRAPH_FEATURE_MODE=off` 时跳过。
- `model/graph_feature_utils.py` 增加 graph cache 文件锁和 atomic replace：
  - 同一 cache 文件只能由一个进程构建；
  - `.npy` 与 `.meta.json` 写入改为临时文件完成后原子替换，降低并发/中断导致半写入 cache 的风险。
- `infer.py` fast checkpoint validation 增加 graph/PCEP 结构参数校验，并兼容旧 baseline4 smoke checkpoint 缺失新增 manifest 字段但参数等于旧默认值的情况。
- `scripts/0521_baseline4_8gpu_parallel.sh` 的 stage1/stage2 后台任务等待逻辑改为等待所有 worker 后统一报错，避免一个 worker 失败时 launcher 提前退出而遗漏其他后台任务状态。
- `scripts/show_extra_results.py` 增加 fast metrics 兼容：当 `metrics.json` 中只有 `count` 字段时，会将其作为 `valid_count` 展示，因此可直接汇总 baseline4 extra inference 输出。

## 2026-05-21 16:42 HKT Baseline4 W&B Credential Preflight

- 用户运行 8-GPU baseline4 launcher 时遇到 `wandb.errors.UsageError: api_key not configured (no-tty)`。
- 确认新 `scripts/0521_baseline4_8gpu_parallel.sh` 没有写入或覆盖 `WANDB_API_KEY` / `WANDB_BASE_URL`；旧 `0509/0513/0518` wrapper 曾硬编码本地 W&B endpoint/key，新 launcher 未复用硬编码 credential，以避免 GitHub 上传时继续泄露 secret。
- 新增 `scripts/wandb_env.local.example`，并在 `.gitignore` 中忽略 `scripts/wandb_env.local`；用户可将本地 W&B `WANDB_BASE_URL` 与 `WANDB_API_KEY` 放入该未跟踪文件。
- 新增 `scripts/check_wandb_auth.py`：
  - online W&B 开启时，启动训练前检查 `WANDB_API_KEY` 或 `~/.netrc` credential；
  - 缺失 credential 时在 preflight 阶段给出清晰错误，不再让 8 个并行 worker 同时抛 no-tty stack trace；
  - `WANDB_MODE=offline/disabled` 或 `LOGGER_BACKEND=none LOG_TO_WANDB=0` 时跳过 credential 检查。
- `scripts/ptv3_experiment_common.sh` 和 `scripts/0521_baseline4_8gpu_parallel.sh` 会自动 source `scripts/wandb_env.local`（若存在）后再构造训练参数。
- 用户要求继续直接硬编码本地 W&B credential；已在 `scripts/ptv3_experiment_common.sh` 和 `scripts/0521_baseline4_8gpu_parallel.sh` 中加入本地 W&B server/key 默认值。若 shell 环境或 `scripts/wandb_env.local` 显式设置同名变量，则仍可覆盖默认值。

## 2026-05-21 19:39 HKT Double-Drug Data Composition and Pair Fusion Iteration

- 针对 double-drug 5-fold 分数偏低的问题，先 review 了 single/double 共用 fast model 的数据与 mask 逻辑：
  - single 辅助行在 double task 中以 `[drug, drug]` 进入模型，但 `synergy` label mask 为 inactive，只参与 expression reconstruction；
  - double 主行以 `[drug1, drug2]` 进入模型，`synergy` label mask active，参与 BCE + expression reconstruction；
  - `_masked_bce` 使用 `1 - mask` 加权，未发现 masked single auxiliary row 进入 binary loss 的证据。
- 新增可控实验开关：
  - `--pair-fusion-mode {symmetric,rich_symmetric,ordered_concat,dual}`，其中 `dual` 使用 `[mean, absdiff, product, drug1, drug2]`；
  - `--pair-type-features`，显式编码 `drug1 == drug2`，区分 single auxiliary row 与 true double pair；
  - `--mse-inactive-label-weight`，控制 active label 缺失样本在 MSE reconstruction 中的权重；
  - `--inactive-label-train-ratio`，支持只用 active double rows 或限制 single auxiliary/masked rows 的比例；
  - `--active-label-sampling-weight`，支持训练采样时提高 active double rows 权重。
- 完成主要 double-drug 5-fold 实验：
  - 原 baseline4 double：AUROC `0.694998`，AUPRC `0.600952`；
  - `dual + pair_type + mse_inactive_label_weight=0.2`：AUROC `0.744865`，AUPRC `0.656621`；
  - double-only active rows（`inactive_label_train_ratio=0`）：AUROC `0.731969`，AUPRC `0.629023`；
  - 限制 single auxiliary/masked rows 到 active rows 的 2 倍（`inactive_label_train_ratio=2`）：AUROC `0.730526`，AUPRC `0.629443`；
  - `dual + pair_type + mse_inactive_label_weight=0.2 + use_ddi + graph_pair_add_scale=0.5`：AUROC `0.756597`，AUPRC `0.661878`。
- 数据角度结论：
  - 只用 double 数据没有提升，反而低于保留全部 single auxiliary 的设置；
  - single auxiliary row 更像是有用的 expression reconstruction 正则，但需要降低其 MSE 权重并显式标注 pair type；
  - double 低分的主要问题更接近 pair fusion 表达不足，而不是 binary loss mask 误写。
- 回归测试：
  - single 默认 5-fold 重新跑完后与前一版一致：AUROC `0.903489`，AUPRC `0.666491`，ACC `0.915167`；
  - 说明新增参数默认值保持 single baseline 行为不变。
- 将 double-drug 默认实验脚本对齐到当前达到目标的设置：
  - `scripts/exp_06_double_pert_pair_5fold.sh` 和 `scripts/exp_08_extra_double_all_train_infer.sh` 默认使用 `PAIR_FUSION_MODE=dual`、`PAIR_TYPE_FEATURES=1`、`MSE_INACTIVE_LABEL_WEIGHT=0.2`、`USE_DDI=1`、`GRAPH_PAIR_ADD_SCALE=0.5`；
  - `scripts/0521_baseline4_8gpu_parallel.sh` 为 exp06/exp08 传入同一组 double 专用默认值；
  - single 相关脚本保留 baseline4 默认值。
- 校验：
  - `python -m py_compile` 覆盖 train/infer/fast model/graph/cache/reference 脚本通过；
  - `bash -n` 覆盖 exp06/exp08/8-GPU launcher/common 脚本通过；
  - `EXP_PREFIX=20260521_1940_double_default_smoke GPU_IDS=0 FOLDS=0 MAX_EPOCHS=1 LIMIT_*_BATCHES=1 LOGGER_BACKEND=none LOG_TO_WANDB=0 RUN_INFERENCE=0 bash scripts/exp_06_double_pert_pair_5fold.sh` 通过，并打印确认默认 double 设置为 `dual + pair_type + mse_inactive=0.2 + use_ddi + graph_pair_add=0.5`。

## 2026-05-21 19:50 HKT Task-Specific Single/Double Rerun Launcher

- 明确 single drug 与 double drug 可使用同一个 `fast_delta` 模型类，但在 pair feature fusion 相关超参数上采用任务级配置。
- `scripts/0521_baseline4_8gpu_parallel.sh` 新增显式 single/double 配置：
  - single 默认：`PAIR_FUSION_MODE=symmetric`、`PAIR_TYPE_FEATURES=0`、`MSE_INACTIVE_LABEL_WEIGHT=1.0`、`USE_DDI=0`、`GRAPH_PAIR_ADD_SCALE=0.0`；
  - double 默认：`PAIR_FUSION_MODE=dual`、`PAIR_TYPE_FEATURES=1`、`MSE_INACTIVE_LABEL_WEIGHT=0.2`、`USE_DDI=1`、`GRAPH_PAIR_ADD_SCALE=0.5`。
- fold stage 中 exp01-05 显式传入 single 配置，exp06 显式传入 double 配置；extra stage 中 exp07 显式传入 single 配置，exp08 显式传入 double 配置。
- 新增 `scripts/0521_baseline4_task_specific_8gpu_parallel.sh`，作为最终重跑入口，默认固定上述 task-specific 配置后复用 `scripts/0521_baseline4_8gpu_parallel.sh` 的并行调度逻辑。
- `bash -n` 覆盖 `scripts/0521_baseline4_8gpu_parallel.sh`、`scripts/0521_baseline4_task_specific_8gpu_parallel.sh`、exp01/06/07/08 通过；`git diff --check` 通过。

## 2026-05-22 16:14 HKT Unseen Cell/Cell-Type Iteration

- 目标：在 2-GPU 服务器上继续优化 `single_cell_type_5fold` 与 `single_cell_5fold` 的 AUPRC，优先尝试参数、covariate encoding、轻量可控模块和数据角度诊断。
- 新增 split-aware covariate UNK encoding：
  - `FastProteinTalkDataset` 支持将 validation/test 中训练 split 未出现的 covariate category 映射到保留的 UNK embedding；
  - 训练时可通过 `--covariate-unk-dropout` 随机替换 enabled covariates 为 UNK，训练共享未知类别表示；
  - `train.py` 新增 `--covariate-unk-for-unseen`、`--covariate-unk-fields`、`--covariate-unk-dropout`，manifest 记录相关配置。
- 新增轻量可控实验模块：
  - `FastDeltaDrugResponseModel` 增加默认关闭的 auxiliary logit heads：`--control-logit-scale`、`--pair-logit-scale`、`--target-logit-scale`、`--covariate-logit-scale`；
  - `train.py`/`infer.py`/`scripts/ptv3_experiment_common.sh` 透传这些参数，默认均为 `0.0`，不改变 baseline 行为；
  - `train.py` 增加 `--positive-label-sampling-weight`，用于温和 oversample positive active-label rows，默认 `1.0` 关闭。
- 修复 fast inference 对 covariate UNK checkpoint 的兼容：
  - `infer.py` 会从 checkpoint manifest 读取 `covariate_unk_fields`；
  - 根据 checkpoint 的训练 split 计算 known covariate values，并在 inference dataset 中映射未知 category；
  - fast model covariate embedding size 使用 checkpoint-aware size，避免 UNK checkpoint 载入时 size mismatch。
- 完成 full 5-fold 实验（均为 1 GPU、batch size 256、logger off）：
  - baseline4 当前参考：cell type AUPRC `0.787186`，cell AUPRC `0.735066`；
  - `MSE_WEIGHT=0.1`：cell type `0.794984`，cell `0.739205`；
  - covariate UNK full fields、dropout `0.15`、`MSE_WEIGHT=0.1`：cell type `0.814780`，cell `0.761629`，为本轮最佳；
  - covariate UNK + `valid_auroc` checkpoint：cell type `0.815947`，cell `0.748006`；
  - covariate UNK + `MSE_WEIGHT=0`：cell type `0.806767`，cell `0.756831`；
  - auxiliary logits（control `0.5`、pair `1.0`、target `0.5`）：cell type `0.811393`，cell `0.742966`；
  - stronger regularization（dropout `0.30`、weight decay `5e-4`、label smoothing `0.03`）：cell type `0.810939`，cell `0.742051`;
  - positive sampler weight `3.0`：cell type `0.810426`，cell `0.728539`;
  - covariate slim variants and field-specific UNK variants did not exceed full-field covariate UNK;
  - no-covariate ablation（`--batch-cov-list` 为空，删除 cell/time/machine/batch 等全部 covariates）：cell type `0.783888`，cell `0.742423`，未超过 full-field covariate UNK;
  - drop-`Cell` covariate only（保留 `machineID_new`、`Cell_plate`、`cell_type`、`batch`、`pert_time`，covariate UNK dropout `0.15`）：cell type `0.815047`，cell `0.758506`，与 full-field covariate UNK 接近但未稳定超过;
  - smaller model (`hidden_dim=256`, `expression_latent_dim=384`, `covariate_embedding_dim=48`) did not improve AUPRC, but is a potential speed/efficiency setting if lower capacity is preferred.
- Data diagnostic:
  - train-only drug response prior alone was weaker than the model;
  - linear mixing model predictions with train-only drug prior only gave tiny improvements (cell type best approx `0.821`, cell best approx `0.766`), so no target-encoding prior was added to the model path.
- Current conclusion:
  - The reliable improvement is covariate UNK encoding, especially for unseen cell/cell-type splits where many covariate categories are absent from train;
  - The 0.85 AUPRC target was not reached without larger architecture/data changes;
  - Current best candidate for these splits is full-field covariate UNK dropout `0.15` with `MSE_WEIGHT=0.1`.
- Validation:
  - `python -m py_compile train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py model/fast_lightning.py scripts/check_wandb_auth.py` passed;
  - smoke tested fast `infer.py` on a covariate UNK checkpoint with one test batch and wrote `outputs/smoke_covunk_infer_20260522/predictions.parquet`.

## 2026-05-22 18:52 HKT Unseen Cell Representation/Loss Exploration

- 在 clean `v2.2beta` 上创建实验分支 `exp/unseen-cell-representation-20260522`，保留 `924688e v2.2beta` 作为回退点。
- 新增默认关闭的 expression/cell 表征实验开关：
  - `--protein-concat-mode {off,pcep,pcep_cell,pcep_dual}`，其中 `pcep_cell` 用 control-expression hidden 查询 protein-expression pooling，`pcep_dual` 同时保留原 context query 和 expression query；
  - `--protein-concat-score-mode {multiply,additive,magnitude}` 与 `--protein-concat-expr-scale`，用于替换 PCEP 中 expression 与 protein attention score 的融合方式；
  - `--aux-covariate-loss-fields`、`--aux-covariate-loss-weight`、`--aux-covariate-loss-label-smoothing`，从 expression hidden 预测指定 covariate，作为 cell/cell-type auxiliary classification loss；
  - `scripts/ptv3_experiment_common.sh` 增加上述参数和 covariate UNK env passthrough，默认全部关闭或保持旧值。
- 校验：
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py` 通过；
  - `bash -n scripts/ptv3_experiment_common.sh scripts/exp_02_single_cell_type_5fold.sh scripts/exp_03_single_cell_5fold.sh` 通过；
  - `pcep_dual + additive + aux cell_type` fast dry-run 前向通过，输出 expression/logit shape 正常。
- 完成 unseen cell 5-fold 实验（1 GPU、batch size 256、logger off、full covariate UNK dropout `0.15` unless noted）：
  - `pcep_dual + additive + topk1024`：AUROC `0.928647`，AUPRC `0.753787`；
  - `pcep_cell + additive + topk512`：AUROC `0.926901`，AUPRC `0.752784`；
  - 原 PCEP 改 `additive` scoring：AUROC `0.925741`，AUPRC `0.749050`；
  - 原 PCEP 改 `magnitude` scoring：AUROC `0.927705`，AUPRC `0.740845`；
  - auxiliary `cell_type` loss weight `0.05`：AUROC `0.929050`，AUPRC `0.750566`；
  - auxiliary `cell_type` loss weight `0.01`：AUROC `0.928162`，AUPRC `0.725546`；
  - `hidden_dim=512`、`expression_latent_dim=768`：AUROC `0.929532`，AUPRC `0.754120`；
  - `hidden_dim=512`、`expression_latent_dim=768`、LR `2e-4`：AUROC `0.928268`，AUPRC `0.715438`；
  - control-expression direct logit scale `0.5`：AUROC `0.926991`，AUPRC `0.756559`；
  - control-expression direct logit scale `1.0`：AUROC `0.928378`，AUPRC `0.751017`。
- Loss-weight sweep:
  - current-code baseline rerun `MSE_WEIGHT=0.10`：AUROC `0.930024`，AUPRC `0.749969`；
  - `MSE_WEIGHT=0.05`：AUROC `0.931399`，AUPRC `0.762206`;
  - `MSE_WEIGHT=0.075`：AUROC `0.934443`，AUPRC `0.767008`;
  - `MSE_WEIGHT=0.20`：AUROC `0.932208`，AUPRC `0.761342`;
  - `MSE_WEIGHT=0.075` + drop `Cell` covariate：AUROC `0.929343`，AUPRC `0.744766`，不采用。
- 对 unseen cell type 的确认实验：
  - full covariate UNK dropout `0.15` + `MSE_WEIGHT=0.075`：AUROC `0.941852`，AUPRC `0.810250`，低于前一轮 cell-type best `0.814780/0.815947`，因此 cell type 仍建议保留 `MSE_WEIGHT=0.10`。
- 当前结论：
  - 直接增强 expression fusion、增大模型容量、或加入 cell-type auxiliary classification 都没有提升 unseen cell；
  - unseen cell 当前最可靠的小幅提升来自 `MSE_WEIGHT=0.075`，AUPRC `0.767008`，比历史 full covariate UNK + `MSE_WEIGHT=0.10` 的 `0.761629` 高约 `+0.0054`；
  - 该设置不适合作为 cell-type 默认，cell-type 仍使用 full covariate UNK dropout `0.15` + `MSE_WEIGHT=0.10`。

## 2026-05-25 11:39 HKT Unseen Cell Deep-Research Plan Implementation

- Implemented additional default-off experiment modules for the unseen-cell plan:
  - train-only KNN drug-response prior features from control-expression prototypes (`--cell-prior-mode knn_drug`);
  - optional learned/fixed prior logit adapters (`--cell-prior-logit-scale`, `--cell-prior-fixed-logit-scale`);
  - cell-conditioned pair FiLM (`--cell-pair-film-scale`);
  - supervised contrastive loss over raw covariates (`--aux-covariate-contrastive-*`);
  - within-covariate ranking loss (`--ranking-loss-*`);
  - fold-train-only gene weighting for MSE (`--mse-gene-weight-mode {variance,pdi,variance_pdi}`);
  - optional pair auxiliary logit gate (`--pair-logit-gate`).
- Data handling changes are limited to the fast training dataset and do not modify processed data or data-processing scripts:
  - `FastProteinTalkDataset` now returns both mapped covariates and raw covariates, so auxiliary losses can use true raw labels while the main model still uses UNK-mapped covariates;
  - optional `prior_features` are passed per row only when enabled.
- Full 5-fold / bottleneck results on `single_cell_5fold`, 1 GPU, batch size 256, full covariate UNK dropout `0.15`:
  - current baseline (`MSE_WEIGHT=0.075`): AUROC `0.934443`, AUPRC `0.767008`;
  - `MSE_WEIGHT=0.05`: AUROC `0.931399`, AUPRC `0.762206`;
  - pair auxiliary logit scale `1.0`: AUROC `0.929817`, AUPRC `0.760380`;
  - pair auxiliary logit scale `2.0`: AUROC `0.931444`, AUPRC `0.763142`;
  - KNN drug prior, learned/fixed prior, contrastive loss, ranking loss, larger hidden dim, positive sampler, MSE inactive reweighting, and variance-weighted MSE were tested on bottleneck folds 2/4 and did not beat the baseline bottleneck pattern.
- Cell-type sanity check:
  - current branch baseline full 5-fold (`MSE_WEIGHT=0.075`): AUROC `0.941852`, AUPRC `0.810250`;
  - `MSE_WEIGHT=0.10 + pair_logit_scale=2.0` was tested on folds 0/1 only and gave AUPRC `0.775353` / `0.714756`, not promising enough to continue.
- Conclusion:
  - no tested lightweight plan component reliably moves unseen cell AUPRC toward `0.85`;
  - the proposed train-only priors are strong as offline diagnostics but degrade the learned model on fold 2, so they remain experimental and default-off;
  - recommended unseen-cell setting remains full covariate UNK dropout `0.15` with `MSE_WEIGHT=0.075`.

## 2026-05-25 15:10 HKT Cell-Type Text Foundation Embedding Experiment

- Added an independent experiment folder `celltype_text_fm/`; root baseline model/training files were not modified.
- Implemented SapBERT-based frozen cell-type semantic embeddings:
  - current 8 `cell_type` labels are expanded into biomedical prompts such as cancer lineage/cell-line descriptions;
  - SapBERT CLS embeddings are averaged per cell type and L2-normalized;
  - row-level `(N, 768)` features are cached under `celltype_text_fm/artifacts/` and passed to the fast model via the existing `prior_features` path.
- Network/download note:
  - direct HuggingFace download was very slow;
  - used `proxy_on2` from `~/.bashrc`, after which SapBERT download and dry-run succeeded.
- Validation:
  - `python -m py_compile celltype_text_fm/text_features.py celltype_text_fm/train_text_celltype.py` passed;
  - `bash -n celltype_text_fm/run_bottleneck_2gpu.sh celltype_text_fm/run_folds_2gpu.sh` passed;
  - dry-run produced `prior_features=(8, 768)`.
- Results on `single_cell_5fold`, 1 GPU per fold, batch size 256, full covariate UNK dropout `0.15`, `MSE_WEIGHT=0.075`:
  - baseline: AUROC `0.934443`, AUPRC `0.767008`;
  - adding SapBERT text feature while keeping categorical `cell_type`, tested on folds 2/4: AUPRC `0.610205 / 0.782888`, unstable;
  - replacing categorical `cell_type` with SapBERT text embedding: AUROC `0.929224`, AUPRC `0.767230`;
  - adding text-logit scale `0.5` on folds 2/4 hurt fold4 and was rejected.
- Conclusion:
  - the cell-type semantic embedding path is deployable and biologically cleaner than raw categorical `cell_type`;
  - current gain is negligible (`+0.00022` AUPRC) and AUROC drops, so it should remain an ablation/optional module rather than a new default.

## 2026-05-25 15:46 HKT Fold0 Covariate Analysis Implementation

- Added a covariate ablation/diagnostic workflow without changing data-processing scripts or data files:
  - `scripts/run_covariate_ablation_fold0_2gpu.sh` runs fold0 unseen-drug and unseen-cell covariate profiles across two GPUs;
  - `scripts/covariate_analysis_report.py` summarizes run manifests plus split-level covariate coverage/unseen-category diagnostics;
  - `scripts/ptv3_experiment_common.sh` now supports optional `BATCH_COV_LIST` env passthrough. Default behavior is unchanged when this env var is unset.
- Completed 38 fresh fold0 runs under `EXP_PREFIX=20260525_covariate_fold0_v1`; all runs reached `fit_completed` and `test_completed`.
- Final report artifacts:
  - `logs/20260525_covariate_fold0_v1_covariate_analysis.md`;
  - `logs/20260525_covariate_fold0_v1_covariate_analysis.json`;
  - `logs/20260525_covariate_fold0_v1_runtime_summary.tsv`.
- Key fold0 results versus the fresh full-covariate baseline:
  - unseen drug fold0 full baseline: AUROC `0.8448`, AUPRC `0.5565`;
  - unseen drug best profile was `drop_batch` (`machineID_new, Cell_plate, Cell, cell_type, pert_time`): AUROC `0.8565`, AUPRC `0.5870`, AUPRC delta `+0.0305`;
  - unseen cell fold0 full baseline: AUROC `0.8949`, AUPRC `0.7871`;
  - unseen cell best profile was `cell_identity_only` (`Cell, cell_type`): AUROC `0.9061`, AUPRC `0.8407`, AUPRC delta `+0.0536`;
  - the more reliable unseen-category variant `cell_identity_covunk015` reached AUROC `0.9088`, AUPRC `0.8393`, AUPRC delta `+0.0522`;
  - full covariate UNK dropout `0.15` reached unseen cell AUROC `0.9189`, AUPRC `0.8394`, AUPRC delta `+0.0524`.
- Split diagnostics:
  - unseen drug fold0 has almost no unseen covariate categories in test, except tiny `Cell_plate`/`batch` tails;
  - unseen cell fold0 has severe covariate shift: `Cell_plate` and `Cell` test rows are `100%` train-unseen, `batch` test rows are `91.67%` train-unseen, and `cell_type` test rows are `27.25%` train-unseen;
  - this supports treating high-cardinality technical covariates as risky in unseen-cell evaluation.

## 2026-05-25 17:22 HKT Target-Expression and FiLM Unseen-Cell Iteration

- Added default-off PDI/PPI-guided target-expression context for the fast model:
  - `--target-expression-mode {off,pdi,pdi_ppi}` builds a cached drug-by-expression-gene weight matrix from existing `pdi_matrix.npy` and optional one-hop `ppi_matrix.npy`;
  - the model converts the cached dense weights into sparse top-k `(gene_index, weight)` buffers, then pools control expression over drug-specific target/neighborhood proteins;
  - fusion is configurable with `--target-expression-fusion-mode {piece,control_add,pair_add}`; the branch is zero-initialized so enabled runs start close to baseline;
  - no data-processing code or source data files were modified.
- Added parameter plumbing to `train.py`, `infer.py`, and `scripts/ptv3_experiment_common.sh`; defaults keep previous baseline behavior unchanged (`target_expression_mode=off`, `cell_pair_film_scale=0`).
- Inference-side target-expression weights are built on the checkpoint protein axis, so extra-data inference can still align expression columns to the training checkpoint axis.
- Added reproducibility scripts:
  - `scripts/run_unseen_cell_targetexpr_film_fold0_2gpu.sh` for fold0 method1/method2 screening;
  - `scripts/run_unseen_cell_targetexpr_pairadd_5fold_2gpu.sh` for the best target-expression pair-add 5-fold setting.
- Fold0 screening on `single_cell_5fold_fold0`, 1 GPU, batch size 256, full covariate UNK dropout `0.15`, `MSE_WEIGHT=0.075`:
  - fresh baseline: AUROC `0.914478`, AUPRC `0.846185`;
  - FiLM-only was negative across scales: best tested AUPRC `0.834991`;
  - target-expression as extra fusion piece was negative: best tested AUPRC `0.837257` after zero-init;
  - target-expression `control_add` was below baseline: best tested AUPRC `0.840263`;
  - target-expression `pair_add` was the only positive fold0 variant: AUROC `0.922204`, AUPRC `0.847389`.
- Full 5-fold result for best method1 setting (`TARGET_EXPRESSION_MODE=pdi_ppi`, `TARGET_EXPRESSION_FUSION_MODE=pair_add`, top-k `256`, PPI top-k `32`, PPI alpha `0.5`, init scale `0.5`):
  - fold AUPRCs: `0.84739 + 0.75910 + 0.66351 + 0.81629 + 0.78267`;
  - average AUROC/AUPRC: `0.932174 / 0.773791`;
  - compared with the current full covariate UNK + `MSE_WEIGHT=0.075` unseen-cell baseline (`0.934443 / 0.767008`), AUPRC improves by about `+0.0068` while AUROC drops by about `-0.0023`.
- Current conclusion:
  - method2 (`cell_pair_film`) should not be adopted for unseen cell;
  - method1 is only useful in the conservative `pair_add` form and gives a small AUPRC gain, not enough to solve the `0.85` target;
  - recommended status is optional ablation / candidate unseen-cell setting, not a global default for every task.

## 2026-05-25 18:02 HKT Target-Neighborhood Score Selection Iteration

- Extended the default-off target-expression branch with explicit neighbor scoring controls:
  - graph-side PPI score normalization via `--target-expression-ppi-norm {raw,row,symmetric}`;
  - hub-aware score penalty via `--target-expression-degree-penalty`;
  - cell-aware candidate reweighting via `--target-expression-cell-gate-mode {off,magnitude,signed}`, scale, and temperature.
- Added `scripts/run_unseen_cell_neighbor_score_fold0_2gpu.sh` for fold0 neighbor-score screening; updated the pair-add 5-fold script to pass the new scoring/gating environment variables.
- Fold0 screening on `single_cell_5fold_fold0`, same base setting as the previous pair-add branch:
  - raw pair-add baseline: AUROC `0.922204`, AUPRC `0.847389`;
  - symmetric degree-normalized PPI: AUROC `0.915830`, AUPRC `0.834186`;
  - raw PPI with degree penalty `0.5`: AUROC `0.921243`, AUPRC `0.844042`;
  - cell-expression magnitude gate scale `1.0`: AUROC `0.922484`, AUPRC `0.848715`;
  - signed expression gate scale `1.0`: AUROC `0.918085`, AUPRC `0.841246`;
  - top-k `128` with magnitude gate: AUROC `0.918404`, AUPRC `0.838637`;
  - top-k `512` with magnitude gate: AUROC `0.915256`, AUPRC `0.835685`.
- Full 5-fold for the best fold0 neighbor selector (`raw PDI/PPI pair-add + magnitude gate scale 1.0`):
  - fold AUPRCs: `0.84871 + 0.76254 + 0.63437 + 0.81394 + 0.73749`;
  - average AUROC/AUPRC: `0.929588 / 0.759410`.
- Current conclusion:
  - degree normalization and hub penalty are not useful for this artifact set; the raw PDI/PPI weights are stronger;
  - cell-aware magnitude gating gives a small fold0 gain, but it is not stable across folds and hurts fold2/fold4;
  - the recommended target-expression setting remains raw PDI/PPI `pair_add` without cell gate (`0.932174 / 0.773791`), while cell-aware gating is kept only as a negative/diagnostic ablation.

## 2026-05-25 19:49 HKT MSE-Gap Architecture Exploration

- Added default-off architecture/loss switches to test whether expression reconstruction can become a stronger auxiliary task:
  - `--response-delta-mode {off,summary,gate}` with `--delta-logit-scale`, `--response-delta-dim`, and `--response-delta-detach`;
  - `--mse-target-mode {all,pdi,pdi_ppi,topvar_pdi}` for reconstruction loss over drug-specific PDI/PPI target proteins;
  - `--mse-weight-schedule {constant,warmup_decay}` with decay controls.
- Added `scripts/run_mse_gap_delta_screen_2gpu.sh` for paired with-MSE / w/o-MSE screening on two GPUs. Defaults keep baseline4 unchanged.
- Validation:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py` passed;
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_mse_gap_delta_screen_2gpu.sh` passed;
  - dry-run and 1-epoch smoke passed for delta-gated response with PDI target MSE.
- Fold0/fold2 screening on `ptv3_main_singledrug / pert_stratified_5fold`, 1 GPU per run, batch size 256:
  - `delta_summary_pdi`: average AUPRC gap `+0.0490`, close to the 5-point target in the initial screen;
  - `delta_gate_pdi`: average AUPRC gap `+0.0245`;
  - PDI/PPI schedule, top-variance PDI schedule, full-gene delta variants, low-dimensional delta variants, and target-only MSE variants were unstable or below target.
- Full 5-fold confirmation for the best fold0/fold2 candidate `delta_summary_pdi`:
  - with MSE: AUROC `0.902626`, AUPRC `0.654613`;
  - w/o MSE: AUROC `0.898656`, AUPRC `0.650194`;
  - gap: AUROC `+0.003970`, AUPRC `+0.004418`;
  - baseline4 remains stronger: AUROC `0.903489`, AUPRC `0.666491`, baseline4 w/o-MSE AUPRC gap `+0.021950`.
- Conclusion:
  - the tested delta-response and target-MSE architectures are useful as ablation modules but should not replace baseline4;
  - the fold0/fold2 MSE-gap gain did not generalize to 5-fold because no-MSE was stronger on fold1/fold3;
  - current reliable baseline remains baseline4, not the new MSE-gap variants.

## 2026-05-25 20:54 HKT MSE-Gap Training Strategy Exploration

- Added optional, default-off true EarlyStopping support:
  - `--early-stopping-patience` and `--early-stopping-min-delta` in `train.py`;
  - `EARLY_STOPPING_PATIENCE` and `EARLY_STOPPING_MIN_DELTA` plumbing in `scripts/ptv3_experiment_common.sh`;
  - default behavior is unchanged because early stopping is disabled unless patience is explicitly set.
- Extended `scripts/run_mse_gap_ckpt_strategy_2gpu.sh`:
  - supports generic fixed-final-epoch strategies such as `last3`, `last5`, `last8`, `last10`, `last20`;
  - supports `VARIANTS="mse"` or `VARIANTS="mse nomse"` so MSE-only sweeps can reuse a fixed no-MSE reference.
- 5-fold paired training-strategy results on `ptv3_main_singledrug / pert_stratified_5fold`, 1 GPU per job, batch size 256:
  - checkpoint metric `valid_auprc`: with/w/o AUPRC `0.652478 / 0.649925`, gap `+0.002553`;
  - checkpoint metric `valid_auroc`: with/w/o AUPRC `0.646351 / 0.651120`, gap `-0.004769`;
  - checkpoint metric `loss2`: with/w/o AUPRC `0.654020 / 0.660823`, gap `-0.006803`;
  - checkpoint metric `total_loss`: with/w/o AUPRC `0.638232 / 0.660823`, gap `-0.022592`;
  - fixed final epoch `last3 / last5 / last8 / last10 / last20` AUPRC gaps were `+0.006938 / +0.008071 / +0.023768 / +0.008423 / +0.012681`;
  - best fixed-epoch result was `last8`, with/w/o AUROC `0.896146 / 0.886069`, with/w/o AUPRC `0.664738 / 0.640970`, AUPRC gap `+0.023768`.
- MSE-weight and schedule sweeps under the best `last8` strategy:
  - default `MSE_WEIGHT=0.25`: AUPRC gap `+0.023768`;
  - `MSE_WEIGHT=0.5`: AUPRC gap `+0.020872`;
  - `MSE_WEIGHT=1.0`: AUPRC gap `-0.002439`;
  - `MSE_WEIGHT=0.5` with warmup-decay to `0.2x`: AUPRC gap `+0.020476`;
  - `MSE_WEIGHT=1.0` with warmup-decay to `0.1x`: AUPRC gap `+0.013399`.
- True EarlyStopping and short-budget best-checkpoint tests:
  - `patience=3`, monitor `valid_auprc`: with/w/o AUPRC `0.662596 / 0.649925`, gap `+0.012671`;
  - `patience=1`, monitor `valid_auprc`: with/w/o AUPRC `0.665790 / 0.667342`, gap `-0.001552`;
  - `MAX_EPOCHS=5` with best valid AUPRC: with/w/o AUPRC `0.677288 / 0.669393`, gap `+0.007895`;
  - `MAX_EPOCHS=8` with best valid AUPRC: with/w/o AUPRC `0.669420 / 0.654441`, gap `+0.014979`.
- Conclusion:
  - fair checkpoint/early-stop/MSE-weight training strategies did not expand the w/o-MSE AUPRC gap to 5 points;
  - the strongest tested training-only setting is fixed final epoch `last8`, but its gap is only `+0.0238`;
  - early stopping should remain optional infrastructure, not a default baseline change for the MSE ablation claim;
  - no data files or data-processing code were modified.

## 2026-05-26 12:05 HKT Model-Size Sweep on Unseen Drug/Cell

- Added `scripts/run_model_size_sweep_2gpu.sh`:
  - runs `fast_delta` capacity sweeps on `ptv3_main_singledrug` unseen-drug (`pert_stratified_5fold`) and unseen-cell (`cell_5fold`) splits;
  - uses two single-GPU workers by default;
  - keeps unseen-drug at baseline4 single-drug settings (`MSE_WEIGHT=0.25`, no covariate UNK);
  - keeps unseen-cell at the current stronger full-covariate UNK setting (`MSE_WEIGHT=0.075`, `COVARIATE_UNK_DROPOUT=0.15`).
- Added `scripts/model_size_sweep_report.py`:
  - summarizes `run_manifest.json` files into markdown/json reports;
  - supports merging multiple sweep prefixes for cross-run comparison.
- Completed full 5-fold sweeps on 2 H200 GPUs:
  - pure capacity / LR `3e-4`: `h128`, `h192`, `h256`, `h384`, `h512`, `h768`;
  - large-model LR `2e-4`: `h384`, `h512`, `h768`;
  - boundary large model: `h1024`, LR `2e-4`.
- Final combined report:
  - `logs/20260526_model_size_combined_report.md`;
  - `logs/20260526_model_size_combined_report.json`.
- Key results, compared against same-day `h384`, LR `3e-4` rerun:
  - unseen drug baseline rerun: AUROC `0.896696`, AUPRC `0.652478`, `68.0s/fold`, `17.2M` params;
  - unseen drug best: `h512`, LR `2e-4`, AUROC `0.903166`, AUPRC `0.677846`, `82.4s/fold`, `26.4M` params;
  - unseen cell baseline rerun: AUROC `0.930355`, AUPRC `0.748441`, `68.2s/fold`, `17.2M` params;
  - unseen cell best: `h768`, LR `2e-4`, AUROC `0.934922`, AUPRC `0.770017`, `100.6s/fold`, `40.8M` params;
  - `h1024`, LR `2e-4` regressed on both tasks, so capacity appears saturated before 1024 hidden dim;
  - `h128` is the fastest tested setting (`~55s/fold`, `5.6M` params) but loses unseen-drug AUPRC by about `0.0115` versus same-day `h384`.
- Current recommendation:
  - for unseen drug performance, use `h512` with LR `2e-4`;
  - for unseen cell only, `h768` with LR `2e-4` is best but slower and only slightly above the historical `MSE_WEIGHT=0.075 + covUNK` reference (`0.767008` AUPRC);
  - for efficiency-sensitive runs, `h128`/`h192` are viable compression ablations, but not the best-performance baseline.

## 2026-05-26 13:27 HKT Default h512 Profile

- Updated the default fast-delta model-size profile from h384 to h512:
  - `HIDDEN_DIM`: `384` -> `512`;
  - `EXPRESSION_LATENT_DIM`: `512` -> `768`;
  - `COVARIATE_EMBEDDING_DIM`: `64` -> `96`.
- Updated default learning rate from `3e-4` to `2e-4` so the no-env-var launcher matches the validated h512 full-suite setting.
- Applied the defaults consistently in:
  - `scripts/ptv3_experiment_common.sh`;
  - `scripts/0521_baseline4_8gpu_parallel.sh`;
  - `train.py`;
  - `infer.py`;
  - `model/fast_delta_model.py`;
  - `scripts/README_ptv3_experiments.md`.
- Reason:
  - full `exp_01` to `exp_08` comparison showed h512 is the better single default than h768 because it is faster, stronger on the primary single unseen-drug split, and has stronger graph/MSE ablation gaps.
- Validation:
  - `bash -n scripts/ptv3_experiment_common.sh && bash -n scripts/0521_baseline4_task_specific_8gpu_parallel.sh && bash -n scripts/0521_baseline4_8gpu_parallel.sh` passed;
  - `python -m py_compile train.py infer.py model/fast_delta_model.py` passed;
  - sourcing `scripts/ptv3_experiment_common.sh` with size/LR variables unset prints `512 768 96 2e-4`.

## 2026-05-26 14:22 HKT Extra Double-Drug test_label Evaluation

- Added `scripts/report_extra_doubledrug_test_label_auprc.py` to recompute updated extra double-drug AUPRC from existing `predictions.parquet` files without rerunning GPU inference.
- Updated Stage-1 extra double-drug standardization to prefer `data/rawdata/update_0526/extra_doubledrug/*_test_label.csv` and preserve raw `test` / `test_label` metadata.
- Updated Step-3 split generation so `ptv3_extra_doubledrug_*` `test_only` excludes `test=0` / `test_label=delete` rows while retaining them in feature tables for audit.
- Updated `infer.py` to carry `test` / `test_label` into predictions and emit `task_by_test_label` metrics when those columns exist.
- Updated training-ready split validation to expect the same filtered extra double-drug `test_only` anchors.
- Added normalized AUPRC reporting as `nauprc = auprc / auprc_baseline`, where `auprc_baseline = positive_count / valid_count`, because raw AUPRC is prevalence-sensitive.
- `scripts/exp_08_extra_double_all_train_infer.sh` now prints and writes the extra double-drug `test_label` AUPRC/nAUPRC report after running Nature, NC, and Guomics inference.
- Validation:
  - `python -m py_compile scripts/report_extra_doubledrug_test_label_auprc.py scripts/show_extra_results.py infer.py model/fast_lightning.py model/training_ready_lightning.py utils/00_standardize_rawdata.py utils/09_build_data_splits.py utils/03_validate_training_ready_outputs.py` passed.
  - `bash -n scripts/exp_08_extra_double_all_train_infer.sh` passed.
  - Existing full extra-double predictions were post-processed successfully with evaluable counts: Guomics `3107`, NC `15155`, Nature `22956`.

## 2026-05-26 16:22 HKT 2-GPU Full-Suite Launcher

- Added `scripts/0526_baseline4_task_specific_2gpu_parallel.sh`.
- The new launcher defaults to:
  - `EXP_PREFIX=20260526_final_task_specific`;
  - `GPU_IDS=0,1`;
  - `DEVICES=1`;
  - `RUN_EXTRA_TASKS=1`.
- It keeps the task-specific single/double fast-delta defaults from `scripts/0521_baseline4_task_specific_8gpu_parallel.sh`.
- It reuses the maintained full-suite scheduler `scripts/0521_baseline4_8gpu_parallel.sh`, so the execution flow remains `exp_01` through `exp_08`, including the updated extra double-drug `test_label` report in `exp_08`.
- Validation:
  - `bash -n scripts/0526_baseline4_task_specific_2gpu_parallel.sh` passed;
  - `bash -n scripts/0521_baseline4_8gpu_parallel.sh` passed;
  - script mode set to executable (`755`).

## 2026-05-26 17:15 HKT h512 0526 Full-Suite Validation

- Reviewed the current h512 baseline4 defaults and confirmed the active profile is:
  - `HIDDEN_DIM=512`;
  - `EXPRESSION_LATENT_DIM=768`;
  - `COVARIATE_EMBEDDING_DIM=96`;
  - `LEARNING_RATE=2e-4`.
- Reviewed `scripts/0526_baseline4_task_specific_2gpu_parallel.sh`:
  - constrains the maintained scheduler to `GPU_IDS=0,1` and `DEVICES=1`;
  - preserves task-specific single/double settings from the baseline4 scheduler.
- Updated `scripts/report_extra_doubledrug_test_label_auprc.py` so the extra double-drug grouped report includes AUROC and ACC in addition to AUPRC, prevalence baseline, nAUPRC, and counts.
- Confirmed current extra double grouped reporting uses `feature_row_index` to join predictions back to `data/rawdata/update_0526/extra_doubledrug`, filters `test=1`, removes `test_label=delete`, and reports:
  - `unseenCell_seenDrugCombo`;
  - `unseenCell_unseenDrugCombo`;
  - `combined`.
- Validation passed:
  - shell syntax checks for the 0526 launcher, 0521 scheduler, exp08 extra double script, and common experiment script;
  - Python compile checks for `train.py`, `infer.py`, model modules, `scripts/show_extra_results.py`, and the grouped extra double report script.
- Full 2-GPU run completed with all runtime rows successful:
  - command: `WANDB_MODE=offline EXP_PREFIX=20260526_h512_nauprc_2gpu_offline_v1 GPU_IDS=0,1 bash scripts/0526_baseline4_task_specific_2gpu_parallel.sh`;
  - runtime summary: `logs/20260526_h512_nauprc_2gpu_offline_v1_runtime_summary.tsv`;
  - launcher log: `logs/20260526_h512_nauprc_2gpu_offline_v1_launcher.log`.
- 5-fold averages from the run:
  - single unseen drug: AUROC `0.903166`, AUPRC `0.677846`, nAUPRC `5.723701`;
  - single unseen cell type: AUROC `0.933192`, AUPRC `0.792645`, nAUPRC `6.098465`;
  - single unseen cell: AUROC `0.927191`, AUPRC `0.751375`, nAUPRC `6.228606`;
  - single no MSE: AUROC `0.893265`, AUPRC `0.651609`, nAUPRC `5.499016`;
  - single no graph: AUROC `0.853110`, AUPRC `0.579863`, nAUPRC `4.876731`;
  - double unseen drug pair: AUROC `0.736534`, AUPRC `0.616861`, nAUPRC `1.526420`.

## 2026-05-27 00:31 HKT MSE-Gap Iteration

- Added default-off MSE-gap experiment switches:
  - `response_base_logit_scale`;
  - `zero_init_delta_head`;
  - `delta_logit_learnable`;
  - `control_expression_dropout`;
  - `mse_pretrain_epochs` / `mse_pretrain_bce_weight`;
  - `delta_teacher_loss_weight`.
- Extended `scripts/run_mse_gap_delta_screen_2gpu.sh` with paired two-GPU presets for expression-delta bridge, MSE warmup, high-dimensional delta, zero-init delta, learnable delta scale, expression dropout, and true-delta teacher experiments.
- Validation passed:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py`;
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_mse_gap_delta_screen_2gpu.sh`.
- Full 5-fold confirmation:
  - current h512 reference MSE gap remains AUPRC `+0.026237`;
  - conservative candidate `learn_delta_b075_g2_init01`: with/w/o MSE AUPRC `0.652089 / 0.644282`, gap `+0.007807`;
  - extreme expression-mediated pressure test `zdelta_b0_g0_dim32`: with/w/o MSE AUPRC `0.244446 / 0.120776`, gap `+0.123670`.
- Conclusion: a >5-point MSE gap can be forced only by destroying base performance. None of the conservative MSE-gap variants should replace the current h512 baseline.
- Detailed report: `docs/2026-05-27_mse_gap_experiment_results.md`.

## 2026-05-28 12:54 HKT Default nAUPRC Reporting and Dose Covariate Experiment

- Added default-off dose covariate support to training and inference:
  - `--use-dose-covariate`;
  - `--dose-covariate-fields`, defaulting to `pert_dose1 pert_dose2`.
- The default no-dose covariate list remains unchanged unless the dose switch is enabled.
- Updated shared experiment scripts so dose-enabled runs pass the same covariate configuration to both train and infer.
- Added `scripts/report_ptv3_exp_results.py` for exp01-exp08 summaries with nAUPRC in the default table.
- Updated model-size and covariate-analysis report scripts so nAUPRC is included by default.
- Did not rerun or modify data standardization, training-ready data construction, or split generation scripts.
- Validation passed:
  - Python compile checks for `train.py`, `infer.py`, and report scripts;
  - shell syntax checks for the common experiment script and relevant exp scripts;
  - dose-enabled dataset load check returned 8 covariates including `pert_dose1` and `pert_dose2`;
  - `git diff` for `utils/00_standardize_rawdata.py`, `utils/02_build_training_ready_data.py`, and `utils/09_build_data_splits.py` was empty.
- Full one-GPU dose run completed:
  - prefix: `20260528_dose_h512_lr2e4_v1`;
  - runtime summary: `logs/20260528_dose_h512_lr2e4_v1_runtime_summary.tsv`.
- Dose 5-fold averages:
  - exp01 single unseen drug: AUROC `0.901005`, AUPRC `0.658997`, nAUPRC `5.567505`;
  - exp02 single unseen cell type: AUROC `0.945831`, AUPRC `0.815531`, nAUPRC `6.188013`;
  - exp03 single unseen cell: AUROC `0.928799`, AUPRC `0.765623`, nAUPRC `6.415258`;
  - exp04 single no MSE: AUROC `0.895101`, AUPRC `0.628842`, nAUPRC `5.299508`;
  - exp05 single no graph: AUROC `0.853265`, AUPRC `0.605805`, nAUPRC `5.107087`;
  - exp06 double unseen drug pair: AUROC `0.820416`, AUPRC `0.768876`, nAUPRC `1.911960`.

## 2026-05-28 23:09 HKT Dose Covariate Parameter Search

- Added `scripts/run_dose_param_search.sh` and `scripts/dose_param_search_report.py` for staged dose-enabled hyperparameter search without changing model size:
  - fixed model profile: hidden dim `512`, expression latent dim `768`, covariate embedding dim `96`;
  - dose enabled by `USE_DOSE_COVARIATE=1` with `pert_dose1 pert_dose2`;
  - search covered learning rate, batch size, dropout, MSE weight, inactive-label MSE weight, ranking loss, and covariate/dropout variants.
- Did not modify or rerun data processing, training-ready construction, or split-generation code.
- Validation passed:
  - `python -m py_compile train.py infer.py scripts/dose_param_search_report.py scripts/report_ptv3_exp_results.py`;
  - `bash -n scripts/run_dose_param_search.sh scripts/ptv3_experiment_common.sh`;
  - `git diff` was empty for `utils/00_standardize_rawdata.py`, `utils/02_build_training_ready_data.py`, `utils/09_build_data_splits.py`, and `data/Data_Process_[1-4].md`.
- Selected dose parameter sets:
  - exp01/04/05 single unseen pert_id and ablations: `mse075_drop010`;
  - exp03 unseen cell: `lr1e4`;
  - exp02 unseen cell type: `ctrl_drop020` after completing all compared stage3 candidates to 5 folds;
  - exp06 double unseen drug pair: `dbl_mse010`;
  - exp07 mat[1-4] extra single: `mse050_lr1e4`;
  - exp08 extra double drug: `lr3e4`.
- Final selected metrics:
  - exp01 single unseen drug: AUROC `0.898136`, AUPRC `0.660333`, nAUPRC `5.574654`;
  - exp04 w/o MSE: AUROC `0.895242`, AUPRC `0.648277`, nAUPRC `5.492080`;
  - exp05 w/o graph: AUROC `0.842970`, AUPRC `0.596229`, nAUPRC `5.023520`;
  - exp02 unseen cell type: AUROC `0.942154`, AUPRC `0.817573`, nAUPRC `6.237695`;
  - exp03 unseen cell: AUROC `0.932460`, AUPRC `0.774338`, nAUPRC `6.493304`;
  - exp06 double unseen drug pair: AUROC `0.808599`, AUPRC `0.736157`, nAUPRC `1.825010`;
  - exp07 extra single mean: AUROC `0.751748`, AUPRC `0.547417`, nAUPRC `2.582783`;
  - exp08 extra double mean: AUROC `0.627824`, AUPRC `0.079844`, nAUPRC `1.918203`.
- Stage1 ablation gaps from the final 5-fold run:
  - w/o MSE gap: AUPRC `+0.012056`;
  - w/o graph gap: AUPRC `+0.064104`.

## 2026-06-01 11:21 HKT Cell-Drug Time-Collapsed Evaluation

- Added `scripts/report_cell_drug_time_eval.py` to report every PTV3 exp01-exp08 result under three evaluation modes:
  - `original`: current row/time-point level evaluation;
  - `cell-drug-bylasttime`: one datapoint per cell-drug or cell-drug-combo using the maximum numeric `pert_time`; ties at the last time are averaged;
  - `cell-drug-avgtime`: one datapoint per cell-drug or cell-drug-combo using the average prediction across all time rows.
- The report computes AUROC, AUPRC, prevalence baseline, nAUPRC, count, positive count, negative count, label-conflict count, and missing-time group count.
- For exp01-exp06, the reporter can materialize fold-level test predictions from existing checkpoints with `infer.py`; this avoids retraining solely for re-evaluation.
- For exp07/exp08, the reporter reads existing extra inference `predictions.parquet` files directly.
- Updated `infer.py` so new prediction files carry `pert_time`, dose, and batch metadata when present. The reporter still joins back to each task `feature_table` by `feature_row_index`, so older prediction files remain evaluable.
- Added `scripts/run_selected_param_full_suite.sh` to rerun the final parameter-search-selected configuration:
  - exp01/04/05: `mse075_drop010`;
  - exp02: `ctrl_drop020`;
  - exp03: `lr1e4`;
  - exp06: `dbl_mse010`;
  - exp07: `mse050_lr1e4`;
  - exp08: `lr3e4`.
- The selected-suite launcher trains all selected experiments, runs extra inference, then calls the cell-drug time-collapsed reporter and writes:
  - `${OUTPUT_DIR}/${BASE_PREFIX}_cell_drug_time_eval.csv`;
  - `${OUTPUT_DIR}/${BASE_PREFIX}_cell_drug_time_eval.json`;
  - `${LOG_DIR}/${BASE_PREFIX}_cell_drug_time_eval.md`.
- Validation before full retraining:
  - `python -m py_compile infer.py scripts/report_cell_drug_time_eval.py scripts/report_ptv3_exp_results.py` passed;
  - `bash -n scripts/run_selected_param_full_suite.sh` passed;
  - extra single and extra double existing prediction smoke checks passed for the three evaluation modes.

## 2026-06-01 12:18 HKT Cell-Drug Full Selected Retraining Results

- Reran the full selected configuration suite with prefix `20260601_cell_drug_selected_full_v1` and `GPU_IDS=0`.
- Runtime summary: `logs/20260601_cell_drug_selected_full_v1_runtime_summary.tsv`.
  - `41/41` commands exited with status `0`;
  - `32` train commands and `9` inference commands completed.
- Cell-drug evaluation outputs:
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_time_eval.csv`;
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_time_eval.json`;
  - `logs/20260601_cell_drug_selected_full_v1_cell_drug_time_eval.md`.
- Summary metrics:
  - exp01 single unseen drug:
    - original: AUROC `0.898153`, AUPRC `0.666963`, nAUPRC `5.632660`;
    - cell-drug-bylasttime: AUROC `0.898841`, AUPRC `0.670592`, nAUPRC `5.687552`;
    - cell-drug-avgtime: AUROC `0.898938`, AUPRC `0.668244`, nAUPRC `5.665939`.
  - exp02 single unseen cell type:
    - original: AUROC `0.942208`, AUPRC `0.818603`, nAUPRC `6.245977`;
    - cell-drug-bylasttime: AUROC `0.941977`, AUPRC `0.818496`, nAUPRC `6.261922`;
    - cell-drug-avgtime: AUROC `0.943112`, AUPRC `0.821349`, nAUPRC `6.278684`.
  - exp03 single unseen cell:
    - original: AUROC `0.932452`, AUPRC `0.777002`, nAUPRC `6.514967`;
    - cell-drug-bylasttime: AUROC `0.931772`, AUPRC `0.776499`, nAUPRC `6.521652`;
    - cell-drug-avgtime: AUROC `0.933672`, AUPRC `0.781928`, nAUPRC `6.566714`.
  - exp04 single w/o MSE:
    - original: AUROC `0.895315`, AUPRC `0.651119`, nAUPRC `5.515379`;
    - cell-drug-bylasttime: AUROC `0.895131`, AUPRC `0.651950`, nAUPRC `5.545950`;
    - cell-drug-avgtime: AUROC `0.896577`, AUPRC `0.655386`, nAUPRC `5.572692`.
  - exp05 single w/o graph:
    - original: AUROC `0.842961`, AUPRC `0.599514`, nAUPRC `5.051623`;
    - cell-drug-bylasttime: AUROC `0.842036`, AUPRC `0.594336`, nAUPRC `5.025268`;
    - cell-drug-avgtime: AUROC `0.844718`, AUPRC `0.601805`, nAUPRC `5.085172`.
  - exp06 double unseen drug pair:
    - original: AUROC `0.810580`, AUPRC `0.754331`, nAUPRC `1.872959`;
    - cell-drug-bylasttime: AUROC `0.796436`, AUPRC `0.714644`, nAUPRC `1.954498`;
    - cell-drug-avgtime: AUROC `0.800986`, AUPRC `0.720658`, nAUPRC `1.971408`;
    - collapsed modes skipped `10` label-conflict cell-drug-combo groups.
  - exp07 extra single:
    - original: AUROC `0.751748`, AUPRC `0.547417`, nAUPRC `2.582783`;
    - cell-drug-bylasttime: AUROC `0.751827`, AUPRC `0.547771`, nAUPRC `2.583577`;
    - cell-drug-avgtime: AUROC `0.751827`, AUPRC `0.547771`, nAUPRC `2.583577`;
    - collapsed modes skipped `28` label-conflict cell-drug groups.
  - exp08 extra double:
    - original: AUROC `0.627824`, AUPRC `0.079844`, nAUPRC `1.918203`;
    - cell-drug-bylasttime: AUROC `0.613931`, AUPRC `0.069015`, nAUPRC `1.822739`;
    - cell-drug-avgtime: AUROC `0.613931`, AUPRC `0.069015`, nAUPRC `1.822739`;
    - collapsed modes skipped `474` label-conflict cell-drug-combo groups.
- For exp07 and exp08, `pert_time` is unavailable for the extra feature rows used here, so `cell-drug-bylasttime` falls back to all rows for those groups and matches `cell-drug-avgtime`.

## 2026-06-01 15:41 HKT Cell-Drug-Dose Time-Collapsed Evaluation

- Updated `scripts/report_cell_drug_time_eval.py` so collapsed evaluation uses one datapoint per cell-drug-dose group:
  - single-drug grouping key: `Cell + pert_id1 + pert_dose1_norm`;
  - double-drug grouping key: `Cell + sorted((pert_id1, pert_dose1_norm), (pert_id2, pert_dose2_norm))`.
- Renamed collapsed report methods to:
  - `cell-drug-dose-bylasttime`;
  - `cell-drug-dose-avgtime`.
- Updated `scripts/run_selected_param_full_suite.sh` so future selected-suite reports write:
  - `outputs/${BASE_PREFIX}_cell_drug_dose_time_eval.csv`;
  - `outputs/${BASE_PREFIX}_cell_drug_dose_time_eval.json`;
  - `logs/${BASE_PREFIX}_cell_drug_dose_time_eval.md`.
- Re-evaluated the existing `20260601_cell_drug_selected_full_v1` predictions without retraining.
- New result document: `docs/2026-06-01_cell_drug_dose_time_eval_results.md`.
- Dose-aware output files:
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.csv`;
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.json`;
  - `logs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.md`.
- exp06 dose-aware collapsed metrics:
  - `cell-drug-dose-bylasttime`: AUROC `0.808264`, AUPRC `0.753058`, nAUPRC `1.867977`;
  - `cell-drug-dose-avgtime`: AUROC `0.812681`, AUPRC `0.757687`, nAUPRC `1.879982`;
  - label-conflict groups dropped from `10` under cell-drug-only grouping to `0` under cell-drug-dose grouping.
- Validation:
  - `python -m py_compile scripts/report_cell_drug_time_eval.py infer.py train.py` passed;
  - `bash -n scripts/run_selected_param_full_suite.sh` passed.

## 2026-06-01 16:03 HKT Extra Subset Reporting Fix

- Corrected the cell-drug-dose report presentation so exp07 and exp08 are reported by extra subset instead of only by `mean_extra`.
- `scripts/report_cell_drug_time_eval.py` now prints separate Markdown sections:
  - `Fold Summary` for exp01-exp06 mean5 rows;
  - `Extra Subset Summary` for exp07/exp08 subset rows;
  - `Extra Mean` as a convenience aggregate only;
  - `Fold Detail` for fold-level rows.
- Regenerated:
  - `logs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.md`;
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.csv`;
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.json`.
- Updated `docs/2026-06-01_cell_drug_dose_time_eval_results.md` so exp07 reports each available mat subset separately and exp08 reports `guomics`, `nature`, and `nc` separately.
- No retraining was run; this was a reporting/presentation correction over existing prediction files.
- Validation:
  - `python -m py_compile scripts/report_cell_drug_time_eval.py` passed;
  - `bash -n scripts/run_selected_param_full_suite.sh` passed.

## 2026-06-01 16:25 HKT exp08 test_label Reporting Fix

- Updated `scripts/report_cell_drug_time_eval.py` so exp08 extra double reports each dataset under three groups:
  - `unseenCell_seenDrugCombo`;
  - `unseenCell_unseenDrugCombo`;
  - `combined`.
- The grouping follows the earlier 2026-05-26 `test_label` reporting contract and uses the `test_label` column carried into prediction metadata.
- Regenerated:
  - `logs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.md`;
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.csv`;
  - `outputs/20260601_cell_drug_selected_full_v1_cell_drug_dose_time_eval.json`.
- Updated `docs/2026-06-01_cell_drug_dose_time_eval_results.md` so exp08 reports `guomics`, `nature`, and `nc` each under the three required groups.
- `mean_extra` continues to aggregate only `combined` rows, so exp08 test-label groups are not double-counted in the convenience aggregate.
- No retraining or inference was run; this was a reporting correction over existing prediction files.
- Validation:
  - `python -m py_compile scripts/report_cell_drug_time_eval.py` passed;
  - `bash -n scripts/run_selected_param_full_suite.sh` passed.

## 2026-06-01 20:55 HKT PTV3 Single-Drug Transcriptome AnnData Export

- Added `utils/12_export_single_drug_transcriptome_anndata.py` to export the single-drug, non-ablation experiment data to AnnData.
- Included exp01, exp02, exp03, and exp07; excluded exp04 `w/o MSE`, exp05 `w/o proteome/graph`, exp06 double-drug, and exp08 extra double-drug.
- Output root: `data/transcriptome_anndata/ptv3_single_nonablation`.
- Wrote common-gene-axis h5ad files under `common_gene_axis/` and native per-task h5ad files under `native_gene_axis/`.
- UniProt accessions were mapped to HGNC gene symbols using HGNC complete set, local UniProt FASTA `GN=`, and UniProt REST fallback.
- Mapping result: `11343` union UniProt features, `11310` mapped, `33` dropped for ambiguous or missing one-to-one HGNC mapping.
- Duplicate UniProt-to-gene mappings are collapsed by row-wise `nanmean`; original accessions remain in `adata.var["source_uniprot_ids"]`.
- Exact split membership is stored in `adata.uns["splits"]` and row-level `is_train/is_valid/is_test` columns, preserving the intentional train/test overlap in `all_train_subset_test`.
- Data README written to `data/transcriptome_anndata/ptv3_single_nonablation/README.md`.
- Detailed processing record written to `data/review/2026-06-01_2055_ptv3_single_transcriptome_anndata_processing.md`.
- Validation:
  - `python -m py_compile utils/12_export_single_drug_transcriptome_anndata.py` passed;
  - all common-axis h5ad files were readable with `anndata.read_h5ad(..., backed="r")`;
  - all common-axis h5ad files have identical `var_names`;
  - checked test rows have matched control rows available.

## 2026-06-01 22:07 HKT LLM Cell-Type Embedding Experiment

- Added frozen LLM cell-type embedding support for the fast training and inference path.
- Added `utils/11_build_cell_type_llm_embeddings.py` to generate cell-type natural-language descriptions with `gpt-5.4` and embeddings with `Qwen/Qwen3-Embedding-8B`.
- Generated the `ptv3` artifact:
  - `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.npz`;
  - `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.json`.
- Artifact details: shape `14 x 4096`; `no` cell type is zero vector; 13 real cell-type embeddings are L2-normalized.
- Added `--cell-type-llm-mode {off,frozen}` and `--cell-type-llm-embedding-path` to `train.py` and `infer.py`.
- `FastProteinTalkDataset` now emits optional `cell_type_features`; `FastDeltaDrugResponseModel` concatenates these frozen features with categorical covariate embeddings before covariate projection.
- Updated `scripts/ptv3_experiment_common.sh` and `scripts/report_cell_drug_time_eval.py` so LLM cell-type settings propagate through train, inference, and materialized fold prediction reports.
- Network/API check used `proxy_on2`; `ALL_PROXY` was unset for OpenAI-compatible Python client calls because the `flow_v2` env lacks SOCKS support.
- Ran exp01 parameter screen with `CELL_TYPE_LLM_MODE=frozen`; selected `mse050` for the full run.
- Ran the full exp01-exp08 suite under prefix `20260601_llm_celltype_full_v1`.
- Final result document: `docs/2026-06-01_llm_celltype_embedding_experiment_results.md`.
- Final report files:
  - `logs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.md`;
  - `outputs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.csv`;
  - `outputs/20260601_llm_celltype_full_v1_cell_drug_dose_time_eval.json`.
- Key mean5 results:
  - exp01 original AUROC/AUPRC/n-AUPRC: `0.896777 / 0.668358 / 5.644481`;
  - exp01 cell-drug-dose-avgtime: `0.897524 / 0.670102 / 5.681576`;
  - exp02 original: `0.938004 / 0.802386 / 6.088709`;
  - exp03 original: `0.930120 / 0.777466 / 6.511693`;
  - exp06 original: `0.805866 / 0.730924 / 1.811493`;
  - exp06 cell-drug-dose-avgtime: `0.808670 / 0.738014 / 1.827928`.
- exp07 reports the available extra single-drug mat subsets; exp08 reports guomics, nature, and nc separately, each under `unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`, and `combined`.
- Validation:
  - `python -m py_compile utils/11_build_cell_type_llm_embeddings.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py train.py infer.py scripts/report_cell_drug_time_eval.py` passed;
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_dose_param_search.sh scripts/exp_01_single_pert_stratified_5fold.sh scripts/exp_07_extra_single_all_train_infer.sh scripts/exp_08_extra_double_all_train_infer.sh` passed.

## 2026-06-02 11:55 HKT LLM Cell-Type Conditioned Unseen-Cell Architecture

- Reviewed the fast unseen-cell `cell_5fold` training path and prior exp03 results.
- Added trainable cell-type conditioning around frozen LLM cell-type embeddings in `model/fast_delta_model.py`.
- Preserved the old default behavior with `--cell-type-llm-fusion-mode covariate`.
- Added new fusion modes:
  - `piece`;
  - `film`;
  - `interaction`;
  - `hybrid`.
- Added `--cell-type-llm-condition-scale`, `--cell-type-llm-logit-scale`, and `--cell-type-llm-dropout` to `train.py` and `infer.py`.
- The raw LLM feature tensor is detached before entering the trainable projector, so the LLM cell-type embedding artifact remains frozen.
- Updated `scripts/ptv3_experiment_common.sh` to pass the new train/infer parameters.
- Added `scripts/run_unseen_cell_llm_condition_search.sh`, a bounded parameter search runner for exp03 unseen-cell with graph, dose, and frozen LLM cell-type embeddings forced on. Default max runtime is `86400` seconds.
- Review record: `data/review_summary/2026-06-02_1155_llm_celltype_condition_unseen_cell_review.md`.
- Validation:
  - GPU dry run passed on `cell_5fold_fold0` with graph, dose, target-expression, and hybrid frozen LLM conditioning;
  - `python -m py_compile train.py infer.py model/fast_delta_model.py dataset/training_ready_fast_dataset.py model/fast_lightning.py` passed;
  - `bash -n scripts/run_unseen_cell_llm_condition_search.sh scripts/ptv3_experiment_common.sh scripts/exp_03_single_cell_5fold.sh` passed.

## 2026-06-02 12:22 HKT LLM Unseen-Cell Fold0 AUPRC Target Reached

- Added residual scaling to the new cell-type LLM `piece`/`interaction` branches and zero-initialized the interaction output after high-scale variants regressed fold0.
- Added `scripts/ensemble_prediction_probs.py` to average prediction parquet probabilities and compute AUROC/AUPRC/nAUPRC.
- Updated `scripts/report_cell_drug_time_eval.py` to forward the new LLM fusion/scale/dropout arguments.
- Ran frozen-LLM, graph-enabled, dose-enabled unseen-cell fold0 screens.
- Best single member:
  - `checkpoints/20260602_1220_llm_targetmag_v1_single_cell_fold0`
  - materialized prediction AUPRC `0.847852`.
- Final two-member ensemble:
  - `20260602_1220_llm_targetmag_v1`
  - `20260602_1226_llm_targetmag_seed7_v1`
- Final output:
  - `outputs/20260602_llm_targetmag_seed42_seed7_ensemble_fold0/predictions.parquet`
  - `outputs/20260602_llm_targetmag_seed42_seed7_ensemble_fold0/metrics.json`
- Final fold0 test metrics:
  - AUROC `0.922863`;
  - AUPRC `0.856793`;
  - nAUPRC `2.819591`;
  - count `1369`.
- Result document: `docs/2026-06-02_llm_celltype_condition_unseen_cell_results.md`.
- Validation:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py scripts/ensemble_prediction_probs.py scripts/report_cell_drug_time_eval.py` passed;
  - `bash -n scripts/ptv3_experiment_common.sh scripts/exp_03_single_cell_5fold.sh scripts/run_unseen_cell_llm_condition_search.sh` passed.

## 2026-06-02 13:29 HKT Single-Model Unseen-Cell Correction

- Corrected the earlier fold0 ensemble claim: it is invalid for the current requirement because the target is single-model 5-fold mean AUPRC, not fold0-only and not probability averaging.
- Added single-model architecture/loss options for further unseen-cell screening:
  - focal BCE in `model/fast_lightning.py`;
  - hard-positive/hard-negative pairwise ranking in `model/fast_lightning.py`;
  - optional `Cell_index` lookup for frozen text embeddings in `train.py` and `infer.py`;
  - optional observed perturb-expression branch in `model/fast_delta_model.py`, default off.
- Generated a frozen Cell-line text embedding artifact:
  - `data/training_ready/ptv3/derived/cell_line_llm_embedding_sapbert_768.npz`;
  - aligned to `Cell_index`, shape `74 x 768`, zero row reserved for `no`.
- Updated `scripts/ptv3_experiment_common.sh` and `scripts/report_cell_drug_time_eval.py` to propagate the new parameters.
- Ran valid single-model fold2 follow-up experiments; none improved the historical fold2 bottleneck (`0.650627`):
  - train-only KNN prior + frozen LLM hybrid: `0.646335`;
  - focal BCE + hard-negative ranking: `0.640760`;
  - frozen Cell-index SapBERT embedding: `0.633792`;
  - no Cell/cell_type categorical covariates + global ranking: `0.627146`;
  - prior + focal + hard ranking: `0.621385`;
  - observed perturb-expression delta branch: `0.601775`.
- Diagnostic probes also did not support a path to `0.85` under the current inputs:
  - non-ensemble tabular probe 5-fold mean `0.7400`;
  - expanded train-only prior mean about `0.7534`;
  - post-hoc perturb-expression/delta probe mean `0.7357`;
  - old `new_features_tanh` probe mean `0.6364`.
- Updated result document:
  - `docs/2026-06-02_llm_celltype_condition_unseen_cell_results.md`.
- Updated review record:
  - `data/review_summary/2026-06-02_1155_llm_celltype_condition_unseen_cell_review.md`.
- Validation:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py scripts/report_cell_drug_time_eval.py` passed;
  - `bash -n scripts/ptv3_experiment_common.sh scripts/exp_03_single_cell_5fold.sh` passed;
  - GPU dry-runs passed for frozen text embedding, graph, dose, hard-ranking/focal, Cell-index embedding, and observed-perturb branches.

## 2026-06-03 16:58 HKT update_0527 Extra Data + Dose Clip10

- Switched PTV3 extra single-drug inputs to `data/rawdata/update_0527/extra_singledrug`.
- Switched PTV3 extra double-drug inputs and the extra double test-label report default to `data/rawdata/update_0527/extra_doubledrug`.
- Added one stage-1 dose cleaner: numeric dose values are coerced with pandas and clipped at `10`; missing dose remains missing. No raw-dose audit columns are kept.
- Extra double-drug standardization now reads independent `pert_dose1` and `pert_dose2` from the update_0527 files. The old single `pert_dose` copy-to-both-slots rule is kept only as a legacy fallback.
- Added stage-2 fail-fast validation that numeric `pert_dose1` / `pert_dose2` must be in `[0, 10]`; negative or unclipped numeric dose now raises before category-index construction.
- Kept the existing dose-index rule: numeric dose maps to `ceil(dose)` and missing dose maps to the `"no"` bucket. Rebuilt `global_meta.json` has `value_to_index["pert_dose"]` max numeric index `10` and `"no"` index `"11"`.
- Updated cell-drug-dose reporting so joined prediction rows use training-ready `pert_dose*` / `pert_dose*_norm` values for dose grouping instead of stale raw prediction columns.
- Rebuilt:
  - `data/standardized`;
  - `data/training_ready`;
  - all data splits with `utils/09_build_data_splits.py --dataset-group all`.
- Axis comparison after rebuild:
  - `protein_index_to_id` unchanged;
  - `pert_index_to_id` changed.
- Because only the perturbation axis changed, regenerated only drug-side derived artifacts:
  - `data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl`;
  - `data/training_ready/ptv3/derived/ddi_matrix.npy` and `.meta.json`;
  - `data/training_ready/ptv3/derived/pdi_matrix.npy` and `.meta.json`.
- Did not run the full `scripts/0427_1.sh` GPU-heavy path because the protein axis did not change.
- Validation:
  - `python -m py_compile utils/00_standardize_rawdata.py utils/01_validate_standardized_outputs.py utils/02_build_training_ready_data.py utils/09_build_data_splits.py train.py infer.py` passed;
  - `python utils/01_validate_standardized_outputs.py` passed;
  - `python utils/02_build_training_ready_data.py` passed;
  - `python utils/09_build_data_splits.py --dataset-group all` passed;
  - acceptance checks confirmed standardized/training-ready numeric dose max `10.0`, min `0.0`, extra source paths under update_0527, extra double `test`/`test_label` columns, and independent clipped double-dose slots;
  - drug embedding shape `(6131, 2048)`, DDI shape `(6131, 6131)`, PDI shape `(6131, 11345)`;
  - `python utils/03_validate_training_ready_outputs.py` passed;
  - one-batch single and double training smokes with dose covariates passed using `--best-ckpt-metric total_loss --logger-backend none`;
  - one-batch extra single and extra double inference smokes passed;
  - `scripts/report_extra_doubledrug_test_label_auprc.py` passed on partial smoke predictions with update_0527 raw metadata;
  - targeted cell-drug-dose token check confirmed grouping uses clipped training-ready dose tokens;
  - `bash -n scripts/exp_07_extra_single_all_train_infer.sh scripts/exp_08_extra_double_all_train_infer.sh scripts/run_all_required_ptv3_experiments.sh scripts/ptv3_experiment_common.sh` passed.

## 2026-06-03 21:04 HKT PTV1 Isolated Pipeline And Formal Experiments

- Added an isolated PTV1 data path under `utils/ptv1/`, covering stage-1 standardization, standardized validation, training-ready build, training-ready validation, and split generation.
- Added PTV1-specific derived builders for ESM protein embeddings, Morgan drug embeddings, and PPI/PDI/DDI graph matrices. Outputs are written only under `data/training_ready/ptv1/derived`.
- Rebuilt final PTV1 data outputs:
  - `ptv1_aivc`: 15002 standardized rows, 15002 x 5576 feature matrix, 923 missing matched controls, 942 self-control rows;
  - `ptv1_extra_singledrug`: 218 standardized rows, 222 x 5576 training-ready feature matrix after 4 matched controls, 4 empty target lists.
- Rebuilt PTV1 splits:
  - `fixed_experiment_type` for exp_11;
  - `pert_id_5fold_fold0..4` for exp_12 with zero fold test-drug overlap;
  - `all_train_subset_test` for exp_13 all-data training;
  - `test_only` for `ptv1_extra_singledrug`.
- Built derived PTV1 artifacts:
  - protein ESM embedding `5578 x 1280`;
  - Morgan drug embedding `128 x 2048`;
  - PPI matrix `5578 x 5578`;
  - DDI matrix `128 x 128`;
  - PDI matrix `128 x 5578`.
- Added PTV1 experiment wrappers under `scripts/ptv1/`, using `--dataset-group ptv1`, graph cache `graph_cache/ptv1`, real graph features, dose covariates, and no Cell-type LLM embedding by default.
- Ran formal PTV1 experiments with prefix `20260603_2037_ptv1`:
  - exp_11 `fixed_experiment_type`: AUROC `0.957636`, AUPRC `0.915971`, n-AUPRC `3.127609`;
  - exp_12 unseen-drug 5-fold mean: AUROC `0.707712`, AUPRC `0.545776`, n-AUPRC `2.212531`;
  - exp_13 extra single-drug overall: AUROC `0.593455`, AUPRC `0.576296`, n-AUPRC `1.121719`, count `218`.
- Exp_13 reference epoch was selected from exp_12 best epochs `9, 4, 4, 7, 5`; mean `5.8`, rounded to epoch `6`, then all-PTV1 training used `max_epochs=7` and `last.ckpt`.
- Added result document:
  - `docs/2026-06-03_ptv1_experiment_results.md`.
- Added review record:
  - `data/review_summary/2026-06-03_2104_ptv1_pipeline_experiment_review.md`.
- Validation:
  - `python -m py_compile utils/ptv1/*.py scripts/ptv1/report_ptv1_exp_results.py` passed;
  - `bash -n scripts/ptv1/*.sh` passed;
  - `python utils/ptv1/01_validate_ptv1_standardized.py` passed;
  - `python utils/ptv1/03_validate_ptv1_training_ready.py` passed;
  - one-batch exp_11, exp_12 fold0, and exp_13 train/infer smokes passed;
  - formal exp_11, exp_12 all five folds, and exp_13 all-data train plus extra inference completed.

## 2026-06-03 22:12 HKT PTV1 Exp12/Exp13 Parameter Tuning

- Added PTV1 parameter-search tooling:
  - `scripts/ptv1/run_ptv1_param_search.sh`;
  - `scripts/ptv1/report_ptv1_param_search.py`.
- Tuned exp_12 unseen-drug on PTV1 with prefix `20260603_2108_ptv1_param_v1`.
- Stage 1 screened folds `0,2,4`; stage 2 promoted full 5-fold candidates `mse025` and `pos_auto_mse050`.
- Best exp_12 setting was `mse025`:
  - `MSE_WEIGHT=0.25`, `LEARNING_RATE=2e-4`, `BATCH_SIZE=256`, `DROPOUT=0.15`, `WEIGHT_DECAY=1e-4`;
  - exp_12 mean improved from baseline AUROC/AUPRC/n-AUPRC `0.707712 / 0.545776 / 2.212531` to `0.731340 / 0.570871 / 2.388662`.
- `pos_auto_mse050` was weaker on exp_12, with mean AUROC/AUPRC/n-AUPRC `0.705360 / 0.553605 / 2.293438`, but gave the best exp_13 extra AUPRC.
- Exp_13 all-PTV1 extra inference results:
  - baseline: AUROC/AUPRC/n-AUPRC `0.593455 / 0.576296 / 1.121719`;
  - `mse025`: `0.595982 / 0.581499 / 1.131846`;
  - `pos_auto_mse050`: `0.594550 / 0.587486 / 1.143500`.
- Added follow-up tuning results to:
  - `docs/2026-06-03_ptv1_experiment_results.md`.
- Added review record:
  - `data/review_summary/2026-06-03_2212_ptv1_tuning_review.md`.
- Validation:
  - `python -m py_compile scripts/ptv1/report_ptv1_param_search.py scripts/ptv1/report_ptv1_exp_results.py` passed before tuning;
  - `bash -n scripts/ptv1/run_ptv1_param_search.sh scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh scripts/ptv1/ptv1_experiment_common.sh` passed before tuning;
  - final static checks were rerun after the documentation update.

## 2026-06-04 11:38 HKT Corrected Cell LLM Clip10 Experiment Setup

- Marked the prior 2026-06-02/2026-06-03 cell-type LLM dose results invalid because the embedding input was tissue-level `cell_type`, not cell-line `Cell`.
- Replaced the LLM embedding builder with `utils/11_build_cell_llm_embeddings.py`.
- Built the corrected artifact:
  - `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz`;
  - `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.json`;
  - shape `74 x 4096`, aligned to `Cell_index`, row 0 `no` is a zero vector.
- Migrated train/infer/report and shell-wrapper public interfaces from `cell_type_llm_*` / `CELL_TYPE_LLM_*` / `--cell-type-llm-*` to `cell_llm_*` / `CELL_LLM_*` / `--cell-llm-*`.
- Removed the old index-column CLI; frozen Cell LLM features now always index by `Cell_index`.
- Added corrected selected-suite runner:
  - `scripts/run_cell_llm_clip10_selected_suite.sh`;
  - default prefix `20260604_cell_llm_dose_clip10_selected_v1`;
  - fail-fast checks for clip10 dose range, 74-row Cell embedding shape, row-0 zero vector, sidecar naming, and `Cell_index` bounds.
- Updated `scripts/run_unseen_cell_llm_condition_search.sh` and PTV1 helpers to use the new `CELL_LLM_*` environment names.
- Updated result document:
  - `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`.
- Added review record:
  - `data/review_summary/2026-06-04_1138_cell_llm_clip10_correction_review.md`.
- Validation:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py scripts/report_cell_drug_time_eval.py utils/11_build_cell_llm_embeddings.py` passed;
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_selected_suite.sh scripts/run_unseen_cell_llm_condition_search.sh scripts/ptv1/ptv1_experiment_common.sh scripts/ptv1/run_ptv1_param_search.sh` passed;
  - artifact checks passed for shape `(74, 4096)`, max `Cell_index=73`, row 0 zero vector, finite nonzero rows, and sidecar without old `cell_type_llm`/`cell_type_count` keys;
  - train/infer help exposes `--cell-llm-*` and no longer exposes `--cell-type-llm-*`.

## 2026-06-04 12:49 HKT Corrected Cell LLM Clip10 Full Exp01-Exp08 Run

- Launched and completed the corrected selected suite with prefix `20260604_cell_llm_dose_clip10_selected_v1`.
- Completed exp01-exp06 as 5-fold runs and exp07/exp08 as all-data extra-inference runs.
- Corrected reference epochs were recomputed from the corrected folds:
  - exp07 from exp01 fold epochs `11, 14, 9, 9, 11`, raw mean `10.8`, selected epoch `11`, all-data `max_epochs=12`;
  - exp08 from exp06 fold epochs `1, 2, 7, 17, 1`, raw mean `5.6`, selected epoch `6`, all-data `max_epochs=7`.
- Main corrected fold means, original metric:
  - exp01 single unseen drug: AUROC/AUPRC/n-AUPRC `0.891681 / 0.661912 / 5.588333`;
  - exp02 single unseen cell type: `0.945987 / 0.811791 / 6.099584`;
  - exp03 single unseen cell: `0.930525 / 0.776145 / 6.475614`;
  - exp04 single w/o MSE: `0.895744 / 0.662743 / 5.591658`;
  - exp05 single w/o graph: `0.844887 / 0.608214 / 5.127502`;
  - exp06 double unseen drug pair: `0.834738 / 0.782355 / 1.941130`.
- Extra corrected means, original metric:
  - exp07 extra single: AUROC/AUPRC/n-AUPRC `0.770714 / 0.569953 / 2.720985`;
  - exp08 extra double: `0.639328 / 0.094090 / 2.134296`.
- Generated report outputs:
  - `logs/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.md`;
  - `outputs/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.csv`;
  - `outputs/20260604_cell_llm_dose_clip10_selected_v1_cell_drug_dose_time_eval.json`;
  - `logs/20260604_cell_llm_dose_clip10_selected_v1_runtime_summary.tsv`.
- Updated result document:
  - `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`.
- Added review record:
  - `data/review_summary/2026-06-04_1249_cell_llm_clip10_full_suite_review.md`.
- Validation:
  - 32 corrected manifests found: exp01-exp06 have 5 folds each, exp07/exp08 have 1 all-data run each;
  - all corrected manifests use `cell_llm_mode=frozen`, `cell_llm_summary.embedding_rows=74`, and no old `cell_type_llm` keys;
  - exp05 manifests use `graph_feature_mode=zero`; all other corrected manifests use `graph_feature_mode=real`;
  - corrected report Markdown/CSV/JSON and runtime summary exist.

## 2026-06-04 16:41 HKT Corrected Cell LLM Clip10 Parameter Search Tooling

- Added a corrected tuning runner:
  - `scripts/run_cell_llm_clip10_param_search.sh`;
  - supports `RUN_MODE=screen`, `RUN_MODE=full`, `RUN_MODE=selected`, and `RUN_MODE=all`;
  - defaults to screen prefix `20260604_cell_llm_clip10_tune_v1`, full prefix `20260604_cell_llm_clip10_tune_v1_full`, and final prefix `20260604_cell_llm_clip10_tuned_selected_v1`.
- Added a corrected tuning report:
  - `scripts/report_cell_llm_clip10_param_search.py`;
  - uses original row/timepoint-level fold metrics for selection;
  - computes the corrected exp01/04/05 weighted stage1 score;
  - ranks exp02/03/06 by mean AUPRC with n-AUPRC and AUROC tie-breaks;
  - emits promoted or selected config names for the runner.
- Encoded the confirmed search plan:
  - stage1 exp01/04/05 shared candidates and graph/no-MSE ablations;
  - stage2 exp03, stage3 exp02, and stage4 exp06 task-specific candidates;
  - exp06 fixed double-drug pair settings `PAIR_FUSION_MODE=dual`, `PAIR_TYPE_FEATURES=1`, `USE_DDI=1`, `GRAPH_PAIR_ADD_SCALE=0.5`.
- Final selected mode uses stage1 parameters for exp07 and stage4 parameters for exp08, with mean-nearest reference epochs from the selected exp01/exp06 5-fold manifests and `last.ckpt` extra-data inference.
- No long tuning or final selected training jobs were launched during this code update.
- Updated result document:
  - `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`.
- Added review record:
  - `data/review_summary/2026-06-04_1641_cell_llm_clip10_param_search_tooling_review.md`.
- Validation:
  - `python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_llm_clip10_param_search.py` passed in `flow_v2`;
  - `bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh` passed;
  - `bash scripts/run_cell_llm_clip10_param_search.sh --help` passed;
  - report smoke test on historical `20260602_llm_dose_param_v1` stage4 artifacts passed and correctly marked old cell-type LLM manifests invalid.

## 2026-06-04 19:36 HKT Cell LLM Clip10 Goal Status Monitor

- Added a compact goal-status monitor:
  - `scripts/cell_llm_clip10_goal_status.py`;
  - reports screen/full/selected manifest counts for the `20260604_cell_llm_clip10_*` prefixes;
  - compares screen progress against the planned stage1-stage4 candidate counts;
  - separates active incomplete manifests from validation errors;
  - validates frozen Cell LLM manifests for `cell_llm_mode=frozen`, `cell_llm_summary.embedding_rows=74`, absence of old `cell_type_llm`, and expected graph mode (`zero` for exp05/single-no-graph, `real` otherwise).
- Confirmed the long screen launcher and post-screen handoff remain active:
  - screen process is continuing under `RUN_MODE=screen`;
  - post-screen launcher waits for the screen completion marker before launching `RUN_MODE=full` and then `RUN_MODE=selected`;
  - target final prefix remains `20260604_cell_llm_clip10_tuned_selected_v1`.
- Current monitor snapshot at the time of this note:
  - stage1 screen `109/135` manifests;
  - stage2/stage3/stage4 screen `0/42` manifests each;
  - one active incomplete manifest for the running `stage1/mse050_pre2` fold;
  - zero Cell LLM or graph-mode validation errors.
- Added review record:
  - `data/review_summary/2026-06-04_1936_cell_llm_clip10_goal_status_monitor_review.md`.
- Validation:
  - `python -m py_compile scripts/cell_llm_clip10_goal_status.py` passed in `flow_v2`;
  - `python scripts/cell_llm_clip10_goal_status.py` passed and reported zero validation errors for completed/current manifests.

## 2026-06-08 11:44 HKT PTV1 Cell LLM Embedding Formal Rerun

- Added a PTV1-specific Cell LLM builder/validator:
  - `utils/ptv1/07_build_ptv1_cell_llm_embeddings.py`;
  - delegates to `utils/11_build_cell_llm_embeddings.py`;
  - defaults to dataset group `ptv1`, `.env`, and `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz`;
  - validates metadata, `Cell_index` alignment, shape `20 x 4096`, row 0 zero, rows 1-19 finite/nonzero, and absence of legacy `cell_type_llm` keys.
- Updated PTV1 scripts:
  - `scripts/ptv1/0427_build_ptv1_derived.sh` can build the PTV1 Cell LLM artifact and supports `SKIP_CELL_LLM_EMBEDDING=1`;
  - `scripts/ptv1/ptv1_experiment_common.sh` now defaults to `CELL_LLM_MODE=frozen` and the PTV1 Cell LLM artifact path;
  - `scripts/ptv1/run_ptv1_param_search.sh` now uses the same frozen PTV1 Cell LLM defaults.
- Generated PTV1 Cell LLM artifact:
  - `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz`;
  - `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.json`;
  - model metadata: description `gpt-5.4`, embedding `Qwen/Qwen3-Embedding-8B`, normalized embeddings.
- Completed formal PTV1 rerun with prefix `20260608_ptv1_cell_llm_v1`:
  - exp_11 fixed experiment type: AUROC/AUPRC/n-AUPRC `0.947288 / 0.893452 / 3.050717`;
  - exp_12 unseen-drug mean5: `0.654052 / 0.511635 / 2.121972`;
  - exp_13 extra single-drug overall: `0.597414 / 0.578210 / 1.125445`;
  - exp_13 reference epochs were `3, 2, 6, 18, 0`, raw mean `5.8`, selected epoch `6`, all-train max epochs `7`.
- Generated report files:
  - `logs/20260608_ptv1_cell_llm_v1_ptv1_exp_results.md`;
  - `outputs/20260608_ptv1_cell_llm_v1_ptv1_exp_results.csv`.
- Updated result document:
  - `docs/2026-06-03_ptv1_experiment_results.md`.
- Added review record:
  - `data/review_summary/2026-06-08_1144_ptv1_cell_llm_rerun_review.md`.
- Validation:
  - `python -m py_compile train.py infer.py utils/11_build_cell_llm_embeddings.py utils/ptv1/*.py scripts/ptv1/report_ptv1_exp_results.py` passed in `flow_v2`;
  - `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh` passed;
  - PTV1 standardized and training-ready validators passed;
  - one-batch exp_11, exp_12 fold0, and exp_13 smoke runs passed;
  - final audit found 7 new manifests with `dataset_group=ptv1`, `cell_llm_mode=frozen`, `cell_llm_summary.embedding_rows=20`, no legacy `cell_type_llm` keys, and exp_13 references only from the new exp_12 folds.

## 2026-06-08 12:42 HKT PTV1 Frozen Cell LLM Fine-tune Search Tooling

- Added a formal frozen Cell LLM fine-tune launcher:
  - `scripts/ptv1/run_ptv1_fine_tune_search.sh`;
  - defaults to prefix `20260608_ptv1_cell_llm_tune_v1`;
  - serializes the full 32-candidate search on one GPU;
  - fixes `CELL_LLM_MODE=frozen`, PTV1 Cell LLM embedding path, `BEST_CKPT_METRIC=valid_auprc`, `LOGGER_BACKEND=none`, `LOG_TO_WANDB=0`, `PROGRESS_BAR=0`;
  - runs exp_11 fixed experiment type, exp_12 folds 0-4, exp_13 direct from exp_11, and exp_13 all-train from the exp_11 selected epoch for each candidate;
  - supports resume by skipping completed training manifests and completed 218-row extra inference outputs.
- Added exp_13-from-exp_11 wrapper:
  - `scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh`;
  - `direct` mode infers `ptv1_extra_singledrug` from the exp_11 best checkpoint;
  - `all_train` mode parses the exp_11 best checkpoint epoch, trains `all_train_subset_test` for `epoch+1`, records `exp13_from_exp11_policy`, and infers extra data from `last.ckpt`.
- Added fine-tune reporter:
  - `scripts/ptv1/report_ptv1_fine_tune_results.py`;
  - summarizes exp_11, exp_12 mean5, exp_13 direct, and exp_13 all-train;
  - defaults the exp_13 eligibility threshold to exp_11 AUPRC `0.883452`;
  - emits exp_11 best, exp_12 best, and exp_13 constrained best selections.
- No full 32-candidate formal search was launched during this code update.
- Smoke validation prefix `20260608_1240_ptv1_fine_tune_smoke`:
  - exp_11 one epoch/one train batch completed on GPU;
  - exp_12 fold0 one epoch/one train batch completed on GPU;
  - exp_13 direct produced 218 predictions;
  - exp_13 all-train used exp_11 epoch `0`, trained one epoch, and produced 218 predictions;
  - smoke report files: `logs/20260608_1240_ptv1_fine_tune_smoke_fine_tune_results.md` and `outputs/20260608_1240_ptv1_fine_tune_smoke_fine_tune_results.tsv`.
- Added review record:
  - `data/review_summary/2026-06-08_1242_ptv1_fine_tune_search_tooling_review.md`.
- Validation:
  - `python -m py_compile train.py infer.py scripts/ptv1/*.py utils/ptv1/*.py utils/11_build_cell_llm_embeddings.py` passed in `flow_v2`;
  - `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh` passed;
  - PTV1 standardized validator, training-ready validator, and Cell LLM `--validate-only` passed;
  - the first smoke attempt inside the sandbox could not initialize CUDA, then the same smoke was rerun with unsandboxed GPU access and passed.

## 2026-06-08 17:05 HKT PTV1 Frozen Cell LLM Fine-tune Full Search

- Launched and completed the formal 32-candidate search with prefix `20260608_ptv1_cell_llm_tune_v1`.
- Runtime window was 2026-06-08 13:19:08 to 17:03:05 HKT on one GPU.
- Final selections:
  - exp_11 best: `mse050_drop010`, AUPRC `0.926324`;
  - exp_12 best: `mse050_target_pdi`, mean5 AUPRC `0.584798`;
  - exp_13 constrained best: `mse025_graph_off`, direct extra AUPRC `0.604631`, with exp_11 AUPRC `0.890020`.
- Generated result files:
  - `logs/20260608_ptv1_cell_llm_tune_v1_fine_tune_results.md`;
  - `outputs/20260608_ptv1_cell_llm_tune_v1_fine_tune_results.tsv`;
  - `docs/2026-06-08_ptv1_frozen_cell_llm_fine_tune_report.md`.
- Formal audit passed:
  - 224/224 training manifests;
  - 32/32 exp_11 random split manifests;
  - 32/32 candidates with exp_12 folds 0-4;
  - 64/64 exp_13 inference manifests;
  - 64/64 exp_13 prediction files with 218 rows;
  - 288/288 runtime summary rows with status 0;
  - all formal training manifests had `dataset_group=ptv1`, `cell_llm_mode=frozen`, `accelerator=gpu`, and full-batch limits of `1.0`.
- Added review record:
  - `data/review_summary/2026-06-08_1705_ptv1_fine_tune_full_results_review.md`.

## 2026-06-08 17:12 HKT PTV1 Fine-tune Exp13 Selection Interpretation Correction

- Corrected the official exp_13 interpretation in `docs/2026-06-08_ptv1_frozen_cell_llm_fine_tune_report.md`.
- Official exp_13 now remains tied to the exp_11-selected parameterization:
  - exp_11 selected config: `mse050_drop010`;
  - exp_13 direct from exp_11 checkpoint: AUPRC/AUROC `0.561480 / 0.580062`;
  - exp_13 all_train with exp_11 parameters: AUPRC/AUROC `0.579478 / 0.603900`.
- Reclassified `mse025_graph_off` as diagnostic cross-candidate exp_13 leaderboard output only, not an official exp_13 selection, because it changes graph parameters relative to the exp_11-selected config.

## 2026-06-08 17:30 HKT PTV1 exp_13 From exp_11 Graph-On Official Rerun

- Added the exp_13-only selected rerun launcher:
  - `scripts/ptv1/run_ptv1_exp13_from_exp11_selected.sh`;
  - pins source exp_11 to `20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split`;
  - pins config to `mse050_drop010`, `MSE_WEIGHT=0.50`, `DROPOUT=0.10`, `MSE_TARGET_MODE=all`, `CELL_LLM_MODE=frozen`, and graph enabled with `GRAPH_FEATURE_MODE=real`, `GRAPH_STRUCTURAL_RP=1`, `GRAPH_DRUG_CONCAT=1`, `GRAPH_LOGIT_SCALE=2.0`;
  - runs only exp_13 `direct` and `all_train` through `scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh`.
- Added the exp_13-only reporter:
  - `scripts/ptv1/report_ptv1_exp13_from_exp11_selected.py`;
  - reports only source exp_11, exp_13 direct, and exp_13 all_train;
  - audits graph-on policy, source checkpoint binding, all_train `exp13_from_exp11_policy`, prediction row counts, and key parameter consistency between source and all_train.
- The first sandboxed launch failed during CUDA initialization in direct inference and left partial `_v1` artifacts; no files were deleted. The completed official rerun therefore used the auto-incremented prefix `20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010`.
- Completed official exp_13 rerun results:
  - source exp_11: AUROC/AUPRC/nAUPRC `0.963021 / 0.926324 / 3.162962`, selected epoch `5`;
  - exp_13 direct: AUROC/AUPRC/nAUPRC `0.580062 / 0.561480 / 1.092882`, rows `218`;
  - exp_13 all_train: AUROC/AUPRC/nAUPRC `0.603900 / 0.579478 / 1.127913`, rows `218`;
  - all_train manifest recorded `selected_epoch=5` and `applied_max_epochs=6`.
- Wrote the official rerun report:
  - `docs/2026-06-08_ptv1_exp13_from_exp11_graphon_rerun_report.md`;
  - `outputs/20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010_exp13_from_exp11_selected_report.tsv`.
- Validation passed:
  - `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`;
  - `python -m py_compile scripts/ptv1/*.py train.py infer.py`;
  - reporter formal audit for prefix `20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v2_mse050_drop010` returned `ok=true`.

## 2026-06-08 21:17 HKT PTV3 Cell + Cell-type LLM Clip10 Selected Suite

- Added frozen `cell_type` LLM embedding support alongside the existing frozen Cell LLM embedding path:
  - builder: `utils/11_build_cell_type_llm_embeddings.py`;
  - dataset sample field: `cell_type_llm_features`;
  - train/infer CLI and manifest fields: `--cell-type-llm-*`;
  - independent model branch in `model/fast_delta_model.py` with covariate/piece/film/interaction/hybrid fusion support.
- Added selected-suite tooling:
  - `scripts/run_cell_celltype_llm_clip10_selected_suite.sh`;
  - `scripts/report_cell_celltype_llm_selected_comparison.py`;
  - explicit `CELL_TYPE_LLM_*` propagation in `scripts/run_cell_llm_clip10_param_search.sh` and `scripts/ptv3_experiment_common.sh`.
- Generated formal cell-type embedding artifact:
  - `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz`;
  - shape `(14, 4096)`, zero row 0, finite rows, metadata `kind=cell_type_llm_embedding`, `input_field=cell_type`, `index_column=cell_type_index`.
- API regeneration using `proxy_on` and `.env` credentials was attempted but chat completion timed out; the formal `_v2` artifact reuses the existing validated Qwen3 cell-type embedding with enriched metadata.
- Completed exp01-exp08 selected suite with prefix `20260608_cell_celltype_llm_clip10_selected_v1`:
  - exp01 mean5 AUPRC/AUROC `0.664372 / 0.895949`;
  - exp02 mean5 AUPRC/AUROC `0.814609 / 0.945281`;
  - exp03 mean5 AUPRC/AUROC `0.768277 / 0.931817`;
  - exp04 mean5 AUPRC/AUROC `0.657075 / 0.906519`;
  - exp05 mean5 AUPRC/AUROC `0.612638 / 0.843826`;
  - exp06 mean5 AUPRC/AUROC `0.756564 / 0.816593`;
  - exp07 extra mean AUPRC/AUROC `0.555530 / 0.764085`;
  - exp08 extra mean AUPRC/AUROC `0.101612 / 0.643297`.
- Compared against `20260604_cell_llm_clip10_tuned_selected_v1`; comparison audit returned `audit_errors=0` and showed AUPRC gains for exp05 (`+0.010999`) and exp08 (`+0.009759`), with decreases for exp01/02/03/04/06/07.
- Wrote reports:
  - `docs/2026-06-08_cell_celltype_llm_clip10_selected_report.md`;
  - `outputs/20260608_cell_celltype_llm_clip10_selected_v1_cell_celltype_llm_comparison.tsv`;
  - `logs/20260608_cell_celltype_llm_clip10_selected_v1_cell_celltype_llm_comparison.md`;
  - `data/review_summary/2026-06-08_2117_cell_celltype_llm_selected_review.md`.

## 2026-06-08 21:42 HKT Cell + Cell-type LLM Clip10 Tuning Launch

- Added tuning-only launch tooling for the new graph + Cell LLM + cell-type LLM architecture:
  - `scripts/run_cell_celltype_llm_clip10_param_search.sh`;
  - `scripts/report_cell_celltype_llm_clip10_param_search.py`.
- Patched `scripts/run_cell_llm_clip10_param_search.sh` so the base clip10 search can use an injected reporter and can validate/resume manifests that include frozen `cell_type_llm` metadata.
- Launched the screen phase with prefix `20260608_cell_celltype_llm_clip10_tune_v1`, full prefix `20260608_cell_celltype_llm_clip10_tune_v1_full`, and selected prefix `20260608_cell_celltype_llm_clip10_tuned_selected_v1`.
- Fixed embeddings for the tuning run:
  - Cell: `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz`;
  - cell_type: `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz`.
- Static checks passed:
  - `bash -n scripts/run_cell_llm_clip10_param_search.sh scripts/run_cell_celltype_llm_clip10_param_search.sh`;
  - `python -m py_compile scripts/report_cell_celltype_llm_clip10_param_search.py scripts/report_cell_llm_clip10_param_search.py`.
- Training started in detached screen mode under `logs/20260608_cell_celltype_llm_clip10_tune_v1_launch.log`; an initial foreground debug launch left a partial checkpoint directory, so the official detached launch used `ALLOW_EXISTING_RUN=1` without deleting artifacts.

## 2026-06-09 14:08 HKT Cell + Cell-type LLM Clip10 Tuned Completion

- Completed the full tuning target for the graph + frozen Cell LLM + frozen cell-type LLM architecture.
- Screen prefix: `20260608_cell_celltype_llm_clip10_tune_v1`.
- Full promoted prefix: `20260608_cell_celltype_llm_clip10_tune_v1_full`.
- Final selected prefix: `20260608_cell_celltype_llm_clip10_tuned_selected_v1`.
- Final selected configs: stage1 `mse050`, stage2 `covdrop010_lr1e4`, stage3 `covdrop010`, stage4 `rank005`.
- Final selected original mean AUPRC/AUROC:
  - exp01 `0.669365 / 0.896935`;
  - exp02 `0.815855 / 0.945795`;
  - exp03 `0.784136 / 0.932252`;
  - exp04 `0.657075 / 0.906519`;
  - exp05 `0.606597 / 0.842547`;
  - exp06 `0.767741 / 0.825790`;
  - exp07 extra `0.560998 / 0.760442`;
  - exp08 extra `0.096357 / 0.650258`.
- Validation passed: `32/32` final selected manifests audited with `audit_errors=0`; exp05 graph zero, all other final runs graph real; Cell and cell-type LLM embeddings frozen and aligned.
- Wrote formal report: `docs/2026-06-09_cell_celltype_llm_clip10_tuned_results.md`.

## 2026-06-09 21:55 HKT PTV1 Cell + Cell-type LLM API Embedding Tuning Completion

- Generated and validated the API-based PTV1 `cell_type` LLM embedding artifact using `proxy_on2` plus `.env` API credentials:
  - `data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz`;
  - shape `(2, 4096)`, row 0 reserved zero vector, row 1 `BREAST` finite/nonzero and normalized;
  - description model `gpt-5.4`, embedding model `Qwen/Qwen3-Embedding-8B`.
- Updated PTV1 defaults so the PTV1 graph + Cell/Cell-type LLM workflow uses the v3 `cell_type` embedding by default while preserving the existing Cell LLM v2 artifact.
- Completed the full PTV1 exp11/exp12/exp13 tuning run with prefix `20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype`:
  - exp11 best `mse075_drop010`, AUPRC/AUROC/nAUPRC `0.914806 / 0.955703 / 3.123631`, epoch `3`;
  - exp12 best `mse075_drop010`, mean5 AUPRC/AUROC/nAUPRC `0.624837 / 0.760972 / 2.663382`;
  - exp13 valid best `mse050_target_pdi`, AUPRC/AUROC/nAUPRC `0.601089 / 0.627653 / 1.169976`, rows `218`;
  - exp13 oracle best `mse050_lr1e4`, AUPRC/AUROC/nAUPRC `0.636390 / 0.652502 / 1.238688`, epoch `16`, rows `218`.
- Wrote the formal PTV1 report:
  - `docs/2026-06-09_ptv1_cell_celltype_llm_exp11_13_tuned_results.md`;
  - generated artifacts `logs/20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype_cell_celltype_tune_results.md` and `outputs/20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype_cell_celltype_tune_results.tsv`.
- Validation passed:
  - Cell LLM and `cell_type` LLM embedding `--validate-only` checks;
  - `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`;
  - `python -m py_compile train.py infer.py utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`;
  - structured TSV and manifest audit: 13 candidates, 65 exp12 folds, exp13 valid/oracle rows `218`, graph `real`, Cell LLM `frozen`, `cell_type` LLM `frozen`, and v3 `cell_type` embedding for official rows.

## 2026-06-10 11:00 HKT PTV1 exp13 Positive-Weight Tuning Tooling

- Added a dedicated PTV1 exp13 positive-weight tuning runner:
  - `scripts/ptv1/run_ptv1_exp13_positive_weight_tune.sh`;
  - default prefix `20260610_ptv1_cell_celltype_llm_exp13_posweight_v1_mse050_target_pdi`;
  - if a requested prefix already has complete artifacts, the runner advances the `_vN_` version without overwriting prior results.
- Added a dedicated reporter:
  - `scripts/ptv1/report_ptv1_exp13_positive_weight_tune.py`;
  - outputs Markdown and TSV with separate exp13 `valid` and diagnostic `oracle` views;
  - audits the oracle checkpoint's exp11 test metrics after selecting each candidate's best exp13 oracle epoch.
- The sweep fixes all non-positive-weight settings to the June 9 exp13 valid-best baseline `mse050_target_pdi`: `MSE_WEIGHT=0.50`, `MSE_TARGET_MODE=pdi`, `LEARNING_RATE=2e-4`, `DROPOUT=0.15`, graph `real` + structural RP + drug concat, Cell LLM frozen v2, and cell-type LLM frozen v3.
- Candidate grid: `0.5`, `1`, `10`, `20`, `50`, `100`, `200`, `500`, and `neg/pos`.
- `neg/pos` is computed from the exp11 fixed train split as negatives `4688`, positives `2353`, ratio `1.9923501912`.
- Added report/review records:
  - `docs/2026-06-10_ptv1_exp13_positive_weight_tuning_report.md`;
  - `data/review_summary/2026-06-10_1100_ptv1_exp13_positive_weight_tuning_review.md`.
- Static checks passed:
  - `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`;
  - `python -m py_compile train.py infer.py scripts/ptv1/report_ptv1_exp13_positive_weight_tune.py scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`.
- Preflight validation passed:
  - `python utils/ptv1/03_validate_ptv1_training_ready.py`;
  - `python utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only --output data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz`;
  - `python utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py --validate-only --output data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz`.

## 2026-06-10 15:34 HKT PTV01-08 Positive-Weight Combo Tuning Runner

- Added a one-click PTV01-08 combo-search runner:
  - `scripts/run_ptv01_08_posweight_combo_tune.sh`.
- The runner follows `docs/2026-06-10_ptv01_08_posweight_combo_tuning_plan.md`:
  - removes `pre1/pre2` and `target_pdi/target_ppi` tuning from the search grid;
  - adds positive-weight configs `10`, `50`, `100`, `200`, `500`, and `neg/pos` via `posw_negpos`;
  - includes `LEARNING_RATE=5e-5` and `DROPOUT=0.50` combinations;
  - supports stage1-heavy screen/full promotion, reduced stage2-stage4 search, and final selected exp01-exp08 execution.
- Multi-GPU scheduling is built in through `GPU_IDS` and `MAX_PARALLEL_JOBS`; screen/full fold jobs and final selected exp01-exp06 folds are launched as single-GPU processes and distributed across the provided GPU list.
- Static checks passed:
  - `bash -n scripts/run_ptv01_08_posweight_combo_tune.sh scripts/ptv3_experiment_common.sh scripts/exp_01_single_pert_stratified_5fold.sh scripts/exp_02_single_cell_type_5fold.sh scripts/exp_03_single_cell_5fold.sh scripts/exp_04_single_no_mse_5fold.sh scripts/exp_05_single_no_pdi_5fold.sh scripts/exp_06_double_pert_pair_5fold.sh scripts/exp_07_extra_single_all_train_infer.sh scripts/exp_08_extra_double_all_train_infer.sh`;
  - `python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_celltype_llm_clip10_param_search.py`.
- Smoke-tested on the current 1-GPU H200 machine with prefix `smoke_ptv01_08_posweight_combo_20260610_153327`:
  - command used `SMOKE_TEST=1 RUN_MODE=screen GPU_IDS=0 MAX_PARALLEL_JOBS=1 STAGES=stage1`;
  - smoke configs: `posw10_lr2e4_drop015` and `posw50_lr5e5_drop050`;
  - all `6/6` fold jobs completed with status `0`;
  - generated 4-decimal Markdown/TSV tuning reports under `logs/` and `outputs/`.
- Manifest audit passed for the smoke run:
  - exp01/exp04 graph `real`, exp05 graph `zero`;
  - Cell LLM and cell-type LLM both `frozen`;
  - `MSE_TARGET_MODE=all`, `MSE_PRETRAIN_EPOCHS=0`;
  - positive weight, learning rate, and dropout matched the requested config tokens.
- Selected-mode smoke also passed with prefix `smoke_ptv01_08_posweight_combo_20260610_153802_selected`:
  - command used `SMOKE_TEST=1 RUN_MODE=selected GPU_IDS=0 MAX_PARALLEL_JOBS=1`;
  - selected exp01-exp06 fold0 jobs completed with status `0`;
  - exp07 and exp08 reference-epoch all-train extra inference completed;
  - exp07/exp08 reference paths resolved from selected exp01/exp06 fold checkpoints.
- Added review record:
  - `data/review_summary/2026-06-10_1534_ptv01_08_posweight_combo_runner_review.md`.

## 2026-06-10 15:59 HKT PTV01-08 Runner Per-GPU Slot Scheduling

- Updated `scripts/run_ptv01_08_posweight_combo_tune.sh` so fold jobs use explicit per-GPU slot scheduling.
- Added `JOBS_PER_GPU`, defaulting to `2`, so an 8-GPU launch now defaults to up to `16` concurrent fold jobs.
- `MAX_PARALLEL_JOBS` remains available as a global cap; if it exceeds `GPU_IDS * JOBS_PER_GPU`, the runner caps it to the slot capacity.
- The scheduler now tracks child PIDs and GPU load counters, releasing the exact GPU slot when each fold job exits.
- Launch logs include per-GPU load, for example `[launch][gpu=0][load=2/2]`.
- Smoke-tested on the current 1-GPU H200 machine with prefix `smoke_ptv01_08_posweight_combo_20260610_155917`:
  - command used `SMOKE_TEST=1 RUN_MODE=screen GPU_IDS=0 JOBS_PER_GPU=2 STAGES=stage1`;
  - all `6/6` fold jobs completed with status `0`;
  - manifest audit confirmed the requested positive weight, learning rate, dropout, graph mode, frozen Cell/cell-type LLM, and `MSE_TARGET_MODE=all`.
- Static checks passed:
  - `bash -n scripts/run_ptv01_08_posweight_combo_tune.sh`;
  - `python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_celltype_llm_clip10_param_search.py`.

## 2026-06-10 17:13 HKT PTV01-08 Tuning Reporter API Forwarding Fix

- Fixed `scripts/report_cell_celltype_llm_clip10_param_search.py` so it forwards the base tuning-report APIs used by the combo runner:
  - `summarize`;
  - `ranked_rows`;
  - `rank_sort_key`;
  - `promotable_rows`.
- Root cause: `scripts/run_ptv01_08_posweight_combo_tune.sh` generates stage1-B combo refinements by importing the Cell + cell-type reporter as a module, but the wrapper only exposed `main()` and `manifest_errors()`. After stage1-A report generation, the runner exited with `AttributeError: module 'r' has no attribute 'ranked_rows'`.
- The wrapper now loads the base Cell LLM tuning reporter once, patches its manifest validator with the Cell + cell-type validator, and reuses that patched module for CLI and imported API calls.
- Verified the failed stage1-B import path against prefix `20260610_ptv01_08_posweight_combo_v1`: `ranked_rows`, `rank_sort_key`, and `promotable_rows` are available, and 39 eligible stage1 rows were ranked.
- Static checks passed:
  - `python -m py_compile scripts/report_cell_celltype_llm_clip10_param_search.py scripts/report_cell_llm_clip10_param_search.py`;
  - `bash -n scripts/run_ptv01_08_posweight_combo_tune.sh`.
- Recovery note: rerunning the same command and prefix should skip completed fold jobs and continue from stage1-B / later stages; no completed artifacts need to be deleted.

## 2026-06-11 14:26 HKT PTV01-08 Positive-Weight Combo Final Report

- Audited the completed `20260610_ptv01_08_posweight_combo_v1` and `20260610_ptv01_08_posweight_combo_selected_v1` artifacts requested by `docs/2026-06-10_ptv01_08_posweight_combo_tuning_plan.md`.
- Found detailed raw reports in `logs/` and `outputs/`, but no durable long-form report under `docs/`.
- Added `docs/2026-06-10_ptv01_08_posweight_combo_tuning_report.md`, generated from:
  - `outputs/20260610_ptv01_08_posweight_combo_v1_param_search_report.tsv`;
  - `outputs/20260610_ptv01_08_posweight_combo_v1_full_param_search_report.tsv`;
  - `outputs/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_dose_time_eval.csv`;
  - selected-run `run_manifest.json` files;
  - GPU/runtime summary TSVs.
- The report records screen/full/final selected artifacts, selected configs, reference epochs, manifest audit, GPU distribution, final mean metrics, delta vs `20260604_cell_llm_clip10_tuned_selected_v1`, extra subset/source deltas, full screen table, and fold detail.
- No checkpoint rerun, GPU inference rerun, or retraining was needed because all required result artifacts were already present and complete.
- Added review record:
  - `data/review_summary/2026-06-11_1426_ptv01_08_posweight_combo_report_review.md`.

## 2026-06-11 22:53 HKT Patient Validation 260605 Inference Tasks

- Reviewed `docs/Data_Process_[1-4].md`, `docs/Training_guideline.md`, the stage-1/stage-2 data builders, split builder, `infer.py`, and exp_07/exp_08 checkpoint manifests before processing `data/rawdata/ptv2drug_patientVali260605`.
- Added `utils/13_build_patient_validation_inference_tasks.py` to build inference-only PTV3 training-ready tasks without expanding `data/training_ready/ptv3/global_meta.json`.
  - Reuses existing PTV3 protein/drug/covariate indices so exp_07/exp_08 checkpoint embedding shapes remain valid.
  - Encodes unseen patient covariates through existing `"no"` category indices.
  - Drops protein columns outside the current PTV3 protein index and records coverage in `data/training_ready/ptv3/patientVali260605_build_summary.json`.
  - Skips rows with more than two unique drug tokens because current exp_07/exp_08 models are two-slot drug models.
  - Marks `P3_lungCancer_2024cell` as non-standard because its matrix contains negative normalized values and cannot follow the standard log1p raw-abundance transform.
- Added `scripts/run_patient_vali260605_infer.sh` for reproducible GPU inference:
  - single-drug tasks use exp_07 best checkpoint `epoch=5-step=426.ckpt`;
  - double-drug tasks use exp_08 best checkpoint `epoch=2-step=234.ckpt`;
  - checkpoint-matching Cell LLM, cell-type LLM, graph, DDI, and dose-covariate arguments are set explicitly.
- Generated 8 inference tasks and test-only splits:
  - standard-compatible: P1 double, P2 single/double subset, P4 double, P5 single/double subset;
  - non-standard/exploratory: P3 single/double.
- Full inference completed successfully on GPU 0:
  - single outputs under `outputs/20260611_patientVali260605_exp07_single/`;
  - double outputs under `outputs/20260611_patientVali260605_exp08_double/`;
  - merged prediction table under `outputs/20260611_patientVali260605_combined/combined_predictions.csv`;
  - strict standard-compatible subset under `outputs/20260611_patientVali260605_combined/combined_predictions_standard_only.csv`.
- Added review record:
  - `data/review_summary/2026-06-11_2253_patient_vali260605_inference_review.md`.

## 2026-06-11 23:43 HKT Patient Validation 260605 v2 Feature Generation and Inference

- Superseded the first `patientVali260605` inference pass with `patientVali260605v2`, following the generalization contract for new feature-bearing entities.
- Updated `utils/13_build_patient_validation_inference_tasks.py`:
  - writes to `data/training_ready_patientVali260605v2`;
  - preserves checkpoint-sensitive PTV3 categorical embedding axes;
  - appends 1,901 patient-only proteins to the protein axis;
  - maps finite negative expression values to `NaN` before applying `log1p` to finite nonnegative values;
  - keeps `batch` and `machineID_new` unseen values mapped to `no`;
  - assigns every patient Cell label a separate `cell_llm_index` so patient Cell LLM features are generated instead of falling back to the old `Cell_index=no` representation.
- Added `utils/14_build_patient_validation_features.py` and generated v2 derived features with `proxy_on2`:
  - drug Morgan fingerprint: `[6131, 2048]`;
  - protein ESM embedding: `[13246, 1280]`, including 1,901 generated appended proteins;
  - PPI matrix: `[13246, 13246]`;
  - DDI matrix: `[6131, 6131]`;
  - PDI matrix: `[6131, 13246]`;
  - patient Cell LLM embedding: `[1117, 4096]`;
  - cell-type LLM embedding reused from PTV3 because all patient cell types are already present: `[14, 4096]`.
- Updated `train.py` and `infer.py` with configurable LLM embedding index columns:
  - `--cell-llm-index-column`, default `Cell_index`;
  - `--cell-type-llm-index-column`, default `cell_type_index`;
  - v2 inference passes `--cell-llm-index-column cell_llm_index`.
- Coverage audit:
  - raw unique proteins: 12,678; unresolved protein columns: 0;
  - raw unique drugs: 23; missing drugs: 0;
  - raw unique patient Cell labels: 1,116; generated Cell LLM labels: 1,116 plus reserved `no`;
  - missing cell types: 0;
  - `P3_lungCancer_2024cell` had 290,916 finite negative values; all were mapped to `NaN`; all generated task matrices have finite negative count 0.
- Skipped 370 rows because exp_07/exp_08 are two-drug-slot models and those rows contain more than two unique drug tokens.
- Updated `scripts/run_patient_vali260605_infer.sh` to use v2 tasks and v2 derived artifacts:
  - single-drug tasks use exp_07 checkpoint `epoch=5-step=426.ckpt`;
  - double-drug tasks use exp_08 checkpoint `epoch=2-step=234.ckpt`.
- Full v2 inference completed successfully:
  - per-task outputs under `outputs/20260611_patientVali260605v2_exp07_single/` and `outputs/20260611_patientVali260605v2_exp08_double/`;
  - combined outputs: `outputs/20260611_patientVali260605v2_combined_predictions.parquet` and `outputs/20260611_patientVali260605v2_combined_predictions.csv`;
  - total prediction rows: 876.
- Added review record:
  - `data/review_summary/2026-06-11_2343_patient_vali260605v2_feature_infer_review.md`.

## 2026-06-12 11:39 HKT Patient Validation 260605 v3 P3 Log2-Scale Reprocessing

- Superseded `patientVali260605v2` with `patientVali260605v3` for inference after auditing P3 expression scale.
- Updated `utils/13_build_patient_validation_inference_tasks.py`:
  - default output root is now `data/training_ready_patientVali260605v3`;
  - `P1/P2/P4/P5` remain raw-abundance matrices and use direct `log1p`;
  - `P3_lungCancer_2024cell` is treated as log2-relative scale and uses `exp2` followed by `log1p`;
  - P3 finite negative values are no longer mapped to `NaN`; all 290,916 negative finite values are converted by `exp2`.
- Updated `utils/14_build_patient_validation_features.py` and `scripts/run_patient_vali260605_infer.sh` to use configurable v3 run naming and v3 Cell LLM artifact paths.
- Regenerated v3 training-ready tasks:
  - total tasks: 8;
  - skipped rows: 370, all because exp_07/exp_08 are two-drug-slot models;
  - P3 transformed range after `exp2 -> log1p`: min approximately `0.00252`, max approximately `5.98289`;
  - all task matrices have finite negative count `0` and infinite count `0`.
- Generated v3 derived features with `proxy_on2`:
  - drug Morgan fingerprint: `[6131, 2048]`;
  - protein ESM embedding: `[13246, 1280]`, including 1,901 generated appended proteins;
  - PPI matrix: `[13246, 13246]`;
  - DDI matrix: `[6131, 6131]`;
  - PDI matrix: `[6131, 13246]`;
  - patient Cell LLM embedding: `[1117, 4096]`;
  - cell-type LLM embedding: `[14, 4096]`.
- Per-task feature coverage audit passed:
  - each task's ordered protein axis is covered by protein embedding, PPI, and PDI columns;
  - each task's perturbation indices are covered by drug fingerprint, DDI, and PDI rows;
  - each task's `cell_llm_index` is covered by the v3 Cell LLM embedding;
  - each task's `cell_type_llm_index` is covered by the cell-type LLM embedding;
  - no missing drugs, no unresolved protein columns, and no missing cell types.
- Full v3 inference completed:
  - single-drug tasks use exp_07 checkpoint `epoch=5-step=426.ckpt`;
  - double-drug tasks use exp_08 checkpoint `epoch=2-step=234.ckpt`;
  - per-task outputs under `outputs/20260612_patientVali260605v3_exp07_single/` and `outputs/20260612_patientVali260605v3_exp08_double/`;
  - combined outputs: `outputs/20260612_patientVali260605v3_combined_predictions.parquet` and `outputs/20260612_patientVali260605v3_combined_predictions.csv`;
  - total prediction rows: 876.
- Added review record:
  - `data/review_summary/2026-06-12_1139_patient_vali260605v3_p3_reprocess_feature_infer_review.md`.

## 2026-06-12 12:04 HKT PTV3 Cell LLM Default Path Fix

- Fixed `train.py::default_derived_paths()` so Cell LLM embedding defaults are dataset-group aware:
  - `ptv1` keeps `cell_llm_embedding_qwen3_4096_v2.npz`;
  - `ptv3` now defaults to the existing `cell_llm_embedding_qwen3_4096.npz`.
- Updated `--cell-llm-embedding-path` help text in `train.py` and `infer.py` so it does not name the wrong dataset-specific artifact.
- Validation passed:
  - `python -m py_compile train.py infer.py`;
  - direct default-path check confirms both PTV1 and PTV3 defaults point to existing artifacts.
- Added review record:
  - `data/review_summary/2026-06-12_1204_ptv3_cell_llm_default_path_fix_review.md`.

## 2026-06-12 17:26 HKT Patient Validation 260605 v3 Readable Output Export

- Added `utils/15_prepare_patient_validation_outputs.py` to prepare shareable patient-validation inference outputs from the existing v3 prediction artifacts without rerunning inference.
- Generated `outputs/0612v3/`:
  - `combined_predictions_readable.csv`, with readable dataset, task, cell, drug-pair, model-source, score-name, unified `prediction_score`, and rank columns;
  - `per_task_csv/*.csv`, converting all 8 per-task parquet prediction files to readable CSV;
  - `task_summary.csv`, `dataset_summary.csv`, `manifest.json`, and `README.md`.
- README summary reports 876 prediction rows across 5 datasets and 8 tasks:
  - 16 single-drug rows scored by exp_07 `pred_response_prob`;
  - 860 double-drug rows scored by exp_08 `pred_synergy_prob`;
  - 370 unsupported multi-drug rows skipped, all inherited from the v3 data-build summary.
- Validation passed:
  - `python -m py_compile utils/15_prepare_patient_validation_outputs.py`;
  - generated per-task CSV row count sums to 876, matching `combined_predictions_readable.csv`.

## 2026-06-12 19:24 HKT exp07/exp08 All-Epoch Checkpoint Retraining

- Retrained the final selected exp07/exp08 all-data runs with the original selected training configuration and `SAVE_TOP_K=-1` so every epoch checkpoint is retained.
- New checkpoint directories:
  - `checkpoints/20260612_ptv01_08_posweight_combo_selected_v1_allckpt_exp07_extra_single_all_train_infer_all_single_for_extra`;
  - `checkpoints/20260612_ptv01_08_posweight_combo_selected_v1_allckpt_exp08_extra_double_all_train_infer_all_single_double_for_extra`.
- The original reference-epoch policy was preserved:
  - exp07 references selected exp01 single unseen-drug folds, selected epoch `5`, applied `max_epochs=6`;
  - exp08 references selected exp06 double unseen-drug-pair folds, selected epoch `2`, applied `max_epochs=3`.
- Validation passed:
  - exp07 has epoch checkpoints `0, 1, 2, 3, 4, 5` plus `last.ckpt`;
  - exp08 has epoch checkpoints `0, 1, 2` plus `last.ckpt`;
  - checked core manifest fields against the original selected runs; task/model/LR/MSE/graph/DDI/dose/Cell LLM settings match, with only `save_top_k` changed from `1` to `-1`.

## 2026-06-12 19:35 HKT Patient Validation 260605 v3 All-Epoch Checkpoint Inference

- Added `utils/16_infer_patient_validation_all_epoch_ckpts.py` to run patientVali260605v3 inference over every saved exp07/exp08 epoch checkpoint and export readable comparison tables.
- Reused the existing v3 standardized tasks and derived features; no drug/protein/cell/cell-type feature artifacts were regenerated.
- Inference sources:
  - exp07 all-epoch checkpoints from `checkpoints/20260612_ptv01_08_posweight_combo_selected_v1_allckpt_exp07_extra_single_all_train_infer_all_single_for_extra`;
  - exp08 all-epoch checkpoints from `checkpoints/20260612_ptv01_08_posweight_combo_selected_v1_allckpt_exp08_extra_double_all_train_infer_all_single_double_for_extra`.
- Generated raw per-task inference outputs under `outputs/20260612_patientVali260605v3_all_epoch_ckpts/` and readable outputs under `outputs/0612v3_all_epoch_ckpts/`:
  - `combined_predictions_readable.csv`;
  - `checkpoint_summary.csv`;
  - `task_checkpoint_summary.csv`;
  - `dataset_checkpoint_summary.csv`;
  - `per_task_checkpoint_csv/*.csv`;
  - `manifest.json` and `README.md`.
- Validation passed:
  - 33 task/checkpoint prediction files were produced;
  - combined readable rows: 2,676 total, including 96 single-drug rows from 6 exp07 checkpoints and 2,580 double-drug rows from 3 exp08 checkpoints;
  - no missing `prediction_score` values;
  - final all-epoch rows (`exp07_epoch005` and `exp08_epoch002`) exactly match the previous `outputs/0612v3` final inference rows: 876/876 merged, max absolute score difference `0.0`.

## 2026-06-15 10:50 HKT exp09 Unified-Head All-Data Training Script

- Added unified fast-delta task-head support so single-drug sensitivity and double-drug synergy share one prediction head and one active-label loss path.
- Updated training and inference code to route `--task-head unified` through active response/synergy labels, manifest metadata, prediction export, and unified evaluation metrics.
- Added `scripts/exp_09_unified_all_train_valid_oracle.sh`:
  - trains on all merged single-drug plus double-drug data, matching exp08 all-data settings;
  - selects valid checkpoints from exp01/exp06 mean reference epochs;
  - evaluates exp07/exp08 extra datasets at valid checkpoints and across every saved epoch for oracle selection.
- Added `scripts/report_exp09_valid_oracle.py` to report valid and oracle metrics with the same detailed AUROC/AUPRC/baseline/n-AUPRC tables used by the selected-results workflow.
- Wired exp09 after exp08 in the required experiment launcher.
- Validation passed:
  - `py_compile` for `train.py`, `infer.py`, `model/fast_lightning.py`, and the new report script;
  - shell syntax checks for exp09 and required experiment launchers;
  - unified-head smoke check confirmed shared active logits and active-label masks.

## 2026-06-15 13:22 HKT exp09 Unified-Head Execution and Results

- Executed exp09 with prefix `20260615_1101_exp09_selectedref_v1` using all merged single-drug and double-drug training data, `task_head=unified`, one shared task-label loss, Cell LLM frozen embeddings, Cell-type LLM frozen embeddings, dose covariates, DDI, and dual pair fusion.
- Completed training to `max_epochs=50` and retained checkpoint epochs `0..49`.
- Evaluated `valid`:
  - exp07 extra single at selected exp01 reference epoch `5`;
  - exp08 extra double at selected exp06 reference epoch `2`;
  - 9 valid task manifests completed.
- Evaluated `oracle`:
  - epochs `0..49`;
  - 50 oracle output roots and 450 oracle task manifests completed.
- Runtime validation passed: `logs/20260615_1101_exp09_selectedref_v1_runtime_summary.tsv` has 460 rows and 0 nonzero statuses.
- Oracle best epochs from mean-extra original AUPRC:
  - exp07 best epoch `2`, AUPRC `0.596656`, AUROC `0.780482`;
  - exp08 best epoch `8`, AUPRC `0.101437`, AUROC `0.655564`.
- Added detailed result documentation:
  - `docs/2026-06-15_exp09_unified_head_valid_oracle_results.md`;
  - generated report `logs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.md`;
  - generated CSV/JSON under `outputs/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.*`.
- Optimized `scripts/report_cell_drug_time_eval.py::collapsed_frame()` from a per-group Python loop to vectorized pandas aggregation after the first report subprocess stalled for about 50 CPU-minutes; regenerated the report from the same completed prediction files without rerunning training or inference.

## 2026-06-15 15:20 HKT exp09 Patient Validation 260605 v3 First-50-Epoch Checkpoint Inference

- Added `utils/17_infer_patient_validation_exp09_all_epoch_ckpts.py` to run patientVali260605v3 inference over exp09 unified-head epoch checkpoints and export outputs matching the previous `outputs/0612v3_all_epoch_ckpts` structure.
- Reused existing standardized patient validation features from `data/training_ready_patientVali260605v3`; source raw data is `data/rawdata/ptv2drug_patientVali260605`.
- Ran only the first 50 exp09 checkpoints, epochs `0..49`, from `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra`.
- Used unified task-head inference for both single-drug sensitivity and double-drug synergy tasks; readable `prediction_score` is exported from `pred_task_prob`.
- Generated raw per-task inference outputs under `outputs/20260615_patientVali260605v3_exp09_all_epoch_ckpts/` and readable outputs under `outputs/0615v3_exp09_all_epoch_ckpts/`:
  - `combined_predictions_readable.csv`;
  - `checkpoint_summary.csv`;
  - `task_checkpoint_summary.csv`;
  - `dataset_checkpoint_summary.csv`;
  - `per_task_checkpoint_csv/*.csv`;
  - `manifest.json` and `README.md`.
- Validation passed:
  - 400 raw prediction parquet files and 400 per-task readable CSV files were produced;
  - combined readable rows: 43,800 total, including 800 single-drug rows and 43,000 double-drug rows;
  - 50 checkpoint labels and 8 patient validation tasks are present;
  - no missing `prediction_score` values; `score_name` is consistently `pred_task_prob`.
