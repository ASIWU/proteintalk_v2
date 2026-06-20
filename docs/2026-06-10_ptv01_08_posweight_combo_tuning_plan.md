# 2026-06-10 PTV exp01-exp08 PosWeight Combo Tuning Plan

- 搜索目标：
- 针对 PTV exp01-exp08 标准实验套件重新做一轮组合调参。
- 本轮不再调 `pre1/pre2`，也不再调 `target_pdi/target_ppi`；`MSE_TARGET_MODE` 固定为默认 `all`，避免把 MSE target-gene 策略和分类 positive weight 混在一起。
- 新增重点参数 `POSITIVE_WEIGHT`，候选固定为 `10`, `50`, `100`, `200`, `500`, `neg/pos`。
- 新增学习率候选 `5e-5`；保留已有 `1e-4`, `2e-4`, `3e-4`。
- 新增 dropout 候选 `0.50`；保留已有 `0.10`, `0.15`, `0.20`。
- 搜索必须包含组合候选，不只做单参数 ablation。
- stage1 做最重搜索；stage2/3/4 使用 stage1 结果缩小搜索空间。

- 固定前提：
- 不重建 raw/training-ready 数据。
- 不改模型结构。
- 沿用当前 exp01-exp08 runner 的 graph + frozen Cell LLM + frozen cell-type LLM 设置，除非单个 stage 的诊断任务本身要求关闭某个模块。
- exp05 仍为 no-graph 诊断，固定 `GRAPH_FEATURE_MODE=zero`；其他 graph-enabled 实验固定 `GRAPH_FEATURE_MODE=real`。
- exp04 仍为 no-MSE 诊断，脚本使用 no-MSE 路径；其他 stage 默认保留 MSE。
- `neg/pos` 在每个 train job 内根据该 job 的 active train labels 计算，并把 resolved positive weight 写入 manifest 和 report。
- 保留 baseline/unweighted 对照行，但 posweight grid 本身只包含 `10/50/100/200/500/neg/pos`。

- 搜索策略：
- stage1：同一候选参数同时跑 exp01、exp04、exp05；exp04 使用 no-MSE 脚本，exp05 使用 graph zero，其余参数与 exp01 相同。
- stage1 score 仍使用：
  `exp01_AUPRC + 0.5 * (exp01_AUPRC - exp04_AUPRC) + 0.5 * (exp01_AUPRC - exp05_AUPRC)`。
- stage1 screen 使用 folds `0 2 4`；stage1 full 使用 folds `0 1 2 3 4`。
- stage1 screen 分两层：
- stage1-A broad screen：覆盖 baseline、单因素和少量核心组合。
- stage1-B combo refine：从 stage1-A 中选 top positive weight、top LR、top dropout 后再做局部组合搜索。
- stage1-A 候选：
  - control：`base`, `mse050`, `mse075_drop010`, `mse050_lr1e4`, `mse050_lr3e4`, `mse050_lr5e5`, `mse050_drop010`, `mse050_drop020`, `mse050_drop050`。
  - posweight anchors：`posw10`, `posw50`, `posw100`, `posw200`, `posw500`, `posw_negpos`，固定 `LR=2e-4`, `DROPOUT=0.15`, `MSE_WEIGHT=0.50`。
  - posweight + LR probes：`posw10_lr5e5`, `posw10_lr1e4`, `posw10_lr3e4`, `posw100_lr5e5`, `posw100_lr1e4`, `posw100_lr3e4`, `posw500_lr5e5`, `posw500_lr1e4`, `posw500_lr3e4`, `posw_negpos_lr5e5`, `posw_negpos_lr1e4`, `posw_negpos_lr3e4`。
  - posweight + dropout probes：`posw10_drop010`, `posw10_drop020`, `posw10_drop050`, `posw100_drop010`, `posw100_drop020`, `posw100_drop050`, `posw500_drop010`, `posw500_drop020`, `posw500_drop050`, `posw_negpos_drop010`, `posw_negpos_drop020`, `posw_negpos_drop050`。
- stage1-B combo refine：
  - 从 stage1-A 选出 top 3 positive-weight family、top 2 LR、top 2 dropout。
  - 跑 `top_posweight x top_lr x top_dropout` 的完整组合，最多 12 个候选。
  - 如果 `dropout=0.50` 在 stage1-A 中没有进入 top 2，但 AUPRC 距 top dropout 小于 `0.0050`，额外保留 `dropout=0.50` 进入 combo refine。
  - 如果 `LR=5e-5` 在 stage1-A 中没有进入 top 2，但 AUROC 或 AUPRC 任一指标距 top LR 小于 `0.0030`，额外保留 `5e-5` 进入 combo refine。
- stage1 promote：
  - stage1-A + stage1-B 合并排序。
  - promote top 4 到 full 5-fold。
  - 若第 4 名和第 5 名 score 差值小于 `0.0020`，最多 promote 到 top 6。
  - full 5-fold winner 才能成为 exp01/04/05/exp07 的正式 selected config。

- stage2 exp03 候选：
- 目标是 unseen-cell，不做 stage1 级别大网格。
- 候选来自：
  - `base`；
  - stage1 full top 2 config；
  - stage1 screen 中最好的 `LR=5e-5` config；
  - stage1 screen 中最好的 `dropout=0.50` config；
  - stage-specific controls：`covdrop010`, `covdrop010_lr5e5`, `covdrop010_drop050`, `drop020`。
- screen 使用 folds `0 2 4`。
- promote top 3 到 full 5-fold；若第 3 名和第 4 名 AUPRC 差值小于 `0.0020`，最多 promote top 4。

- stage3 exp02 候选：
- 目标是 unseen-cell-type，搜索空间同 stage2，但 `covunk_celltype` 替代任何 cell-only unknown covariate candidate。
- 候选来自：
  - `base`；
  - stage1 full top 2 config；
  - stage1 screen 中最好的 `LR=5e-5` config；
  - stage1 screen 中最好的 `dropout=0.50` config；
  - stage-specific controls：`covdrop010`, `covdrop010_lr5e5`, `covdrop010_drop050`, `covunk_celltype`。
- screen 使用 folds `0 2 4`。
- promote top 3 到 full 5-fold；若第 3 名和第 4 名 AUPRC 差值小于 `0.0020`，最多 promote top 4。

- stage4 exp06 候选：
- 目标是 double unseen drug pair，搜索空间比 stage1 小，但保留 posweight 组合检查。
- 所有 exp06 候选固定保留 `PAIR_FUSION_MODE=dual`, `PAIR_TYPE_FEATURES=1`, `USE_DDI=1`, `GRAPH_PAIR_ADD_SCALE=0.5`。
- 候选来自：
  - `base`, `rank005`, `dbl_mse050`；
  - stage1 full top 2 config 映射到 double-drug runner；
  - fixed double posweight probes：`posw10`, `posw50`, `posw100`, `posw_negpos`；
  - compact combos：`posw10_lr5e5`, `posw50_lr5e5`, `posw100_lr5e5`, `posw10_drop050`, `posw50_drop050`, `posw100_drop050`。
- screen 使用 folds `0 2 4`。
- promote top 3 到 full 5-fold；若第 3 名和第 4 名 AUPRC 差值小于 `0.0020`，最多 promote top 4。

- Final selected run：
- 使用 prefix `20260610_ptv01_08_posweight_combo_selected_v1`。
- exp01/04/05 使用 stage1 full 5-fold 最优参数。
- exp02、exp03、exp06 分别使用各自 full 5-fold AUPRC 最优参数。
- exp07 all-data 使用 exp01 selected 参数；reference epoch 从 final exp01 5 folds 的 best epochs 取 mean + nearest rounding，`max_epochs=selected_epoch+1`，用 `last.ckpt` 推理 extra single。
- exp08 all-data 使用 exp06 selected 参数；reference epoch 从 final exp06 5 folds 的 best epochs 取 mean + nearest rounding，`max_epochs=selected_epoch+1`，用 `last.ckpt` 推理 extra double。
- final selected config 不能直接来自 3-fold screen，必须来自 full 5-fold。

- n-GPU 执行策略：
- 用户提供一台机器上的 n 张 GPU，例如 `GPU_IDS=0,1,2,3`。
- runner 必须解析 `GPU_IDS`，并把 train/infer jobs 按 job index 分配到 GPU：`gpu = GPU_IDS[job_index % n_gpu]`。
- 默认 `MAX_PARALLEL_JOBS=n_gpu`，每个 job 使用单 GPU；n=1 时自动顺序执行。
- 不默认使用 DDP；优先用多进程并行跑不同 candidate/fold/job，避免单个小任务占多卡导致 GPU 利用率低。
- 每个 job 单独写 log，runtime summary 记录 `gpu_id`, `stage`, `config`, `exp`, `fold`, `status`, `duration_seconds`, `artifact_path`。
- 如果某 GPU job 失败，只重跑失败 job；已完成且 manifest/config audit 通过的 job 可复用。
- full 5-fold 和 final selected 阶段也按同一 GPU 队列策略并行，不退化为只用 `cuda:0`。

- Report 要求：
- 生成 screen report、full report、final report 三类输出：
  - `logs/20260610_ptv01_08_posweight_combo_v1_param_search_report.md`
  - `outputs/20260610_ptv01_08_posweight_combo_v1_param_search_report.tsv`
  - `docs/2026-06-10_ptv01_08_posweight_combo_tuning_report.md`
  - `logs/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_dose_time_eval.md`
  - `outputs/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_dose_time_eval.csv`
  - `outputs/20260610_ptv01_08_posweight_combo_selected_v1_cell_drug_dose_time_eval.json`
- 所有 Markdown/TSV/CSV 指标保留 4 位小数。
- screen report 至少包含：
  - stage、rank、config、folds、complete；
  - resolved positive weight；
  - LR、dropout、MSE weight、graph mode；
  - exp01 AUPRC/AUROC/nAUPRC；
  - exp04 AUPRC；
  - exp05 AUPRC；
  - gap no-MSE、gap no-graph；
  - stage score；
  - promoted 标记。
- full report 至少包含：
  - 每个 promoted candidate 的 fold-level metrics；
  - mean/std AUPRC、AUROC、nAUPRC；
  - selected config；
  - 每个 stage 的 top candidate 和 runner manifest audit。
- final report 尽可能详细，必须包含三种分类/聚合方式：
  - `original`；
  - `cell-drug-dose-bylasttime`；
  - `cell-drug-dose-avgtime`。
- final report 对 exp01-exp08 都要报告 AUROC、AUPRC、nAUPRC、baseline、count、pos、neg。
- extra data 明细必须展开：
  - exp07 extra single：按 MAT/assay 子集报告，例如 `mat1_480_faims`, `mat1_qe`, `mat2_480_faims`, `mat2_qe`, `mat3_qe`, `mat4_qe`。
  - exp08 extra double：按三类 source 子集报告 `guomics`, `nc`, `nature`。
  - exp08 每个 source 还要保留 `combined`, `unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`。
- final report 需要给出 delta vs previous selected baseline：
  - exp01-exp08 original AUPRC/AUROC delta；
  - extra single mean delta；
  - extra double mean delta；
  - guomics/nc/nature combined delta。

## Test Plan

- 静态检查：
- `python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_llm_clip10_param_search.py`
- `bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh`
- 如果新增 runner/reporter，同步加入 `bash -n` 和 `py_compile`。

- Search 验收：
- stage1 不包含 `pre1`, `pre2`, `target_pdi`, `target_ppi` candidate。
- stage1 至少包含所有 posweight 候选：`10`, `50`, `100`, `200`, `500`, `neg/pos`。
- stage1 至少包含 `LR=5e-5` 和 `DROPOUT=0.50` 的候选。
- stage1-B 至少包含 1 组真实三因素组合：`positive_weight x lr x dropout`。
- stage2/3/4 search space 小于 stage1，且候选来自 stage1 top configs + 少量 stage-specific controls。
- 每个 train manifest 都记录 resolved positive weight；`neg/pos` 不能只写字符串，必须有数值 resolved weight。
- runtime summary 显示所有 `GPU_IDS` 都被使用；不能只有 `cuda:0`。
- 失败 job 可以重跑，但 final report 只能读取 status/pass 的 manifest。

- Full 5-fold 验收：
- promoted candidates 全部完成 5 folds。
- selected 参数来自 full 5-fold，不从 screen 直接定稿。
- full report 中所有 AUPRC/AUROC/nAUPRC 都保留 4 位小数。
- stage1 selected config 对 exp01、exp04、exp05 都有完整 5-fold manifest。
- stage2/3/4 selected config 各自有完整 5-fold manifest。

- Final suite 验收：
- exp01-exp06 各 5 folds；exp07/exp08 各 1 个 all-data run。
- exp07 extra single inference 产物完整。
- exp08 extra double inference 产物完整。
- final report 同时包含 `original`, `cell-drug-dose-bylasttime`, `cell-drug-dose-avgtime`。
- final report 包含 exp07 MAT/assay 子集，以及 exp08 `guomics/nc/nature` 三类 source 子集。
- final report 所有展示指标保留 4 位小数。
- 更新 `docs/2026-04-15_data_standardization_session_summary.md`，并在 `data/review_summary/` 新增带 HKT 小时分钟的 review 文件。

## Assumptions

- 本轮只调训练超参和分类 positive weight，不调 `pre1/pre2` 和 MSE target-gene 模式。
- `POSITIVE_WEIGHT` 只影响 BCE/classification loss；不影响 MSE target gene selection。
- `neg/pos` 使用每个 job 自己的 train split 标签比例，不跨 split 复用。
- stage1 是主要搜索预算所在；stage2/3/4 重点验证可迁移的 top configs 和少量 stage-specific 修正。
- final selection 主指标使用 original row-level AUPRC；final report 同时保留三种分类/聚合方式和 extra 子集明细。
- exp08 不使用 exp01 参数；exp08 使用 exp06 selected 训练参数和 exp06 reference epoch。
- 若某个 candidate 已有完整、同 prefix、同 config、同 fold manifest，可复用；否则重新训练，避免混入旧结果。
