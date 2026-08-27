# 2026-06-05 03:37 HKT Cell LLM Clip10 Tuned Selected Completion Review

Reviewed the corrected Cell LLM clip10 tuning artifacts after the all-mode runner completed.

## Scope

- Screen prefix: `20260604_cell_llm_clip10_tune_v1`
- Full prefix: `20260604_cell_llm_clip10_tune_v1_full`
- Selected prefix: `20260604_cell_llm_clip10_tuned_selected_v1`
- Runner: `scripts/run_cell_llm_clip10_param_search.sh`
- Monitors/reports: `scripts/cell_llm_clip10_goal_status.py`, `scripts/report_cell_llm_clip10_param_search.py`, `scripts/report_cell_drug_time_eval.py`

## Findings

- No blocking issues found in the final artifacts.
- Screen completed all required candidates: stage1 `135/135`, stage2 `42/42`, stage3 `42/42`, stage4 `42/42`.
- Full completed `100/100` promoted manifests with zero validation errors.
- Selected completed `32/32` manifests with zero validation errors.
- Final report artifacts exist:
  - `logs/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.md`
  - `outputs/2026-06/2026-06-04/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.csv`
  - `outputs/2026-06/2026-06-04/20260604_cell_llm_clip10_tuned_selected_v1_cell_drug_dose_time_eval.json`
- Manifest audit confirmed `cell_llm_mode=frozen`, `cell_llm_summary.embedding_rows=74`, no old `cell_type_llm` key, graph mode `zero` for exp05, and graph mode `real` for the other selected experiments.
- exp07 and exp08 reference policies used mean-nearest reference epochs from selected exp01/exp06 and selected `last.ckpt`.

## Selected Configs

- Stage1: `mse050_target_pdi`
- Stage2: `covdrop010_drop010`
- Stage3: `covdrop010_lr1e4`
- Stage4: `drop020_mseinactive010`

## Validation Commands

```bash
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_llm_clip10_param_search.py scripts/cell_llm_clip10_goal_status.py
bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python scripts/cell_llm_clip10_goal_status.py
```
