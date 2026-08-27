# 2026-06-02 18:18 LLM+Dose Parameter Tuning Review

## Scope

- Reviewed the current LLM cell-type embedding plus dose-covariate tuning state for PTV3 exp01..exp08.
- Checked experiment launchers, runtime summaries, checkpoint manifests, and result reports.

## Findings

- `20260601_llm_celltype_full_v1` is a valid frozen LLM cell-type embedding plus dose-covariate full-suite run, but it uses one selected parameter set rather than a complete LLM-specific search.
- `20260601_llm_celltype_exp01_v1` screened exp01 folds 0/2/4 for several configs, but initially did not include the exp04/exp05 ablation gap needed by the stated exp01/04/05 objective.
- No active training process was present before this tuning continuation.
- The available GPU is one idle NVIDIA H200, so the new searches were launched as concurrent single-GPU processes on GPU 0.

## Actions Started

- Started stage1 ablation completion for `20260601_llm_celltype_exp01_v1` with frozen LLM embedding and dose covariates, running exp04/exp05 on folds 0/2/4 for the existing exp01 screen configs.
- Started LLM+dose stage2, stage3, and stage4 screens under `20260602_llm_dose_param_v1`:
  - stage2: exp03 unseen cell, folds 0/2/4.
  - stage3: exp02 unseen cell type, folds 0/2/4.
  - stage4: exp06 double-drug unseen pair, folds 0/2/4.
- Kept hidden size fixed at 512 for these screens.

## Verification Plan

- After screens complete, run `scripts/dose_param_search_report.py` for the relevant prefixes.
- Select stage1 parameters by exp01 AUPRC with exp04/exp05 gap as the secondary objective.
- Select stage2/stage3/stage4 parameters by AUPRC.
- Run exp07 with selected exp01 parameters and exp08 with selected exp06 parameters, then verify n-AUPRC.
