# PTV01-08 Positive-Weight Combo Runner Review

Time: 2026-06-10 15:34 HKT

## Scope

- Added `scripts/run_ptv01_08_posweight_combo_tune.sh` for one-click PTV01-08 positive-weight combo tuning.
- Targeted the plan in `docs/2026-06-10_ptv01_08_posweight_combo_tuning_plan.md`.
- Kept `pre1/pre2` and `target_pdi/target_ppi` out of the tuning grid.
- Added positive-weight configs `10`, `50`, `100`, `200`, `500`, and `neg/pos` (`posw_negpos`).
- Added `LEARNING_RATE=5e-5` and `DROPOUT=0.50` combinations.
- Implemented screen/full/selected/all modes and per-fold multi-GPU scheduling through `GPU_IDS`.
- The selected phase keeps report-compatible checkpoint names while distributing exp01-exp06 folds across the same GPU job queue; exp07/exp08 remain sequential all-train extra-inference jobs after the reference folds exist.

## Static Checks

- Passed:
  - `bash -n scripts/run_ptv01_08_posweight_combo_tune.sh scripts/ptv3_experiment_common.sh scripts/exp_01_single_pert_stratified_5fold.sh scripts/exp_02_single_cell_type_5fold.sh scripts/exp_03_single_cell_5fold.sh scripts/exp_04_single_no_mse_5fold.sh scripts/exp_05_single_no_pdi_5fold.sh scripts/exp_06_double_pert_pair_5fold.sh scripts/exp_07_extra_single_all_train_infer.sh scripts/exp_08_extra_double_all_train_infer.sh`
  - `python -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_cell_celltype_llm_clip10_param_search.py`

## Smoke Test

- Current machine GPU inventory:
  - `0, NVIDIA H200, 143771 MiB`
- Screen command:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
SMOKE_TEST=1 RUN_MODE=screen GPU_IDS=0 MAX_PARALLEL_JOBS=1 STAGES=stage1 \
STAGE1_A_CONFIGS='posw10_lr2e4_drop015' \
STAGE1_B_CONFIGS='posw50_lr5e5_drop050' \
bash scripts/run_ptv01_08_posweight_combo_tune.sh
```

- Prefix: `smoke_ptv01_08_posweight_combo_20260610_153327`.
- Completed jobs: `6/6` with status `0`.
- Report artifacts:
  - `logs/smoke_ptv01_08_posweight_combo_20260610_153327_param_search_report.md`
  - `outputs/2026-06/2026-06-10/smoke_ptv01_08_posweight_combo_20260610_153327_param_search_report.tsv`
  - `logs/smoke_ptv01_08_posweight_combo_20260610_153327_gpu_job_summary.tsv`

## Manifest Audit

- `posw10_lr2e4_drop015` resolved to learning rate `0.0002`, dropout `0.15`, positive weight `10.0`.
- `posw50_lr5e5_drop050` resolved to learning rate `5e-05`, dropout `0.5`, positive weight `50.0`.
- All six smoke manifests recorded:
  - `run_status=fit_completed`
  - `test_status=test_completed`
  - `cell_llm_mode=frozen`
  - `cell_type_llm_mode=frozen`
  - `mse_target_mode=all`
  - exp01/exp04 graph `real`
  - exp05 graph `zero`

## Selected-Mode Smoke Test

- Command:

```bash
SMOKE_TEST=1 RUN_MODE=selected GPU_IDS=0 MAX_PARALLEL_JOBS=1 \
SELECTED_STAGE1_CONFIG='posw10_lr2e4_drop015' \
SELECTED_STAGE2_CONFIG='posw10_lr2e4_drop015' \
SELECTED_STAGE3_CONFIG='posw10_lr2e4_drop015' \
SELECTED_STAGE4_CONFIG='posw10_lr2e4_drop015' \
bash scripts/run_ptv01_08_posweight_combo_tune.sh
```

- Prefix: `smoke_ptv01_08_posweight_combo_20260610_153802_selected`.
- Selected exp01-exp06 fold0 jobs: `6/6` with status `0`.
- exp07 and exp08 completed reference-epoch all-train extra inference.
- Selected fold manifests recorded positive weight `10.0`, learning rate `0.0002`, dropout `0.15`, Cell/cell-type LLM frozen, and `mse_target_mode=all`.

## Full Run Entry Point

Use this on the 8-GPU machine:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 JOBS_PER_GPU=2 RUN_MODE=all \
bash scripts/run_ptv01_08_posweight_combo_tune.sh
```

`JOBS_PER_GPU` defaults to `2`, so the explicit setting above documents the intended 8-GPU behavior. `MAX_PARALLEL_JOBS` can still be used as a lower global cap if needed.

The runner is designed to complete the full search without manual monitoring, but periodic checks of the GPU job summary are useful for early failure detection:

```bash
tail -f logs/20260610_ptv01_08_posweight_combo_v1_gpu_job_summary.tsv
```

## Per-GPU Slot Update

- Updated at 2026-06-10 15:59 HKT.
- Added explicit GPU slot tracking:
  - `JOBS_PER_GPU=2` by default;
  - total default concurrency is `len(GPU_IDS) * JOBS_PER_GPU`;
  - launch logs show slot load, such as `[load=2/2]`;
  - child PIDs are mapped back to GPU indices so slots are released when jobs finish.
- Smoke-tested with:

```bash
SMOKE_TEST=1 RUN_MODE=screen GPU_IDS=0 JOBS_PER_GPU=2 STAGES=stage1 \
STAGE1_A_CONFIGS='posw10_lr2e4_drop015' \
STAGE1_B_CONFIGS='posw50_lr5e5_drop050' \
bash scripts/run_ptv01_08_posweight_combo_tune.sh
```

- Prefix: `smoke_ptv01_08_posweight_combo_20260610_155917`.
- Result: `6/6` fold jobs completed with status `0`; no launch exceeded `load=2/2`.
