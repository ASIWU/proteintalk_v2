# 2026-06-08 12:42 HKT PTV1 Fine-tune Search Tooling Review

## Scope

- Reviewed the existing PTV1 wrappers, shared PTV3 experiment launcher, PTV1 parameter-search reporter, exp_13 all-train script, and current formal PTV1 Cell LLM baseline artifacts.
- Implemented the frozen Cell LLM fine-tune workflow requested for exp_11, exp_12, and exp_13.

## Changes Reviewed

- Added `scripts/ptv1/run_ptv1_fine_tune_search.sh`.
  - Uses formal default prefix `20260608_ptv1_cell_llm_tune_v1`.
  - Encodes the full 32-candidate frozen Cell LLM search space.
  - Runs exp_11, exp_12 folds 0-4, exp_13 direct, and exp_13 all-train serially on one GPU.
  - Keeps logging offline with `LOGGER_BACKEND=none` and `LOG_TO_WANDB=0`.
  - Skips completed fit manifests and completed 218-row extra inference outputs for resume.
- Added `scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh`.
  - `direct` mode infers extra single-drug data from the exp_11 best checkpoint.
  - `all_train` mode parses exp_11 best epoch, trains `all_train_subset_test` for `epoch+1`, records `exp13_from_exp11_policy`, and infers from `last.ckpt`.
- Added `scripts/ptv1/report_ptv1_fine_tune_results.py`.
  - Aggregates exp_11, exp_12 mean5, exp_13 direct, and exp_13 all-train.
  - Marks exp_11 threshold pass/fail at default AUPRC `0.883452`.
  - Reports exp_11 best, exp_12 best, and exp_13 constrained best.
- Updated `scripts/ptv1/ptv1_experiment_common.sh` preflight to compile the new fine-tune reporter.

## Validation

- Static checks passed:
  - `python -m py_compile train.py infer.py scripts/ptv1/*.py utils/ptv1/*.py utils/11_build_cell_llm_embeddings.py`
  - `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`
- Data checks passed:
  - `python utils/ptv1/01_validate_ptv1_standardized.py`
  - `python utils/ptv1/03_validate_ptv1_training_ready.py`
  - `python utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only`
- Reporter smoke passed on compatible existing prefix fragment `20260608_ptv1_cell_llm`.
- New launcher smoke passed with prefix `20260608_1240_ptv1_fine_tune_smoke`.
  - Three training manifests were `fit_completed`.
  - Both exp_13 output manifests had `n_predictions=218`.
  - All smoke training manifests used `dataset_group=ptv1`, `cell_llm_mode=frozen`, and `accelerator=gpu`.

## Notes

- The full formal 32-candidate search was not launched in this review.
- A sandboxed smoke attempt failed because PyTorch could not initialize CUDA; the same smoke completed after rerunning with unsandboxed GPU access.
