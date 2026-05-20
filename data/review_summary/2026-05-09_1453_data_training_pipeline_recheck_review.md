# 2026-05-09 14:53 HKT Data/Training Pipeline Recheck Review

## Scope

- Read and reconciled the current requirements in:
  - `docs/Data_Process_1.md`
  - `docs/Data_Process_2.md`
  - `docs/Data_Process_3.md`
  - `docs/Data_Process_4.md`
  - `docs/Training_gudline.md`
  - `docs/2026-04-15_data_standardization_session_summary.md`
  - recent files in `data/review_summary/`
- Accepted constraints preserved in this pass:
  - full protein axes only; no top2000 axis is allowed;
  - single-drug rows must use `pert_id2 == pert_id1`;
  - training has exactly two losses: expression MSE and the active task BCE;
  - single-drug active loss2 is response/sensitivity;
  - double-drug active loss2 is synergy only;
  - merged single-drug rows inside double-drug training must have synergy masked;
  - `batch` is part of the default batch covariates;
  - `use_target` is enabled by default;
  - `group_size` is permanently ignored.

## Code Changes

- Strengthened `utils/03_validate_training_ready_outputs.py`.
  - Added `batch` to required discrete fields.
  - Required `global_meta.json` to include `value_to_index["batch"]` and sentinel `no`.
  - Added full-axis validation to fail on exact 2000-axis feature/label matrices and on feature axes smaller than the positive source protein count.
  - Added single-drug second-slot validation for primary single-drug tasks and merged single-drug rows in the double-drug task.
- Updated `docs/data_process_summary_04.md` so the documented default `batch_cov_list` includes `batch`.
- Added `scripts/run_ptv3_training_experiments.sh` for the full experiment plan.

## Static Validation

- `py_compile` passed for the data pipeline, training entrypoint, inference entrypoint, dataset, model, and Lightning wrapper.
- `utils/01_validate_standardized_outputs.py` passed.
- `utils/03_validate_training_ready_outputs.py` passed after the new guards.
- Strategy audit passed:
  - PTV3 global protein axis: `11345`;
  - PTV3 global drug axis: `6113`;
  - no checked task has a 2000 protein axis;
  - `batch_index` exists in checked feature tables;
  - single-drug `pert_id2 == pert_id1` and `pert_index2 == pert_index1`;
  - double-drug merged single rows have synergy masked;
  - formal target splits are nonempty and train/valid/test row overlap is zero;
  - all external test-only tasks have expected coverage.

## 8-GPU Smoke Runs

All bounded runs used 8 H200 GPUs, `batch_size=2`, 2 epochs, and 2 train/val/test batches where applicable.

- `ptv3_main_singledrug`, `pert_stratified_5fold_fold0`, response head: passed.
- `ptv3_main_singledrug`, `cell_type_5fold_fold0`, response head: passed.
- `ptv3_main_singledrug`, `cell_5fold_fold0`, response head: passed.
- `ptv3_main_singledrug`, no-MSE ablation, response head: passed.
- `ptv3_main_singledrug`, no-PDI ablation, response head: passed.
- `ptv3_main_doubledrug`, `pert_id_5fold_fold0`, synergy head: passed.
- `ptv3_main_singledrug`, `all_train_subset_test`, response head, `--skip-test`: passed.
- `ptv3_main_doubledrug`, `all_train_subset_test`, synergy head, `--skip-test`: passed.

These smoke runs validate execution, checkpointing, split manifests, and label routing. They do not prove final convergence or final model quality.

## Extra Inference Smoke

All external inference checks used `--limit-batches 2` and wrote 4 prediction rows plus `run_manifest.json`.

- Extra single response inference passed for:
  - `ptv3_extra_singledrug_mat1_480_faims`
  - `ptv3_extra_singledrug_mat1_qe`
  - `ptv3_extra_singledrug_mat2_480_faims`
  - `ptv3_extra_singledrug_mat2_qe`
  - `ptv3_extra_singledrug_mat3_qe`
  - `ptv3_extra_singledrug_mat4_qe`
- Extra double synergy inference passed for:
  - `ptv3_extra_doubledrug_nature`
  - `ptv3_extra_doubledrug_nc`
  - `ptv3_extra_doubledrug_guomics`

## Final Status

- `scripts/run_ptv3_training_experiments.sh` passed `bash -n`.
- Final `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` showed no active GPU compute process.
- The current code supports the requested experiment families at the code-path level.
