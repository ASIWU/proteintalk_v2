# 2026-05-21 15:43 HKT Baseline4 Root Code and Script Review

Scope:
- Reviewed root-level Baseline4 integration in `train.py`, `infer.py`,
  `dataset/training_ready_fast_dataset.py`, `model/fast_delta_model.py`,
  `model/fast_lightning.py`, `model/graph_feature_utils.py`, and
  `scripts/exp_01` through `scripts/exp_08`.
- Reviewed the shared script helper `scripts/ptv3_experiment_common.sh`,
  the 8-GPU launcher `scripts/0521_baseline4_8gpu_parallel.sh`, and the
  reference epoch selector.

Findings:
1. Major operational bug risk: graph feature cache construction is not
   concurrency-safe. `scripts/0521_baseline4_8gpu_parallel.sh` starts up to 8
   fold workers at once, and each worker can call
   `build_or_load_graph_features()` against the same `graph_cache/*.npy` and
   `.meta.json` files. On a clean checkout where `graph_cache` is absent, this
   can cause multiple processes to write the same `.npy` concurrently, or a
   process to load a partially written file after another process has written
   metadata. Recommended fix: add an atomic file lock around graph cache build,
   or prebuild the cache once in the launcher before starting workers.
2. Medium guardrail issue: fast checkpoint config validation does not compare
   every inference-sensitive fast-model hyperparameter. Script defaults match
   training defaults, so the current standard scripts are fine, but manual
   inference can silently drift for settings such as `protein_concat_seed` or
   `graph_pair_add_scale`.
3. Medium operational issue: the 8-GPU launcher waits on background workers
   under `set -e`. If one worker fails, the script can exit before waiting on
   the remaining workers. This is mainly a cleanup/reporting risk; it does not
   affect successful runs.

Checks run:
- `python -m py_compile train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py model/fast_lightning.py model/graph_feature_utils.py model/training_ready_models.py model/training_ready_lightning.py scripts/select_reference_epoch.py`
- `bash -n scripts/ptv3_experiment_common.sh scripts/0521_baseline4_8gpu_parallel.sh scripts/exp_01_single_pert_stratified_5fold.sh scripts/exp_02_single_cell_type_5fold.sh scripts/exp_03_single_cell_5fold.sh scripts/exp_04_single_no_mse_5fold.sh scripts/exp_05_single_no_pdi_5fold.sh scripts/exp_06_double_pert_pair_5fold.sh scripts/exp_07_extra_single_all_train_infer.sh scripts/exp_08_extra_double_all_train_infer.sh`
- `git diff --check`
- Verified current shared-script default `REFERENCE_EPOCH_AGG=mean`.
- Verified existing smoke manifests and outputs show `fast_delta`, batch size
  256, one device, baseline graph mode `real`, w/o graph mode `zero`, and
  extra inference predictions without NaN probabilities.

Conclusion:
- I found one major bug risk before full 8-GPU production use: concurrent graph
  cache creation. The current machine already has the cache, so previous smoke
  tests can pass, but a clean 8-GPU run after GitHub upload would still be at
  risk unless the cache is prebuilt or locked.

Fix follow-up:
- Added `scripts/prebuild_graph_cache.py` and wired it into
  `scripts/0521_baseline4_8gpu_parallel.sh` before parallel workers start.
- Added file locking and atomic cache writes in `model/graph_feature_utils.py`.
- Expanded fast checkpoint config validation in `infer.py`.
- Hardened the 8-GPU launcher wait logic so all background workers are waited
  on before the script reports stage failure.
