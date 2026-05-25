# 2026-05-25 15:46 HKT Covariate Analysis Review

## Scope

- Reviewed current baseline covariate handling in `train.py`, `dataset/training_ready_fast_dataset.py`, and `scripts/ptv3_experiment_common.sh`.
- Implemented a fold0 covariate analysis workflow for:
  - unseen drug: `ptv3_main_singledrug`, `pert_stratified_5fold_fold0`;
  - unseen cell: `ptv3_main_singledrug`, `cell_5fold_fold0`.

## Code Changes

- `scripts/ptv3_experiment_common.sh`
  - Added optional `BATCH_COV_LIST` passthrough.
  - `BATCH_COV_LIST=__none__` emits `--batch-cov-list` with no fields.
  - Default behavior is unchanged when unset.
- `scripts/run_covariate_ablation_fold0_2gpu.sh`
  - Runs covariate profiles across two GPUs.
  - Uses the current fast_delta baseline settings: one GPU, batch size 256, graph features enabled, PCEP enabled.
  - Prebuilds graph cache before parallel workers.
- `scripts/covariate_analysis_report.py`
  - Parses run manifests and runtime summaries.
  - Computes split-level covariate unseen-category diagnostics.
  - Writes markdown and JSON reports.

## Validation

- `bash -n scripts/run_covariate_ablation_fold0_2gpu.sh scripts/ptv3_experiment_common.sh` passed.
- `python -m py_compile scripts/covariate_analysis_report.py train.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py model/fast_lightning.py` passed.
- Dry report generation passed.
- Full covariate ablation under `EXP_PREFIX=20260525_covariate_fold0_v1` completed 38/38 runs with `fit_completed` and `test_completed`.

## Results

- Full report:
  - `logs/20260525_covariate_fold0_v1_covariate_analysis.md`
  - `logs/20260525_covariate_fold0_v1_covariate_analysis.json`
- Fresh full-covariate baseline:
  - unseen drug fold0: AUROC `0.8448`, AUPRC `0.5565`;
  - unseen cell fold0: AUROC `0.8949`, AUPRC `0.7871`.
- Best profiles:
  - unseen drug fold0: `drop_batch`, AUROC `0.8565`, AUPRC `0.5870`;
  - unseen cell fold0: `cell_identity_only`, AUROC `0.9061`, AUPRC `0.8407`;
  - unseen cell reliable UNK variant: `cell_identity_covunk015`, AUROC `0.9088`, AUPRC `0.8393`;
  - unseen cell full covariate UNK dropout `0.15`: AUROC `0.9189`, AUPRC `0.8394`.

## Interpretation

- Unseen drug fold0 does not have strong covariate category shift; removing `batch` was beneficial in this run.
- Unseen cell fold0 has severe high-cardinality covariate shift:
  - `Cell_plate` and `Cell`: `100%` test rows train-unseen;
  - `batch`: `91.67%` test rows train-unseen;
  - `cell_type`: `27.25%` test rows train-unseen.
- Because raw unseen categorical embeddings are untrained, `cell_identity_only` should be treated as a diagnostic, not immediately as a robust default.
- `cell_identity_covunk015` and `full_covunk015` are more defensible default candidates for unseen-cell because train-unseen categories map to learned UNK embeddings.
