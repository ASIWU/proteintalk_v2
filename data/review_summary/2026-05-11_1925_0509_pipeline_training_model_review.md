# 2026-05-11 19:25 HKT 0509 Pipeline / Training / Model Review

## Scope

Reviewed the workflow driven by:

- `scripts/0509_1.sh`
- `scripts/0509_2.sh`
- `scripts/0509_3.sh`
- `scripts/0509_4.sh`

The review followed the shell entrypoints into `scripts/ptv3_experiment_common.sh`, `train.py`, `infer.py`, `dataset/training_ready_dataset.py`, `model/training_ready_lightning.py`, `model/training_ready_models.py`, `utils/09_build_data_splits.py`, and `scripts/select_reference_epoch.py`.

## Findings

1. High: `scripts/0509_3.sh` defaults to `REFERENCE_5FOLD_CKPT_PATH=checkpoints/20260510_single_pert_stratified_5fold`, while `scripts/0509_1.sh` now writes the corresponding 5-fold run under prefix `20260511_single_pert_stratified_5fold`. The existing `20260510` reference manifests are complete, but their checkpoint monitor is `val/total_loss` and `best_ckpt_metric` is missing/null. Current `0509_3.sh` also sets `BEST_CKPT_METRIC=valid_auprc`, but `scripts/select_reference_epoch.py` only validates reference-fold homogeneity and does not require the reference monitor/metric to match the current wrapper intent. In this workspace, the extra-single reference epoch selector therefore chooses median epoch `52` from total-loss-selected checkpoints, not AUPRC-selected checkpoints.

2. Medium: `scripts/0509_4.sh` defaults to `REFERENCE_5FOLD_CKPT_PATH=checkpoints/20260510_double_pert_pair_5fold`, matching the prefix that `scripts/0509_2.sh` would create, but no matching completed double-drug reference manifests are currently present in the workspace. Running `0509_4.sh` now would fail before all-data training with `no run_manifest.json files matched`.

3. Low/security: all four `0509_*` wrappers hardcode `WANDB_API_KEY`. This is operationally convenient for the local server, but it makes the scripts unsafe to share or commit outside the trusted environment. Prefer reading the key from the shell environment or a local untracked file.

## Positive Checks

- `utils/03_validate_training_ready_outputs.py` passed for all current training-ready artifacts.
- `py_compile` passed for `train.py`, `infer.py`, dataset, model, Lightning wrapper, split builder, validator, and reference-epoch selector.
- Current PTV3 formal split counts are non-empty and row-disjoint:
  - `ptv3_main_singledrug/pert_stratified_5fold_fold*`: train about 12.3k, valid about 2.1k, test about 3.6k, zero row overlap.
  - `ptv3_main_doubledrug/pert_id_5fold_fold*`: train about 19.3k, valid about 142-148, test about 344-380, zero row overlap.
- `ptv3_main_doubledrug` train splits include `17,986` merged single-drug auxiliary rows, with `0` such rows in valid/test.
- `all_train_subset_test` overlaps train with valid/test as designed, and `train.py` blocks using this strategy with `trainer.test` unless `--skip-test` is set.
- CPU dry forward checks with the real current graph model passed:
  - single: expression `(1, 10982)`, response logits `(1, 1)`, synergy logits `(1, 1)`
  - double: expression `(1, 11092)`, response logits `(1, 1)`, synergy logits `(1, 1)`

## Training / Model Assessment

The training loop is structurally correct for the current two-loss contract:

- `loss1` is expression MSE with NaN masking.
- `loss2` is only the active task BCE: response for single-drug, synergy for double-drug.
- Double-drug merged single-drug rows contribute expression MSE but have synergy masked out of `loss2`, matching the documented contract.
- Validation/test metrics are aggregated over full evaluated rows, DDP duplicates are deduplicated by `row_index`, and active `val/task_auprc` is available for checkpoint monitoring when full validation contains both classes.

The active model is reasonable for the current accepted graph strategy:

- It uses a PDI-only heterograph to derive protein/drug embeddings.
- It combines control expression value embeddings, graph protein embeddings, perturbation tokens, batch covariates, optional target proteins, CLS pooling, an expression head, and separate response/synergy heads.
- Forward shapes match the full task protein axes and support different extra-task protein axes, which is expected for this graph token model.

Main caveat: the extra-data reference-epoch workflow should require the reference folds to match the intended metric/monitor, or the wrappers should point only at freshly generated matching reference folds.

## Commands Run

- `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile ...`
- `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/03_validate_training_ready_outputs.py`
- `scripts/select_reference_epoch.py` checks for the single and double default reference prefixes.
- `train.py --dry-run-batches 1` for current single and double PTV3 graph-model paths.
