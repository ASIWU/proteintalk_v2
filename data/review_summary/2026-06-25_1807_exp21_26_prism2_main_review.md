# 2026-06-25 18:07 exp_21-exp_26 PRISM2 Main-Label Review

## Scope

- Reviewed `docs/Data_Process_1.md` through `docs/Data_Process_4.md` and `docs/Training_guideline.md`.
- Implemented an isolated PRISM2 branch for exp_21-exp_26 without overwriting `data/training_ready` or exp_01-exp_08 artifacts.

## Code Changes

- `utils/02_build_training_ready_data.py`
  - Added opt-in `--include-prism2-main-tasks`.
  - Added `ptv3_main_singledrug_prism2` filtering on `PRISM2nd_label_total`.
  - Added `ptv3_main_doubledrug_prism2aux`, using native double-drug `synergy` labels and PRISM2-filtered single auxiliary rows.
- `utils/09_build_data_splits.py`
  - Added main-single split families for `ptv3_main_singledrug_prism2`.
  - Added double pair split support for `ptv3_main_doubledrug_prism2aux`.
- `utils/03_validate_training_ready_outputs.py`
  - Added PRISM2 label/filter validation and split coverage checks.
  - Added auxiliary-label checks for PRISM2-aux double task.
- `train.py`, `infer.py`, `dataset/training_ready_fast_dataset.py`, `model/graph_feature_utils.py`
  - Added mmap-safe `.npy` loading fallback via `utils/npy_io.py`.
  - `ptv3_main_singledrug_prism2` defaults to `PRISM2nd_label_total`.
- `scripts/ptv3_experiment_common.sh`
  - Added `TRAINING_READY_ROOT` forwarding.
- Added exp_21-exp_26 scripts and PRISM2 data/graph preparation scripts.

## Data Checks

- Built with `flow_v2`:
  - `bash scripts/build_exp21_26_prism2_data.sh`
- Output root:
  - `data/training_ready_prism2_main`
- Validation:
  - `utils/03_validate_training_ready_outputs.py --output-root data/training_ready_prism2_main` passed.
- PRISM2 single data summary:
  - feature rows: `8157`
  - control rows: `424`
  - non-control rows: `7733`
  - PRISM2 counts: `non-responsive=5727`, `sensitive=2006`
  - PRISM2-only rows included: `165`
  - PRISM1-only rows included: `0`

## Graph Check

- Old and new PTV3 `pert_index` / `protein_index` hashes matched exactly.
- Reused `data/training_ready/ptv3/derived` through `data/training_ready_prism2_main/ptv3/derived`.
- Shape checks:
  - DDI: `(6131, 6131)`
  - PDI/DPI: `(6131, 11345)`
  - PPI: `(11345, 11345)`
  - drug embedding: `(6131, 2048)`
  - protein embedding: `(11345, 1280)`
- Graph rebuild was not required. If future index hashes diverge, use `scripts/exp21_26_rebuild_prism2_graphs.sh` in tmux session `gpu2`.

## Runtime

- Launched exp_21-exp_26 training in tmux session `gpu2`:
  - `TRAINING_READY_ROOT=data/training_ready_prism2_main`
  - `EXP_PREFIX=20260625_1807_prism2_exp21_26_nowandb`
  - `LOGGER_BACKEND=tensorboard`
  - `LOG_TO_WANDB=0`
- Runtime summary target:
  - `logs/20260625_1807_prism2_exp21_26_nowandb_runtime_summary.tsv`
