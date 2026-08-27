# 2026-06-20 14:56 HKT PTV3/PTV1 Label Alignment Review

## Scope

- Reviewed PTV1-flow `cell_5fold` raw split semantics, masking behavior for `-1` labels, prediction CSV limitations, and PTV3 fast-task split/set-info requirements.
- Reviewed PTV3 fast dataset loading, exp03 runner wiring, report materialization, and new label-aligned comparison outputs.

## Findings

- The original PTV3 vs PTV1-flow AUPRC gap was primarily a label/objective mismatch. On fold2, PTV1 predictions scored high on PTV1 labels but poorly on PTV3 PRISM labels, while PTV3 did the reverse.
- PTV1 `predictions.csv` has an unreliable `experiment_type` column after `-1` label masking; corrected comparisons must reconstruct valid experiment key order from raw `loo_label.csv` and `pheno.csv`.
- The derived PTV3 task preserves PTV3 feature/control/set-info contracts and masks PTV1 `-1` labels instead of converting them to negatives.
- GPFS mmap can fail on this environment with `OSError: [Errno 19] No such device`; the fast dataset now falls back to normal `np.load`.

## Validation

- `py_compile` passed for modified Python files.
- Shell syntax checks passed for modified experiment scripts.
- New label-aligned task smoke test loaded artifacts, split indices, set info, and one dataset item successfully.
- GPU training/inference ran in `gpu2:brainctl` under `flow_v2`.
- Corrected 5-fold comparison has `common_label_mismatch=0` for all folds.
- Final selected PTV3 config `20260620_ptv3_ptv1label_cell_v1_covdrop020_mse075` reached mean AP/AUPRC `0.862943`, exceeding PTV1-flow cell_5fold mean AP `0.860341`.
