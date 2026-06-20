# PTV1 Exp12/Exp13 Tuning Review - 2026-06-03 22:12 HKT

## Scope

- Follow-up tuning for low PTV1 exp_12 and exp_13 results.
- Added reusable tuning helpers:
  - `scripts/ptv1/run_ptv1_param_search.sh`
  - `scripts/ptv1/report_ptv1_param_search.py`
- Search prefix: `20260603_2108_ptv1_param_v1`.

## Experiments Reviewed

- Stage 1 folds: `pert_id_5fold_fold0`, `fold2`, `fold4`.
- Stage 2 full folds: `pert_id_5fold_fold0..4`.
- Exp_13 extra inference was run for:
  - `mse025`;
  - `pos_auto_mse050`.

## Findings

- `MSE_WEIGHT=0.25` is the best exp_12 setting found.
  - Baseline exp_12 mean: AUROC `0.707712`, AUPRC `0.545776`, n-AUPRC `2.212531`.
  - Tuned `mse025` exp_12 mean: AUROC `0.731340`, AUPRC `0.570871`, n-AUPRC `2.388662`.
- `pos_auto_mse050` is weaker for exp_12 but best for exp_13 extra AUPRC.
  - Exp_12 mean: AUROC `0.705360`, AUPRC `0.553605`, n-AUPRC `2.293438`.
  - Exp_13 extra: AUROC `0.594550`, AUPRC `0.587486`, n-AUPRC `1.143500`.
- `mse025` exp_13 extra also improves over baseline, but less than `pos_auto_mse050`.
  - Exp_13 extra: AUROC `0.595982`, AUPRC `0.581499`, n-AUPRC `1.131846`.
- Combination configs did not help:
  - `mse025_pos_auto` stage-1 mean AUPRC `0.511107`.
  - `mse025_focal` stage-1 mean AUPRC `0.450210`.
- The `pos_auto_mse050` exp_13 reference policy required `REFERENCE_ALLOW_MIXED_CONFIG=1`, because `POSITIVE_WEIGHT=auto` is fold-specific by design.

## Residual Risk

- Exp_13 has only 218 labeled extra rows, so the observed AUPRC gain is small and may be sensitive to cell composition.
- Exp_12 and exp_13 prefer different settings: `mse025` for unseen-drug validation, `pos_auto_mse050` for extra single-drug AUPRC.
- No new model architecture changes were made; this was loss/optimizer-level tuning only.

## Validation

- `bash -n scripts/ptv1/run_ptv1_param_search.sh` passed after adding combination configs.
- `python -m py_compile scripts/ptv1/report_ptv1_param_search.py` passed.
- Final static checks were rerun after documentation updates.
