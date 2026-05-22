# 2026-05-22 18:52 HKT Unseen Cell Representation Review

- Scope: reviewed and extended the fast unseen-cell path without touching data files or data processing scripts.
- Git control: created branch `exp/unseen-cell-representation-20260522` from `924688e v2.2beta`; committed model/script switches in `b1eda35` and `63b3193`.
- Code changes reviewed:
  - `model/fast_delta_model.py`: added default-off `pcep_cell`/`pcep_dual` modes, expression score modes, and expression-hidden covariate auxiliary heads.
  - `model/fast_lightning.py`: added optional auxiliary covariate classification loss, default weight `0.0`.
  - `train.py`/`infer.py`: added argument passthrough and checkpoint-compatible inference handling for new PCEP score args.
  - `scripts/ptv3_experiment_common.sh`: added env passthrough for protein concat score mode, aux covariate loss, and covariate UNK.
- Validation:
  - `python -m py_compile train.py infer.py model/fast_delta_model.py model/fast_lightning.py` passed.
  - `bash -n scripts/ptv3_experiment_common.sh scripts/exp_02_single_cell_type_5fold.sh scripts/exp_03_single_cell_5fold.sh` passed.
  - Fast dry-run with `pcep_dual + additive + aux cell_type` produced expected expression/logit shapes.
- Experiment review:
  - Expression fusion variants, auxiliary cell-type classification loss, larger hidden size, and direct control-expression logit heads did not improve unseen cell AUPRC.
  - `MSE_WEIGHT=0.075` with full covariate UNK dropout `0.15` improved unseen cell to AUROC `0.934443`, AUPRC `0.767008`.
  - The same `MSE_WEIGHT=0.075` did not improve unseen cell type; cell type remains better with the previous `MSE_WEIGHT=0.10` setup.
- Risk:
  - New architecture switches are default-off, so v2.2beta behavior is preserved when defaults are used.
  - The observed unseen-cell improvement is modest and should be rerun on the standard 8-GPU workflow before becoming the paper/default setting.
