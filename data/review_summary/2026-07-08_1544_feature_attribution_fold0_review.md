# 2026-07-08 15:44 HKT Feature Attribution Fold0 Review

Scope:
- Reviewed the PTV3 fast-delta training launcher, fold scripts, manifest fields, random-control support, graph-cache behavior, and fold0 split artifacts for the planned exp01/exp03 feature attribution matrix.

Findings:
- `train.py` already records the needed manifest fields for `have_mse_loss`, graph mode, target-protein length, PCEP mode, covariates, control-expression mode/path, and drug embedding path.
- `scripts/ptv3_experiment_common.sh` did not expose `--drug-embedding-path`; added optional `DRUG_EMBEDDING_PATH` passthrough for train and infer.
- The graph feature cache key does not distinguish drug embedding content beyond shape, so zero-Morgan runs need an isolated graph-cache directory to avoid reusing real-Morgan graph features.
- Existing `CONTROL_EXPRESSION_MODE=random_saved` can load the zero-control matrix because it indexes any saved expression matrix by perturb row.

Changes:
- Added zero artifact builder, feature-attribution fold0 runner, and report/validation script.
- Generated and validated zero-control and zero-Morgan artifacts.
- Launched the exp01/exp03 fold0 matrix on tmux `gpu2` with prefix `20260708_feature_attr`.

Verification:
- `python -m py_compile train.py scripts/build_feature_attribution_zero_artifacts.py scripts/report_feature_attribution_fold0.py`
- `bash -n scripts/ptv3_experiment_common.sh`
- `bash -n scripts/run_exp01_exp03_fold0_feature_attribution.sh`
- Reporter smoke with `--allow-incomplete` to `/tmp`
- Confirmed `gpu2` tmux session and required CPU artifacts before launch.
