# 2026-06-12 10:23 HKT Patient Validation 260605 Expression Scale Review

## Scope

- Checked whether original PTV3/PTV1 training-ready expression matrices were correctly `log1p` transformed.
- Audited `data/rawdata/ptv2drug_patientVali260605` matrix scales to decide which patient datasets are raw abundance and which are already log2/normalized.
- No new inference or feature generation was run.

## Evidence

- `docs/Data_Process_2.md` requires each expression matrix to be `log1p` transformed while preserving `NaN`.
- `utils/02_build_training_ready_data.py::Stage1Cache.load_expression_matrix` reads stage-1 `expression_matrix.npy`, casts to `float32`, applies `np.log1p` to finite entries, and keeps `NaN`.
- `utils/ptv1/02_build_ptv1_training_ready.py` reuses the same stage-2 builder for PTV1.
- Direct sampled checks confirmed `data/training_ready/ptv3` and `data/training_ready/ptv1` matrices match `log1p(data/standardized/.../expression_matrix.npy)` for native expression-backed rows.

## Patient Matrix Scale Judgment

- P1 ovarian: raw abundance scale. Values are all nonnegative, median approximately `4.73e5`, direct `log1p` median approximately `13.07`.
- P2 lung2020: raw abundance scale. Values are all nonnegative, median approximately `4.72e6`, direct `log1p` median approximately `15.37`.
- P3 lung2024: not raw abundance. Values are centered near 0 with many valid negatives (`290,916` finite negative entries), sample-wise row medians are near 0, and max is approximately `8.63`. This is consistent with log2-normalized or log2-relative expression rather than raw intensity.
- P4 colon: raw abundance scale. Values are all nonnegative, median approximately `3.81e5`, direct `log1p` median approximately `12.85`.
- P5 breast: raw abundance scale. Values are all nonnegative, median approximately `9.88e5`, direct `log1p` median approximately `13.80`.

## Judgment

- Original PTV3 and PTV1 training-ready data were correctly transformed with `log1p`.
- For patient validation data, P1/P2/P4/P5 should use direct `log1p(raw)`.
- P3 should not be processed as `negative_to_nan_then_log1p`; its negative values are valid log2-scale values, not invalid raw abundance values.
- If P3 is used, the mathematically consistent transform is `log1p(2 ** x)` rather than mapping negative values to `NaN`.
- Caveat: P3 appears centered/relative, not absolute log2 raw intensity. `log1p(2 ** x)` would produce relative-abundance-scale values with median approximately `0.695`, far below the original PTV3/PTV1 `log1p(raw)` range. Therefore P3 should be flagged as scale-incompatible unless an additional calibration strategy is chosen.
