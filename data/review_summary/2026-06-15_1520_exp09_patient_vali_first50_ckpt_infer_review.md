# 2026-06-15 15:20 HKT exp09 patientVali260605v3 first-50 checkpoint inference review

## Scope

- Reviewed and used `utils/17_infer_patient_validation_exp09_all_epoch_ckpts.py` for exp09 unified-head patient validation inference.
- Ran exp09 checkpoints `epoch=0.ckpt` through `epoch=49.ckpt` on the existing v3 patient validation training-ready data.
- Exported raw outputs to `outputs/2026-06/2026-06-15/20260615_patientVali260605v3_exp09_all_epoch_ckpts/` and readable outputs to `outputs/2026-06/2026-06-15/0615v3_exp09_all_epoch_ckpts/`.

## Validation

- Confirmed 400 raw `predictions.parquet` files were produced: 50 checkpoints times 8 patient validation tasks.
- Confirmed 400 readable per-task checkpoint CSV files were produced.
- Confirmed `combined_predictions_readable.csv` has 43,800 rows: 800 single-drug rows and 43,000 double-drug rows.
- Confirmed checkpoint coverage is exactly epochs `0..49`.
- Confirmed 8 task names are present and `prediction_score` has no missing values.
- Confirmed readable `score_name` is consistently `pred_task_prob`, matching exp09 unified-head inference.

## Notes

- The user requested only the first 50 epochs; later exp09 checkpoints, if any are added in future runs, are intentionally not included.
- Existing standardized features from `data/training_ready_patientVali260605v3` were reused; raw patient validation data was not regenerated.
