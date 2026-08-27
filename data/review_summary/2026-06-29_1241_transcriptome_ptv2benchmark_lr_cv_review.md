# 2026-06-29 12:41 HKT Transcriptome PTV2-Benchmark LR CV Review

## Scope

- Reviewed the existing CPA/BioLord generated-transcriptome downstream MLP scripts from the 2026-06-26 workflow.
- Checked the PTV2 benchmark reference defaults in `baseline/ptv2_benchmark_260514/PTV1_model_20260428`.
- Reviewed the CPA/BioLord h5ad usage contract in `docs/2026-06-26_official_fixed_residual_transcriptome_usage.md`.

## Findings

- The downstream task can reuse prediction h5ad `.X` as 9843-dimensional generated transcriptome input and `PRISM1st_label_total` as the exp01/02/03 binary label.
- The benchmark-aligned settings requested for this MLP map cleanly to hidden size, dropout, activation, batch size, AdamW, weight decay, and fixed epoch count.
- Benchmark mechanisms such as ODE dynamics, proteomics MSE reconstruction, SWAG, validation scheduler, early stopping, controls, covariates, graphs, dose, and time do not map to this generated-transcriptome plus Morgan-fingerprint MLP input.
- Selecting LR by mean5 test AUPRC is intentionally test-selected and should be reported as such, not as an independent holdout estimate.

## Action

- Added `--merge-val-into-train`, `--no-validation`, and `--fixed-epochs` support to `scripts/train_transcriptome_downstream_mlp.py`.
- Added the LR-CV launcher and shell wrapper:
  - `scripts/run_transcriptome_downstream_ptv2benchmark_lr_cv.py`
  - `scripts/run_transcriptome_downstream_ptv2benchmark_lr_cv.sh`
- Added `scripts/report_transcriptome_downstream_ptv2benchmark_lr_cv.py` for best-LR selection, best fold detail, all-LR mean5 summaries, and anomaly reporting.
- Verified py_compile, cpa/biolord exp01 fold0 data smoke, and one-epoch cpa exp01 fold0 training smoke.
