# Exp32 Screen Scope And Exp31 FT/Zero-shot Review

- Review time: 2026-07-21 11:09 HKT
- Scope: epoch-2 Exp32 combined virtual-screen output, score-distribution visualization, and the apparent similarity between Exp31 fine-tuned and zero-shot metrics.

## Exp32 Combined Screen Scope

- `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_predictions.csv` has 83,642 rows and 18 columns.
- It contains the complete fixed-condition query screen: 2 devices (`B`, `CAC`) x 13 biological sample pairs x 3,217 drugs = 83,642 device/sample/drug predictions.
- Every `(device, sample_pair_id, drug_id)` key is unique. Each device contributes exactly 41,821 rows and covers all 13 samples and all 3,217 drugs.
- The file excludes the 13 baseline-control rows per device because those rows provide model inputs rather than screening queries. It also excludes expression predictions and ground-truth metrics; Exp32 saved only unified-head sensitivity probabilities.
- B and CAC remain separate instrument-specific predictions. They have not been averaged into one score per biological sample/drug pair.
- Generated the requested epoch-2 score-distribution artifacts with the existing validated plotting script:
  - `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_score_distribution.png`;
  - `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_score_distribution.pdf`.
- Plot inputs contain finite probabilities in `[0,1]`. B mean/median are `0.0682/0.0050`; CAC mean/median are `0.0610/0.0032`; each curve contains 41,821 predictions.

## Exp31 Fine-tuned Versus Zero-shot Audit

- Fine-tuning did run and FT inference did not reuse the epoch-2 initialization checkpoint. The four FT manifests are `fit_completed`, and their selected checkpoints are new files at steps 108-120. Model summaries report 32.4 million trainable parameters.
- Each fine-tune uses only 388 BRCA training rows and 104 BRCA validation rows, then evaluates on 1,827 non-BRCA rows. With batch size 128, this is approximately four training batches per epoch and at most 120 optimizer steps over 30 epochs, using learning rate `1e-5`.
- The raw probabilities are not nearly identical. Paired FT-versus-zero-shot comparisons are:

| label | Pearson | Spearman | probability MAE | 0.5-threshold flip | FT - zero-shot AUPRC | FT - zero-shot AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `sensitive_early` | 0.7873 | 0.7635 | 0.1308 | 9.30% | -0.0048 | -0.0039 |
| `sensitive_late` | 0.7718 | 0.7602 | 0.1351 | 10.18% | +0.0125 | +0.0052 |
| `disease_control_early` | 0.6681 | 0.6723 | 0.2828 | 38.04% | +0.0588 | +0.0561 |
| `disease_control_late` | 0.7952 | 0.7471 | 0.1151 | 10.34% | +0.0254 | +0.0280 |

- Therefore, the similar-looking aggregate metrics do not indicate a no-op fine-tune. AUROC and AUPRC depend mainly on ranking, so sizeable probability calibration changes can yield smaller metric changes when much of the ordering is retained.
- Transfer is deliberately difficult: training/validation are BRCA-only, while test is CM/CRC/NSCLC/PDAC. Fine-tuning can improve the BRCA validation objective without guaranteeing an equally large cross-tumor improvement.
- The two sensitive labels are strongly imbalanced and have only 30-40 positive BRCA training rows. Their non-BRCA test positive rates are 6.46% and 11.55%, making AUPRC changes comparatively noisy and limiting the stable task-specific signal available to fine-tuning.
- `disease_control_early` is the clearest successful transfer rather than a near-tie: AUPRC increases from `0.5658` to `0.6246` and AUROC from `0.6261` to `0.6823`. It is also the label selected by BRCA validation AUPRC.
- No checkpoint-path, output-alignment, or inference-row mismatch was found. The similarity is consistent with limited-data, low-learning-rate, cross-domain fine-tuning rather than an execution error.

## Verification

- Confirmed the `/root/tmp/proteintalk_v2/...` CSV resolves to the same shared-workspace file under `/mnt/shared-storage-gpfs2/.../proteintalk_v2/...`.
- Confirmed CSV shape, column set, device/sample/drug coverage, unique query keys, and prediction semantics.
- Confirmed the PNG/PDF were generated successfully and visually inspected the PNG.
- Confirmed FT/zero-shot prediction alignment by exact `feature_row_index` before calculating paired differences.
