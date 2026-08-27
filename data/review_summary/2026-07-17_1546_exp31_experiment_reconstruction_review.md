# 2026-07-17 15:46 HKT Exp31 Experiment Reconstruction Review

## Scope

- Reconstructed the purpose, data contract, split, training setup, and results of `20260709_exp31_rnaseq` from the exp31 report, training history, build summary, runner, reporter, raw label table, and result JSON.
- This was a read-only review of the experiment implementation and artifacts; no training or inference was rerun and no experiment artifact was modified.

## Experiment Reconstruction

- Exp31 adapted the exp09 unified single+double checkpoint to PDX clinical response prediction from baseline RNA-seq and drug features.
- Four independent binary endpoints were trained:
  - early and late sensitivity: CR/PR versus SD/PD;
  - early and late disease control: CR/PR/SD versus PD.
- The raw response table confirms these binary mappings. It does not document a numeric time cutoff for `early` versus `late`; the columns should therefore be described as the source table's early/late response definitions.
- The split was sample-level: 30 BRCA PDX samples / 388 treatment rows for training, 8 BRCA samples / 104 rows for validation, and 139 non-BRCA samples / 1,827 rows for held-out testing. The test cancers were CRC, PDAC, NSCLC, and CM.
- The non-BRCA test set contained 1,420 single-drug and 407 double-drug rows. The entire kept dataset contained 177 samples and 2,319 treatment rows after dropping 2 rows without cancer type.
- The exp09 protein axis of 11,092 proteins was retained for checkpoint compatibility; RNA-seq covered 10,159 proteins. Missing RNA values remained `NaN`, and fine-tuning used classification only with MSE disabled.
- All four fine-tunes started from `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/last.ckpt`, used the unified head, learning rate `1e-5`, batch size `128`, at most 30 epochs, and selected the checkpoint with maximum BRCA validation AUPRC.
- The `synergy` field was only a compatibility carrier for the clinical binary label and did not represent biological drug synergy.

## Overall Results

| endpoint | BRCA valid AUPRC | non-BRCA FT AUROC | non-BRCA zero-shot AUROC | non-BRCA FT AUPRC | non-BRCA zero-shot AUPRC | AUPRC change |
|---|---:|---:|---:|---:|---:|---:|
| sensitive early | 0.8537 | 0.6170 | 0.6297 | 0.1805 | 0.1473 | +0.0332 (+22.6%) |
| sensitive late | 0.8362 | 0.6511 | 0.6297 | 0.1411 | 0.0839 | +0.0571 (+68.0%) |
| disease control early | 0.8480 | 0.6602 | 0.5768 | 0.6078 | 0.4724 | +0.1354 (+28.7%) |
| disease control late | 0.6496 | 0.6733 | 0.5698 | 0.2549 | 0.1539 | +0.1010 (+65.6%) |

- The reporter selected `sensitive_early` because it had the highest BRCA validation AUPRC (`0.8537`). Its held-out non-BRCA AUPRC improved from `0.1473` to `0.1805`, while AUROC decreased slightly from `0.6297` to `0.6170`.
- Fine-tuning improved AUPRC and AUROC in every cancer-type subgroup for every endpoint compared with exp09 zero-shot.
- The treatment-type result was less uniform. Single-drug metrics improved for all four endpoints. Double-drug AUPRC/AUROC declined for sensitive early, declined slightly for sensitive late and disease-control early, and improved only for disease-control late.
- In the selected sensitive-early double-drug subgroup, fine-tuned AUROC was `0.4216` and AUPRC was `0.2470`, below its positive-rate baseline of `0.2703` (`nAUPRC=0.9138`).

## Interpretation And Limitations

- The experiment supports the claim that BRCA PDX label fine-tuning generally improved non-BRCA precision-recall performance over direct exp09 zero-shot inference, especially for single drugs.
- The evidence does not support a general improvement for drug combinations.
- The four endpoints have different clinical meanings and positive prevalences. `sensitive_early` being reporter-selected does not make it intrinsically superior to the other endpoints; it only won the predefined raw BRCA-validation-AUPRC rule.
- Different test metrics identify different numerical leaders: disease-control early has the highest raw test AUPRC (`0.6078`), disease-control late the highest test AUROC (`0.6733`), and sensitive late the highest test nAUPRC (`2.1841`). These are not directly interchangeable endpoint comparisons.
- Validation contains only 8 BRCA PDX samples, while the 1,827 test rows come from 139 PDX samples with repeated treatments. No confidence intervals or sample-clustered uncertainty estimates were reported, so the apparent gains should be treated as preliminary rather than definitive.
- Five SMILES-only drugs were added with Morgan/DDI features but all-zero new PDI rows, and the sample Cell LLM channel used tissue-level offline fallback embeddings rather than sample-specific API-generated embeddings.

## Sources Reviewed

- `docs/2026-07-09_exp31_rnaseq_pdx_ft_benchmark_results.md`
- `docs/training_history.md`
- `data/review_summary/2026-07-09_1516_exp31_rnaseq_ft_feasibility_review.md`
- `data/review_summary/2026-07-09_1609_exp31_rnaseq_pdx_ft_tooling_review.md`
- `data/training_ready_exp31_rnaseq/ptv3/exp31_rnaseq_build_summary.json`
- `outputs/2026-07/2026-07-09/20260709_exp31_rnaseq_pdx_ft_benchmark_summary.json`
- `scripts/exp_31_rnaseq_pdx_ft_benchmark.sh`
- `scripts/report_exp31_rnaseq_pdx_ft_benchmark.py`
- `utils/31_build_exp31_rnaseq_training_ready.py`
- `data/rawdata/rna_seq/260618pdx_pct_sample_info_with_smiles_check_comboAB.csv`
