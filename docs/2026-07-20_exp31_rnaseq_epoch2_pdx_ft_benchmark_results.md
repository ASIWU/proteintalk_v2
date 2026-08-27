# Exp31 RNA-seq PDX Fine-tune Benchmark

- Generated: `2026-07-20T13:03:44+00:00`
- Experiment prefix: `20260720_exp31_rnaseq_epoch2`
- Split: `brca_ft_valid_nonbrca_test` (BRCA train/valid, non-BRCA test)
- Init checkpoint: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`.
- Label carrier note: exp31 writes each clinical label into `synergy` only for unified-head compatibility; it is not a biological synergy label.

## Status

- All expected preflight and result files passed reporter checks.

## Selected Setting

- Selected label: `disease_control_early`
- BRCA valid AUPRC: `0.8504`
- Non-BRCA test AUPRC: `0.6246`
- Non-BRCA test AUROC: `0.6823`

## Overall Metrics

| label | source | status | valid AUPRC | n | pos | AUROC | AUPRC | base | nAUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sensitive_early | ft | ok | 0.6569 | 1827 | 211 | 0.6998 | 0.2324 | 0.1155 | 2.0120 |
| sensitive_early | zeroshot | ok |  | 1827 | 211 | 0.7037 | 0.2372 | 0.1155 | 2.0537 |
| sensitive_late | ft | ok | 0.5824 | 1827 | 118 | 0.7251 | 0.1735 | 0.0646 | 2.6859 |
| sensitive_late | zeroshot | ok |  | 1827 | 118 | 0.7200 | 0.1610 | 0.0646 | 2.4920 |
| disease_control_early | ft | ok | 0.8504 | 1827 | 813 | 0.6823 | 0.6246 | 0.4450 | 1.4036 |
| disease_control_early | zeroshot | ok |  | 1827 | 813 | 0.6261 | 0.5658 | 0.4450 | 1.2715 |
| disease_control_late | ft | ok | 0.5285 | 1827 | 254 | 0.6610 | 0.2448 | 0.1390 | 1.7608 |
| disease_control_late | zeroshot | ok |  | 1827 | 254 | 0.6331 | 0.2194 | 0.1390 | 1.5783 |

## Stratified Metrics

| label | source | group | value | n | pos | AUROC | AUPRC | base | nAUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sensitive_early | ft | cancer_type | CM | 420 | 62 | 0.7349 | 0.3956 | 0.1476 | 2.6801 |
| sensitive_early | ft | cancer_type | CRC | 456 | 26 | 0.8061 | 0.1973 | 0.0570 | 3.4610 |
| sensitive_early | ft | cancer_type | NSCLC | 399 | 59 | 0.5969 | 0.1899 | 0.1479 | 1.2839 |
| sensitive_early | ft | cancer_type | PDAC | 552 | 64 | 0.7428 | 0.2754 | 0.1159 | 2.3754 |
| sensitive_early | ft | treatment_type | double | 407 | 110 | 0.5054 | 0.2884 | 0.2703 | 1.0671 |
| sensitive_early | ft | treatment_type | single | 1420 | 101 | 0.5946 | 0.1193 | 0.0711 | 1.6773 |
| sensitive_early | zeroshot | cancer_type | CM | 420 | 62 | 0.7526 | 0.4314 | 0.1476 | 2.9227 |
| sensitive_early | zeroshot | cancer_type | CRC | 456 | 26 | 0.7417 | 0.2687 | 0.0570 | 4.7120 |
| sensitive_early | zeroshot | cancer_type | NSCLC | 399 | 59 | 0.5833 | 0.1772 | 0.1479 | 1.1981 |
| sensitive_early | zeroshot | cancer_type | PDAC | 552 | 64 | 0.7622 | 0.3319 | 0.1159 | 2.8628 |
| sensitive_early | zeroshot | treatment_type | double | 407 | 110 | 0.6179 | 0.3300 | 0.2703 | 1.2210 |
| sensitive_early | zeroshot | treatment_type | single | 1420 | 101 | 0.6001 | 0.1350 | 0.0711 | 1.8980 |
| sensitive_late | ft | cancer_type | CM | 420 | 21 | 0.8087 | 0.2747 | 0.0500 | 5.4938 |
| sensitive_late | ft | cancer_type | CRC | 456 | 16 | 0.8085 | 0.1389 | 0.0351 | 3.9582 |
| sensitive_late | ft | cancer_type | NSCLC | 399 | 33 | 0.5633 | 0.1324 | 0.0827 | 1.6013 |
| sensitive_late | ft | cancer_type | PDAC | 552 | 48 | 0.7432 | 0.2282 | 0.0870 | 2.6242 |
| sensitive_late | ft | treatment_type | double | 407 | 67 | 0.5804 | 0.2263 | 0.1646 | 1.3745 |
| sensitive_late | ft | treatment_type | single | 1420 | 51 | 0.6115 | 0.0724 | 0.0359 | 2.0164 |
| sensitive_late | zeroshot | cancer_type | CM | 420 | 21 | 0.8268 | 0.3092 | 0.0500 | 6.1846 |
| sensitive_late | zeroshot | cancer_type | CRC | 456 | 16 | 0.7983 | 0.1991 | 0.0351 | 5.6752 |
| sensitive_late | zeroshot | cancer_type | NSCLC | 399 | 33 | 0.5934 | 0.1090 | 0.0827 | 1.3176 |
| sensitive_late | zeroshot | cancer_type | PDAC | 552 | 48 | 0.7561 | 0.2880 | 0.0870 | 3.3119 |
| sensitive_late | zeroshot | treatment_type | double | 407 | 67 | 0.6185 | 0.2256 | 0.1646 | 1.3706 |
| sensitive_late | zeroshot | treatment_type | single | 1420 | 51 | 0.6203 | 0.0993 | 0.0359 | 2.7647 |
| disease_control_early | ft | cancer_type | CM | 420 | 179 | 0.6658 | 0.5934 | 0.4262 | 1.3924 |
| disease_control_early | ft | cancer_type | CRC | 456 | 192 | 0.6912 | 0.6014 | 0.4211 | 1.4283 |
| disease_control_early | ft | cancer_type | NSCLC | 399 | 160 | 0.6619 | 0.5495 | 0.4010 | 1.3704 |
| disease_control_early | ft | cancer_type | PDAC | 552 | 282 | 0.7065 | 0.7087 | 0.5109 | 1.3872 |
| disease_control_early | ft | treatment_type | double | 407 | 272 | 0.5813 | 0.7524 | 0.6683 | 1.1258 |
| disease_control_early | ft | treatment_type | single | 1420 | 541 | 0.6370 | 0.4896 | 0.3810 | 1.2852 |
| disease_control_early | zeroshot | cancer_type | CM | 420 | 179 | 0.6590 | 0.6172 | 0.4262 | 1.4481 |
| disease_control_early | zeroshot | cancer_type | CRC | 456 | 192 | 0.6003 | 0.5820 | 0.4211 | 1.3821 |
| disease_control_early | zeroshot | cancer_type | NSCLC | 399 | 160 | 0.5643 | 0.4383 | 0.4010 | 1.0930 |
| disease_control_early | zeroshot | cancer_type | PDAC | 552 | 282 | 0.6773 | 0.6922 | 0.5109 | 1.3549 |
| disease_control_early | zeroshot | treatment_type | double | 407 | 272 | 0.6134 | 0.7221 | 0.6683 | 1.0806 |
| disease_control_early | zeroshot | treatment_type | single | 1420 | 541 | 0.5512 | 0.4154 | 0.3810 | 1.0904 |
| disease_control_late | ft | cancer_type | CM | 420 | 47 | 0.5957 | 0.2294 | 0.1119 | 2.0498 |
| disease_control_late | ft | cancer_type | CRC | 456 | 47 | 0.7090 | 0.2489 | 0.1031 | 2.4145 |
| disease_control_late | ft | cancer_type | NSCLC | 399 | 48 | 0.5593 | 0.1509 | 0.1203 | 1.2547 |
| disease_control_late | ft | cancer_type | PDAC | 552 | 112 | 0.7007 | 0.3333 | 0.2029 | 1.6425 |
| disease_control_late | ft | treatment_type | double | 407 | 105 | 0.5940 | 0.3161 | 0.2580 | 1.2254 |
| disease_control_late | ft | treatment_type | single | 1420 | 149 | 0.5903 | 0.1496 | 0.1049 | 1.4262 |
| disease_control_late | zeroshot | cancer_type | CM | 420 | 47 | 0.6203 | 0.2305 | 0.1119 | 2.0597 |
| disease_control_late | zeroshot | cancer_type | CRC | 456 | 47 | 0.6615 | 0.2274 | 0.1031 | 2.2059 |
| disease_control_late | zeroshot | cancer_type | NSCLC | 399 | 48 | 0.5535 | 0.1329 | 0.1203 | 1.1044 |
| disease_control_late | zeroshot | cancer_type | PDAC | 552 | 112 | 0.6736 | 0.3707 | 0.2029 | 1.8269 |
| disease_control_late | zeroshot | treatment_type | double | 407 | 105 | 0.5687 | 0.2954 | 0.2580 | 1.1452 |
| disease_control_late | zeroshot | treatment_type | single | 1420 | 149 | 0.5586 | 0.1461 | 0.1049 | 1.3924 |

## Data Build

- Build summary: `data/training_ready_exp31_rnaseq/ptv3/exp31_rnaseq_build_summary.json`
- Added SMILES-only drugs: `5`
- Dropped missing cancer_type rows: `2`
- RNA axis coverage: `10159` / `11092` proteins.
