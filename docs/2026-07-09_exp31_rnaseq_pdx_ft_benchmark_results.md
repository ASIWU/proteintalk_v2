# Exp31 RNA-seq PDX Fine-tune Benchmark

- Generated: `2026-07-09T08:15:32+00:00`
- Experiment prefix: `20260709_exp31_rnaseq`
- Split: `brca_ft_valid_nonbrca_test` (BRCA train/valid, non-BRCA test)
- Init checkpoint: exp09 unified single+double checkpoint.
- Label carrier note: exp31 writes each clinical label into `synergy` only for unified-head compatibility; it is not a biological synergy label.

## Status

- All expected preflight and result files passed reporter checks.

## Selected Setting

- Selected label: `sensitive_early`
- BRCA valid AUPRC: `0.8537`
- Non-BRCA test AUPRC: `0.1805`
- Non-BRCA test AUROC: `0.6170`

## Overall Metrics

| label | source | status | valid AUPRC | n | pos | AUROC | AUPRC | base | nAUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sensitive_early | ft | ok | 0.8537 | 1827 | 211 | 0.6170 | 0.1805 | 0.1155 | 1.5633 |
| sensitive_early | zeroshot | ok |  | 1827 | 211 | 0.6297 | 0.1473 | 0.1155 | 1.2754 |
| sensitive_late | ft | ok | 0.8362 | 1827 | 118 | 0.6511 | 0.1411 | 0.0646 | 2.1841 |
| sensitive_late | zeroshot | ok |  | 1827 | 118 | 0.6297 | 0.0839 | 0.0646 | 1.2998 |
| disease_control_early | ft | ok | 0.8480 | 1827 | 813 | 0.6602 | 0.6078 | 0.4450 | 1.3658 |
| disease_control_early | zeroshot | ok |  | 1827 | 813 | 0.5768 | 0.4724 | 0.4450 | 1.0615 |
| disease_control_late | ft | ok | 0.6496 | 1827 | 254 | 0.6733 | 0.2549 | 0.1390 | 1.8335 |
| disease_control_late | zeroshot | ok |  | 1827 | 254 | 0.5698 | 0.1539 | 0.1390 | 1.1069 |

## Stratified Metrics

| label | source | group | value | n | pos | AUROC | AUPRC | base | nAUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sensitive_early | ft | cancer_type | CM | 420 | 62 | 0.6191 | 0.2051 | 0.1476 | 1.3896 |
| sensitive_early | ft | cancer_type | CRC | 456 | 26 | 0.7613 | 0.1825 | 0.0570 | 3.2003 |
| sensitive_early | ft | cancer_type | NSCLC | 399 | 59 | 0.6063 | 0.1960 | 0.1479 | 1.3256 |
| sensitive_early | ft | cancer_type | PDAC | 552 | 64 | 0.7054 | 0.2450 | 0.1159 | 2.1133 |
| sensitive_early | ft | treatment_type | double | 407 | 110 | 0.4216 | 0.2470 | 0.2703 | 0.9138 |
| sensitive_early | ft | treatment_type | single | 1420 | 101 | 0.6291 | 0.1294 | 0.0711 | 1.8187 |
| sensitive_early | zeroshot | cancer_type | CM | 420 | 62 | 0.5969 | 0.1635 | 0.1476 | 1.1078 |
| sensitive_early | zeroshot | cancer_type | CRC | 456 | 26 | 0.6599 | 0.1080 | 0.0570 | 1.8933 |
| sensitive_early | zeroshot | cancer_type | NSCLC | 399 | 59 | 0.5879 | 0.1754 | 0.1479 | 1.1860 |
| sensitive_early | zeroshot | cancer_type | PDAC | 552 | 64 | 0.6825 | 0.2295 | 0.1159 | 1.9794 |
| sensitive_early | zeroshot | treatment_type | double | 407 | 110 | 0.5803 | 0.3202 | 0.2703 | 1.1847 |
| sensitive_early | zeroshot | treatment_type | single | 1420 | 101 | 0.5905 | 0.0941 | 0.0711 | 1.3227 |
| sensitive_late | ft | cancer_type | CM | 420 | 21 | 0.6770 | 0.0762 | 0.0500 | 1.5231 |
| sensitive_late | ft | cancer_type | CRC | 456 | 16 | 0.8243 | 0.1474 | 0.0351 | 4.1995 |
| sensitive_late | ft | cancer_type | NSCLC | 399 | 33 | 0.5919 | 0.1273 | 0.0827 | 1.5395 |
| sensitive_late | ft | cancer_type | PDAC | 552 | 48 | 0.6722 | 0.2018 | 0.0870 | 2.3210 |
| sensitive_late | ft | treatment_type | double | 407 | 67 | 0.4932 | 0.1868 | 0.1646 | 1.1348 |
| sensitive_late | ft | treatment_type | single | 1420 | 51 | 0.6509 | 0.0853 | 0.0359 | 2.3759 |
| sensitive_late | zeroshot | cancer_type | CM | 420 | 21 | 0.6251 | 0.0632 | 0.0500 | 1.2647 |
| sensitive_late | zeroshot | cancer_type | CRC | 456 | 16 | 0.7327 | 0.0663 | 0.0351 | 1.8886 |
| sensitive_late | zeroshot | cancer_type | NSCLC | 399 | 33 | 0.5775 | 0.0988 | 0.0827 | 1.1948 |
| sensitive_late | zeroshot | cancer_type | PDAC | 552 | 48 | 0.6657 | 0.1724 | 0.0870 | 1.9820 |
| sensitive_late | zeroshot | treatment_type | double | 407 | 67 | 0.5435 | 0.1885 | 0.1646 | 1.1448 |
| sensitive_late | zeroshot | treatment_type | single | 1420 | 51 | 0.5965 | 0.0518 | 0.0359 | 1.4422 |
| disease_control_early | ft | cancer_type | CM | 420 | 179 | 0.6094 | 0.5371 | 0.4262 | 1.2601 |
| disease_control_early | ft | cancer_type | CRC | 456 | 192 | 0.7042 | 0.6192 | 0.4211 | 1.4707 |
| disease_control_early | ft | cancer_type | NSCLC | 399 | 160 | 0.6619 | 0.5506 | 0.4010 | 1.3730 |
| disease_control_early | ft | cancer_type | PDAC | 552 | 282 | 0.7013 | 0.6921 | 0.5109 | 1.3547 |
| disease_control_early | ft | treatment_type | double | 407 | 272 | 0.5690 | 0.7418 | 0.6683 | 1.1099 |
| disease_control_early | ft | treatment_type | single | 1420 | 541 | 0.6477 | 0.4999 | 0.3810 | 1.3120 |
| disease_control_early | zeroshot | cancer_type | CM | 420 | 179 | 0.5540 | 0.4246 | 0.4262 | 0.9962 |
| disease_control_early | zeroshot | cancer_type | CRC | 456 | 192 | 0.5646 | 0.4864 | 0.4211 | 1.1551 |
| disease_control_early | zeroshot | cancer_type | NSCLC | 399 | 160 | 0.5322 | 0.4009 | 0.4010 | 0.9996 |
| disease_control_early | zeroshot | cancer_type | PDAC | 552 | 282 | 0.6271 | 0.5924 | 0.5109 | 1.1595 |
| disease_control_early | zeroshot | treatment_type | double | 407 | 272 | 0.6137 | 0.7475 | 0.6683 | 1.1184 |
| disease_control_early | zeroshot | treatment_type | single | 1420 | 541 | 0.5323 | 0.3870 | 0.3810 | 1.0159 |
| disease_control_late | ft | cancer_type | CM | 420 | 47 | 0.5741 | 0.1285 | 0.1119 | 1.1482 |
| disease_control_late | ft | cancer_type | CRC | 456 | 47 | 0.7618 | 0.2602 | 0.1031 | 2.5244 |
| disease_control_late | ft | cancer_type | NSCLC | 399 | 48 | 0.5832 | 0.1734 | 0.1203 | 1.4414 |
| disease_control_late | ft | cancer_type | PDAC | 552 | 112 | 0.7100 | 0.3487 | 0.2029 | 1.7186 |
| disease_control_late | ft | treatment_type | double | 407 | 105 | 0.5605 | 0.3164 | 0.2580 | 1.2266 |
| disease_control_late | ft | treatment_type | single | 1420 | 149 | 0.6494 | 0.2103 | 0.1049 | 2.0041 |
| disease_control_late | zeroshot | cancer_type | CM | 420 | 47 | 0.5389 | 0.1153 | 0.1119 | 1.0307 |
| disease_control_late | zeroshot | cancer_type | CRC | 456 | 47 | 0.6000 | 0.1297 | 0.1031 | 1.2581 |
| disease_control_late | zeroshot | cancer_type | NSCLC | 399 | 48 | 0.5038 | 0.1182 | 0.1203 | 0.9827 |
| disease_control_late | zeroshot | cancer_type | PDAC | 552 | 112 | 0.6130 | 0.2788 | 0.2029 | 1.3743 |
| disease_control_late | zeroshot | treatment_type | double | 407 | 105 | 0.5309 | 0.2763 | 0.2580 | 1.0709 |
| disease_control_late | zeroshot | treatment_type | single | 1420 | 149 | 0.5360 | 0.1143 | 0.1049 | 1.0892 |

## Data Build

- Build summary: `data/training_ready_exp31_rnaseq/ptv3/exp31_rnaseq_build_summary.json`
- Added SMILES-only drugs: `5`
- Dropped missing cancer_type rows: `2`
- RNA axis coverage: `10159` / `11092` proteins.
