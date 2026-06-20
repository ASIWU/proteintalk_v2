# PTV1 exp13 Positive-Weight Tuning Report

- base prefix: `20260610_ptv1_cell_celltype_llm_exp13_posweight_v1_mse050_target_pdi`
- scope: exp11 train/valid/test plus exp13 test only; no exp12 and no all_train
- fixed baseline config: `MSE_WEIGHT=0.50`, `MSE_TARGET_MODE=pdi`, `LR=2e-4`, `DROPOUT=0.15`
- fixed architecture: graph `real` + structural RP + drug concat, Cell LLM frozen v2, cell-type LLM frozen v3
- exp13 valid: exp11 validation-selected checkpoint inferred on exp13 test
- exp13 oracle: diagnostic best exp13 test AUPRC over saved exp11 epoch checkpoints; not official validation selection
- audit status: `ok`

## Final selections

- best valid: `posw10` requested `10` resolved `10.000000`; exp13 valid AUPRC/AUROC/nAUPRC 0.604275 / 0.625969 / 1.176179
- best oracle: `posw500` requested `500` resolved `500.000000`; exp13 oracle AUPRC/AUROC/nAUPRC 0.667455 / 0.662820 / 1.299153; epoch `19`

## Candidate summary

| candidate | requested | resolved | exp11 valid AUPRC | exp11 epoch | exp11 test AUPRC | exp11 AUROC | exp11 nAUPRC | exp11 rows | exp13 valid AUPRC | valid AUROC | valid nAUPRC | valid rows | oracle epoch | oracle ckpt | exp13 oracle AUPRC | oracle AUROC | oracle nAUPRC | oracle rows | oracle ckpt exp11 AUPRC | oracle ckpt exp11 AUROC | oracle ckpt exp11 nAUPRC | oracle exp11 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| posw0p5 | 0.5 | 0.500000 | 0.818904 | 14 | 0.911105 | 0.953892 | 3.110996 | 799 | 0.582069 | 0.613165 | 1.132956 | 218 | 39 | epoch=39-step=1120.ckpt | 0.619851 | 0.627316 | 1.206496 | 218 | 0.844285 | 0.908729 | 2.882837 | 799 |
| posw1 | 1 | 1.000000 | 0.811033 | 16 | 0.883922 | 0.939407 | 3.018179 | 799 | 0.601089 | 0.627653 | 1.169976 | 218 | 23 | epoch=23-step=672.ckpt | 0.614654 | 0.618725 | 1.196381 | 218 | 0.862569 | 0.929491 | 2.945267 | 799 |
| posw10 | 10 | 10.000000 | 0.833798 | 10 | 0.844017 | 0.929101 | 2.881923 | 799 | 0.604275 | 0.625969 | 1.176179 | 218 | 25 | epoch=25-step=728.ckpt | 0.607439 | 0.619230 | 1.182337 | 218 | 0.865957 | 0.926015 | 2.956836 | 799 |
| posw20 | 20 | 20.000000 | 0.845389 | 8 | 0.836604 | 0.933148 | 2.856611 | 799 | 0.571953 | 0.592613 | 1.113265 | 218 | 15 | epoch=15-step=448.ckpt | 0.617974 | 0.625632 | 1.202842 | 218 | 0.804448 | 0.912643 | 2.746813 | 799 |
| posw50 | 50 | 50.000000 | 0.834664 | 5 | 0.830049 | 0.932596 | 2.834229 | 799 | 0.592383 | 0.578883 | 1.153031 | 218 | 17 | epoch=17-step=504.ckpt | 0.611583 | 0.628748 | 1.190403 | 218 | 0.844183 | 0.919968 | 2.882488 | 799 |
| posw100 | 100 | 100.000000 | 0.843378 | 10 | 0.805004 | 0.931185 | 2.748709 | 799 | 0.586055 | 0.604826 | 1.140713 | 218 | 24 | epoch=24-step=700.ckpt | 0.626203 | 0.631696 | 1.218859 | 218 | 0.800939 | 0.885550 | 2.734829 | 799 |
| posw200 | 200 | 200.000000 | 0.815752 | 11 | 0.730368 | 0.902235 | 2.493865 | 799 | 0.599582 | 0.595814 | 1.167043 | 218 | 13 | epoch=13-step=392.ckpt | 0.635189 | 0.649975 | 1.236349 | 218 | 0.773823 | 0.895333 | 2.642241 | 799 |
| posw500 | 500 | 500.000000 | 0.800807 | 6 | 0.725874 | 0.914537 | 2.478519 | 799 | 0.588234 | 0.601078 | 1.144956 | 218 | 19 | epoch=19-step=560.ckpt | 0.667455 | 0.662820 | 1.299153 | 218 | 0.747199 | 0.910793 | 2.551332 | 799 |
| posw_negpos | neg/pos | 1.992350 | 0.805145 | 8 | 0.877783 | 0.938227 | 2.997216 | 799 | 0.574627 | 0.603310 | 1.118470 | 218 | 16 | epoch=16-step=476.ckpt | 0.635858 | 0.652670 | 1.237653 | 218 | 0.831588 | 0.908914 | 2.839483 | 799 |

## Audit

- all expected candidates completed
- exp11 selected-test rows are `799` for every candidate
- exp13 valid/oracle rows are `218` for every candidate
- all official rows pass graph/Cell LLM/cell-type LLM/MSE target policy checks
