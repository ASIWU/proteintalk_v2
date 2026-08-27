# exp01/exp03 Fold0 Feature Attribution Results

- Generated: `2026-07-08T08:46:28+00:00`
- Prefix: `20260708_feature_attr`
- Task: `ptv3_main_singledrug` / `response`
- Checkpoint root: `checkpoints`

## Validation
- All expected manifests are present and match the variant matrix.

## Metrics
| panel | variant | status | MSE | AUPRC | baseline | nAUPRC | AUROC | ACC | count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp01 | full_mse | ok | True | 0.696256 | 0.136389 | 5.104917 | 0.898813 | 0.916808 | 3534 |
| exp01 | full_nomse | ok | False | 0.702725 | 0.136389 | 5.152346 | 0.911140 | 0.913413 | 3534 |
| exp01 | random_control | ok | False | 0.708677 | 0.136389 | 5.195983 | 0.927187 | 0.919921 | 3534 |
| exp01 | no_control_pcep | ok | False | 0.735266 | 0.136389 | 5.390934 | 0.924270 | 0.908319 | 3534 |
| exp01 | pcep_off | ok | False | 0.665255 | 0.136389 | 4.877617 | 0.901126 | 0.918223 | 3534 |
| exp01 | cov_none | ok | False | 0.750769 | 0.136389 | 5.504603 | 0.925042 | 0.919072 | 3534 |
| exp01 | graph_zero | ok | False | 0.597593 | 0.136389 | 4.381525 | 0.851556 | 0.906055 | 3534 |
| exp01 | target_zero | ok | False | 0.706809 | 0.136389 | 5.182292 | 0.896657 | 0.916242 | 3534 |
| exp01 | morgan_zero | ok | False | 0.697962 | 0.136389 | 5.117420 | 0.916270 | 0.912847 | 3534 |
| exp01 | morgan_only | ok | False | 0.599638 | 0.136389 | 4.396514 | 0.850685 | 0.910583 | 3534 |
| exp01 | graph_only | ok | False | 0.644649 | 0.136389 | 4.726536 | 0.899283 | 0.902660 | 3534 |
| exp01 | target_only | ok | False | 0.427465 | 0.136389 | 3.134149 | 0.822926 | 0.866440 | 3534 |
| exp01 | cov_only | ok | False | 0.367696 | 0.136389 | 2.695928 | 0.808786 | 0.863611 | 3534 |
| exp01 | control_only | ok | False | 0.389201 | 0.136389 | 2.853602 | 0.810380 | 0.863611 | 3534 |
| exp03 | full_mse | ok | True | 0.691005 | 0.075992 | 9.093090 | 0.916434 | 0.953409 | 8843 |
| exp03 | full_nomse | ok | False | 0.714316 | 0.075992 | 9.399839 | 0.923149 | 0.956350 | 8843 |
| exp03 | random_control | ok | False | 0.686148 | 0.075992 | 9.029183 | 0.917591 | 0.954880 | 8843 |
| exp03 | no_control_pcep | ok | False | 0.746276 | 0.075992 | 9.820410 | 0.941045 | 0.955897 | 8843 |
| exp03 | pcep_off | ok | False | 0.643828 | 0.075992 | 8.472280 | 0.912545 | 0.945381 | 8843 |
| exp03 | cov_none | ok | False | 0.714865 | 0.075992 | 9.407070 | 0.926269 | 0.955558 | 8843 |
| exp03 | graph_zero | ok | False | 0.638572 | 0.075992 | 8.403111 | 0.893046 | 0.947755 | 8843 |
| exp03 | target_zero | ok | False | 0.658970 | 0.075992 | 8.671539 | 0.908543 | 0.937917 | 8843 |
| exp03 | morgan_zero | ok | False | 0.712227 | 0.075992 | 9.372350 | 0.929642 | 0.959177 | 8843 |
| exp03 | morgan_only | ok | False | 0.740411 | 0.075992 | 9.743233 | 0.912014 | 0.959064 | 8843 |
| exp03 | graph_only | ok | False | 0.678724 | 0.075992 | 8.931478 | 0.914451 | 0.953975 | 8843 |
| exp03 | target_only | ok | False | 0.623793 | 0.075992 | 8.208627 | 0.890831 | 0.946851 | 8843 |
| exp03 | cov_only | ok | False | 0.079229 | 0.075992 | 1.042591 | 0.503322 | 0.924008 | 8843 |
| exp03 | control_only | ok | False | 0.117204 | 0.075992 | 1.542315 | 0.604188 | 0.924008 | 8843 |

## Attribution
## exp01 fold0 unseen-drug
- Largest full-model AUPRC drop: `graph` via `graph_zero` (0.105132).
- Strongest only-feature AUPRC: `graph` via `graph_only` (0.644649).

| feature | ablation variant | full AUPRC | ablated AUPRC | AUPRC drop | nAUPRC drop | AUROC drop |
|---|---|---:|---:|---:|---:|---:|
| graph | graph_zero | 0.702725 | 0.597593 | 0.105132 | 0.770821 | 0.059584 |
| morgan | morgan_zero | 0.702725 | 0.697962 | 0.004763 | 0.034925 | -0.005130 |
| target | target_zero | 0.702725 | 0.706809 | -0.004084 | -0.029946 | 0.014483 |
| covariate | cov_none | 0.702725 | 0.750769 | -0.048044 | -0.352257 | -0.013902 |
| control_pcep_total | no_control_pcep | 0.702725 | 0.735266 | -0.032541 | -0.238589 | -0.013131 |
| pcep | pcep_off | 0.702725 | 0.665255 | 0.037470 | 0.274728 | 0.010014 |

| feature | only-feature variant | AUPRC | nAUPRC | AUROC | count |
|---|---|---:|---:|---:|---:|
| morgan | morgan_only | 0.599638 | 4.396514 | 0.850685 | 3534 |
| graph | graph_only | 0.644649 | 4.726536 | 0.899283 | 3534 |
| target | target_only | 0.427465 | 3.134149 | 0.822926 | 3534 |
| covariate | cov_only | 0.367696 | 2.695928 | 0.808786 | 3534 |
| control_pcep | control_only | 0.389201 | 2.853602 | 0.810380 | 3534 |

## exp03 fold0 unseen-cell
- Largest full-model AUPRC drop: `graph` via `graph_zero` (0.075744).
- Strongest only-feature AUPRC: `morgan` via `morgan_only` (0.740411).

| feature | ablation variant | full AUPRC | ablated AUPRC | AUPRC drop | nAUPRC drop | AUROC drop |
|---|---|---:|---:|---:|---:|---:|
| graph | graph_zero | 0.714316 | 0.638572 | 0.075744 | 0.996729 | 0.030102 |
| morgan | morgan_zero | 0.714316 | 0.712227 | 0.002089 | 0.027490 | -0.006493 |
| target | target_zero | 0.714316 | 0.658970 | 0.055345 | 0.728300 | 0.014605 |
| covariate | cov_none | 0.714316 | 0.714865 | -0.000549 | -0.007231 | -0.003120 |
| control_pcep_total | no_control_pcep | 0.714316 | 0.746276 | -0.031960 | -0.420570 | -0.017896 |
| pcep | pcep_off | 0.714316 | 0.643828 | 0.070487 | 0.927560 | 0.010604 |

| feature | only-feature variant | AUPRC | nAUPRC | AUROC | count |
|---|---|---:|---:|---:|---:|
| morgan | morgan_only | 0.740411 | 9.743233 | 0.912014 | 8843 |
| graph | graph_only | 0.678724 | 8.931478 | 0.914451 | 8843 |
| target | target_only | 0.623793 | 8.208627 | 0.890831 | 8843 |
| covariate | cov_only | 0.079229 | 1.042591 | 0.503322 | 8843 |
| control_pcep | control_only | 0.117204 | 1.542315 | 0.604188 | 8843 |
