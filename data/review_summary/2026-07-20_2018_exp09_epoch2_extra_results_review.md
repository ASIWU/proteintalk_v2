# 2026-07-20 20:18 HKT Exp09 Epoch-2 Extra Results Review

## Scope

- Summarized the formal extra-set inference results for `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`.
- Reviewed all six single-drug extra tasks covering mat1-4 and all three double-drug extra tasks (`guomics`, `nature`, and `nc`) under:
  `outputs/2026-06/2026-06-15/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_oracle_epoch002`.
- Cross-checked every task's `metrics.json`, `run_manifest.json`, and `predictions.parquet`; recomputed valid positive/negative counts from `task_label` and verified them against the recorded AUPRC baseline.
- Compared epoch 2 with the same 50-checkpoint oracle sweep to establish its per-task and macro AUPRC ranks.
- This was a read-only review; no inference, metric, code, data, or output artifact was changed.

## Run Identity and Semantics

- All nine manifests reference the exact checkpoint `.../epoch=2.ckpt` and use `split_strategy=test_only`, `split_name=test`, unified head, with no batch limit.
- For single-drug tasks, the unified task metric is the response/sensitivity result.
- For double-drug tasks, the unified task metric is the synergy result; the files also report `unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`, and combined strata.

## Single-drug Extra Headline Results

| Task | AUROC | AUPRC | Baseline | nAUPRC | ACC | N | Positive | Negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mat1 480-FAIMS | 0.734584 | 0.510744 | 0.202917 | 2.517005 | 0.820245 | 17,140 | 3,478 | 13,662 |
| mat1 QE | 0.735535 | 0.510156 | 0.202917 | 2.514111 | 0.821704 | 17,140 | 3,478 | 13,662 |
| mat2 480-FAIMS | 0.827275 | 0.688985 | 0.215819 | 3.192422 | 0.856721 | 13,882 | 2,996 | 10,886 |
| mat2 QE | 0.828739 | 0.692599 | 0.215819 | 3.209167 | 0.857585 | 13,882 | 2,996 | 10,886 |
| mat3 QE | 0.735866 | 0.513986 | 0.210834 | 2.437877 | 0.812241 | 18,332 | 3,865 | 14,467 |
| mat4 QE | 0.820893 | 0.663466 | 0.205937 | 3.221687 | 0.853762 | 12,295 | 2,532 | 9,763 |

- Six-task unweighted macro mean: AUROC `0.780482`, AUPRC `0.596656`, baseline `0.209041`, nAUPRC `2.848712`, ACC `0.837043`.
- Epoch 2 ranks first among the 50 checkpoints by six-task macro single-extra AUPRC.
- Device agreement is strong: mat1 QE versus 480-FAIMS differs by only about `0.00059` AUPRC; mat2 differs by about `0.00361`.
- Strongest single task by AUPRC is mat2 QE (`0.692599`); mat4 has the strongest lift over baseline by nAUPRC (`3.221687`).
- Mat1 and mat3 are the weaker absolute-AUPRC groups (about `0.51`) but still deliver approximately `2.44-2.52x` baseline AUPRC.

## Double-drug Extra Combined Results

| Task | AUROC | AUPRC | Baseline | nAUPRC | ACC | N | Positive | Negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| guomics | 0.716144 | 0.093393 | 0.034617 | 2.697860 | 0.929522 | 5,633 | 195 | 5,438 |
| nature | 0.655606 | 0.069343 | 0.035112 | 1.974927 | 0.938151 | 68,182 | 2,394 | 65,788 |
| nc | 0.574506 | 0.093782 | 0.073045 | 1.283895 | 0.637611 | 15,155 | 1,107 | 14,048 |

- Three-task unweighted macro mean: AUROC `0.648752`, AUPRC `0.085506`, baseline `0.047592`, nAUPRC `1.985561`, ACC `0.835095`.
- Guomics has the best discrimination and relative enrichment: AUROC `0.716144`, nAUPRC `2.697860`.
- Nature is intermediate and retains about `1.97x` baseline enrichment.
- NC has a similar absolute AUPRC to guomics but a much higher class baseline; its nAUPRC is only `1.283895` and AUROC is `0.574506`, making it the weakest double-extra subset.

## Double-drug Stratified Results

| Task | Stratum | AUROC | AUPRC | Baseline | nAUPRC | N | Positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| guomics | unseen cell / seen combo | 0.975000 | 0.750000 | 0.047619 | 15.750000 | 42 | 2 |
| guomics | unseen cell / unseen combo | 0.715036 | 0.084821 | 0.034520 | 2.457162 | 5,591 | 193 |
| nature | unseen cell / seen combo | 0.661646 | 0.107198 | 0.059693 | 1.795834 | 16,652 | 994 |
| nature | unseen cell / unseen combo | 0.627137 | 0.047022 | 0.027169 | 1.730742 | 51,530 | 1,400 |
| nc | unseen cell / seen combo | 0.652680 | 0.171501 | 0.091881 | 1.866544 | 2,057 | 189 |
| nc | unseen cell / unseen combo | 0.556850 | 0.080477 | 0.070087 | 1.148238 | 13,098 | 918 |

- All three double datasets perform better on seen drug combinations than unseen combinations.
- The apparently excellent guomics seen-combination metric is based on only 42 rows and 2 positives, so it is not a stable headline estimate.

## Position Within the 50-checkpoint Sweep

- Single-extra macro AUPRC: epoch 2 ranks `1/50`.
- Double-extra macro AUPRC: epoch 2 ranks `46/50`; the best double macro epoch is epoch 8 with AUPRC `0.101437`.
- Epoch-2 task-specific AUPRC ranks:
  - mat1 480-FAIMS `2/50`; mat1 QE `1/50`;
  - mat2 480-FAIMS `3/50`; mat2 QE `2/50`;
  - mat3 QE `1/50`; mat4 QE `2/50`;
  - guomics `44/50`; nature `3/50`; nc `49/50`.
- Therefore epoch 2 is an excellent single-extra checkpoint, but it is not a generally strong double-extra checkpoint; its double aggregate is held back especially by NC and guomics relative to their other epochs.
