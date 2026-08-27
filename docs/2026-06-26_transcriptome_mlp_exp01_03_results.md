# 2026-06-26 Transcriptome MLP exp01-exp03 Results

Generated: `2026-06-26T18:56:22`.
Prefix: `20260626_transcriptome_mlp_official_fixed_residual_v1`.

## Setup

- Input transcriptome: CPA/BioLord prediction h5ad `.X`, 9843 common-HGNC features.
- Drug input: Morgan fingerprint, 2048 bits, keyed by `drug_id` with SMILES fallback.
- Label: `PRISM1st_label_total`, `sensitive=1`, `non-responsive=0`.
- Model: independent transcriptome and drug towers, concat fusion, one BCE logit.
- Validation selection: mean validation AUPRC across screen folds `0,2,4`; tie-breakers are n-AUPRC, AUROC, then validation loss.

## Tuning Winners

| branch | exp | winner | folds | val AUROC | val AUPRC | val n-AUPRC | val loss | hidden | drug hidden | dropout | lr | weight decay |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpa | exp01 | lowdrop_lowlr | 3/3 | 0.857208 | 0.653085 | 4.017597 | 1.917790 | 512 | 128 | 0.1 | 0.0003 | 0.0001 |
| cpa | exp02 | lowdrop_lowlr | 3/3 | 0.902255 | 0.726041 | 3.709979 | 1.116233 | 512 | 128 | 0.1 | 0.0003 | 0.0001 |
| cpa | exp03 | lowdrop_lowlr | 3/3 | 0.929722 | 0.813131 | 6.000052 | 0.691031 | 512 | 128 | 0.1 | 0.0003 | 0.0001 |
| biolord | exp01 | lowdrop_lowlr | 3/3 | 0.858742 | 0.650968 | 4.004616 | 3.088505 | 512 | 128 | 0.1 | 0.0003 | 0.0001 |
| biolord | exp02 | lowdrop_lowlr | 3/3 | 0.899229 | 0.716790 | 3.665898 | 1.148473 | 512 | 128 | 0.1 | 0.0003 | 0.0001 |
| biolord | exp03 | regularized | 3/3 | 0.925269 | 0.806722 | 6.066418 | 0.776587 | 256 | 128 | 0.3 | 0.001 | 0.0003 |

## Final Mean5 Test

| branch | exp | folds | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| biolord | exp01 | 5/5 | 0.827308 | 0.584105 | 0.118838 | 4.921827 | 17986 | 2137 | 15849 |
| biolord | exp02 | 5/5 | 0.944421 | 0.813494 | 0.173980 | 6.116543 | 17986 | 2137 | 15849 |
| biolord | exp03 | 5/5 | 0.934373 | 0.778016 | 0.154930 | 6.502679 | 17986 | 2137 | 15849 |
| cpa | exp01 | 5/5 | 0.834701 | 0.579502 | 0.118838 | 4.883284 | 17986 | 2137 | 15849 |
| cpa | exp02 | 5/5 | 0.942125 | 0.809793 | 0.173980 | 6.142308 | 17986 | 2137 | 15849 |
| cpa | exp03 | 5/5 | 0.932881 | 0.777893 | 0.154930 | 6.462344 | 17986 | 2137 | 15849 |

## Final Fold Detail

| branch | exp | fold | config | best epoch | val AUPRC | test AUROC | test AUPRC | baseline | n-AUPRC | count | pos | neg |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| biolord | exp01 | 0 | lowdrop_lowlr | 7 | 0.754780 | 0.758275 | 0.496352 | 0.120910 | 4.105153 | 3606 | 436 | 3170 |
| biolord | exp01 | 1 | lowdrop_lowlr | 29 | 0.555774 | 0.875264 | 0.700477 | 0.120022 | 5.836227 | 3591 | 431 | 3160 |
| biolord | exp01 | 2 | lowdrop_lowlr | 2 | 0.660689 | 0.761739 | 0.421883 | 0.108467 | 3.889506 | 3614 | 392 | 3222 |
| biolord | exp01 | 3 | lowdrop_lowlr | 10 | 0.646188 | 0.888799 | 0.706706 | 0.112594 | 6.276596 | 3597 | 405 | 3192 |
| biolord | exp01 | 4 | lowdrop_lowlr | 8 | 0.539401 | 0.852465 | 0.595104 | 0.132197 | 4.501653 | 3578 | 473 | 3105 |
| biolord | exp02 | 0 | lowdrop_lowlr | 2 | 0.608553 | 0.942520 | 0.736292 | 0.145419 | 5.063232 | 7750 | 1127 | 6623 |
| biolord | exp02 | 1 | lowdrop_lowlr | 6 | 0.755778 | 0.913047 | 0.716570 | 0.106180 | 6.748630 | 5453 | 579 | 4874 |
| biolord | exp02 | 2 | lowdrop_lowlr | 7 | 0.747815 | 0.968966 | 0.918451 | 0.224599 | 4.089293 | 374 | 84 | 290 |
| biolord | exp02 | 3 | lowdrop_lowlr | 4 | 0.746205 | 0.939276 | 0.893828 | 0.326531 | 2.737348 | 196 | 64 | 132 |
| biolord | exp02 | 4 | lowdrop_lowlr | 10 | 0.760335 | 0.958295 | 0.802329 | 0.067173 | 11.944211 | 4213 | 283 | 3930 |
| biolord | exp03 | 0 | regularized | 4 | 0.763714 | 0.917139 | 0.847786 | 0.303871 | 2.789951 | 1369 | 416 | 953 |
| biolord | exp03 | 1 | regularized | 12 | 0.841871 | 0.935287 | 0.775532 | 0.105141 | 7.376104 | 5174 | 544 | 4630 |
| biolord | exp03 | 2 | regularized | 9 | 0.801788 | 0.892094 | 0.638256 | 0.183423 | 3.479697 | 1303 | 239 | 1064 |
| biolord | exp03 | 3 | regularized | 10 | 0.852840 | 0.956934 | 0.815072 | 0.112056 | 7.273781 | 5408 | 606 | 4802 |
| biolord | exp03 | 4 | regularized | 10 | 0.854216 | 0.970411 | 0.813432 | 0.070161 | 11.593862 | 4732 | 332 | 4400 |
| cpa | exp01 | 0 | lowdrop_lowlr | 2 | 0.757281 | 0.766085 | 0.465903 | 0.120910 | 3.853319 | 3606 | 436 | 3170 |
| cpa | exp01 | 1 | lowdrop_lowlr | 12 | 0.534433 | 0.884191 | 0.718591 | 0.120022 | 5.987145 | 3591 | 431 | 3160 |
| cpa | exp01 | 2 | lowdrop_lowlr | 6 | 0.633429 | 0.782478 | 0.428588 | 0.108467 | 3.951317 | 3614 | 392 | 3222 |
| cpa | exp01 | 3 | lowdrop_lowlr | 11 | 0.637295 | 0.882485 | 0.689912 | 0.112594 | 6.127439 | 3597 | 405 | 3192 |
| cpa | exp01 | 4 | lowdrop_lowlr | 8 | 0.545153 | 0.858264 | 0.594515 | 0.132197 | 4.497200 | 3578 | 473 | 3105 |
| cpa | exp02 | 0 | lowdrop_lowlr | 2 | 0.686286 | 0.940786 | 0.729083 | 0.145419 | 5.013655 | 7750 | 1127 | 6623 |
| cpa | exp02 | 1 | lowdrop_lowlr | 11 | 0.750115 | 0.914510 | 0.707065 | 0.106180 | 6.659112 | 5453 | 579 | 4874 |
| cpa | exp02 | 2 | lowdrop_lowlr | 11 | 0.744263 | 0.965887 | 0.912177 | 0.224599 | 4.061361 | 374 | 84 | 290 |
| cpa | exp02 | 3 | lowdrop_lowlr | 7 | 0.734978 | 0.929806 | 0.874450 | 0.326531 | 2.678003 | 196 | 64 | 132 |
| cpa | exp02 | 4 | lowdrop_lowlr | 6 | 0.758373 | 0.959635 | 0.826189 | 0.067173 | 12.299407 | 4213 | 283 | 3930 |
| cpa | exp03 | 0 | lowdrop_lowlr | 10 | 0.746947 | 0.911282 | 0.825446 | 0.303871 | 2.716431 | 1369 | 416 | 953 |
| cpa | exp03 | 1 | lowdrop_lowlr | 19 | 0.858595 | 0.928673 | 0.779258 | 0.105141 | 7.411545 | 5174 | 544 | 4630 |
| cpa | exp03 | 2 | lowdrop_lowlr | 2 | 0.835985 | 0.898823 | 0.677632 | 0.183423 | 3.694373 | 1303 | 239 | 1064 |
| cpa | exp03 | 3 | lowdrop_lowlr | 6 | 0.883471 | 0.954275 | 0.828888 | 0.112056 | 7.397077 | 5408 | 606 | 4802 |
| cpa | exp03 | 4 | lowdrop_lowlr | 9 | 0.850758 | 0.971353 | 0.778242 | 0.070161 | 11.092296 | 4732 | 332 | 4400 |

## Exception Summary

- No missing, non-OK, or one-class final folds detected.
