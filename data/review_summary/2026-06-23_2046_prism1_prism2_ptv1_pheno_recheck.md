# 2026-06-23 20:46 HKT PRISM1/PRISM2 vs PTV1 pheno Recheck

## Scope

- Rechecked the claim that `PRISM1st_label_total` is fully consistent with legacy PTV1 `pheno.csv`.
- Used direct `sample_id` alignment through PTV1 `control_id.csv`, avoiding possible `(Cell, pert_id, time)` parser or aggregation artifacts.
- Compared PTV1 `pheno.csv` against `PRISM1st_label_total`, `PRISM1st_label`, `PRISM1st_label_desalt_unique`, and `PRISM2nd_label_total` from `data/rawdata/update_0623/260513ptv3_EGH_28602sampinfo_with_smiles_check_prism1_label_add_prism2_label_add_machine_details.csv`.
- Checked current exp02/exp03 manifests to confirm which label key was used for training.

## Main Finding

PTV1 `pheno.csv` is exactly aligned with `PRISM2nd_label_total`, not with `PRISM1st_label_total`, in the current `update_0623` file.

## Sample-ID Direct Comparison

For each PTV1 row, `sample_id` was read from `*_control_id.csv` first column and joined directly to the raw `update_0623` metadata.

| benchmark | split | PTV1 valid rows | PRISM1 comparable | PRISM1 missing label | PRISM1 mismatch | PRISM1 mismatch rate | PRISM2 comparable | PRISM2 mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cell_5fold | test | 7492 | 7410 | 82 | 936 | 12.6316% | 7492 | 0 |
| cell_5fold | train | 27338 | 27050 | 288 | 3451 | 12.7579% | 27338 | 0 |
| cell_5fold | valid | 2630 | 2590 | 40 | 293 | 11.3127% | 2630 | 0 |
| cell_type_5fold | test | 7492 | 7410 | 82 | 936 | 12.6316% | 7492 | 0 |
| cell_type_5fold | train | 25212 | 24944 | 268 | 3050 | 12.2274% | 25212 | 0 |
| cell_type_5fold | valid | 4756 | 4696 | 60 | 694 | 14.7785% | 4756 | 0 |

The test split PRISM1 confusion table is:

| PTV1 pheno | PRISM1 label 0 | PRISM1 label 1 |
|---:|---:|---:|
| 0 | 5266 | 306 |
| 1 | 630 | 1208 |

For `PRISM2nd_label_total`, every PTV1-valid row is comparable and the mismatch count is 0.

## PTV1 Dataset Key-Level Comparison

This reproduces the PTV1 dataset behavior by requiring both 6h and 24h rows for a key and using the first row's `pheno` label.

| benchmark | split | PTV1 valid keys | PRISM1 comparable | PRISM1 missing/conflict | PRISM1 mismatch | PRISM1 mismatch rate | PRISM2 comparable | PRISM2 mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cell_5fold | test | 3673 | 3633 | 40 | 456 | 12.5516% | 3673 | 0 |
| cell_5fold | train | 13384 | 13244 | 140 | 1678 | 12.6699% | 13384 | 0 |
| cell_5fold | valid | 1308 | 1288 | 20 | 146 | 11.3354% | 1308 | 0 |
| cell_type_5fold | test | 3673 | 3633 | 40 | 456 | 12.5516% | 3673 | 0 |
| cell_type_5fold | train | 12461 | 12329 | 132 | 1501 | 12.1745% | 12461 | 0 |
| cell_type_5fold | valid | 2231 | 2203 | 28 | 323 | 14.6618% | 2231 | 0 |

## Training Label Used by Current exp02/exp03

The final `20260623_130336_update0623_nowandb` exp02/exp03 manifests still use PRISM1 as the response label:

- `task_name`: `ptv3_main_singledrug`
- `task_head`: `response`
- `effective_key1`: `PRISM1st_label_total`
- `task_label_key`: `PRISM1st_label_total`

This was confirmed in:

- `checkpoints/20260623_130336_update0623_nowandb_single_cell_type_fold0/run_manifest.json`
- `checkpoints/20260623_130336_update0623_nowandb_single_cell_fold0/run_manifest.json`

## Conclusion

The earlier mismatch result was not caused by a bad key parser. Direct sample-id alignment confirms that PTV1 `pheno.csv` does not equal current `PRISM1st_label_total`; it equals `PRISM2nd_label_total`. The missed point was that `update_0623` carries the PTV1-consistent label in `PRISM2nd_label_total`, while exp02/exp03 still train on `PRISM1st_label_total`.
