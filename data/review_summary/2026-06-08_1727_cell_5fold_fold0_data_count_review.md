# 2026-06-08 17:27 HKT Cell 5-Fold Fold0 Data Count Review

## Scope

- Read `docs/Data_Process_1.md` through `docs/Data_Process_4.md` to confirm the repository data-processing and split workflow.
- Checked the PTV3 main single-drug unseen-cell experiment path used by `scripts/exp_03_single_cell_5fold.sh`.
- Verified `cell_5fold_fold0` train/valid/test counts from the split manifest, pkl index files, and sample-id count TSV.

## Sources

- `docs/Data_Process_1.md`
- `docs/Data_Process_2.md`
- `docs/Data_Process_3.md`
- `docs/Data_Process_4.md`
- `scripts/exp_03_single_cell_5fold.sh`
- `data/training_ready/ptv3/splits/ptv3_main_singledrug/split_manifest.json`
- `data/training_ready/ptv3/splits/ptv3_main_singledrug/ptv3_main_singledrug_three_5fold_sample_id_split_counts.tsv`
- `data/training_ready/ptv3/splits/ptv3_main_singledrug/*_indices_cell_5fold_fold0.pkl`

## Findings

- `cell_5fold_fold0` belongs to `ptv3_main_singledrug` and is launched by `scripts/exp_03_single_cell_5fold.sh`.
- The split policy is 5-fold group split by `Cell`.
- Fold0 data counts are:
  - train: `12330`
  - valid/val: `4287`
  - test: `1369`
- `valid_indices_cell_5fold_fold0.pkl` and `val_indices_cell_5fold_fold0.pkl` are count-equivalent compatibility aliases.
- The sample-id count TSV reports unique counts equal to total counts for all three fold0 splits, with `missing_from_raw_info_count=0` and `duplicate_sample_id_count=0`.
- `split_manifest.json` reports zero train/valid/test overlaps for this fold.
