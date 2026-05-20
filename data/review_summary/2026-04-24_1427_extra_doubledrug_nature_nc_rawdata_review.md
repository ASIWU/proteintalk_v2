# 2026-04-24 14:27 Extra Double-Drug Nature / NC Rawdata Review

## Scope

Reviewed the updated extra double-drug rawdata files:

- `data/rawdata/extra_doubeldrug/260424nc_drugComb_info_unique_with_smiles.csv`
- `data/rawdata/extra_doubeldrug/260424nature_drugComb_info_unique_with_smiles.csv`

## Findings

- The existing task specs still referenced the old `20260411...csv` filenames, while the current raw checkout contains the new `260424...with_smiles.csv` files.
- Both updated files are metadata-only tables. They contain sample metadata, labels, drug names, target text, and SMILES columns, but no perturbation proteome matrix columns.
- The current parsing rules for NC / Nature column names were mostly still compatible, but the code needed to prefer the new files and retain the newly added audit fields.

## Changes Made

- Updated `utils/00_standardize_rawdata.py` so `ptv3_extra_doubledrug_nc` and `ptv3_extra_doubledrug_nature` prefer the new `260424` files, with old `20260411` files retained as fallback candidates.
- Preserved raw SMILES audit columns for both files.
- Preserved NC-specific raw audit columns: `anchor_lib`, `group`, `group1`, and `Cell2`.
- Preserved Nature-specific raw audit columns: `Tissue`, `Cancer.Type`, `Anchor.Pathway`, `Library.Pathway`, and `Synergy?`.

## Rerun Results

- Stage-1 validation passed.
- Stage-2 validation passed.
- `ptv3_extra_doubledrug_nc`: stage-1 `16394 x 0`; stage-2 processed / feature `16412 x 11343`, with `18` matched control rows.
- `ptv3_extra_doubledrug_nature`: stage-1 `23389 x 0`; stage-2 processed / feature `23415 x 11343`, with `26` matched control rows.

## Notes

The `0` protein dimension in stage 1 is expected for these two tasks because the updated raw files do not include native perturbation proteome matrices. Stage 2 keeps the perturbation rows and appends matched control proteomes.
