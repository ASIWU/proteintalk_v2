# Organoid CellType-augmented matrix review

- Review time: 2026-07-10 14:34:41 +0800
- Scope: determine whether the two newly added CellType-augmented matrices resolve the remaining exp09 organoid-inference metadata problems
- Reviewed files:
  - `data/rawdata/rna_seq/260709_B260625QC_merged_replicates_add_CellType.csv`
  - `data/rawdata/rna_seq/260709_CAC260627QC_merged_replicates_add_CellType.csv`
- Compared against:
  - `data/rawdata/organoid/260702_B260625QC_merged_replicates.csv`
  - `data/rawdata/organoid/260702_CAC260627QC_merged_replicates.csv`
  - `data/rawdata/rna_seq/260709_samp_inf.csv`
- No raw files were modified and no GPU task was launched.

## Structure

| file | rows | columns | expression columns | row identity |
|---|---:|---:|---:|---|
| B CellType augmented | 13 | 7,418 | 7,416 | `B260625_PTV2_1` through `_13` |
| CAC CellType augmented | 13 | 8,284 | 8,282 | `CAC260627_PTV2_1` through `_13` |

- Both files are sample-by-protein matrices.
- The first column is named `prot_gene`, but its values are sample IDs. It must be interpreted as `sample_id` in downstream code without modifying the raw file.
- The second column is `cell_type` and contains the tissue labels `Lung`, `Pancreas`, and `Colon`.
- All 13 sample IDs and cell-type labels are complete and unique in each file.

## Cell-type mapping validation

Both files explicitly encode the same mapping:

- `PTV2_1` and `PTV2_13`: Pancreas
- `PTV2_5`: Colon
- all remaining `PTV2_2` through `PTV2_12`: Lung

The mappings have zero mismatches against the English tissue column in `260709_samp_inf.csv`. This removes the need to infer tissue type from a separate numeric-suffix join when building the model task.

## Expression integrity validation

- B sample-ID set matches the original B matrix exactly.
- CAC sample-ID set matches the original CAC matrix exactly.
- After matching each sample row to the corresponding original sample column, both matrices have zero expression-value mismatches, including identical missing-value positions.
- Maximum absolute finite-value difference is `0.0` for both sources.
- UniProt IDs parsed from the augmented protein headers match the original `Protein.Group` values in exactly the same order.
- The only full-header text differences are missing gene symbols represented as `_NA` instead of the original empty/NaN gene value:
  - B: `Q6ZSR9_NA`
  - CAC: `Q56UQ5_NA`, `Q6ZSR9_NA`
- Missing fractions remain unchanged: B 13.42%, CAC 11.62%.

## Impact on remaining inference decisions

Resolved:

1. The full proteomics sample ID and tissue type are now carried in the same row.
2. `Lung`, `Pancreas`, and `Colon` can map directly to existing exp09 `cell_type` categories and existing cell-type LLM rows.
3. The augmented files are lossless transpositions of the original expression matrices and are safe to use as source inputs.

Not resolved, but not a data-format blocker:

1. The new organoid sample IDs are still unseen by the checkpoint's learned `Cell` vocabulary and frozen Cell-LLM table. The checkpoint-compatible policy remains to retain the raw sample ID for audit/output while mapping categorical `Cell_index` and Cell-LLM input to `no`, unless a separate custom organoid Cell-LLM artifact is intentionally built.
2. `Cell_plate` and `batch` remain unavailable and must use the checkpoint's `no` category.
3. The augmented files contain no `pert_time` or dose information. The exposure-time and dose policies must still be supplied by the inference task definition.
4. `pat_ID` is not present in the augmented files. It is optional for model inference; including it in reports still requires the numeric-suffix join to `260709_samp_inf.csv`.

## Conclusion

The new files fully resolve sample-to-tissue assignment and can replace the original matrix-plus-separate-tissue-join as the primary organoid expression inputs. They do not turn the organoid sample IDs into checkpoint-known `Cell` categories and do not provide exposure conditions. Those fields require explicit checkpoint-compatible settings rather than additional expression data.
