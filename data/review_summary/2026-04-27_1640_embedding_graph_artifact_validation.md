# 2026-04-27 16:40 Embedding / Graph Artifact Validation

## Scope

- Validated artifacts generated under `data/training_ready/ptv3/derived` against `data/training_ready/ptv3/global_meta.json`.
- Checked protein embedding, drug embedding, PPI, DDI, and PDI outputs.

## Results

- `protein_embedding_esm.pkl`
  - Shape is `(11345, 1280)`, matching `protein_index` row count.
  - `item_to_index` and `index_to_item` match `global_meta.json`.
  - All values are finite; zero-row count is `0`.
  - `unresolved_items` is empty.
  - `sequence_fallback_items` contains 9 entries: 7 missing FASTA sequences plus `control` and `no`.
  - Non-blocking note: payload does not contain `embedding_dim`, because this artifact appears to have been generated before the metadata field was added.
- `drug_embedding_morgan_2048.pkl`
  - Shape is `(6113, 2048)`, matching `pert_index` row count.
  - `item_to_index` and `index_to_item` match `global_meta.json`.
  - Values are binary `0.0` / `1.0`; all finite.
  - Zero rows are exactly `control`, `PC`, and `no`, which correspond to empty-SMILES fallback items.
  - `unresolved_items` is empty.
- `ppi_matrix.npy`
  - Shape is `(11345, 11345)`, matching `protein_index x protein_index`.
  - Matrix is finite, symmetric, diagonal-zero, and has value range `0.0` to `0.9999976`.
  - `control` and `no` rows/columns are all zero.
  - Nonzero entries: `1113814`; nonzero rows/columns: `10420`.
- `ddi_matrix.npy`
  - Shape is `(6113, 6113)`, matching `pert_index x pert_index`.
  - Matrix is finite, symmetric, diagonal-one, and has value range `0.0` to `1.0`.
  - Special fallback rows `control`, `PC`, and `no` each have only the diagonal nonzero.
  - Nonzero entries: `37048959`; all rows/columns have at least one nonzero value due the diagonal.
- `pdi_matrix.npy`
  - Shape is `(6113, 11345)`, matching `pert_index x protein_index`.
  - Matrix is finite and has value range `0.0` to `0.9990000`.
  - `control`, `PC`, and `no` perturbation rows are all zero.
  - `control` and `no` protein columns are all zero.
  - Nonzero entries: `384620`; nonzero rows: `5308`; nonzero columns: `8427`.

## Conclusion

- The generated PTV3 embedding and graph artifacts are aligned to `global_meta.json` and are usable.
- The only issue found is a metadata-only gap in the protein embedding pickle: missing `embedding_dim` field despite the actual matrix dimension being correct.
