# 2026-04-27 16:12 PDI Self-Contained Builder Review

## Scope

- Reviewed the PDI workflow in `utils/05_build_graph_matrices_from_global_meta.py` after the STITCH resources were moved to `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/stitch_db`.
- Checked whether PDI can be generated without running the old helper scripts separately.

## Findings

- Before this update, the current PDI command still depended on external preprocessing choices: a readable links TSV, a chemical InChIKey path, and a protein mapping JSON or online mapping.
- The STITCH links TSV in `stitch_db` is not readable by the current user permissions, but the parquet copy is readable and contains `chemical`, `protein`, and score columns.
- The local SQLite database `uniprot_to_string.db` contains table `mapping(alias, string_protein_id)`, which is sufficient to derive UniProt-to-STRING mappings directly inside the current script.

## Changes Applied

- Added default `--stitch-db-dir` support pointing to `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/stitch_db`.
- Added auto-discovery for:
  - `protein_chemical.links.detailed.v5.0.parquet`
  - `chemicals.inchikeys.v5.0.tsv`
  - `uniprot_to_string.db`
- Added parquet row-group streaming for PDI links so the 28G parquet can be processed without loading it all into memory.
- Added SQLite UniProt mapping loading, replacing the need to pre-export `--protein-node-mapping-json`.
- Kept backward compatibility for `--links-tsv`, `--links-path`, `--pert-to-flat-json`, and `--protein-node-mapping-json`.

## Verification

- `python -m py_compile utils/05_build_graph_matrices_from_global_meta.py`
- `python utils/05_build_graph_matrices_from_global_meta.py pdi --help`
- Resolved the real `stitch_db` paths and inspected the first parquet row group.
- Loaded a sample UniProt-to-STRING mapping from the real SQLite DB.
- Ran a synthetic end-to-end PDI smoke test with a temporary STITCH directory and verified matrix shape, score value, and `.meta.json` matched-link count.
