# 2026-04-27 15:59 PDI Usage Review

## Scope

- Reviewed `utils/05_build_graph_matrices_from_global_meta.py` PDI implementation.
- Compared it with the original PDI-related scripts:
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0303_inchikey2chemical_id.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0303_newpdi.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0303_uniprotid2protein_experimental.py`
  - `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/0304_buildpdi.py`

## Findings

- Current PDI builder writes matrix shape `len(pert_index) x len(protein_index)`, i.e. rows are perturbations and columns are proteins.
- Original `0304_buildpdi.py` wrote protein-by-drug orientation. The current orientation is intentional for direct lookup by perturbation row in the training-ready dataloader.
- Current code replaces the original multi-step PDI workflow:
  - `0303_inchikey2chemical_id.py` maps `pert_id -> flat_chemical_id`.
  - `0303_uniprotid2protein_experimental.py` maps UniProt IDs to STRING protein IDs.
  - `0303_newpdi.py` filters the large STITCH chemical-protein links table.
  - `0304_buildpdi.py` builds the matrix.
- In `utils/05_build_graph_matrices_from_global_meta.py pdi`, chemical mapping can be generated on the fly from `global_meta["pertid_to_smiles"]` plus `chemicals.inchikeys.v5.0.tsv`.
- Protein mapping still needs a UniProt-to-STRING JSON or online STRING mapping. For reproducibility, a local JSON exported from `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/uniprot_to_string.db` is preferred.
- The large link TSV at `/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/protein_chemical.links.detailed.v5.0.tsv` was not readable in this session due file permissions, while the parquet copy is readable. The current CLI name is `--links-tsv`, and parquet cannot currently be streamed because the parser passes a chunk size.

## Recommendation

- Use `utils/05_build_graph_matrices_from_global_meta.py pdi` instead of the old scripts.
- Provide `--chemical-inchikey-tsv /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/chemicals.inchikeys.v5.0.tsv`.
- Provide a local `--protein-node-mapping-json` exported from the SQLite mapping database.
- Use the readable STITCH TSV if permissions allow; otherwise patch the current builder to stream the parquet row groups directly.
