# 2026-04-27 15:07 Drug / Graph Coverage Review

## Scope

- Reviewed `utils/04_build_embeddings_from_global_meta.py` drug embedding behavior.
- Reviewed `utils/05_build_graph_matrices_from_global_meta.py` PPI, DDI, and PDI matrix shape / missing-mapping behavior.
- Checked current generated artifacts under `data/training_ready/*/derived`.

## Drug Embedding Findings

- The drug embedding builder allocates `embedding_matrix` with one row per `global_meta.json["pert_index"]` entry.
- It only computes Morgan fingerprints for perturbations with non-empty SMILES that parse into an RDKit `Mol`.
- Perturbations with `pert_id == "no"`, missing SMILES, or invalid SMILES are skipped, recorded in `unresolved_items`, and left as zero rows.
- PTV3 metadata has 6,113 perturbation entities: 6,110 parse into RDKit molecules and 3 are missing / special (`control`, `PC`, `no`).
- PTV1 metadata has 148 perturbation entities: 146 parse into RDKit molecules and 2 are missing / special (`#3`, `no`).
- No current `drug_embedding_morgan_2048.pkl` artifact was found under `data/training_ready/*/derived`.
- Under the same standard now used for protein embedding, where every entity should be sent through the generator with a fallback input, the current drug embedding builder is incomplete.

## RDKit Empty-SMILES Check

- `Chem.MolFromSmiles("")` returns a valid empty molecule.
- Morgan fingerprint for the empty molecule is an all-zero bit vector.
- This means an empty-SMILES fallback can make every perturbation pass through RDKit fingerprint generation, but fallback rows will still be zero vectors by chemistry.

## DDI Findings

- DDI allocates a square matrix of shape `len(pert_index) x len(pert_index)`.
- It computes fingerprints only for perturbations with valid non-empty SMILES.
- Missing / special / invalid perturbations keep all-zero rows and columns.
- No current DDI matrix artifact was found under `data/training_ready/*/derived`.
- Empty fingerprint fallback should be handled carefully for DDI, because RDKit Tanimoto similarity of empty fingerprint vs empty fingerprint is `1.0`, which would create artificial similarity among all missing/special perturbations.

## PPI Findings

- PPI allocates a square matrix of shape `len(protein_index) x len(protein_index)`.
- `control` and `no` are included in the matrix shape and remain zero rows/columns.
- Protein IDs without graph edges or without matching external node IDs remain zero rows/columns.
- If no edges remain after filtering to metadata proteins, the builder raises an error instead of writing an empty full-size matrix.
- No current PPI matrix artifact was found under `data/training_ready/*/derived`.

## PDI Findings

- PDI allocates a rectangular matrix of shape `len(pert_index) x len(protein_index)`.
- Perturbations without chemical mapping and proteins without node mapping remain zero-only rows/columns.
- If external links have no matching mapped chemical/protein pairs, the builder still writes a full-size zero matrix.
- No current PDI matrix artifact was found under `data/training_ready/*/derived`.

## Verification

- `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile utils/04_build_embeddings_from_global_meta.py utils/05_build_graph_matrices_from_global_meta.py`
- Counted RDKit parse coverage for `data/training_ready/ptv3/global_meta.json` and `data/training_ready/ptv1/global_meta.json`.
- Checked generated artifact presence under `data/training_ready/*/derived`.

## Follow-up Fix

- At 2026-04-27 15:15 HKT, drug embedding was updated so every `pert_index` entity passes through RDKit fingerprint generation. Missing / invalid / special values use empty-SMILES fallback and are recorded in `smiles_fallback_items`.
- PTV3 drug embedding was regenerated at `data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl`; shape is `(6113, 2048)`, matching 6,113 `pert_index` entities.
- DDI was updated so every `pert_index` entity has a fingerprint. PTV3 DDI was generated at `data/training_ready/ptv3/derived/ddi_matrix.npy`; shape is `(6113, 6113)`, and metadata records `fingerprint_count: 6113`.
- PPI now writes a full-size zero matrix plus warning metadata when no external edges remain after filtering, instead of failing before producing an artifact.
- PDI metadata now records expected axis counts and matched-link counts for shape auditing.
