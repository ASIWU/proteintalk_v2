# Model SMILES Input Contract Review

- Review time: 2026-07-22 20:51 HKT (UTC+08:00)
- Scope: identify the exact SMILES source and numerical drug input for every active ProteinTalk model/data family, and explain the earlier Exp32 colleague-mapping legacy-SMILES column.
- This review changed no model code, checkpoint, global metadata, derived feature artifact, or prediction output.

## Core Contract

- Training and inference networks do not read raw SMILES text directly.
- Each row supplies two perturbation indices. The dataset resolves both indices into two rows of `drug_embedding_morgan_2048.pkl`; the fast model therefore receives a two-slot numeric Morgan tensor, while legacy graph models use the same Morgan rows as drug-node features.
- The maintained Morgan artifacts use RDKit `AllChem.GetMorganGenerator(radius=2, fpSize=2048)` and are `float32` binary matrices aligned exactly to `global_meta.json["pert_index"]`.
- `includeChirality` is not passed. In the installed RDKit 2025.03.6 it defaults to `False`. A direct enantiomer check produced identical default fingerprints and different fingerprints only after explicitly setting `includeChirality=True`.
- Therefore the selected source string can retain stereochemical annotations, but the current Morgan model-input branch is chirality-insensitive.
- The DDI matrix also uses radius-2/2048 Morgan fingerprints with the same default chirality setting. PDI construction uses the same canonical SMILES to produce an InChIKey for STITCH lookup. Exp09/Exp31/Exp32 graph features are derived from PDI/DDI/PPI artifacts, so they are secondary numerical features rather than raw SMILES text.

## SMILES Sources By Model Family

### PTV3 main models, Exp01-Exp30, Exp32, Prism2, and patient-validation inference

- Stage-1 main/extra single and extra-double standardization selects `Smiles_with_chiral`, then legacy lower-case `smiles`, then `Smiles_no_chiral`; double-drug fields use the corresponding slot-specific columns.
- A first-seen-by-task-order canonical map is then fixed per perturbation ID and stored in `data/training_ready/ptv3/global_meta.json["pertid_to_smiles"]`.
- The numerical input is `data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl`, shape `(6131, 2048)`, with 6,128 non-empty SMILES entries and zero-vector fallbacks for `control`, `PC`, and `no`.
- Exp32 explicitly reuses this exact artifact. Exp32's 3,217 screened drugs all have `exp32_model_smiles` exactly equal to a selected raw `Smiles_with_chiral` value.
- The Exp32, Prism2, and patient-validation artifact copies resolve to or hash-identically match the main PTV3 artifact (`36a38db707d719483a238d5ea149b3c88d56e967d59ee7ee08837557d50a2d15`).

### PTV1 models

- PTV1 main-drug SMILES come from `data/rawdata/ptv1/ptv1.csv` column `SMILES`.
- PTV1 extra-single drugs not supplied by the PTV1 main registry use the PTV3 canonical `pertid_to_smiles` lookup.
- The resulting PTV1 artifact is `data/training_ready/ptv1/derived/drug_embedding_morgan_2048.pkl`, shape `(128, 2048)`, with 126 non-empty SMILES and zero-vector fallbacks for `#3` and `no`.

### Exp31 RNA-seq/PDX fine-tuning

- The first 6,131 drug rows are copied without modification from the main PTV3 Morgan artifact.
- Five new `exp31smiles::*` IDs are sourced from the Exp31 raw sample-info `smiles_a` / `smiles_b` values and converted with the same radius-2/2048/default-no-chirality Morgan generator.
- The resulting artifact is `data/training_ready_exp31_rnaseq/ptv3/derived/drug_embedding_morgan_2048.pkl`, shape `(6136, 2048)`.

### Transcriptome downstream MLP and intentional ablations

- The transcriptome downstream MLP first looks up the main PTV3 Morgan artifact by `drug_id`; only an ID miss triggers an on-the-fly radius-2/2048 fingerprint from exported `drug_smiles_primary`, which itself comes from the standardized task SMILES.
- Feature-attribution `zero_morgan` runs intentionally replace the real drug artifact with an all-zero matrix. Those are ablation inputs and have no SMILES-derived numerical signal.

## Runtime Verification

- Fresh radius-2/2048 regeneration from each family's current `global_meta.json["pertid_to_smiles"]` matched every stored artifact row exactly:
  - PTV3: `6131 / 6131`;
  - PTV1: `128 / 128`;
  - Exp31: `6136 / 6136`.
- PTV3 artifact SHA-256: `36a38db707d719483a238d5ea149b3c88d56e967d59ee7ee08837557d50a2d15`.
- PTV1 artifact SHA-256: `c93cbf09a411c60e9718c80f3d1b01ecd9192ad35abb2f2d39aa6c59dbd5fa76`.
- Exp31 artifact SHA-256: `d1431246ee3bf25e39bbb35810b4af218a51f578a5862c6fa48a916ff1d1a54c`.

## Why The First Exp32 Mapping Used Lower-case `smiles`

- The mapping request asked for raw drug name, `smile`, `Smiles_no_chiral`, and `Smiles_with_chiral`. The raw tables contain a literal lower-case `smiles` field (or `smiles1` / `smiles2`), so the first mapping exported that raw field alongside the two explicitly named variants.
- The mistake was presentation/schema naming: it was labelled simply `smiles`, without making clear that it was raw legacy metadata and not the standardized Exp32 model-SMILES field.
- That lower-case field was never used to rerun Exp32, generate the Exp32 Morgan artifact, or produce the prediction probabilities. The mapping was created only after inference to explain IDs to a colleague.
- The corrected colleague-facing file uses `raw_legacy_smiles` and separately publishes `exp32_model_smiles`; the original mapping is retained only as historical provenance.
