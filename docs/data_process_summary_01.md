# Data Process Summary 01

## Scope

This document summarizes the current raw-data standardization workflow implemented for the requirements in [docs/Data_Process.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process.md:1).

The pipeline standardizes raw data under `data/rawdata/` into reproducible task-level outputs under `data/standardized/`.

## What Was Implemented

### 1. Reproducible Python pipeline

Two scripts were added as the main entry points:

- [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)
- [utils/01_validate_standardized_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/01_validate_standardized_outputs.py:1)

`00_standardize_rawdata.py` reads the raw files, standardizes each task, writes output artifacts, and builds global metadata.

`01_validate_standardized_outputs.py` validates that all generated task outputs are internally consistent.

### 2. Task-level standardized outputs

For each task, the pipeline writes:

- `info.csv`
- `expression_matrix.npy`
- `protein_order.json`
- `sample_ids.json`
- `sample_id_to_row_index.json`
- `expression_dict.pkl` when the task is small enough to materialize safely

The task outputs are stored under:

- `data/standardized/ptv3/tasks/...`
- `data/standardized/ptv1/tasks/...`

### 3. Global metadata outputs

The pipeline writes:

- `data/standardized/ptv3/global_meta.json`
- `data/standardized/ptv1/global_meta.json`
- `data/standardized/file_audit.json`

These files record global indices, mapping tables, and per-task audit details.

## Current Task Coverage

### PTV3 tasks

- `ptv3_main_singledrug`
- `ptv3_main_doubledrug`
- `ptv3_extra_baseline`
- `ptv3_extra_singledrug_mat1_480_faims`
- `ptv3_extra_singledrug_mat1_qe`
- `ptv3_extra_singledrug_mat2_480_faims`
- `ptv3_extra_singledrug_mat2_qe`
- `ptv3_extra_singledrug_mat3_qe`
- `ptv3_extra_singledrug_mat4_qe`
- `ptv3_extra_doubledrug_guomics`
- `ptv3_extra_doubledrug_nc`
- `ptv3_extra_doubledrug_nature`

### PTV1 tasks

- `ptv1_aivc`
- `ptv1_extra_singledrug`

`ptv1` is processed into its own isolated output space and does not share index space with `ptv3`.

## Important Standardization Rules

### 1. Unified sample info schema

All task `info.csv` files are aligned to the same standard columns, including:

- `sample_id`
- `machineID_new`
- `Cell_plate`
- `Cell`
- `cell_type`
- `pert_id1`
- `pert_id2`
- `batch`
- `pert_time`
- `pert_dose1`
- `pert_dose2`
- `PRISM1st_label_total`
- `PRISM2nd_label_total`
- `instrument`
- `cell_pertid_time`
- `drugname`
- `smiles`
- `target_protein_list`
- `control`
- `synergy`

### 2. File-specific protein name to UniProt rules

Protein columns are not parsed with one loose fallback. Each expression file uses an explicit rule:

- `data/rawdata/singledrug/20250113_ptv3_unique_mat_28602samp_10982prot_finall_v2.csv`
  Rule: `UniProtID_GeneSymbol`, take the token before the first `_`
- `data/rawdata/doubledrug/20260417ptv3_J_3509samp_9112prot_finall_edit.csv`
  Rule: column name is already a UniProt accession
- `data/rawdata/extra_baseline/260102ptv3_unseenCell_baselineProt.csv`
  Rule: column name is already a UniProt accession
- `data/rawdata/ptv1/aivc.csv`
  Rule: dot-delimited descriptor, take the first token that matches a UniProt accession

For files with explicit rules, unresolved protein columns raise an error instead of being silently accepted.

### 3. Control definition

The current control rule follows the confirmed workflow rule:

- a row is a control if raw `control == "control"`
- or raw `control == sample_id`

After normalization, both cases become self-control rows.

The control pool used for extra-data matching is built from:

- main single-drug controls
- main double-drug controls
- extra baseline controls

### 4. Extra-data baseline matching

Extra datasets without their own baseline proteome use the control pool above.

Matching priority is:

1. `Machine Match`
2. `Type Match`
3. `Batch Match`
4. `Plate Match`

The selected match and match quality are written into audit columns such as:

- `control_match_level`
- `control_match_source_task`
- `control_match_pool_kind`
- `control_match_score`

### 5. Unified external perturbation naming

When an extra dataset cannot be mapped to an existing confirmed `pert_id`, the pipeline uses deterministic fallback namespaces instead of mixed ad hoc names:

- `extid::...`
- `extsmiles::...`
- `extname::...`
- `extunk::...`

This makes unresolved external compounds reproducible across reruns.

### 6. Extra target protein mapping

Extra-data `target_protein_list` is built using:

- raw target gene text from the extra files
- PRISM drug-name lookup
- the mapping file `data/rawdata/extra_singledrug/20260318_prism1st_target_gene_uniprotID_map.csv`

This is used for extra single-drug and extra double-drug tasks.

### 7. Extra baseline cleanup

The unmatched extra baseline rows below are deleted directly, per user confirmation:

- `S1_B`
- `S1_CAC`
- `S1_O`

### 8. PTV1 main-task rules

`ptv1_aivc` now applies the ptv1-specific workflow rather than the old one-task shortcut:

- `aivc.csv` is treated as the expression source and `aivc_info.csv` as the info projection
- `NY_label -> PRISM1st_label_total`
- `Library_dose -> pert_dose1`
- `Anchor_dose -> pert_dose2`
- `pert_id -> pert_id1`
- `Anchor_id -> pert_id2`
- smiles / targets are resolved from `data/rawdata/ptv1/ptv1.csv`
- when both `pert_id1` and `pert_id2` are present, task-level `smiles` and `target_protein_list` are merged across both sides
- control rows are defined only by `pert_time == 0`; perturbed rows point to a deterministic representative control within the same `(BioRep, protein_plate)` group

### 9. PTV1 extra single-drug rules

`ptv1_extra_singledrug` now has its own standardized task:

- sample rows come from `data/rawdata/ptv1_extra_singledrug/test12091214_sample_predictions_E115id.csv`
- `E115_id` is used directly as `pert_id1`
- `cell` is used as both `Cell` and `Cell_plate`
- control rows are matched from `ptv1_aivc` by `cell -> protein_plate`
- smiles / targets are read from the stage-1 `ptv3` global meta payload
- because the raw prediction file is model-expanded, the current code keeps one row per unique `(cell, E115_id)` pair
- `ground_truth` is taken from the `ppODE_swa1` row
- `ground_truth` is written into `PRISM2nd_label_total`

## Current Raw File Assumptions

The current implementation is aligned to the latest raw files, including:

- main double-drug info:
  `data/rawdata/doubledrug/20260417ptv3_J_3509sampinfo.csv`
- main double-drug expression:
  `data/rawdata/doubledrug/20260417ptv3_J_3509samp_9112prot_finall_edit.csv`
- Guomics extra double-drug file:
  `data/rawdata/extra_doubeldrug/260417ptv3_Guomics_drug_combo_unique_with_smlies.csv`
- ptv1 extra prediction file:
  `data/rawdata/ptv1_extra_singledrug/test12091214_sample_predictions_E115id.csv`

If any of these raw files change again, rerun the pipeline and validator.

## Current Output Status

Current reference sizes:

- ptv3 tasks below remain from the previous validated full-run summary
- ptv1 tasks below were rechecked in the focused ptv1 verification run

- `ptv3_main_singledrug`: `28602 x 10982`
- `ptv3_main_doubledrug`: `3509 x 9112`
- `ptv3_extra_baseline`: `75 x 10169`
- `ptv3_extra_singledrug_mat1_480_faims`: `70143 x 0`
- `ptv3_extra_singledrug_mat1_qe`: `70143 x 0`
- `ptv3_extra_singledrug_mat2_480_faims`: `37087 x 0`
- `ptv3_extra_singledrug_mat2_qe`: `37087 x 0`
- `ptv3_extra_singledrug_mat3_qe`: `93313 x 0`
- `ptv3_extra_singledrug_mat4_qe`: `42169 x 0`
- `ptv3_extra_doubledrug_guomics`: `9009 x 0`
- `ptv3_extra_doubledrug_nc`: `22975 x 0`
- `ptv3_extra_doubledrug_nature`: `23400 x 0`
- `ptv1_aivc`: `15002 x 5576`
- `ptv1_extra_singledrug`: `182 x 0`

## Validation

The standard validation command is:

```bash
source ~/.bashrc
conda activate flow_v2
python utils/01_validate_standardized_outputs.py
```

For this ptv1 workflow update, a targeted temp-root build was revalidated and returned `Validation passed.` for:

- `ptv1_aivc`
- `ptv1_extra_singledrug`

## Recorded Questions

The code still records this raw-data ambiguity explicitly instead of hiding it:

- `ptv1_aivc`: many `(BioRep, protein_plate)` groups contain multiple valid `pert_time == 0` control rows, so the current workflow records candidate counts and chooses one deterministic representative control sample id per group

The following ptv1-extra behavior is now fixed by rule rather than left open:

- `test12091214_sample_predictions_E115id.csv` is deduplicated by `(cell, E115_id)`
- `ground_truth` is read from the `ppODE_swa1` row for each deduplicated sample
- disagreement from other models is preserved only as audit context

## Notes For Future Updates

- If a raw file name or schema changes, update the explicit parsing logic in `00_standardize_rawdata.py`.
- If a new expression file is added, define its protein-column-to-UniProt rule explicitly before accepting it.
- If extra target mappings improve, extend `20260318_prism1st_target_gene_uniprotID_map.csv` first, then rerun the pipeline.
