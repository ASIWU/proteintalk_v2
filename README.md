# ProteinTalk Raw Data Standardization

This repository contains a reproducible pipeline to standardize ProteinTalk raw data under `data/rawdata/` into task-level outputs under `data/standardized/`.

The main workflow follows [docs/Data_Process.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process.md:1).

## Main Scripts

- [utils/00_standardize_rawdata.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/00_standardize_rawdata.py:1)
  Standardize raw data and write all output artifacts.
- [utils/01_validate_standardized_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/01_validate_standardized_outputs.py:1)
  Validate shapes, sample ids, and protein-order consistency.
- [utils/02_build_training_ready_data.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/02_build_training_ready_data.py:1)
  Build stage-2 training-ready task tables, expression matrices, and aligned `global_meta.json`.
- [utils/03_validate_training_ready_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/03_validate_training_ready_outputs.py:1)
  Validate stage-2 task outputs and metadata index consistency.
- [utils/04_build_embeddings_from_global_meta.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/04_build_embeddings_from_global_meta.py:1)
  Build ligand/drug and protein embeddings aligned to stage-2 `global_meta.json`.
- [utils/05_build_graph_matrices_from_global_meta.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/05_build_graph_matrices_from_global_meta.py:1)
  Build PPI, DDI, and PDI matrices aligned to stage-2 `global_meta.json`.
- [utils/06_export_uniprot_ids_from_global_meta.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/06_export_uniprot_ids_from_global_meta.py:1)
  Export a one-accession-per-line UniProt txt list from stage-2 `global_meta.json`.
- [docs/data_process_summary_01.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/data_process_summary_01.md:1)
  Summary of the implemented workflow and key rules.

## Environment

Use the project conda environment:

```bash
source ~/.bashrc
conda activate flow_v2
```

## How To Run

### 1. Standardize all raw data

From the repository root:

```bash
source ~/.bashrc
conda activate flow_v2
python utils/00_standardize_rawdata.py
```

### 2. Validate the generated outputs

```bash
source ~/.bashrc
conda activate flow_v2
python utils/01_validate_standardized_outputs.py
```

If everything is correct, the validator ends with:

```text
Validation passed.
```

### 3. Build training-ready data

The embedding and graph builders read the stage-2 `global_meta.json`, so run this after stage-1 standardization:

```bash
source ~/.bashrc
conda activate flow_v2
python utils/02_build_training_ready_data.py
python utils/03_validate_training_ready_outputs.py
```

This writes:

- `data/training_ready/ptv3/global_meta.json`
- `data/training_ready/ptv1/global_meta.json`
- task-level `processed.csv`, feature tables, and aligned expression matrices

### 4. Generate ligand/protein embeddings and graph matrices

The examples below use `ptv3`. For `ptv1`, replace `ptv3` in the input and output paths.

#### Ligand / drug embedding

Build Morgan fingerprint embeddings ordered by `global_meta.json["pert_index"]`:

```bash
python utils/04_build_embeddings_from_global_meta.py drug \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-pkl data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl \
  --radius 2 \
  --n-bits 2048
```

#### Protein embedding

Build ESM mean-pooled embeddings ordered by `global_meta.json["protein_index"]`:

Export the UniProt accession list first if you need to download a FASTA from UniProt:

```bash
python utils/06_export_uniprot_ids_from_global_meta.py \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-txt data/training_ready/ptv3/derived/uniprot_ids.txt \
  --audit-json data/training_ready/ptv3/derived/uniprot_ids.audit.json
```

```bash
python utils/04_build_embeddings_from_global_meta.py protein \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-pkl data/training_ready/ptv3/derived/protein_embedding_esm.pkl \
  --fasta /path/to/uniprot_sprot.fasta \
  --model-name facebook/esm2_t33_650M_UR50D \
  --batch-size 4 \
  --max-length 1024
```

`--max-length` is the tokenizer input sequence limit, not the embedding feature
dimension. The embedding feature dimension comes from the selected model
`hidden_size`; for `facebook/esm2_t33_650M_UR50D`, the output matrix has 1280
feature columns.

If the runtime cannot download the model, pass a local model directory to `--model-name`.

#### PPI matrix

Build a protein-protein matrix ordered by `protein_index`. The edge table should have `prot1/prot2`, `protein1/protein2`, or `source/target` columns plus STRING-like score columns such as `combined_score`, `experimental`, `database`, `coexpression`, or `textmining`. The PPI builder does not apply a confidence-score threshold. By default it also does not apply top-k pruning; pass `--topk N` only if you intentionally want to keep the strongest N neighbors per protein.

```bash
python utils/05_build_graph_matrices_from_global_meta.py ppi \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --edge-path /path/to/protein_links.tsv \
  --output-npy data/training_ready/ptv3/derived/ppi_matrix.npy \
  --node-mapping-json /path/to/uniprot_to_string.json
```

Use `--allow-online-mapping` only when network access is available and a local mapping JSON is not available.

#### DDI matrix

Build a drug-drug Tanimoto similarity matrix ordered by `pert_index`:

```bash
python utils/05_build_graph_matrices_from_global_meta.py ddi \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-npy data/training_ready/ptv3/derived/ddi_matrix.npy \
  --radius 2 \
  --n-bits 2048
```

#### PDI matrix

Build a perturbation-drug by protein matrix with rows ordered by `pert_index` and columns ordered by `protein_index`. The PDI builder can resolve the STITCH links parquet, chemical InChIKey TSV, and UniProt-to-STRING SQLite database from one `stitch_db` directory.

```bash
python utils/05_build_graph_matrices_from_global_meta.py pdi \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-npy data/training_ready/ptv3/derived/pdi_matrix.npy \
  --stitch-db-dir /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/stitch_db
```

If you already have prebuilt mappings, keep using `--pert-to-flat-json` and `--protein-node-mapping-json`. If the STITCH links are outside `stitch_db`, pass `--links-path`; both parquet and delimited tables are supported.

```bash
--pert-to-flat-json /path/to/pert_to_flat.json \
--protein-node-mapping-json /path/to/uniprot_to_string.json \
--links-path /path/to/protein_chemical.links.detailed.v5.0.parquet
```

Embedding outputs are `.pkl` payloads with `embedding_matrix`, `item_to_index`, `index_to_item`, and `unresolved_items`. Matrix outputs are `.npy` files and each graph builder also writes a sibling `.meta.json`.

### 5. Visualize graph matrix value distributions

After generating PPI, PDI, and DDI, use the distribution visualizer to inspect sparsity and nonzero score ranges before deciding any downstream normalization.

```bash
python utils/08_visualize_graph_matrix_distributions.py \
  --derived-dir data/training_ready/ptv3/derived \
  --output-dir data/training_ready/ptv3/derived/graph_value_distributions
```

The script loads one matrix at a time, samples large value arrays, and writes:

- `ppi_distribution.png`
- `pdi_distribution.png`
- `ddi_distribution.png`
- `nonzero_distribution_overlay.png`
- `graph_matrix_distribution_summary.csv`
- `graph_matrix_distribution_summary.json`
- one per-matrix `*_distribution_summary.json`

Useful options:

```bash
# Plot only selected matrices
python utils/08_visualize_graph_matrix_distributions.py --matrices ppi pdi

# Increase sample size for smoother histograms
python utils/08_visualize_graph_matrix_distributions.py \
  --sample-size 5000000 \
  --nonzero-sample-size 5000000

# Override one matrix path
python utils/08_visualize_graph_matrix_distributions.py \
  --ppi-npy data/training_ready/ptv3/derived/ppi_matrix.npy
```

The all-value histograms include zeros, so sparse matrices will be dominated by zero. Use the nonzero histograms and `graph_matrix_distribution_summary.csv` when choosing transformations for edge weights.

## Optional Arguments

Both scripts support `--help`.

Examples:

```bash
python utils/00_standardize_rawdata.py --help
python utils/01_validate_standardized_outputs.py --help
```

The standardizer also supports overriding the output root if needed.

Example:

```bash
python utils/00_standardize_rawdata.py --output-root data/standardized_test
python utils/01_validate_standardized_outputs.py --output-root data/standardized_test
```

## What The Pipeline Produces

For each task, the pipeline writes a task directory like:

- `data/standardized/ptv3/tasks/<task_name>/`
- `data/standardized/ptv1/tasks/<task_name>/`

Each task directory contains:

- `info.csv`
- `expression_matrix.npy`
- `protein_order.json`
- `sample_ids.json`
- `sample_id_to_row_index.json`
- `expression_dict.pkl` when materialization is enabled for that task

The pipeline also writes:

- `data/standardized/file_audit.json`
- `data/standardized/ptv3/global_meta.json`
- `data/standardized/ptv1/global_meta.json`

## Current Tasks

### PTV3

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

### PTV1

- `ptv1_aivc`

## Important Behavior

- `ptv1` is isolated from `ptv3` and uses its own output root and meta index.
- Protein parsing uses explicit per-file rules instead of a generic fallback.
- Extra-data control matching uses the confirmed control rule:
  raw `control == "control"` or raw `control == sample_id`.
- Extra-data `target_protein_list` uses the mapping file:
  `data/rawdata/extra_singledrug/20260318_prism1st_target_gene_uniprotID_map.csv`

## When To Rerun

Rerun the standardizer and validator whenever:

- a raw csv is modified
- a raw file is renamed
- a new raw dataset is added
- the mapping file for targets is updated
- the control-matching rule changes

## Quick Check After A Rerun

1. Run `python utils/00_standardize_rawdata.py`
2. Run `python utils/01_validate_standardized_outputs.py`
3. Check `data/standardized/file_audit.json`
4. Spot-check the affected task `info.csv`
5. If expression files changed, spot-check `protein_order.json`

## Related Documents

- [docs/Data_Process.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process.md:1)
- [docs/data_process_summary_01.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/data_process_summary_01.md:1)
- [docs/2026-04-15_data_standardization_session_summary.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/2026-04-15_data_standardization_session_summary.md:1)
