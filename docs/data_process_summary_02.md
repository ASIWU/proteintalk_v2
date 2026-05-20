# Data Process Summary 02

## Scope

This document summarizes the stage-2 training-ready pipeline implemented from [docs/Data_Process_2.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process_2.md:1).

The new pipeline uses stage-1 outputs under `data/standardized/` as input and writes stage-2 outputs under `data/training_ready/`.

## New Scripts

- [utils/02_build_training_ready_data.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/02_build_training_ready_data.py:1)
  Builds processed task tables, aligned log1p expression matrices, updated `global_meta.json`, and task-level feature tables.
- [utils/03_validate_training_ready_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/03_validate_training_ready_outputs.py:1)
  Validates stage-2 task outputs and metadata contracts.
- [utils/04_build_embeddings_from_global_meta.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/04_build_embeddings_from_global_meta.py:1)
  Provides drug and protein embedding builders aligned to stage-2 `global_meta.json`.
- [utils/05_build_graph_matrices_from_global_meta.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/05_build_graph_matrices_from_global_meta.py:1)
  Provides PPI, DDI, and PDI matrix builders aligned to stage-2 index spaces.

## What Stage 2 Produces

For each final training task, the pipeline writes a task directory under:

- `data/training_ready/ptv3/tasks/<task_name>/`
- `data/training_ready/ptv1/tasks/<task_name>/`

Each task directory contains:

- `processed.csv`
- `processed_expression_matrix.npy`
- `processed_ordered_protein_index.json`
- `processed_ordered_protein_uniprot.json`
- `processed_sample_ids.json`
- `feature_table.csv`
- `feature_table.parquet` when parquet is available, otherwise `feature_table.pkl`
- `feature_expression_matrix.npy`
- `feature_ordered_protein_index.json`
- `feature_ordered_protein_uniprot.json`
- `feature_sample_ids.json`
- `feature_loading_manifest.json`

The pipeline also writes:

- `data/training_ready/ptv3/global_meta.json`
- `data/training_ready/ptv1/global_meta.json`
- `data/training_ready/file_audit.json`

## Implemented Rules

### 1. Processed-row filtering

- `ptv3_main_singledrug`
  Non-control rows with empty `PRISM1st_label_total` are removed.
- `ptv3_main_doubledrug`
  Non-control rows with empty `synergy` are removed.
- `ptv3_extra_singledrug_*`
  Non-control rows with empty `PRISM2nd_label_total` are removed.
- `ptv3_extra_doubledrug_*`
  Non-control rows with empty `PRISM1st_label_total` are removed.
- `ptv1_aivc`
  No new label-based row filter was applied because `Data_Process_2.md` did not define one for PTV1.
- `ptv1_extra_singledrug`
  No additional stage-2 non-control filter is applied. Stage-1 already keeps one row per unique `(cell, E115_id)` pair and stage-2 only appends matched control rows.

### 2. Control-row handling

- A row is treated as a control row only when `control == sample_id` or `control == "control"`.
- Existing control rows are always preserved.
- For `extra` tasks, matched control rows are appended from stage-1 source tasks before filtering.
- Because filtering only applies to non-control rows, appended control rows are never removed.

### 3. `target_protein_list` conversion

- Stage-1 `target_protein_list` values are treated as UniProt ID lists.
- Stage-2 `target_protein_list` values are saved as JSON lists of protein indices.
- The original UniProt list is preserved in `target_protein_uniprot_list`.
- Any UniProt target missing from the dataset-level `protein_index` is dropped from the index list and recorded in `target_protein_missing_from_index`.

### 4. Global metadata updates

Stage-2 `global_meta.json` now includes:

- `protein_index`
- `protein_index_to_id`
- `pert_index`
- `pert_index_to_id`
- `pertid_to_smiles`
- `pertid_to_target_protein_list`
- `pertid_to_target_uniprot_list`
- `pertid_to_missing_target_uniprot`
- `value_to_index`
- `categorical_normalization_rules`
- `special_values`

Special values:

- `protein_index`: append `"control"` and `"no"` if missing
- `pert_index`: append `"no"` if missing

### 5. Discrete feature normalization

For `machineID_new`, `Cell_plate`, `Cell`, and `cell_type`, the stage-2 builder normalizes values by:

- uppercasing
- replacing punctuation, spaces, hyphens, and separators with `_`
- collapsing repeated `_`

For `pert_time`, `pert_dose1`, and `pert_dose2`:

- `pert_time` numeric values are canonicalized to compact float text, for example `10.0 -> 10`
- `pert_dose1` / `pert_dose2` numeric values are canonicalized to compact float text and then mapped by `ceil(dose)`, stored as string indices, for example `0.2 -> "1"` and `1.1 -> "2"`
- missing `pert_dose` values map to `"no"`, and `value_to_index["pert_dose"]["no"]` is stored as `string(max_numeric_index + 1)`
- non-numeric fallback is still the uppercase canonical text rule, although the current standardized data contains only numeric or missing dose values

All mappings include `"no"`, and for `pert_dose1` / `pert_dose2` one shared mapping is used. In task CSV/parquet outputs, cast `pert_dose1_index` / `pert_dose2_index` to `int` inside the dataloader if the model expects integer category ids.

### 6. Expression handling

- All stage-2 expression matrices are built from stage-1 matrices after in-memory `log1p`.
- `NaN` positions remain `NaN`.
- Task-level ordered protein lists are the union of source-task proteins, sorted by dataset-level `protein_index`.
- Missing proteins in a source matrix are expanded as `NaN`.
- Rows with no native expression data become all-`NaN` vectors.
- `feature_loading_manifest.json` records the row-alignment contract used by dataloaders:
  `feature_table` row -> `expression_row_index` -> `feature_expression_matrix.npy`.

### 7. Feature-table composition

- `single drug` and `ptv1_aivc` feature tables use only their own processed rows.
- `double drug` feature tables merge:
  - their own processed rows
  - processed rows from `ptv3_main_singledrug`
- `extra data` feature tables use:
  - their own processed rows
  - appended matched control rows

## Important Implementation Decisions To Review

These points describe the concrete behavior implemented by the current codebase and recorded in `data/training_ready/file_audit.json`. Item 2 is a deliberate deviation from the original `Data_Process_2.md` requirement; the other items are implementation choices or underspecified areas:

1. Output root:
   Stage-2 writes to `data/training_ready/` instead of overwriting `data/standardized/`.
2. Feature-file format:
   The feature table is stored as CSV plus parquet/pickle, and the aligned expression matrix stays in `.npy` rather than embedding full expression vectors directly into CSV cells. `feature_loading_manifest.json` records how to read the aligned matrix directly in the dataloader. This does not satisfy the original `Data_Process_2.md` requirement that expanded expression be written directly into the feature CSV.
3. Extra-task control timing:
   Matched control rows are appended before filtering, so if an extra task loses all non-control rows it can still keep control rows.
4. Missing target proteins:
   Target UniProt IDs absent from `protein_index` are dropped from the index list instead of being forced to a synthetic protein index.
5. PDI matrix orientation:
   The new PDI builder writes `matrix[pert_index, protein_index]`.
6. PTV1 filtering:
   `ptv1_aivc` still keeps all rows, and `ptv1_extra_singledrug` also skips any extra stage-2 non-control filter after its stage-1 unique `(cell, E115_id)` selection.
7. PTV1 extra label source:
   `ptv1_extra_singledrug` uses the `ppODE_swa1` row as the label source for each unique `(cell, E115_id)` pair, while any disagreement from other model rows is recorded only as audit context.

## Auxiliary Builder Notes

- `utils/04_build_embeddings_from_global_meta.py`
  now accepts common FASTA header formats such as plain UniProt headers and `sp|P12345|...`.
- `utils/05_build_graph_matrices_from_global_meta.py`
  keeps matrix shapes exactly aligned to `global_meta.json`, applies the reference `ppi_string.py` edge filter logic for PPI, and preserves duplicated perturbation/protein aliases instead of overwriting them onto a single graph node.

## Builder README

Use the `flow_v2` environment before running any of the auxiliary builders:

```bash
source ~/.bashrc
conda activate flow_v2
```

The examples below use `ptv3`. For `ptv1`, replace the `global_meta.json` path and output directory.

### 1. Drug embedding

Inputs:

- `data/training_ready/ptv3/global_meta.json`

Command:

```bash
python utils/04_build_embeddings_from_global_meta.py drug \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-pkl data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl \
  --radius 2 \
  --n-bits 2048
```

Output:

- pickle payload with `embedding_matrix`, `item_to_index`, `index_to_item`, and `unresolved_items`

### 2. Protein embedding

Inputs:

- `data/training_ready/ptv3/global_meta.json`
- a UniProt FASTA file
- a transformers model name or local model directory

Command:

```bash
python utils/04_build_embeddings_from_global_meta.py protein \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-pkl data/training_ready/ptv3/derived/protein_embedding_esm.pkl \
  --fasta /path/to/uniprot_sprot.fasta \
  --model-name facebook/esm2_t33_650M_UR50D \
  --batch-size 4 \
  --max-length 1024
```

If the runtime cannot download models, pass a local `--model-name` path instead.

### 3. PPI matrix

Inputs:

- `data/training_ready/ptv3/global_meta.json`
- a STRING-like protein edge table with supported score columns
- optional UniProt-to-node mapping JSON

Command:

```bash
python utils/05_build_graph_matrices_from_global_meta.py ppi \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --edge-path /path/to/protein_links.tsv \
  --output-npy data/training_ready/ptv3/derived/ppi_matrix.npy \
  --node-mapping-json /path/to/uniprot_to_string.json \
  --topk 100
```

If you do not have a local mapping file, you can use `--allow-online-mapping`.

### 4. DDI matrix

Inputs:

- `data/training_ready/ptv3/global_meta.json`

Command:

```bash
python utils/05_build_graph_matrices_from_global_meta.py ddi \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --output-npy data/training_ready/ptv3/derived/ddi_matrix.npy \
  --radius 2 \
  --n-bits 2048
```

### 5. PDI matrix

Inputs:

- `data/training_ready/ptv3/global_meta.json`
- a chemical-protein links TSV
- either a prebuilt `pert_id -> flat_chemical_id` JSON or a chemical InChIKey TSV
- optional UniProt-to-protein-node mapping JSON

Command:

```bash
python utils/05_build_graph_matrices_from_global_meta.py pdi \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --links-tsv /path/to/chemical_protein_links.tsv \
  --output-npy data/training_ready/ptv3/derived/pdi_matrix.npy \
  --chemical-inchikey-tsv /path/to/chemical_inchikey.tsv \
  --protein-node-mapping-json /path/to/uniprot_to_string.json \
  --chunksize 500000
```

If you already resolved chemicals onto flat IDs, replace `--chemical-inchikey-tsv ...` with `--pert-to-flat-json /path/to/pert_to_flat.json`.

All matrix builders also write a sibling `.meta.json` file that records axis definitions and build parameters.

## Dataloader Example

The training loader should treat the task feature table as the row-level source of truth, then use `feature_loading_manifest.json` to locate aligned expression rows.

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainingReadyDataset(Dataset):
    def __init__(
        self,
        task_dir: str | Path,
        *,
        drug_embedding_pkl: str | Path | None = None,
        protein_embedding_pkl: str | Path | None = None,
        ppi_npy: str | Path | None = None,
        ddi_npy: str | Path | None = None,
        pdi_npy: str | Path | None = None,
    ) -> None:
        self.task_dir = Path(task_dir)
        with (self.task_dir / "feature_loading_manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        feature_native = Path(self.manifest["feature_table_native_path"])
        feature_csv = Path(self.manifest["feature_table_csv_path"])
        if feature_native.suffix == ".parquet" and feature_native.exists():
            self.feature_table = pd.read_parquet(feature_native)
        elif feature_native.suffix == ".pkl" and feature_native.exists():
            self.feature_table = pd.read_pickle(feature_native)
        else:
            self.feature_table = pd.read_csv(feature_csv, low_memory=False)

        self.expression_matrix = np.load(self.manifest["expression_matrix_path"])
        self.ordered_protein_index = json.loads(Path(self.manifest["ordered_protein_index_path"]).read_text())

        self.drug_embedding = self._load_embedding_matrix(drug_embedding_pkl)
        self.protein_embedding = self._load_embedding_matrix(protein_embedding_pkl)
        self.ppi = np.load(ppi_npy) if ppi_npy else None
        self.ddi = np.load(ddi_npy) if ddi_npy else None
        self.pdi = np.load(pdi_npy) if pdi_npy else None

    @staticmethod
    def _load_embedding_matrix(path: str | Path | None):
        if path is None:
            return None
        import pickle

        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return payload["embedding_matrix"]

    def __len__(self) -> int:
        return len(self.feature_table)

    def __getitem__(self, idx: int) -> dict:
        row = self.feature_table.iloc[idx]
        expression_row_index = int(row[self.manifest["expression_row_index_column"]])
        expression = torch.tensor(self.expression_matrix[expression_row_index], dtype=torch.float32)

        pert_index1 = int(row["pert_index1"])
        pert_index2 = int(row["pert_index2"])
        target_raw = row["target_protein_list"]
        target_protein_index = target_raw if isinstance(target_raw, list) else json.loads(target_raw)

        item = {
            "sample_id": row["sample_id"],
            "expression": expression,
            "machine_id_index": int(row["machineID_new_index"]),
            "cell_plate_index": int(row["Cell_plate_index"]),
            "cell_index": int(row["Cell_index"]),
            "cell_type_index": int(row["cell_type_index"]),
            "pert_time_index": int(row["pert_time_index"]),
            "pert_dose1_index": int(row["pert_dose1_index"]),
            "pert_dose2_index": int(row["pert_dose2_index"]),
            "pert_index1": pert_index1,
            "pert_index2": pert_index2,
            "target_protein_index": torch.tensor(target_protein_index, dtype=torch.long),
        }

        if self.drug_embedding is not None:
            item["drug_embedding1"] = torch.tensor(self.drug_embedding[pert_index1], dtype=torch.float32)
            item["drug_embedding2"] = torch.tensor(self.drug_embedding[pert_index2], dtype=torch.float32)
        if self.protein_embedding is not None and target_protein_index:
            item["target_protein_embedding"] = torch.tensor(
                self.protein_embedding[target_protein_index],
                dtype=torch.float32,
            )
        if self.ddi is not None:
            item["ddi_row1"] = torch.tensor(self.ddi[pert_index1], dtype=torch.float32)
            item["ddi_row2"] = torch.tensor(self.ddi[pert_index2], dtype=torch.float32)
        if self.pdi is not None:
            item["pdi_row1"] = torch.tensor(self.pdi[pert_index1], dtype=torch.float32)
            item["pdi_row2"] = torch.tensor(self.pdi[pert_index2], dtype=torch.float32)
        if self.ppi is not None and target_protein_index:
            item["target_ppi_subgraph"] = torch.tensor(
                self.ppi[np.ix_(target_protein_index, target_protein_index)],
                dtype=torch.float32,
            )

        return item
```

Practical notes:

- `expression_row_index` is the only supported way to align `feature_table` rows onto `feature_expression_matrix.npy`.
- `feature_ordered_protein_index.json` or `ordered_protein_index_path` defines the protein-axis order of each expression vector.
- `target_protein_list` is variable-length, so batching usually needs a custom `collate_fn` with padding or masks.
- `pert_dose1_index` and `pert_dose2_index` should be cast with `int(...)` inside `__getitem__`, because the metadata stores dose bin ids as strings.

## Current Validated Matrix Shapes

Validated with [utils/03_validate_training_ready_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/03_validate_training_ready_outputs.py:1).

Current shape references mix two validation scopes:

- `ptv3` entries remain from the previous validated full-run summary
- `ptv1` entries were rechecked in the focused ptv1 temp-root verification run for this workflow update

The entries below are the shapes of `processed_expression_matrix.npy` and `feature_expression_matrix.npy`. They are not the shapes of `feature_table.csv`. The validator also checks that each feature table has the same row count and row order as its feature expression matrix.

- `ptv3_main_singledrug`: processed expression `18568 x 10982`, feature expression `18568 x 10982`
- `ptv3_main_doubledrug`: processed expression `1798 x 9112`, feature expression `20366 x 11076`
- `ptv3_extra_singledrug_mat1_480_faims`: processed expression `15859 x 10169`, feature expression `15859 x 10169`
- `ptv3_extra_singledrug_mat1_qe`: processed expression `15859 x 10169`, feature expression `15859 x 10169`
- `ptv3_extra_singledrug_mat2_480_faims`: processed expression `12163 x 10169`, feature expression `12163 x 10169`
- `ptv3_extra_singledrug_mat2_qe`: processed expression `12163 x 10169`, feature expression `12163 x 10169`
- `ptv3_extra_singledrug_mat3_qe`: processed expression `17643 x 10982`, feature expression `17643 x 10982`
- `ptv3_extra_singledrug_mat4_qe`: processed expression `11106 x 10982`, feature expression `11106 x 10982`
- `ptv3_extra_doubledrug_guomics`: processed expression `9012 x 10982`, feature expression `9012 x 10982`
- `ptv3_extra_doubledrug_nc`: processed expression `22993 x 11267`, feature expression `22993 x 11267`
- `ptv3_extra_doubledrug_nature`: processed expression `23426 x 11267`, feature expression `23426 x 11267`
- `ptv1_aivc`: processed expression `15002 x 5576`, feature expression `15002 x 5576`
- `ptv1_extra_singledrug`: processed expression `186 x 5576`, feature expression `186 x 5576`

## Current Output Caveats

- `feature_table.csv` does not contain expanded expression vectors. Dataloaders must follow `feature_loading_manifest.json` and use `expression_row_index` to read from `feature_expression_matrix.npy`.
- Current stage-2 outputs contain many all-`NaN` expression rows because rows without native expression data are retained and expanded rather than dropped or imputed. Current `feature_expression_matrix.npy` all-`NaN` row counts are:
  - `ptv3_extra_doubledrug_guomics`: `9009 / 9012`
  - `ptv3_extra_doubledrug_nc`: `22975 / 22993`
  - `ptv3_extra_doubledrug_nature`: `23400 / 23426`
  - `ptv3_extra_singledrug_mat1_480_faims`: `15834 / 15859`
  - `ptv3_extra_singledrug_mat1_qe`: `15834 / 15859`
  - `ptv3_extra_singledrug_mat2_480_faims`: `12138 / 12163`
  - `ptv3_extra_singledrug_mat2_qe`: `12138 / 12163`
  - `ptv3_extra_singledrug_mat3_qe`: `17609 / 17643`
  - `ptv3_extra_singledrug_mat4_qe`: `11072 / 11106`
  - `ptv1_aivc`: `50 / 15002`
  - `ptv1_extra_singledrug`: `182 / 186`
- For the current extra double-drug outputs, all non-control rows have non-empty `PRISM1st_label_total` and `synergy`, but empty `PRISM2nd_label_total`. This is a property of the current processed data, not an extra stage-2 filter.
- For `ptv1_extra_singledrug`, the current stage-2 output contains `182` all-`NaN` perturbation rows and `4` appended control rows with native expression. This matches the current four-cell control matching strategy and the fixed stage-1 `(cell, E115_id) + ppODE_swa1` rule.

## Validation

Commands used:

```bash
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python utils/02_build_training_ready_data.py
/mnt/shared-storage-user/wuhao/miniconda3/bin/conda run -n flow_v2 python utils/03_validate_training_ready_outputs.py
```

Focused ptv1 verification also used:

```bash
source ~/.bashrc
conda activate flow_v2
python utils/03_validate_training_ready_outputs.py --output-root /tmp/<ptv1_verify>/training_ready
```

Result:

```text
Validation passed.
```

Scope:

- The validation command checks the outputs written by `utils/02_build_training_ready_data.py`: processed tables, feature tables, aligned expression matrices, ordered protein index files, row-order alignment, index ranges, and row-filter rules.
- The validation command does not execute `utils/04_build_embeddings_from_global_meta.py` or `utils/05_build_graph_matrices_from_global_meta.py`.
