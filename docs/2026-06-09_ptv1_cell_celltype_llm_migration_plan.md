# PTV1 Cell + Cell-Type LLM Architecture Migration Plan

Date: 2026-06-09 HKT

## Summary

This plan migrates the new `graph + Cell LLM embedding + cell_type LLM embedding + dose covariate` architecture to the PTV1 exp11-13 workflow.

PTV1 currently has valid `Cell` labels, but its `cell_type` field is all `no`. Because the existing PTV3 `cell_type` semantics are tissue/lineage labels, not molecular subtypes, PTV1 should assign all real PTV1 breast cancer cell lines to `BREAST` and keep `no=0` as the reserved missing label.

The migration should first prove that the old PTV1 parameterization can train and infer with the new architecture. After that, tune exp11 and exp12 independently. exp13 should not be tuned directly in this stage; it should report exp11 checkpoint transfer to exp13 in two ways:

- `valid`: the exp11 valid-selected best checkpoint directly inferred on exp13.
- `oracle`: all epoch checkpoints from the same exp11 run inferred on exp13, selecting the best exp13 AUPRC as a diagnostic test-label upper bound.

## Data Processing

### PTV1 `cell_type` Assignment

- Update PTV1 standardization so both `ptv1_aivc` and `ptv1_extra_singledrug` assign:
  - `cell_type=BREAST` for rows with a real non-empty `Cell`.
  - `cell_type=no` only for missing or reserved rows.
- Rebuild:
  - `data/standardized/ptv1`
  - `data/training_ready/ptv1`
  - PTV1 split artifacts
  - PTV1 derived graph and embedding-dependent artifacts
- Expected `data/training_ready/ptv1/global_meta.json` result:
  - `value_to_index.Cell` remains 20 rows with `no=0`.
  - `value_to_index.cell_type == {"no": 0, "BREAST": 1}`.

### LLM Embeddings

- Keep Cell LLM aligned by `Cell_index`.
- Add PTV1 cell_type LLM aligned by `cell_type_index`.
- Generate artifacts under `data/training_ready/ptv1/derived/`:
  - `cell_llm_embedding_qwen3_4096_v2.npz`
  - `cell_type_llm_embedding_qwen3_4096_v2.npz`
- Run generation with:

```bash
source ~/.bashrc
proxy_on
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
```

- Use `.env` keys already supported by the embedding scripts:
  - `openai_api_key` or `OPENAI_API_KEY`
  - `openai_base_url` or `OPENAI_BASE_URL`

### Prompt Policy

Cell prompt:

- Describe a specific breast cancer cell line used in proteomics drug-response experiments.
- Include cell line name, observed aliases, breast cancer context, known lineage or molecular traits where appropriate, and why those traits may affect perturbation response.
- Use 2-4 sentences, no bullets.

Cell-type prompt:

- Describe the tissue/lineage label `BREAST`, matching PTV3 `cell_type` semantics.
- Mention breast cancer lineage, common molecular context, proteomic drug-response relevance, and why lineage may affect perturbation response.
- Do not encode luminal/basal/HER2 subtype as `cell_type`; that would change field semantics.

## Code And Experiment Changes

### Training Interface

No model-body change should be required. Existing `train.py`, `infer.py`, `dataset/training_ready_fast_dataset.py`, and `model/fast_delta_model.py` already support:

- `CELL_LLM_*`
- `CELL_TYPE_LLM_*`
- graph features
- dose covariates

PTV1 wrappers should be updated or specialized so all new architecture runs use:

- `GRAPH_FEATURE_MODE=real`
- `GRAPH_STRUCTURAL_RP=1`
- `GRAPH_DRUG_CONCAT=1`
- `GRAPH_LOGIT_SCALE=2.0`
- `USE_DOSE_COVARIATE=1`
- `DOSE_COVARIATE_FIELDS="pert_dose1 pert_dose2"`
- `CELL_LLM_MODE=frozen`
- `CELL_LLM_FUSION_MODE=covariate`
- `CELL_LLM_CONDITION_SCALE=0.0`
- `CELL_LLM_LOGIT_SCALE=0.0`
- `CELL_LLM_DROPOUT=0.0`
- `CELL_TYPE_LLM_MODE=frozen`
- `CELL_TYPE_LLM_FUSION_MODE=covariate`
- `CELL_TYPE_LLM_CONDITION_SCALE=0.0`
- `CELL_TYPE_LLM_LOGIT_SCALE=0.0`
- `CELL_TYPE_LLM_DROPOUT=0.0`

Use new prefixes to avoid overwriting Cell-only results:

- old-parameter proof run: `20260609_ptv1_cell_celltype_llm_v1`
- tuning run: `20260609_ptv1_cell_celltype_llm_tune_v1`

### Old-Parameter Proof Run

Run the current old PTV1 baseline parameterization with the new architecture enabled:

- exp11: PTV1 fixed experiment-type random split.
- exp12: PTV1 unseen-drug 5-fold.
- exp13 `valid`: infer `ptv1_extra_singledrug` from the exp11 valid-selected best checkpoint.
- exp13 `oracle`: infer `ptv1_extra_singledrug` from every exp11 epoch checkpoint and report the best exp13 AUPRC.

This step is a runability and baseline check, not the final tuning result.

### exp11 Tuning

- Tune exp11 independently.
- Reuse the existing PTV1 fine-tune grid as the starting point, but remove any graph-off candidate.
- Every candidate must use graph + Cell LLM + cell_type LLM.
- Save all epoch checkpoints for oracle transfer evaluation.
- Formal exp11 selection is based on exp11 valid-selected performance, not exp13 oracle.
- For every exp11 candidate, report exp13 transfer:
  - valid exp13 AUPRC/AUROC from the exp11 valid-selected best checkpoint.
  - oracle exp13 AUPRC/AUROC from the best exp13-performing epoch checkpoint.

### exp12 Tuning

- Tune exp12 independently from exp11.
- Reuse the existing PTV1 exp12/fine-tune grid as the starting point, again removing graph-off candidates.
- Every candidate must use graph + Cell LLM + cell_type LLM.
- Formal exp12 selection is based on 5-fold mean AUPRC.
- Report fold-level AUPRC/AUROC/nAUPRC and mean/std metrics.

### exp13 Reporting

exp13 should be reported as transfer from exp11 checkpoints:

- `valid`: direct inference using exp11 manifest `best_model_path`.
- `oracle`: direct inference using the best exp13 AUPRC among all exp11 epoch checkpoints.

The report must clearly label oracle as a diagnostic upper bound because it uses exp13 labels for checkpoint selection.

Do not let graph-off candidates enter official exp13 reporting.

## Implementation Targets

Suggested additions:

- `utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py`
- `scripts/ptv1/run_ptv1_cell_celltype_llm_v1.sh`
- `scripts/ptv1/run_ptv1_cell_celltype_llm_fine_tune_search.sh`
- `scripts/ptv1/run_ptv1_exp13_oracle_from_exp11_ckpts.sh`
- `scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`

Suggested formal report:

- `docs/2026-06-09_ptv1_cell_celltype_llm_exp11_13_tuned_results.md`

Also update:

- `docs/2026-04-15_data_standardization_session_summary.md`
- `data/review_summary/<timestamp>_ptv1_cell_celltype_llm_migration_review.md`

## Test Plan

Static checks:

```bash
bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh
python -m py_compile train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py utils/ptv1/*.py scripts/ptv1/*.py
```

Data audit:

- PTV1 `cell_type` mapping is exactly `no=0, BREAST=1`.
- `ptv1_aivc` and `ptv1_extra_singledrug` real Cell rows have `cell_type_index=1`.
- Cell embedding has 20 rows, row 0 zero, rows 1-19 finite and nonzero.
- cell_type embedding has 2 rows, row 0 zero, row 1 finite and nonzero.
- `Cell_index` and `cell_type_index` ranges never exceed embedding rows.

Experiment audit:

- All new formal/tuning manifests have `dataset_group=ptv1`.
- All exp11/12/13 formal rows have `graph_feature_mode=real`.
- All formal rows have `cell_llm_mode=frozen` and `cell_type_llm_mode=frozen`.
- exp13 prediction rows equal 218.
- No report row uses `mse025_graph_off` or any `GRAPH_FEATURE_MODE=off` candidate.

exp13 audit:

- `valid` checkpoint path equals exp11 manifest `best_model_path`.
- `oracle` scans all available exp11 epoch checkpoints.
- `oracle` reports epoch, checkpoint path, AUPRC, AUROC, nAUPRC, rows.
- `oracle` is marked diagnostic and not used as formal model selection.

## Assumptions

- PTV1 `cell_type` should follow PTV3 tissue/lineage semantics, so all real PTV1 breast cancer cell lines map to `BREAST`.
- exp11 and exp12 are tuned independently.
- exp13 is evaluated only as direct transfer from exp11 checkpoints in this stage.
- Existing PTV1 Cell-only artifacts are read-only historical baselines and should not be overwritten.
- New architecture outputs must use `cell_celltype_llm` prefixes.
