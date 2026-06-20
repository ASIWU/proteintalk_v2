# PTV1 Cell LLM Rerun Review

Review time: 2026-06-08 11:44 HKT.

## Scope Reviewed

- PTV1 Cell LLM builder and validation wrapper.
- PTV1 derived build and experiment shell defaults.
- Generated PTV1 Cell LLM artifact.
- Formal rerun artifacts for prefix `20260608_ptv1_cell_llm_v1`.

## Code Findings

- `utils/ptv1/07_build_ptv1_cell_llm_embeddings.py` delegates to the shared Cell LLM builder and enforces PTV1-only defaults.
- `scripts/ptv1/ptv1_experiment_common.sh` defaults to `CELL_LLM_MODE=frozen` and `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz`.
- `scripts/ptv1/0427_build_ptv1_derived.sh` can generate the PTV1 Cell LLM artifact and can skip it with `SKIP_CELL_LLM_EMBEDDING=1`.
- `scripts/ptv1/run_ptv1_param_search.sh` now carries the same frozen Cell LLM defaults for future PTV1 searches.

## Artifact Findings

- PTV1 Cell LLM artifact:
  - `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz`;
  - shape `20 x 4096`;
  - metadata `dataset_group=ptv1`, `kind=cell_llm_embedding`, `input_field=Cell`, `index_column=Cell_index`;
  - row 0 is the reserved zero vector;
  - rows 1-19 are finite and nonzero;
  - no legacy `cell_type_llm` key was found in the sidecar.
- PTV1 feature-table alignment:
  - `ptv1_aivc` has 15002 rows and observed Cell indices 1-19;
  - `ptv1_extra_singledrug` has 222 rows and observed Cell indices 2, 7, 11, and 18.

## Experiment Findings

- Formal prefix: `20260608_ptv1_cell_llm_v1`.
- Exp_11 fixed experiment type:
  - AUROC `0.947288`;
  - AUPRC `0.893452`;
  - n-AUPRC `3.050717`;
  - count `799`.
- Exp_12 unseen-drug 5-fold:
  - mean AUROC `0.654052`;
  - mean AUPRC `0.511635`;
  - mean n-AUPRC `2.121972`;
  - total count `13137`.
- Exp_13 all-PTV1 plus extra single-drug inference:
  - reference epochs `3, 2, 6, 18, 0`;
  - selected epoch `6`;
  - all-train max epochs `7`;
  - extra overall AUROC `0.597414`;
  - extra overall AUPRC `0.578210`;
  - extra overall n-AUPRC `1.125445`;
  - predictions written for 218 extra rows.
- Generated reports:
  - `logs/20260608_ptv1_cell_llm_v1_ptv1_exp_results.md`;
  - `outputs/20260608_ptv1_cell_llm_v1_ptv1_exp_results.csv`.

## Validation Performed

- `python -m py_compile train.py infer.py utils/11_build_cell_llm_embeddings.py utils/ptv1/*.py scripts/ptv1/report_ptv1_exp_results.py`
- `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`
- `python utils/ptv1/01_validate_ptv1_standardized.py`
- `python utils/ptv1/03_validate_ptv1_training_ready.py`
- `python utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only`
- one-batch exp_11 smoke with frozen Cell LLM.
- one-batch exp_12 fold0 smoke with frozen Cell LLM.
- exp_13 smoke with frozen Cell LLM and one-batch extra inference.
- final manifest audit for 7 new manifests:
  - all `dataset_group=ptv1`;
  - all `cell_llm_mode=frozen`;
  - all `cell_llm_summary.embedding_rows=20`;
  - no legacy `cell_type_llm` key;
  - exp_13 references only the new `20260608_ptv1_cell_llm_v1` exp_12 folds.

## Residual Risks

- The Cell LLM descriptions and embeddings depend on the configured OpenAI-compatible API in `.env`; the generated artifact should be treated as the fixed artifact for this rerun.
- The rerun enabled frozen Cell LLM features but kept all other baseline hyperparameters, so it is a formal baseline rerun rather than a tuned Cell LLM search.
