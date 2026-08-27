# 2026-06-12 12:04 HKT - PTV3 Cell LLM Default Path Fix Review

## Scope
- Reviewed reviewer comment about PTV3 `--cell-llm-mode frozen` failing when `--cell-llm-embedding-path` is omitted.
- Checked `train.py`, `infer.py`, and existing generated Cell LLM artifacts under `data/training_ready/*/derived`.

## Finding
- `default_derived_paths()` used `cell_llm_embedding_qwen3_4096_v2.npz` for all dataset groups.
- Existing PTV3 generated artifact is `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz`.
- Therefore standard PTV3 training/inference with frozen Cell LLM mode could resolve to a missing default file.

## Fix
- Updated `train.py::default_derived_paths()` to choose Cell LLM artifact by dataset group:
  - `ptv1`: `cell_llm_embedding_qwen3_4096_v2.npz`
  - non-`ptv1` groups including `ptv3`: `cell_llm_embedding_qwen3_4096.npz`
- Updated `--cell-llm-embedding-path` help text in both `train.py` and `infer.py` to avoid naming the wrong dataset-specific default.

## Validation
- `python -m py_compile train.py infer.py` passed in `flow_v2`.
- Direct default-path check passed:
  - `ptv1`: `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz`, exists
  - `ptv3`: `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz`, exists
