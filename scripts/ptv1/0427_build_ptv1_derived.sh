#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
GLOBAL_META="${GLOBAL_META:-data/training_ready/ptv1/global_meta.json}"
DERIVED_DIR="${DERIVED_DIR:-data/training_ready/ptv1/derived}"
FASTA="${FASTA:-data/training_ready/ptv3/derived/idmapping_2026_04_27.fasta}"
ESM_MODEL="${ESM_MODEL:-/mnt/shared-storage-user/beam/wuhao/hf_cache/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c}"
PPI_EDGE_PATH="${PPI_EDGE_PATH:-/root/beam_wuhao/H100/vcc_data/westlake/20250410_6508308PPI_protein_links_detailed_v12_both_prot1&2_.csv}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz}"
CELL_LLM_ENV_FILE="${CELL_LLM_ENV_FILE:-.env}"
CELL_TYPE_LLM_ENV_FILE="${CELL_TYPE_LLM_ENV_FILE:-${CELL_LLM_ENV_FILE}}"
CELL_LLM_PROXY_COMMAND="${CELL_LLM_PROXY_COMMAND:-proxy_on}"

embedding_args=(
  --global-meta "${GLOBAL_META}"
  --derived-dir "${DERIVED_DIR}"
  --fasta "${FASTA}"
  --model-name "${ESM_MODEL}"
)
if [[ "${SKIP_PROTEIN_EMBEDDING:-0}" == "1" ]]; then
  embedding_args+=(--skip-protein)
fi
if [[ "${SKIP_DRUG_EMBEDDING:-0}" == "1" ]]; then
  embedding_args+=(--skip-drug)
fi
"${PYTHON_BIN}" utils/ptv1/05_build_ptv1_embeddings.py "${embedding_args[@]}"

graph_args=(
  --global-meta "${GLOBAL_META}"
  --derived-dir "${DERIVED_DIR}"
  --ppi-edge-path "${PPI_EDGE_PATH}"
)
if [[ "${SKIP_PPI:-0}" == "1" ]]; then
  graph_args+=(--skip-ppi)
fi
if [[ "${SKIP_PDI:-0}" == "1" ]]; then
  graph_args+=(--skip-pdi)
fi
if [[ "${SKIP_DDI:-0}" == "1" ]]; then
  graph_args+=(--skip-ddi)
fi
"${PYTHON_BIN}" utils/ptv1/06_build_ptv1_graph_matrices.py "${graph_args[@]}"

ensure_llm_proxy() {
  if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.bashrc"
  fi
  if [[ -n "${CELL_LLM_PROXY_COMMAND}" ]]; then
    if command -v "${CELL_LLM_PROXY_COMMAND}" >/dev/null 2>&1; then
      "${CELL_LLM_PROXY_COMMAND}"
    else
      echo "[llm] proxy command not found, continuing without it: ${CELL_LLM_PROXY_COMMAND}" >&2
    fi
  fi
}

if [[ "${SKIP_CELL_LLM_EMBEDDING:-0}" != "1" || "${SKIP_CELL_TYPE_LLM_EMBEDDING:-0}" != "1" ]]; then
  ensure_llm_proxy
fi

if [[ "${SKIP_CELL_LLM_EMBEDDING:-0}" != "1" ]]; then
  cell_llm_args=(
    --output "${CELL_LLM_EMBEDDING_PATH}"
    --env-file "${CELL_LLM_ENV_FILE}"
  )
  if [[ "${FORCE_CELL_LLM_EMBEDDING:-0}" == "1" ]]; then
    cell_llm_args+=(--force)
  fi
  "${PYTHON_BIN}" utils/ptv1/07_build_ptv1_cell_llm_embeddings.py "${cell_llm_args[@]}"
fi

if [[ "${SKIP_CELL_TYPE_LLM_EMBEDDING:-0}" != "1" ]]; then
  cell_type_llm_args=(
    --output "${CELL_TYPE_LLM_EMBEDDING_PATH}"
    --env-file "${CELL_TYPE_LLM_ENV_FILE}"
  )
  if [[ "${FORCE_CELL_TYPE_LLM_EMBEDDING:-0}" == "1" ]]; then
    cell_type_llm_args+=(--force)
  fi
  "${PYTHON_BIN}" utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py "${cell_type_llm_args[@]}"
fi
