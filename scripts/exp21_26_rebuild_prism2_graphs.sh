#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready_prism2_main}"
DERIVED_ROOT="${TRAINING_READY_ROOT}/ptv3/derived"
GLOBAL_META="${TRAINING_READY_ROOT}/ptv3/global_meta.json"

FASTA_PATH="${FASTA_PATH:-data/training_ready/ptv3/derived/idmapping_2026_04_27.fasta}"
ESM_MODEL_NAME="${ESM_MODEL_NAME:-/root/beam_wuhao/hf_cache/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c}"
PPI_EDGE_PATH="${PPI_EDGE_PATH:-/root/beam_wuhao/H100/vcc_data/westlake/20250410_6508308PPI_protein_links_detailed_v12_both_prot1&2_.csv}"
STITCH_DB_DIR="${STITCH_DB_DIR:-/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/stitch_db}"

mkdir -p "${DERIVED_ROOT}"

"${PYTHON_BIN}" utils/04_build_embeddings_from_global_meta.py drug \
  --global-meta "${GLOBAL_META}" \
  --output-pkl "${DERIVED_ROOT}/drug_embedding_morgan_2048.pkl" \
  --radius 2 \
  --n-bits 2048

"${PYTHON_BIN}" utils/04_build_embeddings_from_global_meta.py protein \
  --global-meta "${GLOBAL_META}" \
  --output-pkl "${DERIVED_ROOT}/protein_embedding_esm.pkl" \
  --fasta "${FASTA_PATH}" \
  --model-name "${ESM_MODEL_NAME}" \
  --batch-size "${PROTEIN_EMBED_BATCH_SIZE:-4}" \
  --max-length "${PROTEIN_EMBED_MAX_LENGTH:-1024}"

"${PYTHON_BIN}" utils/05_build_graph_matrices_from_global_meta.py ddi \
  --global-meta "${GLOBAL_META}" \
  --output-npy "${DERIVED_ROOT}/ddi_matrix.npy" \
  --radius 2 \
  --n-bits 2048

"${PYTHON_BIN}" utils/05_build_graph_matrices_from_global_meta.py pdi \
  --global-meta "${GLOBAL_META}" \
  --output-npy "${DERIVED_ROOT}/pdi_matrix.npy" \
  --stitch-db-dir "${STITCH_DB_DIR}" \
  --protein-mapping-db "${STITCH_DB_DIR}/uniprot_to_string.db"

"${PYTHON_BIN}" utils/05_build_graph_matrices_from_global_meta.py ppi \
  --global-meta "${GLOBAL_META}" \
  --edge-path "${PPI_EDGE_PATH}" \
  --output-npy "${DERIVED_ROOT}/ppi_matrix.npy"

"${PYTHON_BIN}" utils/08_visualize_graph_matrix_distributions.py \
  --derived-dir "${DERIVED_ROOT}" \
  --output-dir "${DERIVED_ROOT}/graph_value_distributions"
