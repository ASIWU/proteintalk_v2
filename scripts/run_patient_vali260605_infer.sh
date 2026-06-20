#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_NAME="${RUN_NAME:-patientVali260605v3}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-20260612_${RUN_NAME}}"
TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready_${RUN_NAME}}"
DERIVED_ROOT="${TRAINING_READY_ROOT}/ptv3/derived"
TASK_PREFIX="ptv3_${RUN_NAME}"

EXP07_CKPT="${EXP07_CKPT:-checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp07_extra_single_all_train_infer_all_single_for_extra/epoch=5-step=426.ckpt}"
EXP08_CKPT="${EXP08_CKPT:-checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra/epoch=2-step=234.ckpt}"

COMMON_ARGS=(
  --dataset-group ptv3
  --training-ready-root "${TRAINING_READY_ROOT}"
  --model-type fast_delta
  --split-strategy test_only
  --split-name test
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --protein-embedding-path "${DERIVED_ROOT}/protein_embedding_esm.pkl"
  --drug-embedding-path "${DERIVED_ROOT}/drug_embedding_morgan_2048.pkl"
  --ppi-matrix-path "${DERIVED_ROOT}/ppi_matrix.npy"
  --pdi-matrix-path "${DERIVED_ROOT}/pdi_matrix.npy"
  --ddi-matrix-path "${DERIVED_ROOT}/ddi_matrix.npy"
  --graph-structural-rp
  --graph-drug-concat
  --cell-llm-mode frozen
  --cell-llm-embedding-path "${DERIVED_ROOT}/cell_llm_embedding_${RUN_NAME}_qwen3_4096.npz"
  --cell-llm-index-column cell_llm_index
  --cell-type-llm-mode frozen
  --cell-type-llm-embedding-path "${DERIVED_ROOT}/cell_type_llm_embedding_qwen3_4096_v2.npz"
  --cell-type-llm-index-column cell_type_llm_index
  --batch-cov-list machineID_new Cell_plate Cell cell_type batch pert_time pert_dose1 pert_dose2
  --allow-checkpoint-config-mismatch
)

run_single() {
  local task_name="$1"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u infer.py \
    "${COMMON_ARGS[@]}" \
    --task-name "${task_name}" \
    --task-head response \
    --checkpoint-path "${EXP07_CKPT}" \
    --output-dir "outputs/${OUTPUT_PREFIX}_exp07_single/${task_name}"
}

run_double() {
  local task_name="$1"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u infer.py \
    "${COMMON_ARGS[@]}" \
    --task-name "${task_name}" \
    --task-head synergy \
    --checkpoint-path "${EXP08_CKPT}" \
    --graph-pair-add-scale 0.5 \
    --pair-fusion-mode dual \
    --pair-type-features \
    --use-ddi \
    --output-dir "outputs/${OUTPUT_PREFIX}_exp08_double/${task_name}"
}

run_single "${TASK_PREFIX}_p2_lung2020_single"
run_single "${TASK_PREFIX}_p3_lung2024_single"
run_single "${TASK_PREFIX}_p5_breast_single"

run_double "${TASK_PREFIX}_p1_ovarian_double"
run_double "${TASK_PREFIX}_p2_lung2020_double"
run_double "${TASK_PREFIX}_p3_lung2024_double"
run_double "${TASK_PREFIX}_p4_colon_double"
run_double "${TASK_PREFIX}_p5_breast_double"
