#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

GPU_IDS="${GPU_IDS:-0,1}"
DEVICES="${DEVICES:-2}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SPLIT_PREFIX="${SPLIT_PREFIX:-pert_stratified_5fold_fold}"
EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_fast_single_unseen}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-0.25}"
POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-none}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
GRAPH_FEATURE_MODE="${GRAPH_FEATURE_MODE:-real}"
GRAPH_FEATURE_DIM="${GRAPH_FEATURE_DIM:-128}"
GRAPH_INIT_SCALE="${GRAPH_INIT_SCALE:-0.1}"
GRAPH_STRUCTURAL_RP="${GRAPH_STRUCTURAL_RP:-1}"
GRAPH_MULTIHOP="${GRAPH_MULTIHOP:-0}"
GRAPH_DRUG_CONCAT="${GRAPH_DRUG_CONCAT:-1}"
GRAPH_LOGIT_SCALE="${GRAPH_LOGIT_SCALE:-2.0}"
GRAPH_JUMP_FUSION="${GRAPH_JUMP_FUSION:-concat}"
GRAPH_JUMP_GATE="${GRAPH_JUMP_GATE:-softmax}"
GRAPH_JUMP_TEMPERATURE="${GRAPH_JUMP_TEMPERATURE:-1.0}"
PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-off}"
PROTEIN_CONCAT_DIM="${PROTEIN_CONCAT_DIM:-64}"
PROTEIN_CONCAT_TOPK="${PROTEIN_CONCAT_TOPK:-512}"
PROTEIN_CONCAT_INIT_SCALE="${PROTEIN_CONCAT_INIT_SCALE:-0.1}"
PRECISION="${PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGGER_BACKEND="${LOGGER_BACKEND:-tensorboard}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1.0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

graph_extra_args=()
if [[ "${GRAPH_STRUCTURAL_RP}" == "1" || "${GRAPH_STRUCTURAL_RP}" == "true" || "${GRAPH_STRUCTURAL_RP}" == "yes" ]]; then
  graph_extra_args+=(--graph-structural-rp)
fi
if [[ "${GRAPH_MULTIHOP}" == "1" || "${GRAPH_MULTIHOP}" == "true" || "${GRAPH_MULTIHOP}" == "yes" ]]; then
  graph_extra_args+=(--graph-multihop)
fi
if [[ "${GRAPH_DRUG_CONCAT}" == "1" || "${GRAPH_DRUG_CONCAT}" == "true" || "${GRAPH_DRUG_CONCAT}" == "yes" ]]; then
  graph_extra_args+=(--graph-drug-concat)
fi

for fold in ${FOLDS}; do
  python new_version/train.py \
    --dataset-group ptv3 \
    --task-name ptv3_main_singledrug \
    --split-strategy "${SPLIT_PREFIX}${fold}" \
    --experiment-name "${EXP_PREFIX}_fold${fold}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --mse-weight "${MSE_WEIGHT}" \
    --positive-weight "${POSITIVE_WEIGHT}" \
    --label-smoothing "${LABEL_SMOOTHING}" \
    --graph-feature-mode "${GRAPH_FEATURE_MODE}" \
    --graph-feature-dim "${GRAPH_FEATURE_DIM}" \
    --graph-init-scale "${GRAPH_INIT_SCALE}" \
    --graph-logit-scale "${GRAPH_LOGIT_SCALE}" \
    --graph-jump-fusion "${GRAPH_JUMP_FUSION}" \
    --graph-jump-gate "${GRAPH_JUMP_GATE}" \
    --graph-jump-temperature "${GRAPH_JUMP_TEMPERATURE}" \
    --protein-concat-mode "${PROTEIN_CONCAT_MODE}" \
    --protein-concat-dim "${PROTEIN_CONCAT_DIM}" \
    --protein-concat-topk "${PROTEIN_CONCAT_TOPK}" \
    --protein-concat-init-scale "${PROTEIN_CONCAT_INIT_SCALE}" \
    "${graph_extra_args[@]}" \
    --accelerator gpu \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}" \
    --logger-backend "${LOGGER_BACKEND}" \
    --limit-train-batches "${LIMIT_TRAIN_BATCHES}" \
    --limit-val-batches "${LIMIT_VAL_BATCHES}" \
    --limit-test-batches "${LIMIT_TEST_BATCHES}"
done
