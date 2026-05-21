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
METHODS="${METHODS:-graph128_struct_drugcat_logit2_no_pos graph128_struct_drugcat_logit2_zero_no_pos}"
SPLIT_PREFIX="${SPLIT_PREFIX:-pert_stratified_5fold_fold}"
EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_fast_single_unseen_sweep}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
PRECISION="${PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
CKPT_DIR="${CKPT_DIR:-new_version/checkpoints}"
LOG_ROOT="${LOG_ROOT:-new_version/runtime_logs/${EXP_PREFIX}}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
mkdir -p "${LOG_ROOT}"

common_args=(
  --dataset-group ptv3
  --task-name ptv3_main_singledrug
  --batch-size "${BATCH_SIZE}"
  --max-epochs "${MAX_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --accelerator gpu
  --devices "${DEVICES}"
  --strategy ddp
  --precision "${PRECISION}"
  --num-workers "${NUM_WORKERS}"
  --logger-backend "${LOGGER_BACKEND}"
  --checkpoint-dir "${CKPT_DIR}"
  --save-top-k 1
  --no-save-last-ckpt
  --monitor val/task_auprc
  --monitor-mode max
  --log-every-n-steps 49
)

method_args() {
  local method="$1"
  case "${method}" in
    default)
      printf '%s\n' --mse-weight 0.25 --positive-weight auto --target-protein-max-length 32 --graph-feature-mode off
      ;;
    cls_only)
      printf '%s\n' --no-mse-loss --positive-weight auto --target-protein-max-length 32 --graph-feature-mode off
      ;;
    low_mse)
      printf '%s\n' --mse-weight 0.05 --positive-weight auto --target-protein-max-length 32 --graph-feature-mode off
      ;;
    no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode off
      ;;
    graph_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 64
      ;;
    graph_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 64
      ;;
    no_graph_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode off
      ;;
    graph128_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128
      ;;
    graph128_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128
      ;;
    graph64_scale025_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 64 --graph-init-scale 0.25
      ;;
    graph64_scale05_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 64 --graph-init-scale 0.5
      ;;
    graph128_scale025_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-init-scale 0.25
      ;;
    baseline3)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-init-scale 0.1
      ;;
    baseline3_zero)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-init-scale 0.1
      ;;
    graph128_struct_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp
      ;;
    graph128_struct_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp
      ;;
    graph128_struct_drugcat_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat
      ;;
    graph128_struct_drugcat_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat
      ;;
    graph128_struct_logit05_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-logit-scale 0.5
      ;;
    graph128_struct_logit05_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-logit-scale 0.5
      ;;
    graph128_struct_drugcat_logit05_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 0.5
      ;;
    graph128_struct_drugcat_logit05_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 0.5
      ;;
    graph128_struct_drugcat_logit1_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 1.0
      ;;
    graph128_struct_drugcat_logit1_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 1.0
      ;;
    graph128_struct_drugcat_logit2_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 2.0
      ;;
    graph128_struct_drugcat_logit2_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 2.0
      ;;
    graph128_jump_sparse_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-multihop --graph-drug-concat --graph-logit-scale 2.0 --graph-jump-fusion selective --graph-jump-gate sparsemax
      ;;
    graph128_jump_sparse_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-multihop --graph-drug-concat --graph-logit-scale 2.0 --graph-jump-fusion selective --graph-jump-gate sparsemax
      ;;
    graph128_jump_sparse_pcep_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-multihop --graph-drug-concat --graph-logit-scale 2.0 --graph-jump-fusion selective --graph-jump-gate sparsemax --protein-concat-mode pcep --protein-concat-dim 64 --protein-concat-topk 512 --protein-concat-init-scale 0.1
      ;;
    graph128_jump_sparse_pcep_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-multihop --graph-drug-concat --graph-logit-scale 2.0 --graph-jump-fusion selective --graph-jump-gate sparsemax --protein-concat-mode pcep --protein-concat-dim 64 --protein-concat-topk 512 --protein-concat-init-scale 0.1
      ;;
    graph128_struct_drugcat_logit2_pcep_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 2.0 --protein-concat-mode pcep --protein-concat-dim 64 --protein-concat-topk 512 --protein-concat-init-scale 0.1
      ;;
    graph128_struct_drugcat_logit2_pcep_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-drug-concat --graph-logit-scale 2.0 --protein-concat-mode pcep --protein-concat-dim 64 --protein-concat-topk 512 --protein-concat-init-scale 0.1
      ;;
    graph128_multihop_drugcat_logit2_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode real --graph-feature-dim 128 --graph-structural-rp --graph-multihop --graph-drug-concat --graph-logit-scale 2.0
      ;;
    graph128_multihop_drugcat_logit2_zero_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 32 --graph-feature-mode zero --graph-feature-dim 128 --graph-structural-rp --graph-multihop --graph-drug-concat --graph-logit-scale 2.0
      ;;
    graph_hidden512_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --hidden-dim 512 --expression-latent-dim 768 --target-protein-max-length 64 --graph-feature-mode real --graph-feature-dim 64
      ;;
    target64)
      printf '%s\n' --mse-weight 0.25 --positive-weight auto --target-protein-max-length 64 --graph-feature-mode off
      ;;
    hidden512)
      printf '%s\n' --mse-weight 0.25 --positive-weight auto --hidden-dim 512 --expression-latent-dim 768 --target-protein-max-length 64 --graph-feature-mode off
      ;;
    hidden512_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --hidden-dim 512 --expression-latent-dim 768 --target-protein-max-length 64 --graph-feature-mode off
      ;;
    cls_no_pos)
      printf '%s\n' --no-mse-loss --positive-weight none --target-protein-max-length 32 --graph-feature-mode off
      ;;
    mse010_no_pos)
      printf '%s\n' --mse-weight 0.10 --positive-weight none --target-protein-max-length 32 --graph-feature-mode off
      ;;
    low_mse_no_pos)
      printf '%s\n' --mse-weight 0.05 --positive-weight none --target-protein-max-length 32 --graph-feature-mode off
      ;;
    target64_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --target-protein-max-length 64 --graph-feature-mode off
      ;;
    smooth05_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --label-smoothing 0.05 --target-protein-max-length 32 --graph-feature-mode off
      ;;
    dropout25_no_pos)
      printf '%s\n' --mse-weight 0.25 --positive-weight none --dropout 0.25 --target-protein-max-length 32 --graph-feature-mode off
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      return 2
      ;;
  esac
}

echo "[sweep] EXP_PREFIX=${EXP_PREFIX}"
echo "[sweep] METHODS=${METHODS}"
echo "[sweep] FOLDS=${FOLDS}"
echo "[sweep] SPLIT_PREFIX=${SPLIT_PREFIX}"
echo "[sweep] GPU_IDS=${GPU_IDS}; DEVICES=${DEVICES}; MAX_EPOCHS=${MAX_EPOCHS}; BATCH_SIZE=${BATCH_SIZE}"

for method in ${METHODS}; do
  mapfile -t extra_args < <(method_args "${method}")
  for fold in ${FOLDS}; do
    experiment="${EXP_PREFIX}_${method}_fold${fold}"
    log_file="${LOG_ROOT}/${experiment}.log"
    echo "[run] ${experiment}"
    python new_version/train.py \
      "${common_args[@]}" \
      "${extra_args[@]}" \
      --split-strategy "${SPLIT_PREFIX}${fold}" \
      --experiment-name "${experiment}" \
      > "${log_file}" 2>&1
    echo "[done] ${experiment} log=${log_file}"
  done
done

python new_version/summarize_runs.py \
  --checkpoint-dir "${CKPT_DIR}" \
  --prefix "${EXP_PREFIX}" \
  --output "${LOG_ROOT}/summary.tsv"

echo "[summary] ${LOG_ROOT}/summary.tsv"
