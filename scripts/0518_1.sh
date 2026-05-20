#!/usr/bin/env bash
set -euo pipefail

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
cd /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2

export HF_DATASETS_CACHE=/home/wuhao/beam_wuhao/cache
export HF_HUB_CACHE=/mnt/shared-storage-user/beam/wuhao/hf_cache
export WANDB_CACHE_DIR=/mnt/shared-storage-user/beam/wuhao/wandb_cache
export WANDB_ARTIFACT_CACHE=10GB
export http_proxy=http://wuhao:Za8ZkuZapFh3v2KJf5ytMIbmcu0tyYHmuAqE9QzkUxX1Zwif4GQU9IiT9BNf@proxy.h.pjlab.org.cn:23128
export https_proxy=http://wuhao:Za8ZkuZapFh3v2KJf5ytMIbmcu0tyYHmuAqE9QzkUxX1Zwif4GQU9IiT9BNf@proxy.h.pjlab.org.cn:23128
export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"
# change the base url when restart the server
export WANDB_BASE_URL="http://100.96.30.112:8080"
export WANDB_API_KEY="local-f401d2b9276fb4a6dd1db4c1efee0512723ce6fb"

export NUM_WORKERS="${NUM_WORKERS:-16}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
export CKPT_DIR="${CKPT_DIR:-checkpoints}"
export LOG_DIR="${LOG_DIR:-logs}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

DOUBLE_EXP_PREFIX="${DOUBLE_EXP_PREFIX:-20260513_double_pert_pair_5fold}"
EXTRA_DOUBLE_EXP_PREFIX="${EXTRA_DOUBLE_EXP_PREFIX:-20260513_extra_double_all_train_infer}"
SINGLE_NO_PDI_EXP_PREFIX="${SINGLE_NO_PDI_EXP_PREFIX:-20260513_single_no_pdi_5fold}"
RESTART_STAMP="${RESTART_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"

ptv3_completed_train_test_run() {
  local exp_name="$1"
  local manifest="${CKPT_DIR}/${exp_name}/run_manifest.json"
  [[ -f "${manifest}" ]] || return 1
  "${PYTHON_BIN}" - "${manifest}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

completed = (
    manifest.get("run_status") == "fit_completed"
    and manifest.get("test_status") == "test_completed"
)
raise SystemExit(0 if completed else 1)
PY
}

ptv3_archive_existing_run() {
  local exp_name="$1"
  local src
  local archive_dir
  local dst

  for src in "${CKPT_DIR}/${exp_name}" "${LOG_DIR}/${exp_name}" "${OUTPUT_DIR}/${exp_name}"; do
    if [[ -e "${src}" ]]; then
      archive_dir="$(dirname "${src}")/_archived_failed_restarts"
      dst="${archive_dir}/$(basename "${src}")_${RESTART_STAMP}"
      if [[ -e "${dst}" ]]; then
        echo "[error] archive target already exists: ${dst}" >&2
        exit 1
      fi
      mkdir -p "${archive_dir}"
      echo "[resume] archive incomplete run artifact: ${src} -> ${dst}"
      mv "${src}" "${dst}"
    fi
  done
}

read -r -a CANDIDATE_DOUBLE_FOLDS <<< "${CANDIDATE_DOUBLE_FOLDS:-2 3 4}"
pending_double_folds=()
for fold in "${CANDIDATE_DOUBLE_FOLDS[@]}"; do
  exp_name="${DOUBLE_EXP_PREFIX}_double_pert_pair_fold${fold}"
  if ptv3_completed_train_test_run "${exp_name}"; then
    echo "[resume] skip completed double fold ${fold}: ${exp_name}"
  else
    ptv3_archive_existing_run "${exp_name}"
    pending_double_folds+=("${fold}")
  fi
done

if ((${#pending_double_folds[@]} > 0)); then
  pending_double_folds_text="${pending_double_folds[*]}"
  echo "[resume] run double pert-pair folds: ${pending_double_folds_text}"
  EXP_PREFIX="${DOUBLE_EXP_PREFIX}" \
  LOG_TO_WANDB="${LOG_TO_WANDB:-1}" \
  FOLDS="${pending_double_folds_text}" \
  MAX_EPOCHS="${MAX_EPOCHS:-100}" \
  LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
  BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-loss2}" \
  BATCH_SIZE="${BATCH_SIZE:-64}" \
  bash scripts/exp_06_double_pert_pair_5fold.sh
else
  echo "[resume] double pert-pair candidate folds already completed; skip exp_06"
fi

missing_reference_folds=()
for fold in 0 1 2 3 4; do
  exp_name="${DOUBLE_EXP_PREFIX}_double_pert_pair_fold${fold}"
  if ! ptv3_completed_train_test_run "${exp_name}"; then
    missing_reference_folds+=("${fold}")
  fi
done
if ((${#missing_reference_folds[@]} > 0)); then
  echo "[error] cannot start extra double run; incomplete reference folds: ${missing_reference_folds[*]}" >&2
  exit 1
fi

EXP_PREFIX="${EXTRA_DOUBLE_EXP_PREFIX}" \
LOG_TO_WANDB="${LOG_TO_WANDB:-1}" \
RUN_INFERENCE="${RUN_INFERENCE:-1}" \
MAX_EPOCHS="${MAX_EPOCHS:-100}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-loss2}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
REFERENCE_5FOLD_CKPT_PATH="${REFERENCE_5FOLD_CKPT_PATH:-${CKPT_DIR}/${DOUBLE_EXP_PREFIX}}" \
REFERENCE_EPOCH_AGG="${REFERENCE_EPOCH_AGG:-median}" \
REFERENCE_EPOCH_MIN_COUNT="${REFERENCE_EPOCH_MIN_COUNT:-5}" \
SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}" \
SCHEDULER_NAME="${SCHEDULER_NAME:-}" \
bash scripts/exp_08_extra_double_all_train_infer.sh

EXP_PREFIX="${SINGLE_NO_PDI_EXP_PREFIX}" \
LOG_TO_WANDB="${LOG_TO_WANDB:-1}" \
FOLDS="${FOLDS:-0 1 2 3 4}" \
MAX_EPOCHS="${MAX_EPOCHS:-100}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-loss2}" \
BATCH_SIZE="${BATCH_SIZE:-128}" \
bash scripts/exp_05_single_no_pdi_5fold.sh
