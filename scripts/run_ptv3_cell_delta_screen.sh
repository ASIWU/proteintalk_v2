#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
BASE_PREFIX="${BASE_PREFIX:-$(date +%Y%m%d_%H%M)_ptv3_cell_delta_screen}"
GPU_IDS="${GPU_IDS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-}"
FOLDS="${FOLDS:-2}"
METHODS="${METHODS:-delta_gate_mse050 delta_summary_learn_mse050 delta_gate_teacher025 ctrl_drug_delta pos_sample_delta}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
RUN_REPORT="${RUN_REPORT:-1}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda:0}"
REPORT_INFER_BATCH_SIZE="${REPORT_INFER_BATCH_SIZE:-512}"
REPORT_NUM_WORKERS="${REPORT_NUM_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv3_cell_delta_screen}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz}"
TASK_NAME="${TASK_NAME:-ptv3_main_singledrug}"
SPLIT_STRATEGY_PREFIX="${SPLIT_STRATEGY_PREFIX:-cell_5fold_fold}"
GPU_JOB_SUMMARY="${GPU_JOB_SUMMARY:-${LOG_DIR}/${BASE_PREFIX}_gpu_job_summary.tsv}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 2
fi
if [[ -z "${MAX_PARALLEL_JOBS}" ]]; then
  MAX_PARALLEL_JOBS=$((JOBS_PER_GPU * ${#GPU_ARRAY[@]}))
fi

"${PYTHON_BIN}" -m py_compile \
  train.py \
  infer.py \
  dataset/training_ready_fast_dataset.py \
  model/fast_delta_model.py \
  model/fast_lightning.py \
  model/graph_feature_utils.py \
  scripts/report_cell_drug_time_eval.py

printf "method\tfold\tgpu_id\tstatus\tstart_utc\tend_utc\tduration_sec\tlog_path\tcsv_path\n" > "${GPU_JOB_SUMMARY}"

COMMON_ENV=(
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
  "ALLOW_EXISTING_RUN=${ALLOW_EXISTING_RUN}"
  "CELL_LLM_MODE=frozen"
  "CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
  "CELL_LLM_FUSION_MODE=covariate"
  "CELL_LLM_CONDITION_SCALE=0.0"
  "CELL_LLM_LOGIT_SCALE=0.0"
  "CELL_LLM_DROPOUT=0.0"
  "CELL_TYPE_LLM_MODE=frozen"
  "CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH}"
  "CELL_TYPE_LLM_FUSION_MODE=covariate"
  "CELL_TYPE_LLM_CONDITION_SCALE=0.0"
  "CELL_TYPE_LLM_LOGIT_SCALE=0.0"
  "CELL_TYPE_LLM_DROPOUT=0.0"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=${LOGGER_BACKEND:-none}"
  "LOG_TO_WANDB=${LOG_TO_WANDB:-0}"
  "PROGRESS_BAR=${PROGRESS_BAR:-0}"
  "RUN_DATA_VALIDATION=${RUN_DATA_VALIDATION:-0}"
  "RUN_INFERENCE=0"
  "RUN_PREFLIGHT=0"
  "MODEL_TYPE=fast_delta"
  "HIDDEN_DIM=512"
  "EXPRESSION_LATENT_DIM=768"
  "COVARIATE_EMBEDDING_DIM=96"
  "BATCH_SIZE=${BATCH_SIZE:-256}"
  "INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-256}"
  "PRECISION=${PRECISION:-bf16-mixed}"
  "NUM_WORKERS=${NUM_WORKERS:-4}"
  "MAX_EPOCHS=${MAX_EPOCHS:-50}"
  "LEARNING_RATE=${LEARNING_RATE:-5e-5}"
  "DROPOUT=${DROPOUT:-0.15}"
  "WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}"
  "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-1.0}"
  "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-1.0}"
  "LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES:-1.0}"
  "SAVE_TOP_K=1"
  "SAVE_LAST_CKPT=1"
  "BEST_CKPT_METRIC=valid_auprc"
  "MSE_WEIGHT=0.50"
  "MSE_WEIGHT_SCHEDULE=constant"
  "MSE_FINAL_WEIGHT_MULTIPLIER=1.0"
  "MSE_PRETRAIN_EPOCHS=0"
  "MSE_PRETRAIN_BCE_WEIGHT=0.0"
  "MSE_TARGET_MODE=all"
  "MSE_INACTIVE_LABEL_WEIGHT=1.0"
  "RANKING_LOSS_WEIGHT=0.0"
  "POSITIVE_WEIGHT=none"
  "ACTIVE_LABEL_SAMPLING_WEIGHT=1.0"
  "POSITIVE_LABEL_SAMPLING_WEIGHT=1.0"
  "COVARIATE_UNK_FOR_UNSEEN=0"
  "COVARIATE_UNK_FIELDS="
  "COVARIATE_UNK_DROPOUT=0.0"
  "CONTROL_EXPRESSION_DROPOUT=0.0"
  "GRAPH_FEATURE_MODE=real"
  "GRAPH_STRUCTURAL_RP=1"
  "GRAPH_DRUG_CONCAT=1"
  "GRAPH_LOGIT_SCALE=2.0"
  "GRAPH_CACHE_DIR=${GRAPH_CACHE_DIR}"
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
  "CKPT_DIR=${CKPT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "TASK_NAME=${TASK_NAME}"
  "SPLIT_STRATEGY_PREFIX=${SPLIT_STRATEGY_PREFIX}"
)

method_env() {
  local method="$1"
  case "${method}" in
    selected_rerun)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "DELTA_LOGIT_LEARNABLE=0" \
        "DELTA_TEACHER_LOSS_WEIGHT=0.0"
      ;;
    delta_gate_mse050)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=gate" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=0" \
        "DELTA_TEACHER_LOSS_WEIGHT=0.0"
      ;;
    delta_summary_learn_mse050)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.25" \
        "DELTA_LOGIT_LEARNABLE=1" \
        "DELTA_TEACHER_LOSS_WEIGHT=0.0"
      ;;
    delta_gate_teacher025)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=gate" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1" \
        "DELTA_TEACHER_LOSS_WEIGHT=0.25"
      ;;
    ctrl_drug_delta)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1" \
        "DELTA_TEACHER_LOSS_WEIGHT=0.10" \
        "CONTROL_DRUG_INTERACTION_MODE=full" \
        "CONTROL_DRUG_INTERACTION_SCALE=1.0" \
        "CONTROL_DRUG_LOGIT_SCALE=1.0"
      ;;
    pos_sample_delta)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1" \
        "DELTA_TEACHER_LOSS_WEIGHT=0.10" \
        "POSITIVE_WEIGHT=auto" \
        "POSITIVE_LABEL_SAMPLING_WEIGHT=3.0"
      ;;
    topvar_delta_sched)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=1.0" \
        "DELTA_LOGIT_LEARNABLE=1" \
        "MSE_TARGET_MODE=topvar_pdi" \
        "MSE_TARGET_TOPK=768" \
        "MSE_TARGET_VARIANCE_TOPK=4096" \
        "MSE_TARGET_VARIANCE_SCALE=2.0" \
        "MSE_WEIGHT=0.35" \
        "MSE_WEIGHT_SCHEDULE=warmup_decay" \
        "MSE_DECAY_START_EPOCH_FRAC=0.4" \
        "MSE_FINAL_WEIGHT_MULTIPLIER=0.35"
      ;;
    covunk_cell)
      printf '%s\n' \
        "COVARIATE_UNK_FOR_UNSEEN=1" \
        "COVARIATE_UNK_FIELDS=Cell" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop030)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.30" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop010)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.10" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop040)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.40" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop050)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.50" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop030_delta)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.30" \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1"
      ;;
    covdrop030_covunk)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.30" \
        "COVARIATE_UNK_FOR_UNSEEN=1" \
        "COVARIATE_UNK_FIELDS=Cell" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop030_prior_delta)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.30" \
        "CELL_PRIOR_MODE=knn_drug" \
        "CELL_PRIOR_K=8" \
        "CELL_PRIOR_TEMPERATURE=0.2" \
        "CELL_PRIOR_LOGIT_SCALE=1.0" \
        "CELL_PRIOR_FIXED_LOGIT_SCALE=1.0" \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1"
      ;;
    covdrop030_control_aux)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.30" \
        "CONTROL_LOGIT_SCALE=1.0" \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1"
      ;;
    covdrop020_mse000)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.0" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_mse025)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.25" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_mse075)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_lr1e4)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "LEARNING_RATE=1e-4" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_lr2e4)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "LEARNING_RATE=2e-4" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_lr1e4_mse025)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "LEARNING_RATE=1e-4" \
        "MSE_WEIGHT=0.25" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_drop010)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "DROPOUT=0.10" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_drop020)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "DROPOUT=0.20" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_validauroc)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "BEST_CKPT_METRIC=valid_auroc" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    covdrop020_loss2)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "BEST_CKPT_METRIC=loss2" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    traj_drug_mse075_s1)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "RESPONSE_TRAJECTORY_MODE=drug" \
        "RESPONSE_TRAJECTORY_DIM=64" \
        "TRAJECTORY_LOGIT_SCALE=1.0" \
        "TRAJECTORY_LOGIT_LEARNABLE=1"
      ;;
    traj_drug_mse075_s2)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "RESPONSE_TRAJECTORY_MODE=drug" \
        "RESPONSE_TRAJECTORY_DIM=64" \
        "TRAJECTORY_LOGIT_SCALE=2.0" \
        "TRAJECTORY_LOGIT_LEARNABLE=1"
      ;;
    traj_gate_mse075_s1)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "RESPONSE_TRAJECTORY_MODE=gate" \
        "RESPONSE_TRAJECTORY_DIM=64" \
        "TRAJECTORY_LOGIT_SCALE=1.0" \
        "TRAJECTORY_LOGIT_LEARNABLE=1"
      ;;
    traj_summary_mse075_s1)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "RESPONSE_TRAJECTORY_MODE=summary" \
        "RESPONSE_TRAJECTORY_DIM=64" \
        "TRAJECTORY_LOGIT_SCALE=1.0" \
        "TRAJECTORY_LOGIT_LEARNABLE=1"
      ;;
    traj_drug_detach_mse075_s1)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "RESPONSE_TRAJECTORY_MODE=drug" \
        "RESPONSE_TRAJECTORY_DIM=64" \
        "RESPONSE_TRAJECTORY_DETACH=1" \
        "TRAJECTORY_LOGIT_SCALE=1.0" \
        "TRAJECTORY_LOGIT_LEARNABLE=1"
      ;;
    traj_drug_teacher025)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "RESPONSE_TRAJECTORY_MODE=drug" \
        "RESPONSE_TRAJECTORY_DIM=64" \
        "TRAJECTORY_LOGIT_SCALE=1.0" \
        "TRAJECTORY_LOGIT_LEARNABLE=1" \
        "DELTA_TEACHER_LOSS_WEIGHT=0.25"
      ;;
    traj_drug_dim128_mse075_s1)
      printf '%s\n' \
        "COVARIATE_UNK_DROPOUT=0.20" \
        "MSE_WEIGHT=0.75" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0" \
        "RESPONSE_TRAJECTORY_MODE=drug" \
        "RESPONSE_TRAJECTORY_DIM=128" \
        "TRAJECTORY_LOGIT_SCALE=1.0" \
        "TRAJECTORY_LOGIT_LEARNABLE=1"
      ;;
    cell_prior_knn)
      printf '%s\n' \
        "CELL_PRIOR_MODE=knn_drug" \
        "CELL_PRIOR_K=8" \
        "CELL_PRIOR_TEMPERATURE=0.2" \
        "CELL_PRIOR_LOGIT_SCALE=1.0" \
        "CELL_PRIOR_FIXED_LOGIT_SCALE=1.0" \
        "RESPONSE_DELTA_MODE=off" \
        "DELTA_LOGIT_SCALE=0.0"
      ;;
    cell_prior_delta)
      printf '%s\n' \
        "CELL_PRIOR_MODE=knn_drug" \
        "CELL_PRIOR_K=8" \
        "CELL_PRIOR_TEMPERATURE=0.2" \
        "CELL_PRIOR_LOGIT_SCALE=1.0" \
        "CELL_PRIOR_FIXED_LOGIT_SCALE=1.0" \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1"
      ;;
    rank_cell_delta)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1" \
        "RANKING_LOSS_WEIGHT=0.05" \
        "RANKING_LOSS_GROUP_FIELD=Cell" \
        "RANKING_LOSS_HARD_NEGATIVES=32" \
        "RANKING_LOSS_HARD_POSITIVES=16"
      ;;
    focal_delta)
      printf '%s\n' \
        "RESPONSE_DELTA_MODE=summary" \
        "RESPONSE_DELTA_DIM=64" \
        "RESPONSE_DELTA_DETACH=1" \
        "DELTA_LOGIT_SCALE=0.5" \
        "DELTA_LOGIT_LEARNABLE=1" \
        "FOCAL_LOSS=1" \
        "FOCAL_GAMMA=2.0" \
        "FOCAL_ALPHA=0.25"
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      return 2
      ;;
  esac
}

utc_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

fold_complete() {
  local prefix="$1"
  local fold="$2"
  local manifest="${CKPT_DIR}/${prefix}_exp03_single_cell_5fold_single_cell_fold${fold}/run_manifest.json"
  [[ -f "${manifest}" ]] || return 1
  "${PYTHON_BIN}" - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
if manifest.get("run_status") != "fit_completed":
    raise SystemExit(1)
if not Path(manifest.get("best_model_path") or path.parent / "last.ckpt").exists():
    raise SystemExit(1)
PY
}

run_report() {
  local gpu_id="$1"
  local prefix="$2"
  local method="$3"
  local fold="$4"
  local csv_path="${OUTPUT_DIR}/${prefix}_cell_drug_time_eval.csv"
  local json_path="${OUTPUT_DIR}/${prefix}_cell_drug_time_eval.json"
  local md_path="${LOG_DIR}/${prefix}_cell_drug_time_eval.md"
  if ! CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" scripts/report_cell_drug_time_eval.py \
    --prefix "${prefix}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --materialize-fold-predictions \
    --device "${REPORT_DEVICE}" \
    --infer-batch-size "${REPORT_INFER_BATCH_SIZE}" \
    --num-workers "${REPORT_NUM_WORKERS}" \
    --csv-out "${csv_path}" \
    --json-out "${json_path}" \
    --format markdown > "${md_path}"; then
    echo "[report-fail] method=${method} fold=${fold} md=${md_path}" >&2
    return 1
  fi
  echo "[report] method=${method} fold=${fold} csv=${csv_path} md=${md_path}"
}

run_one() {
  local gpu_id="$1"
  local method="$2"
  local fold="$3"
  local prefix="${BASE_PREFIX}_${method}"
  local exp_prefix="${prefix}_exp03_single_cell_5fold"
  local log_path="${LOG_DIR}/${prefix}_fold${fold}.job.log"
  local csv_path="${OUTPUT_DIR}/${prefix}_cell_drug_time_eval.csv"
  local time_summary="${LOG_DIR}/${prefix}_runtime_summary.tsv"
  local start_utc end_utc start_sec end_sec status

  if [[ "${SKIP_COMPLETED}" == "1" ]] && fold_complete "${prefix}" "${fold}"; then
    echo "[skip] method=${method} fold=${fold}"
    if [[ "${RUN_REPORT}" == "1" ]]; then
      run_report "${gpu_id}" "${prefix}" "${method}" "${fold}" >> "${log_path}" 2>&1
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${method}" "${fold}" "${gpu_id}" "0" "$(utc_now)" "$(utc_now)" "0" "${log_path}" "${csv_path}" >> "${GPU_JOB_SUMMARY}"
    return 0
  fi

  echo "[run][gpu=${gpu_id}] method=${method} fold=${fold}"
  start_utc="$(utc_now)"
  start_sec="$(date +%s)"
  status=0
  mapfile -t EXTRA_ENV < <(method_env "${method}")
  env "${COMMON_ENV[@]}" "${EXTRA_ENV[@]}" \
    "GPU_IDS=${gpu_id}" \
    "FOLDS=${fold}" \
    "EXP_PREFIX=${exp_prefix}" \
    "TIME_SUMMARY_PATH=${time_summary}" \
    bash scripts/exp_03_single_cell_5fold.sh > "${log_path}" 2>&1 || status="$?"
  if [[ "${status}" == "0" && "${RUN_REPORT}" == "1" ]]; then
    run_report "${gpu_id}" "${prefix}" "${method}" "${fold}" >> "${log_path}" 2>&1 || status="$?"
  fi
  end_utc="$(utc_now)"
  end_sec="$(date +%s)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${method}" "${fold}" "${gpu_id}" "${status}" "${start_utc}" "${end_utc}" "$((end_sec - start_sec))" "${log_path}" "${csv_path}" >> "${GPU_JOB_SUMMARY}"
  if [[ "${status}" == "0" ]]; then
    echo "[done][gpu=${gpu_id}] method=${method} fold=${fold}"
  else
    echo "[fail][gpu=${gpu_id}] method=${method} fold=${fold}; log=${log_path}" >&2
  fi
  return "${status}"
}

jobs=()
read -r -a FOLD_ARRAY <<< "${FOLDS}"
for method in ${METHODS}; do
  method_env "${method}" >/dev/null
  for fold in "${FOLD_ARRAY[@]}"; do
    jobs+=("${method}|${fold}")
  done
done

running=0
failures=0
pids=()
pid_labels=()

wait_one() {
  local pid="$1"
  local label="$2"
  if ! wait "${pid}"; then
    failures=$((failures + 1))
    echo "[error] job failed: ${label}" >&2
  fi
  running=$((running - 1))
}

for i in "${!jobs[@]}"; do
  while (( running >= MAX_PARALLEL_JOBS )); do
    wait_one "${pids[0]}" "${pid_labels[0]}"
    pids=("${pids[@]:1}")
    pid_labels=("${pid_labels[@]:1}")
  done
  IFS='|' read -r method fold <<< "${jobs[$i]}"
  gpu_id="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
  run_one "${gpu_id}" "${method}" "${fold}" &
  pids+=("$!")
  pid_labels+=("${method}/fold${fold}")
  running=$((running + 1))
done

for i in "${!pids[@]}"; do
  wait_one "${pids[$i]}" "${pid_labels[$i]}"
done

if (( failures > 0 )); then
  echo "[error] ${failures} job(s) failed; inspect ${GPU_JOB_SUMMARY}" >&2
  exit 1
fi

echo "[done] PTV3 cell delta screen complete; summary=${GPU_JOB_SUMMARY}"
