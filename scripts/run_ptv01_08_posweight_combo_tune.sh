#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

usage() {
  cat <<'EOF'
Usage:
  GPU_IDS=0,1,2,3 JOBS_PER_GPU=2 RUN_MODE=all bash scripts/run_ptv01_08_posweight_combo_tune.sh
  SMOKE_TEST=1 RUN_MODE=screen bash scripts/run_ptv01_08_posweight_combo_tune.sh

Modes:
  screen    Run stage1-A/B screen and stage2-stage4 screen.
  full      Run full 5-fold for candidates promoted from an existing screen.
  selected  Run final exp01-exp08 selected suite from an existing full run.
  all       Run stage1 screen, stage1 full, stage2-stage4 screen/full, selected.

Notes:
  - This runner intentionally excludes pre1/pre2 and target_pdi/target_ppi tuning.
  - Config posw_negpos passes POSITIVE_WEIGHT=auto; train.py resolves and records
    the numeric neg/pos weight per train split.
  - Jobs are scheduled across GPU_IDS as independent single-GPU fold jobs.
  - JOBS_PER_GPU defaults to 2; MAX_PARALLEL_JOBS can cap the total job count.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
RUN_MODE="${RUN_MODE:-screen}"
BASE_PREFIX_WAS_SET="${BASE_PREFIX+x}"
SCREEN_PREFIX_WAS_SET="${SCREEN_PREFIX+x}"
FULL_PREFIX_WAS_SET="${FULL_PREFIX+x}"
SELECTED_PREFIX_WAS_SET="${SELECTED_PREFIX+x}"
STAGES_WAS_SET="${STAGES+x}"
SCREEN_FOLDS_WAS_SET="${SCREEN_FOLDS+x}"
FULL_FOLDS_WAS_SET="${FULL_FOLDS+x}"
FINAL_FOLDS_WAS_SET="${FINAL_FOLDS+x}"
SCREEN_MIN_FOLDS_WAS_SET="${SCREEN_MIN_FOLDS+x}"
FULL_MIN_FOLDS_WAS_SET="${FULL_MIN_FOLDS+x}"
RUN_REPORT_WAS_SET="${RUN_REPORT+x}"
RUN_INPUT_VALIDATION_WAS_SET="${RUN_INPUT_VALIDATION+x}"
GPU_JOB_SUMMARY_WAS_SET="${GPU_JOB_SUMMARY+x}"
BASE_PREFIX="${BASE_PREFIX:-20260610_ptv01_08_posweight_combo_v1}"
SCREEN_PREFIX="${SCREEN_PREFIX:-${BASE_PREFIX}}"
FULL_PREFIX="${FULL_PREFIX:-${BASE_PREFIX}_full}"
SELECTED_PREFIX="${SELECTED_PREFIX:-20260610_ptv01_08_posweight_combo_selected_v1}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv01_08_posweight_combo}"
GPU_IDS="${GPU_IDS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-}"
STAGES="${STAGES:-stage1 stage2 stage3 stage4}"
SCREEN_FOLDS="${SCREEN_FOLDS:-0 2 4}"
FULL_FOLDS="${FULL_FOLDS:-0 1 2 3 4}"
FINAL_FOLDS="${FINAL_FOLDS:-0 1 2 3 4}"
SCREEN_MIN_FOLDS="${SCREEN_MIN_FOLDS:-3}"
FULL_MIN_FOLDS="${FULL_MIN_FOLDS:-5}"
PROMOTION_MARGIN="${PROMOTION_MARGIN:-0.002}"
RUN_TUNING_REPORT="${RUN_TUNING_REPORT:-1}"
RUN_REPORT="${RUN_REPORT:-1}"
RUN_INPUT_VALIDATION="${RUN_INPUT_VALIDATION:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda:0}"
REPORT_INFER_BATCH_SIZE="${REPORT_INFER_BATCH_SIZE:-512}"
REPORT_NUM_WORKERS="${REPORT_NUM_WORKERS:-4}"
TUNING_REPORT_SCRIPT="${TUNING_REPORT_SCRIPT:-scripts/report_cell_celltype_llm_clip10_param_search.py}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz}"
GPU_JOB_SUMMARY="${GPU_JOB_SUMMARY:-${LOG_DIR}/${BASE_PREFIX}_gpu_job_summary.tsv}"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  smoke_stamp="$(date +%Y%m%d_%H%M%S)"
  if [[ -z "${BASE_PREFIX_WAS_SET}" ]]; then
    BASE_PREFIX="smoke_ptv01_08_posweight_combo_${smoke_stamp}"
  fi
  if [[ -z "${SCREEN_PREFIX_WAS_SET}" ]]; then
    SCREEN_PREFIX="${BASE_PREFIX}"
  fi
  if [[ -z "${FULL_PREFIX_WAS_SET}" ]]; then
    FULL_PREFIX="${BASE_PREFIX}_full"
  fi
  if [[ -z "${SELECTED_PREFIX_WAS_SET}" ]]; then
    SELECTED_PREFIX="${BASE_PREFIX}_selected"
  fi
  if [[ -z "${SCREEN_FOLDS_WAS_SET}" ]]; then
    SCREEN_FOLDS="0"
  fi
  if [[ -z "${FULL_FOLDS_WAS_SET}" ]]; then
    FULL_FOLDS="0"
  fi
  if [[ -z "${FINAL_FOLDS_WAS_SET}" ]]; then
    FINAL_FOLDS="0"
  fi
  if [[ -z "${SCREEN_MIN_FOLDS_WAS_SET}" ]]; then
    SCREEN_MIN_FOLDS="1"
  fi
  if [[ -z "${FULL_MIN_FOLDS_WAS_SET}" ]]; then
    FULL_MIN_FOLDS="1"
  fi
  if [[ -z "${STAGES_WAS_SET}" ]]; then
    STAGES="stage1"
  fi
  RUN_MODE="${RUN_MODE:-screen}"
  if [[ -z "${RUN_REPORT_WAS_SET}" ]]; then
    RUN_REPORT="0"
  fi
  if [[ -z "${RUN_INPUT_VALIDATION_WAS_SET}" ]]; then
    RUN_INPUT_VALIDATION="0"
  fi
  export MAX_EPOCHS="${MAX_EPOCHS:-1}"
  export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1}"
  export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"
  export LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1}"
  export INFER_LIMIT_BATCHES="${INFER_LIMIT_BATCHES:-1}"
  export NUM_WORKERS="${NUM_WORKERS:-0}"
  export LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
  export LOG_TO_WANDB="${LOG_TO_WANDB:-0}"
  export PROGRESS_BAR="${PROGRESS_BAR:-0}"
fi
if [[ -z "${GPU_JOB_SUMMARY_WAS_SET}" ]]; then
  GPU_JOB_SUMMARY="${LOG_DIR}/${BASE_PREFIX}_gpu_job_summary.tsv"
fi

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}" "${GRAPH_CACHE_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 || -z "${GPU_ARRAY[0]}" ]]; then
  echo "[error] GPU_IDS is empty" >&2
  exit 2
fi
if ! [[ "${JOBS_PER_GPU}" =~ ^[0-9]+$ ]] || [[ "${JOBS_PER_GPU}" -lt 1 ]]; then
  echo "[error] JOBS_PER_GPU must be a positive integer" >&2
  exit 2
fi
GPU_SLOT_CAPACITY=$((${#GPU_ARRAY[@]} * JOBS_PER_GPU))
if [[ -z "${MAX_PARALLEL_JOBS}" ]]; then
  MAX_PARALLEL_JOBS="${GPU_SLOT_CAPACITY}"
fi
if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL_JOBS}" -lt 1 ]]; then
  echo "[error] MAX_PARALLEL_JOBS must be >= 1" >&2
  exit 2
fi
if [[ "${MAX_PARALLEL_JOBS}" -gt "${GPU_SLOT_CAPACITY}" ]]; then
  echo "[warn] MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS} exceeds GPU slot capacity ${GPU_SLOT_CAPACITY}; capping to ${GPU_SLOT_CAPACITY}" >&2
  MAX_PARALLEL_JOBS="${GPU_SLOT_CAPACITY}"
fi

init_gpu_job_summary() {
  if [[ ! -f "${GPU_JOB_SUMMARY}" ]]; then
    printf "stage\tconfig\texp_suffix\tfold\tgpu_id\tstatus\tstart_utc\tend_utc\tduration_sec\tlog_path\n" > "${GPU_JOB_SUMMARY}"
  fi
}

utc_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

validate_inputs() {
  "${PYTHON_BIN}" - "${CELL_LLM_EMBEDDING_PATH}" "${CELL_TYPE_LLM_EMBEDDING_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cell_path = Path(sys.argv[1])
cell_type_path = Path(sys.argv[2])

def load_matrix(path: Path):
    if not path.exists():
        raise SystemExit(f"missing embedding artifact: {path}")
    payload = np.load(path, allow_pickle=False)
    if "embedding_matrix" not in payload.files:
        raise SystemExit(f"{path} must contain embedding_matrix")
    meta = {}
    if "meta_json" in payload.files:
        meta = json.loads(str(payload["meta_json"].item()))
    return payload["embedding_matrix"], meta

cell_matrix, cell_meta = load_matrix(cell_path)
cell_type_matrix, cell_type_meta = load_matrix(cell_type_path)
if cell_matrix.shape != (74, 4096):
    raise SystemExit(f"{cell_path} expected shape (74, 4096), got {cell_matrix.shape}")
if cell_meta.get("kind") != "cell_llm_embedding" or cell_meta.get("index_column") != "Cell_index":
    raise SystemExit(f"{cell_path} has invalid metadata: {cell_meta}")
if cell_type_matrix.shape != (14, 4096):
    raise SystemExit(f"{cell_type_path} expected shape (14, 4096), got {cell_type_matrix.shape}")
if cell_type_meta.get("kind") != "cell_type_llm_embedding" or cell_type_meta.get("index_column") != "cell_type_index":
    raise SystemExit(f"{cell_type_path} has invalid metadata: {cell_type_meta}")
for name, matrix in (("Cell", cell_matrix), ("cell_type", cell_type_matrix)):
    if not np.allclose(matrix[0], 0.0):
        raise SystemExit(f"{name} embedding row 0 must be zero")
    if not np.isfinite(matrix[1:]).all() or np.any(np.all(np.isclose(matrix[1:], 0.0), axis=1)):
        raise SystemExit(f"{name} non-no rows must be finite and nonzero")

root = Path("data/training_ready/ptv3")
meta = json.loads((root / "global_meta.json").read_text())
if len(meta["value_to_index"]["Cell"]) != 74 or int(meta["value_to_index"]["Cell"].get("no", -1)) != 0:
    raise SystemExit("global Cell mapping must contain 74 values with no=0")
if len(meta["value_to_index"]["cell_type"]) != 14 or int(meta["value_to_index"]["cell_type"].get("no", -1)) != 0:
    raise SystemExit("global cell_type mapping must contain 14 values with no=0")

for task in meta.get("task_names", []):
    table_path = root / "tasks" / str(task) / "feature_table.parquet"
    csv_path = table_path.with_suffix(".csv")
    columns = ["Cell_index", "cell_type_index", "pert_dose1", "pert_dose2"]
    if table_path.exists():
        df = pd.read_parquet(table_path, columns=columns)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, usecols=columns, low_memory=False)
    else:
        raise SystemExit(f"missing feature_table for {task}")
    if int(pd.to_numeric(df["Cell_index"], errors="coerce").fillna(0).astype(int).max()) >= cell_matrix.shape[0]:
        raise SystemExit(f"{task}: Cell_index exceeds Cell embedding rows")
    if int(pd.to_numeric(df["cell_type_index"], errors="coerce").fillna(0).astype(int).max()) >= cell_type_matrix.shape[0]:
        raise SystemExit(f"{task}: cell_type_index exceeds cell_type embedding rows")
    for field in ("pert_dose1", "pert_dose2"):
        numeric = pd.to_numeric(df[field], errors="coerce").dropna()
        if not numeric.empty and (float(numeric.min()) < 0.0 or float(numeric.max()) > 10.0):
            raise SystemExit(f"{task}: {field} outside clip10 range")
print("[validate] Cell and cell_type LLM artifacts passed")
PY
}

COMMON_ENV=(
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
  "ALLOW_EXISTING_RUN=${ALLOW_EXISTING_RUN}"
  "CELL_LLM_MODE=frozen"
  "CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
  "CELL_LLM_FUSION_MODE=${CELL_LLM_FUSION_MODE:-covariate}"
  "CELL_LLM_CONDITION_SCALE=${CELL_LLM_CONDITION_SCALE:-0.0}"
  "CELL_LLM_LOGIT_SCALE=${CELL_LLM_LOGIT_SCALE:-0.0}"
  "CELL_LLM_DROPOUT=${CELL_LLM_DROPOUT:-0.0}"
  "CELL_TYPE_LLM_MODE=frozen"
  "CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH}"
  "CELL_TYPE_LLM_FUSION_MODE=${CELL_TYPE_LLM_FUSION_MODE:-covariate}"
  "CELL_TYPE_LLM_CONDITION_SCALE=${CELL_TYPE_LLM_CONDITION_SCALE:-0.0}"
  "CELL_TYPE_LLM_LOGIT_SCALE=${CELL_TYPE_LLM_LOGIT_SCALE:-0.0}"
  "CELL_TYPE_LLM_DROPOUT=${CELL_TYPE_LLM_DROPOUT:-0.0}"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=${LOGGER_BACKEND:-none}"
  "LOG_TO_WANDB=${LOG_TO_WANDB:-0}"
  "PROGRESS_BAR=${PROGRESS_BAR:-0}"
  "RUN_DATA_VALIDATION=${RUN_DATA_VALIDATION:-0}"
  "RUN_INFERENCE=1"
  "RUN_PREFLIGHT=${RUN_PREFLIGHT:-0}"
  "MODEL_TYPE=fast_delta"
  "HIDDEN_DIM=512"
  "EXPRESSION_LATENT_DIM=768"
  "COVARIATE_EMBEDDING_DIM=96"
  "INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-256}"
  "PRECISION=${PRECISION:-bf16-mixed}"
  "NUM_WORKERS=${NUM_WORKERS:-4}"
  "MAX_EPOCHS=${MAX_EPOCHS:-50}"
  "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-1.0}"
  "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-1.0}"
  "LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES:-1.0}"
  "INFER_LIMIT_BATCHES=${INFER_LIMIT_BATCHES:-}"
  "SAVE_TOP_K=1"
  "SAVE_LAST_CKPT=1"
  "BEST_CKPT_METRIC=valid_auprc"
  "MSE_WEIGHT_SCHEDULE=constant"
  "MSE_FINAL_WEIGHT_MULTIPLIER=1.0"
  "MSE_PRETRAIN_EPOCHS=0"
  "MSE_PRETRAIN_BCE_WEIGHT=0.0"
  "MSE_TARGET_MODE=all"
  "MSE_INACTIVE_LABEL_WEIGHT=1.0"
  "RANKING_LOSS_WEIGHT=0.0"
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
)

CONFIG_ENV=()
set_config_env() {
  local name="$1"
  CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "POSITIVE_WEIGHT=none")
  case "${name}" in
    base)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none")
      ;;
    mse050) ;;
    mse075)
      CONFIG_ENV[4]="MSE_WEIGHT=0.75"
      ;;
    mse075_drop010)
      CONFIG_ENV[2]="DROPOUT=0.10"; CONFIG_ENV[4]="MSE_WEIGHT=0.75"
      ;;
    mse050_lr5e5)
      CONFIG_ENV[0]="LEARNING_RATE=5e-5"
      ;;
    mse050_lr1e4)
      CONFIG_ENV[0]="LEARNING_RATE=1e-4"
      ;;
    mse050_lr3e4)
      CONFIG_ENV[0]="LEARNING_RATE=3e-4"
      ;;
    mse050_drop010)
      CONFIG_ENV[2]="DROPOUT=0.10"
      ;;
    mse050_drop020|drop020)
      CONFIG_ENV[2]="DROPOUT=0.20"; CONFIG_ENV[4]="MSE_WEIGHT=0.25"
      ;;
    mse050_drop050|drop050)
      CONFIG_ENV[2]="DROPOUT=0.50"; CONFIG_ENV[4]="MSE_WEIGHT=0.25"
      ;;
    covdrop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covdrop010_lr5e5)
      CONFIG_ENV=("LEARNING_RATE=5e-5" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covdrop010_drop050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.50" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covunk_cell)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=Cell")
      ;;
    covunk_celltype)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=cell_type")
      ;;
    dbl_mse050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none" "MSE_INACTIVE_LABEL_WEIGHT=0.50")
      ;;
    rank005)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=none" "RANKING_LOSS_WEIGHT=0.05")
      ;;
    posw*)
      local pos_value=""
      if [[ "${name}" == posw_negpos* ]]; then
        pos_value="auto"
      elif [[ "${name}" =~ ^posw([0-9]+) ]]; then
        pos_value="${BASH_REMATCH[1]}"
      else
        echo "[error] cannot parse positive-weight config: ${name}" >&2
        exit 2
      fi
      CONFIG_ENV[5]="POSITIVE_WEIGHT=${pos_value}"
      if [[ "${name}" == *"_lr5e5"* ]]; then
        CONFIG_ENV[0]="LEARNING_RATE=5e-5"
      fi
      if [[ "${name}" == *"_lr1e4"* ]]; then
        CONFIG_ENV[0]="LEARNING_RATE=1e-4"
      fi
      if [[ "${name}" == *"_lr2e4"* ]]; then
        CONFIG_ENV[0]="LEARNING_RATE=2e-4"
      fi
      if [[ "${name}" == *"_lr3e4"* ]]; then
        CONFIG_ENV[0]="LEARNING_RATE=3e-4"
      fi
      if [[ "${name}" == *"_drop010"* ]]; then
        CONFIG_ENV[2]="DROPOUT=0.10"
      fi
      if [[ "${name}" == *"_drop015"* ]]; then
        CONFIG_ENV[2]="DROPOUT=0.15"
      fi
      if [[ "${name}" == *"_drop020"* ]]; then
        CONFIG_ENV[2]="DROPOUT=0.20"
      fi
      if [[ "${name}" == *"_drop050"* ]]; then
        CONFIG_ENV[2]="DROPOUT=0.50"
      fi
      ;;
    *)
      echo "[error] unknown config: ${name}" >&2
      exit 2
      ;;
  esac
  return 0
}

DOUBLE_FIXED_ENV=(
  "PAIR_FUSION_MODE=dual"
  "PAIR_TYPE_FEATURES=1"
  "MSE_INACTIVE_LABEL_WEIGHT=0.2"
  "USE_DDI=1"
  "GRAPH_PAIR_ADD_SCALE=0.5"
)

stage1_a_configs() {
  if [[ -n "${STAGE1_A_CONFIGS:-}" ]]; then
    echo "${STAGE1_A_CONFIGS}"
  elif [[ "${SMOKE_TEST}" == "1" ]]; then
    echo "mse050 posw10_lr5e5_drop050"
  else
    echo "base mse050 mse075_drop010 mse050_lr1e4 mse050_lr3e4 mse050_lr5e5 mse050_drop010 mse050_drop020 mse050_drop050 posw10 posw50 posw100 posw200 posw500 posw_negpos posw10_lr5e5 posw10_lr1e4 posw10_lr3e4 posw100_lr5e5 posw100_lr1e4 posw100_lr3e4 posw500_lr5e5 posw500_lr1e4 posw500_lr3e4 posw_negpos_lr5e5 posw_negpos_lr1e4 posw_negpos_lr3e4 posw10_drop010 posw10_drop020 posw10_drop050 posw100_drop010 posw100_drop020 posw100_drop050 posw500_drop010 posw500_drop020 posw500_drop050 posw_negpos_drop010 posw_negpos_drop020 posw_negpos_drop050"
  fi
}

config_lr_token() {
  local config="$1"
  if [[ "${config}" == *"_lr5e5"* || "${config}" == "mse050_lr5e5" || "${config}" == "covdrop010_lr5e5" ]]; then
    echo "lr5e5"
  elif [[ "${config}" == *"_lr1e4"* || "${config}" == "mse050_lr1e4" ]]; then
    echo "lr1e4"
  elif [[ "${config}" == *"_lr3e4"* || "${config}" == "mse050_lr3e4" ]]; then
    echo "lr3e4"
  else
    echo "lr2e4"
  fi
}

config_dropout_token() {
  local config="$1"
  if [[ "${config}" == *"_drop010"* || "${config}" == "mse050_drop010" || "${config}" == "mse075_drop010" ]]; then
    echo "drop010"
  elif [[ "${config}" == *"_drop020"* || "${config}" == "mse050_drop020" || "${config}" == "drop020" ]]; then
    echo "drop020"
  elif [[ "${config}" == *"_drop050"* || "${config}" == "mse050_drop050" || "${config}" == "drop050" || "${config}" == "covdrop010_drop050" ]]; then
    echo "drop050"
  else
    echo "drop015"
  fi
}

fold_complete() {
  local exp_prefix="$1"
  local fold_suffix="$2"
  local expected_graph_mode="$3"
  local fold="$4"
  "${PYTHON_BIN}" - "${CKPT_DIR}" "${exp_prefix}" "${fold_suffix}" "${expected_graph_mode}" "${fold}" <<'PY'
import json
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
exp_prefix = sys.argv[2]
fold_suffix = sys.argv[3]
expected_graph_mode = sys.argv[4]
fold = sys.argv[5]
manifest_path = ckpt_dir / f"{exp_prefix}_{fold_suffix}{fold}" / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(1)
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit(1)
checks = [
    manifest.get("run_status") == "fit_completed",
    manifest.get("test_status") == "test_completed",
    manifest.get("cell_llm_mode") == "frozen",
    manifest.get("cell_type_llm_mode") == "frozen",
    manifest.get("graph_feature_mode") == expected_graph_mode,
]
cell = manifest.get("cell_llm_summary")
cell_type = manifest.get("cell_type_llm_summary")
checks.extend([
    isinstance(cell, dict) and cell.get("embedding_rows") == 74,
    isinstance(cell_type, dict) and cell_type.get("embedding_rows") == 14,
    str(manifest.get("mse_target_mode", "all")).lower() == "all",
    int(manifest.get("mse_pretrain_epochs", 0)) == 0,
])
raise SystemExit(0 if all(checks) else 1)
PY
}

running_jobs=0
job_index=0
failures=0
GPU_LOAD=()
for gpu_index in "${!GPU_ARRAY[@]}"; do
  GPU_LOAD["${gpu_index}"]=0
done
declare -A JOB_GPU_INDEX=()
RESERVED_GPU_INDEX=""

wait_for_one_job() {
  local finished_pid=""
  local status=0
  if wait -n -p finished_pid; then
    status=0
  else
    status="$?"
    if [[ "${status}" -ne 127 ]]; then
      failures=$((failures + 1))
    fi
  fi
  if [[ -n "${finished_pid}" && -n "${JOB_GPU_INDEX[${finished_pid}]+x}" ]]; then
    local gpu_index="${JOB_GPU_INDEX[${finished_pid}]}"
    if [[ "${GPU_LOAD[${gpu_index}]}" -gt 0 ]]; then
      GPU_LOAD["${gpu_index}"]=$((GPU_LOAD["${gpu_index}"] - 1))
    fi
    unset "JOB_GPU_INDEX[${finished_pid}]"
  fi
  if [[ "${running_jobs}" -gt 0 ]]; then
    running_jobs=$((running_jobs - 1))
  fi
  return 0
}

reserve_gpu_index() {
  local start_index offset gpu_index
  while true; do
    if [[ "${running_jobs}" -lt "${MAX_PARALLEL_JOBS}" ]]; then
      start_index=$((job_index % ${#GPU_ARRAY[@]}))
      for ((offset = 0; offset < ${#GPU_ARRAY[@]}; offset++)); do
        gpu_index=$(((start_index + offset) % ${#GPU_ARRAY[@]}))
        if [[ "${GPU_LOAD[${gpu_index}]}" -lt "${JOBS_PER_GPU}" ]]; then
          RESERVED_GPU_INDEX="${gpu_index}"
          return 0
        fi
      done
    fi
    wait_for_one_job
  done
}

wait_for_all_jobs() {
  while [[ "${running_jobs}" -gt 0 ]]; do
    wait_for_one_job
  done
  if [[ "${failures}" -ne 0 ]]; then
    echo "[error] ${failures} job(s) failed; inspect ${LOG_DIR}/*ptv01_08_posweight_combo*.job.log and ${GPU_JOB_SUMMARY}" >&2
    exit 1
  fi
}

launch_fold_job() {
  local prefix="$1"
  local stage="$2"
  local config="$3"
  local exp_suffix="$4"
  local script_path="$5"
  local fold_suffix="$6"
  local expected_graph_mode="$7"
  local fold="$8"
  shift 8

  local run_prefix="${prefix}_${stage}_${config}"
  local exp_prefix="${run_prefix}_${exp_suffix}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && fold_complete "${exp_prefix}" "${fold_suffix}" "${expected_graph_mode}" "${fold}"; then
    echo "[skip] stage=${stage} config=${config} exp=${exp_suffix} fold=${fold}"
    return
  fi

  set_config_env "${config}"
  local gpu_index
  reserve_gpu_index
  gpu_index="${RESERVED_GPU_INDEX}"
  local gpu_id="${GPU_ARRAY[${gpu_index}]}"
  GPU_LOAD["${gpu_index}"]=$((GPU_LOAD["${gpu_index}"] + 1))
  job_index=$((job_index + 1))
  running_jobs=$((running_jobs + 1))

  local log_path="${LOG_DIR}/${exp_prefix}_${fold_suffix}${fold}.job.log"
  local time_summary="${LOG_DIR}/${run_prefix}_runtime_summary.tsv"
  echo "[launch][gpu=${gpu_id}][load=${GPU_LOAD[${gpu_index}]}/${JOBS_PER_GPU}] stage=${stage} config=${config} exp=${exp_suffix} fold=${fold}"
  (
    local start_utc end_utc start_sec end_sec status
    start_utc="$(utc_now)"
    start_sec="$(date +%s)"
    status=0
    env "${COMMON_ENV[@]}" "$@" "${CONFIG_ENV[@]}" \
      "GPU_IDS=${gpu_id}" \
      "FOLDS=${fold}" \
      "EXP_PREFIX=${exp_prefix}" \
      "TIME_SUMMARY_PATH=${time_summary}" \
      bash "${script_path}" > "${log_path}" 2>&1 || status="$?"
    end_utc="$(utc_now)"
    end_sec="$(date +%s)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${stage}" "${config}" "${exp_suffix}" "${fold}" "${gpu_id}" "${status}" \
      "${start_utc}" "${end_utc}" "$((end_sec - start_sec))" "${log_path}" >> "${GPU_JOB_SUMMARY}"
    exit "${status}"
  ) &
  JOB_GPU_INDEX["$!"]="${gpu_index}"
}

launch_selected_fold_job() {
  local exp_suffix="$1"
  local script_path="$2"
  local config="$3"
  local fold_suffix="$4"
  local expected_graph_mode="$5"
  local fold="$6"
  shift 6

  local exp_prefix="${SELECTED_PREFIX}_${exp_suffix}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && fold_complete "${exp_prefix}" "${fold_suffix}" "${expected_graph_mode}" "${fold}"; then
    echo "[skip][selected] config=${config} exp=${exp_suffix} fold=${fold}"
    return
  fi

  set_config_env "${config}"
  local gpu_index
  reserve_gpu_index
  gpu_index="${RESERVED_GPU_INDEX}"
  local gpu_id="${GPU_ARRAY[${gpu_index}]}"
  GPU_LOAD["${gpu_index}"]=$((GPU_LOAD["${gpu_index}"] + 1))
  job_index=$((job_index + 1))
  running_jobs=$((running_jobs + 1))

  local log_path="${LOG_DIR}/${exp_prefix}_${fold_suffix}${fold}.selected.job.log"
  local time_summary="${LOG_DIR}/${SELECTED_PREFIX}_runtime_summary.tsv"
  echo "[launch][selected][gpu=${gpu_id}][load=${GPU_LOAD[${gpu_index}]}/${JOBS_PER_GPU}] config=${config} exp=${exp_suffix} fold=${fold}"
  (
    local start_utc end_utc start_sec end_sec status
    start_utc="$(utc_now)"
    start_sec="$(date +%s)"
    status=0
    env "${COMMON_ENV[@]}" "$@" "${CONFIG_ENV[@]}" \
      "GPU_IDS=${gpu_id}" \
      "FOLDS=${fold}" \
      "EXP_PREFIX=${exp_prefix}" \
      "TIME_SUMMARY_PATH=${time_summary}" \
      bash "${script_path}" > "${log_path}" 2>&1 || status="$?"
    end_utc="$(utc_now)"
    end_sec="$(date +%s)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "selected" "${config}" "${exp_suffix}" "${fold}" "${gpu_id}" "${status}" \
      "${start_utc}" "${end_utc}" "$((end_sec - start_sec))" "${log_path}" >> "${GPU_JOB_SUMMARY}"
    exit "${status}"
  ) &
  JOB_GPU_INDEX["$!"]="${gpu_index}"
}

run_stage_configs() {
  local prefix="$1"
  local stage="$2"
  local folds="$3"
  local configs="$4"
  local config fold
  failures=0
  for config in ${configs}; do
    for fold in ${folds}; do
      case "${stage}" in
        stage1)
          launch_fold_job "${prefix}" "${stage}" "${config}" exp01_single_pert_stratified_5fold scripts/exp_01_single_pert_stratified_5fold.sh single_pert_stratified_fold real "${fold}"
          launch_fold_job "${prefix}" "${stage}" "${config}" exp04_single_no_mse_5fold scripts/exp_04_single_no_mse_5fold.sh single_no_mse_fold real "${fold}"
          launch_fold_job "${prefix}" "${stage}" "${config}" exp05_single_no_graph_5fold scripts/exp_05_single_no_pdi_5fold.sh single_no_graph_fold zero "${fold}"
          ;;
        stage2)
          launch_fold_job "${prefix}" "${stage}" "${config}" exp03_single_cell_5fold scripts/exp_03_single_cell_5fold.sh single_cell_fold real "${fold}"
          ;;
        stage3)
          launch_fold_job "${prefix}" "${stage}" "${config}" exp02_single_cell_type_5fold scripts/exp_02_single_cell_type_5fold.sh single_cell_type_fold real "${fold}"
          ;;
        stage4)
          launch_fold_job "${prefix}" "${stage}" "${config}" exp06_double_pert_pair_5fold scripts/exp_06_double_pert_pair_5fold.sh double_pert_pair_fold real "${fold}" "${DOUBLE_FIXED_ENV[@]}"
          ;;
        *)
          echo "[error] unknown stage=${stage}" >&2
          exit 2
          ;;
      esac
    done
  done
  wait_for_all_jobs
}

run_tuning_report() {
  local prefix="$1"
  local min_folds="$2"
  local report_path="${LOG_DIR}/${prefix}_param_search_report.md"
  "${PYTHON_BIN}" "${TUNING_REPORT_SCRIPT}" \
    --base-prefix "${prefix}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --stage all \
    --min-folds "${min_folds}" \
    --precision 4 \
    --format markdown | tee "${report_path}"
  "${PYTHON_BIN}" "${TUNING_REPORT_SCRIPT}" \
    --base-prefix "${prefix}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --stage all \
    --min-folds "${min_folds}" \
    --precision 4 \
    --format tsv > "${OUTPUT_DIR}/${prefix}_param_search_report.tsv"
  echo "[report] ${report_path}"
}

promoted_configs() {
  local prefix="$1"
  local stage="$2"
  local min_folds="$3"
  "${PYTHON_BIN}" "${TUNING_REPORT_SCRIPT}" \
    --base-prefix "${prefix}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --stage "${stage}" \
    --min-folds "${min_folds}" \
    --promotion-margin "${PROMOTION_MARGIN}" \
    --emit-promoted-configs
}

selected_config() {
  local prefix="$1"
  local stage="$2"
  local min_folds="$3"
  "${PYTHON_BIN}" "${TUNING_REPORT_SCRIPT}" \
    --base-prefix "${prefix}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --stage "${stage}" \
    --min-folds "${min_folds}" \
    --emit-selected-config
}

stage1_combo_refine_configs() {
  "${PYTHON_BIN}" - "${TUNING_REPORT_SCRIPT}" "${SCREEN_PREFIX}" "${CKPT_DIR}" "${OUTPUT_DIR}" "${SCREEN_MIN_FOLDS}" <<'PY'
import importlib.util
import math
import re
import sys
from argparse import Namespace
from pathlib import Path

script, prefix, ckpt, out, min_folds = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
spec = importlib.util.spec_from_file_location("r", script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
rows = mod.ranked_rows(mod.summarize(Namespace(
    base_prefix=prefix,
    checkpoint_root=ckpt,
    output_root=out,
    stage="stage1",
    format="json",
    precision=4,
    min_folds=min_folds,
    promotion_margin=0.002,
    emit_promoted_configs=False,
    emit_selected_config=False,
)))
rows = [r for r in rows if r.get("stage") == "stage1" and r.get("complete") and r.get("valid_cell_llm")]
rows.sort(key=lambda r: mod.rank_sort_key(r), reverse=True)

def score(r):
    value = r.get("rank_metric")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return value if math.isfinite(value) else -math.inf

def pos_family(config):
    if config.startswith("posw_negpos"):
        return "posw_negpos"
    m = re.match(r"^(posw[0-9]+)", config)
    return m.group(1) if m else None

def lr_token(config):
    if "_lr5e5" in config or config.endswith("lr5e5"):
        return "lr5e5"
    if "_lr1e4" in config or config.endswith("lr1e4"):
        return "lr1e4"
    if "_lr3e4" in config or config.endswith("lr3e4"):
        return "lr3e4"
    return "lr2e4"

def drop_token(config):
    if "_drop010" in config or config.endswith("drop010"):
        return "drop010"
    if "_drop020" in config or config.endswith("drop020"):
        return "drop020"
    if "_drop050" in config or config.endswith("drop050"):
        return "drop050"
    return "drop015"

def best_by(fn, allowed=None):
    best = {}
    for row in rows:
        key = fn(str(row.get("config", "")))
        if key is None:
            continue
        if allowed is not None and key not in allowed:
            continue
        if key not in best or score(row) > score(best[key]):
            best[key] = row
    return [key for key, _ in sorted(best.items(), key=lambda item: score(item[1]), reverse=True)]

pos = best_by(pos_family)[:3] or ["posw10", "posw100", "posw_negpos"]
lrs = best_by(lr_token, {"lr5e5", "lr1e4", "lr2e4", "lr3e4"})[:2] or ["lr2e4", "lr5e5"]
drops = best_by(drop_token, {"drop010", "drop015", "drop020", "drop050"})[:2] or ["drop015", "drop050"]

configs = []
for p in pos:
    for lr in lrs:
        for drop in drops:
            configs.append(f"{p}_{lr}_{drop}")
configs = list(dict.fromkeys(configs))[:12]
print(" ".join(configs))
PY
}

top_stage1_configs_for_stage234() {
  local prefix="$1"
  local min_folds="$2"
  "${PYTHON_BIN}" - "${TUNING_REPORT_SCRIPT}" "${prefix}" "${CKPT_DIR}" "${OUTPUT_DIR}" "${min_folds}" <<'PY'
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

script, prefix, ckpt, out, min_folds = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
spec = importlib.util.spec_from_file_location("r", script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
rows = mod.promotable_rows(mod.summarize(Namespace(
    base_prefix=prefix, checkpoint_root=ckpt, output_root=out, stage="stage1",
    format="json", precision=4, min_folds=min_folds, promotion_margin=0.002,
    emit_promoted_configs=False, emit_selected_config=False,
)), "stage1")
print(" ".join(str(r["config"]) for r in rows[:2]))
PY
}

stage2_configs() {
  if [[ -n "${STAGE2_CONFIGS:-}" ]]; then
    echo "${STAGE2_CONFIGS}"
    return
  fi
  local derived_prefix="${1:-${FULL_PREFIX}}"
  local min_folds="${2:-${FULL_MIN_FOLDS}}"
  local stage1_top
  stage1_top="$(top_stage1_configs_for_stage234 "${derived_prefix}" "${min_folds}" 2>/dev/null || true)"
  echo "base ${stage1_top} covdrop010 covdrop010_lr5e5 covdrop010_drop050 drop020"
}

stage3_configs() {
  if [[ -n "${STAGE3_CONFIGS:-}" ]]; then
    echo "${STAGE3_CONFIGS}"
    return
  fi
  local derived_prefix="${1:-${FULL_PREFIX}}"
  local min_folds="${2:-${FULL_MIN_FOLDS}}"
  local stage1_top
  stage1_top="$(top_stage1_configs_for_stage234 "${derived_prefix}" "${min_folds}" 2>/dev/null || true)"
  echo "base ${stage1_top} covdrop010 covdrop010_lr5e5 covdrop010_drop050 covunk_celltype"
}

stage4_configs() {
  if [[ -n "${STAGE4_CONFIGS:-}" ]]; then
    echo "${STAGE4_CONFIGS}"
    return
  fi
  local derived_prefix="${1:-${FULL_PREFIX}}"
  local min_folds="${2:-${FULL_MIN_FOLDS}}"
  local stage1_top
  stage1_top="$(top_stage1_configs_for_stage234 "${derived_prefix}" "${min_folds}" 2>/dev/null || true)"
  echo "base rank005 dbl_mse050 ${stage1_top} posw10 posw50 posw100 posw_negpos posw10_lr5e5 posw50_lr5e5 posw100_lr5e5 posw10_drop050 posw50_drop050 posw100_drop050"
}

run_stage1_screen() {
  local configs_a configs_b
  configs_a="$(stage1_a_configs)"
  echo "[stage1-A] configs=${configs_a}"
  run_stage_configs "${SCREEN_PREFIX}" stage1 "${SCREEN_FOLDS}" "${configs_a}"
  if [[ "${RUN_TUNING_REPORT}" == "1" ]]; then
    run_tuning_report "${SCREEN_PREFIX}" "${SCREEN_MIN_FOLDS}"
  fi
  configs_b="${STAGE1_B_CONFIGS:-$(stage1_combo_refine_configs)}"
  if [[ -n "${configs_b}" ]]; then
    echo "[stage1-B] configs=${configs_b}"
    run_stage_configs "${SCREEN_PREFIX}" stage1 "${SCREEN_FOLDS}" "${configs_b}"
    if [[ "${RUN_TUNING_REPORT}" == "1" ]]; then
      run_tuning_report "${SCREEN_PREFIX}" "${SCREEN_MIN_FOLDS}"
    fi
  fi
}

run_stage234_screen() {
  local derived_prefix="$1"
  local derived_min_folds="$2"
  if [[ "${STAGES}" == *"stage2"* ]]; then
    run_stage_configs "${SCREEN_PREFIX}" stage2 "${SCREEN_FOLDS}" "$(stage2_configs "${derived_prefix}" "${derived_min_folds}")"
  fi
  if [[ "${STAGES}" == *"stage3"* ]]; then
    run_stage_configs "${SCREEN_PREFIX}" stage3 "${SCREEN_FOLDS}" "$(stage3_configs "${derived_prefix}" "${derived_min_folds}")"
  fi
  if [[ "${STAGES}" == *"stage4"* ]]; then
    run_stage_configs "${SCREEN_PREFIX}" stage4 "${SCREEN_FOLDS}" "$(stage4_configs "${derived_prefix}" "${derived_min_folds}")"
  fi
  if [[ "${RUN_TUNING_REPORT}" == "1" ]]; then
    run_tuning_report "${SCREEN_PREFIX}" "${SCREEN_MIN_FOLDS}"
  fi
}

run_screen() {
  if [[ "${STAGES}" == *"stage1"* ]]; then
    run_stage1_screen
  fi
  run_stage234_screen "${SCREEN_PREFIX}" "${SCREEN_MIN_FOLDS}"
}

run_stage1_full() {
  local configs
  configs="${STAGE1_FULL_CONFIGS:-$(promoted_configs "${SCREEN_PREFIX}" stage1 "${SCREEN_MIN_FOLDS}")}"
  echo "[stage1-full] configs=${configs}"
  run_stage_configs "${FULL_PREFIX}" stage1 "${FULL_FOLDS}" "${configs}"
  if [[ "${RUN_TUNING_REPORT}" == "1" ]]; then
    run_tuning_report "${FULL_PREFIX}" "${FULL_MIN_FOLDS}"
  fi
}

run_stage234_full() {
  local stage configs
  for stage in stage2 stage3 stage4; do
    if [[ "${STAGES}" != *"${stage}"* ]]; then
      continue
    fi
    configs="$(promoted_configs "${SCREEN_PREFIX}" "${stage}" "${SCREEN_MIN_FOLDS}")"
    echo "[${stage}-full] configs=${configs}"
    run_stage_configs "${FULL_PREFIX}" "${stage}" "${FULL_FOLDS}" "${configs}"
  done
  if [[ "${RUN_TUNING_REPORT}" == "1" ]]; then
    run_tuning_report "${FULL_PREFIX}" "${FULL_MIN_FOLDS}"
  fi
}

run_full() {
  if [[ "${STAGES}" == *"stage1"* ]]; then
    run_stage1_full
  fi
  run_stage234_full
}

run_selected_exp() {
  local exp_suffix="$1"
  local script_path="$2"
  local config="$3"
  shift 3
  set_config_env "${config}"
  echo "[selected] exp=${exp_suffix} config=${config}"
  env "${COMMON_ENV[@]}" "$@" "${CONFIG_ENV[@]}" \
    "GPU_IDS=${GPU_ARRAY[0]}" \
    "FOLDS=${FINAL_FOLDS}" \
    "EXP_PREFIX=${SELECTED_PREFIX}_${exp_suffix}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${SELECTED_PREFIX}_runtime_summary.tsv" \
    "REFERENCE_EPOCH_AGG=mean" \
    "REFERENCE_EPOCH_ROUNDING=nearest" \
    "REFERENCE_EPOCH_MIN_COUNT=${FULL_MIN_FOLDS}" \
    bash "${script_path}"
}

run_selected() {
  local stage1_config stage2_config stage3_config stage4_config
  local fold
  stage1_config="${SELECTED_STAGE1_CONFIG:-$(selected_config "${FULL_PREFIX}" stage1 "${FULL_MIN_FOLDS}")}"
  stage2_config="${SELECTED_STAGE2_CONFIG:-$(selected_config "${FULL_PREFIX}" stage2 "${FULL_MIN_FOLDS}")}"
  stage3_config="${SELECTED_STAGE3_CONFIG:-$(selected_config "${FULL_PREFIX}" stage3 "${FULL_MIN_FOLDS}")}"
  stage4_config="${SELECTED_STAGE4_CONFIG:-$(selected_config "${FULL_PREFIX}" stage4 "${FULL_MIN_FOLDS}")}"
  echo "[selected-configs] stage1=${stage1_config}; stage2_exp03=${stage2_config}; stage3_exp02=${stage3_config}; stage4_exp06=${stage4_config}"

  failures=0
  for fold in ${FINAL_FOLDS}; do
    launch_selected_fold_job exp01_single_pert_stratified_5fold scripts/exp_01_single_pert_stratified_5fold.sh "${stage1_config}" single_pert_stratified_fold real "${fold}"
    launch_selected_fold_job exp04_single_no_mse_5fold scripts/exp_04_single_no_mse_5fold.sh "${stage1_config}" single_no_mse_fold real "${fold}"
    launch_selected_fold_job exp05_single_no_graph_5fold scripts/exp_05_single_no_pdi_5fold.sh "${stage1_config}" single_no_graph_fold zero "${fold}"
    launch_selected_fold_job exp02_single_cell_type_5fold scripts/exp_02_single_cell_type_5fold.sh "${stage3_config}" single_cell_type_fold real "${fold}"
    launch_selected_fold_job exp03_single_cell_5fold scripts/exp_03_single_cell_5fold.sh "${stage2_config}" single_cell_fold real "${fold}"
    launch_selected_fold_job exp06_double_pert_pair_5fold scripts/exp_06_double_pert_pair_5fold.sh "${stage4_config}" double_pert_pair_fold real "${fold}" "${DOUBLE_FIXED_ENV[@]}"
  done
  wait_for_all_jobs

  run_selected_exp exp07_extra_single_all_train_infer scripts/exp_07_extra_single_all_train_infer.sh "${stage1_config}" \
    "REFERENCE_5FOLD_CKPT_PATH=${CKPT_DIR}/${SELECTED_PREFIX}_exp01_single_pert_stratified_5fold"
  run_selected_exp exp08_extra_double_all_train_infer scripts/exp_08_extra_double_all_train_infer.sh "${stage4_config}" \
    "${DOUBLE_FIXED_ENV[@]}" \
    "REFERENCE_5FOLD_CKPT_PATH=${CKPT_DIR}/${SELECTED_PREFIX}_exp06_double_pert_pair_5fold"

  if [[ "${RUN_REPORT}" == "1" ]]; then
    local report_markdown="${LOG_DIR}/${SELECTED_PREFIX}_cell_drug_dose_time_eval.md"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/report_cell_drug_time_eval.py \
      --prefix "${SELECTED_PREFIX}" \
      --checkpoint-root "${CKPT_DIR}" \
      --output-root "${OUTPUT_DIR}" \
      --materialize-fold-predictions \
      --device "${REPORT_DEVICE}" \
      --infer-batch-size "${REPORT_INFER_BATCH_SIZE}" \
      --num-workers "${REPORT_NUM_WORKERS}" \
      --precision 4 \
      --csv-out "${OUTPUT_DIR}/${SELECTED_PREFIX}_cell_drug_dose_time_eval.csv" \
      --json-out "${OUTPUT_DIR}/${SELECTED_PREFIX}_cell_drug_dose_time_eval.json" \
      --format markdown | tee "${report_markdown}"
    echo "[report] ${report_markdown}"
  fi
}

run_all() {
  run_stage1_screen
  run_stage1_full
  run_stage234_screen "${FULL_PREFIX}" "${FULL_MIN_FOLDS}"
  run_stage234_full
  run_selected
}

echo "[settings] RUN_MODE=${RUN_MODE}; BASE_PREFIX=${BASE_PREFIX}; SCREEN_PREFIX=${SCREEN_PREFIX}; FULL_PREFIX=${FULL_PREFIX}; SELECTED_PREFIX=${SELECTED_PREFIX}"
echo "[settings] GPU_IDS=${GPU_IDS}; JOBS_PER_GPU=${JOBS_PER_GPU}; MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}; GPU_SLOT_CAPACITY=${GPU_SLOT_CAPACITY}; STAGES=${STAGES}; SMOKE_TEST=${SMOKE_TEST}"
init_gpu_job_summary
"${PYTHON_BIN}" -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py "${TUNING_REPORT_SCRIPT}"
bash -n scripts/ptv3_experiment_common.sh \
  scripts/exp_01_single_pert_stratified_5fold.sh \
  scripts/exp_02_single_cell_type_5fold.sh \
  scripts/exp_03_single_cell_5fold.sh \
  scripts/exp_04_single_no_mse_5fold.sh \
  scripts/exp_05_single_no_pdi_5fold.sh \
  scripts/exp_06_double_pert_pair_5fold.sh \
  scripts/exp_07_extra_single_all_train_infer.sh \
  scripts/exp_08_extra_double_all_train_infer.sh
if [[ "${RUN_INPUT_VALIDATION}" == "1" ]]; then
  validate_inputs
fi

case "${RUN_MODE}" in
  screen) run_screen ;;
  full) run_full ;;
  selected) run_selected ;;
  all) run_all ;;
  *)
    echo "[error] unknown RUN_MODE=${RUN_MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

echo "[done] ptv01-08 posweight combo tuning runner finished; RUN_MODE=${RUN_MODE}; BASE_PREFIX=${BASE_PREFIX}"
