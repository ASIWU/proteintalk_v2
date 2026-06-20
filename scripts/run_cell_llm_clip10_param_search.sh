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
  RUN_MODE=screen   bash scripts/run_cell_llm_clip10_param_search.sh
  RUN_MODE=full     bash scripts/run_cell_llm_clip10_param_search.sh
  RUN_MODE=selected bash scripts/run_cell_llm_clip10_param_search.sh

Modes:
  screen   Run 3-fold parameter screens for stage1-stage4.
  full     Promote top screen candidates and run full 5-fold under FULL_PREFIX.
  selected Run final exp01-exp08 suite from full 5-fold selected configs.
  all      Run screen, full, then selected.

Useful overrides:
  SCREEN_PREFIX, FULL_PREFIX, SELECTED_PREFIX, GPU_IDS, STAGES,
  STAGE1_CONFIGS, STAGE2_CONFIGS, STAGE3_CONFIGS, STAGE4_CONFIGS,
  SELECTED_STAGE1_CONFIG, SELECTED_STAGE2_CONFIG, SELECTED_STAGE3_CONFIG,
  SELECTED_STAGE4_CONFIG.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
RUN_MODE="${RUN_MODE:-screen}"
SCREEN_PREFIX="${SCREEN_PREFIX:-${BASE_PREFIX:-20260604_cell_llm_clip10_tune_v1}}"
FULL_PREFIX="${FULL_PREFIX:-${SCREEN_PREFIX}_full}"
SELECTED_PREFIX="${SELECTED_PREFIX:-20260604_cell_llm_clip10_tuned_selected_v1}"
SCREEN_FOLDS="${SCREEN_FOLDS:-0 2 4}"
FULL_FOLDS="${FULL_FOLDS:-0 1 2 3 4}"
FINAL_FOLDS="${FINAL_FOLDS:-0 1 2 3 4}"
STAGES="${STAGES:-stage1 stage2 stage3 stage4}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz}"
RUN_TUNING_REPORT="${RUN_TUNING_REPORT:-1}"
TUNING_REPORT_SCRIPT="${TUNING_REPORT_SCRIPT:-scripts/report_cell_llm_clip10_param_search.py}"
RUN_REPORT="${RUN_REPORT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SCREEN_MIN_FOLDS="${SCREEN_MIN_FOLDS:-3}"
FULL_MIN_FOLDS="${FULL_MIN_FOLDS:-5}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda:0}"
REPORT_INFER_BATCH_SIZE="${REPORT_INFER_BATCH_SIZE:-512}"
REPORT_NUM_WORKERS="${REPORT_NUM_WORKERS:-4}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

validate_inputs() {
  "${PYTHON_BIN}" - "${CELL_LLM_EMBEDDING_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

embedding_path = Path(sys.argv[1])
if not embedding_path.exists():
    raise SystemExit(f"missing Cell LLM embedding artifact: {embedding_path}")

payload = np.load(embedding_path, allow_pickle=False)
if "embedding_matrix" not in payload.files:
    raise SystemExit(f"{embedding_path} must contain embedding_matrix")
matrix = payload["embedding_matrix"]
if matrix.shape != (74, 4096):
    raise SystemExit(f"{embedding_path} expected shape (74, 4096), got {matrix.shape}")
if not np.allclose(matrix[0], 0.0):
    raise SystemExit(f"{embedding_path} row 0 must be the zero vector")
nonzero = matrix[1:]
if not np.isfinite(nonzero).all() or np.any(np.all(np.isclose(nonzero, 0.0), axis=1)):
    raise SystemExit(f"{embedding_path} non-no rows must be finite and nonzero")

sidecar_path = embedding_path.with_suffix(".json")
if not sidecar_path.exists():
    raise SystemExit(f"missing Cell LLM sidecar: {sidecar_path}")
sidecar_text = sidecar_path.read_text(encoding="utf-8")
for forbidden in ("cell_" + "type_llm", "cell_" + "type_count"):
    if forbidden in sidecar_text:
        raise SystemExit(f"{sidecar_path} contains forbidden old key: {forbidden}")

training_ready_root = Path("data/training_ready/ptv3")
meta = json.load((training_ready_root / "global_meta.json").open())
if len(meta["value_to_index"]["Cell"]) != 74:
    raise SystemExit('global_meta value_to_index["Cell"] must contain 74 values')
if int(meta["value_to_index"]["Cell"].get("no", -1)) != 0:
    raise SystemExit('global_meta value_to_index["Cell"]["no"] must be 0')

tasks = [
    "ptv3_main_singledrug",
    "ptv3_main_doubledrug",
    "ptv3_extra_singledrug_mat1_480_faims",
    "ptv3_extra_singledrug_mat1_qe",
    "ptv3_extra_singledrug_mat2_480_faims",
    "ptv3_extra_singledrug_mat2_qe",
    "ptv3_extra_singledrug_mat3_qe",
    "ptv3_extra_singledrug_mat4_qe",
    "ptv3_extra_doubledrug_nature",
    "ptv3_extra_doubledrug_nc",
    "ptv3_extra_doubledrug_guomics",
]
for task in tasks:
    task_dir = training_ready_root / "tasks" / task
    table_path = task_dir / "feature_table.parquet"
    if table_path.exists():
        df = pd.read_parquet(table_path, columns=["Cell_index", "pert_dose1", "pert_dose2"])
    else:
        csv_path = task_dir / "feature_table.csv"
        if not csv_path.exists():
            raise SystemExit(f"missing feature_table for {task}")
        df = pd.read_csv(csv_path, usecols=["Cell_index", "pert_dose1", "pert_dose2"], low_memory=False)
    max_cell_index = pd.to_numeric(df["Cell_index"], errors="coerce").fillna(0).astype(int).max()
    if int(max_cell_index) >= matrix.shape[0]:
        raise SystemExit(f"{task}: max Cell_index {max_cell_index} exceeds embedding rows {matrix.shape[0]}")
    for field in ("pert_dose1", "pert_dose2"):
        numeric = pd.to_numeric(df[field], errors="coerce").dropna()
        if not numeric.empty and (float(numeric.max()) > 10.0 or float(numeric.min()) < 0.0):
            raise SystemExit(
                f"{task}: {field} outside clip10 range, min={float(numeric.min())}, max={float(numeric.max())}"
            )
print("[validate] Cell LLM artifact and clip10 dose checks passed")
PY
}

COMMON_ENV=(
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
  "ALLOW_EXISTING_RUN=${ALLOW_EXISTING_RUN:-0}"
  "CELL_LLM_MODE=frozen"
  "CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
  "CELL_LLM_FUSION_MODE=${CELL_LLM_FUSION_MODE:-covariate}"
  "CELL_LLM_CONDITION_SCALE=${CELL_LLM_CONDITION_SCALE:-0.0}"
  "CELL_LLM_LOGIT_SCALE=${CELL_LLM_LOGIT_SCALE:-0.0}"
  "CELL_LLM_DROPOUT=${CELL_LLM_DROPOUT:-0.0}"
  "CELL_TYPE_LLM_MODE=${CELL_TYPE_LLM_MODE:-off}"
  "CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH:-}"
  "CELL_TYPE_LLM_FUSION_MODE=${CELL_TYPE_LLM_FUSION_MODE:-covariate}"
  "CELL_TYPE_LLM_CONDITION_SCALE=${CELL_TYPE_LLM_CONDITION_SCALE:-0.0}"
  "CELL_TYPE_LLM_LOGIT_SCALE=${CELL_TYPE_LLM_LOGIT_SCALE:-0.0}"
  "CELL_TYPE_LLM_DROPOUT=${CELL_TYPE_LLM_DROPOUT:-0.0}"
  "GPU_IDS=${GPU_IDS}"
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
  "INFER_BATCH_SIZE=256"
  "PRECISION=${PRECISION:-bf16-mixed}"
  "NUM_WORKERS=${NUM_WORKERS:-4}"
  "MAX_EPOCHS=${MAX_EPOCHS:-50}"
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
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
  "CKPT_DIR=${CKPT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
)

CONFIG_ENV=()
set_config_env() {
  local name="$1"
  CONFIG_ENV=()
  case "${name}" in
    base)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    mse050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse075)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.75")
      ;;
    mse100)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=1.00")
      ;;
    mse050_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse075_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.75")
      ;;
    mse100_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=1.00")
      ;;
    mse050_lr1e4)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_lr3e4)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_warmdecay)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_WEIGHT_SCHEDULE=warmup_decay" "MSE_FINAL_WEIGHT_MULTIPLIER=0.25")
      ;;
    mse050_pre1)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_PRETRAIN_EPOCHS=1" "MSE_PRETRAIN_BCE_WEIGHT=0.0")
      ;;
    mse050_pre2)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_PRETRAIN_EPOCHS=2" "MSE_PRETRAIN_BCE_WEIGHT=0.0")
      ;;
    mse050_target_pdi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=pdi")
      ;;
    mse050_target_ppi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=pdi_ppi")
      ;;
    covdrop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covdrop010_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covdrop010_drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covdrop010_bs128)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=128" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covdrop010_lr1e4)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    covdrop010_lr3e4)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    bs128)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=128" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr1e4)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr3e4)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    ctrl_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "CONTROL_EXPRESSION_DROPOUT=0.10")
      ;;
    covunk_cell)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=Cell")
      ;;
    covunk_celltype)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=cell_type")
      ;;
    dbl_mse010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    dbl_mse050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.50")
      ;;
    rank005)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "RANKING_LOSS_WEIGHT=0.05")
      ;;
    drop020_lr1e4)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    drop020_lr3e4)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    drop020_bs128)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=128" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    drop020_mseinactive010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    drop020_mseinactive050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.50")
      ;;
    *)
      echo "[error] unknown config: ${name}" >&2
      exit 2
      ;;
  esac
}

missing_folds() {
  local exp_prefix="$1"
  local fold_suffix="$2"
  local expected_graph_mode="$3"
  local requested_folds="$4"
  "${PYTHON_BIN}" - "${CKPT_DIR}" "${exp_prefix}" "${fold_suffix}" "${expected_graph_mode}" "${requested_folds}" <<'PY'
import json
import os
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
exp_prefix = sys.argv[2]
fold_suffix = sys.argv[3]
expected_graph_mode = sys.argv[4]
requested_folds = sys.argv[5].split()
allow_cell_type_llm = os.environ.get("CELL_TYPE_LLM_MODE", "off") != "off"
missing = []
for fold in requested_folds:
    manifest_path = ckpt_dir / f"{exp_prefix}_{fold_suffix}{fold}" / "run_manifest.json"
    if not manifest_path.exists():
        missing.append(fold)
        continue
    text = manifest_path.read_text(encoding="utf-8")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        missing.append(fold)
        continue
    if manifest.get("run_status") != "fit_completed" or manifest.get("test_status") != "test_completed":
        missing.append(fold)
        continue
    errors = []
    if manifest.get("cell_llm_mode") != "frozen":
        errors.append(f"cell_llm_mode={manifest.get('cell_llm_mode')!r}")
    summary = manifest.get("cell_llm_summary")
    if not isinstance(summary, dict) or summary.get("embedding_rows") != 74:
        errors.append("cell_llm_summary.embedding_rows!=74")
    if allow_cell_type_llm:
        if manifest.get("cell_type_llm_mode") != "frozen":
            errors.append(f"cell_type_llm_mode={manifest.get('cell_type_llm_mode')!r}")
        cell_type_summary = manifest.get("cell_type_llm_summary")
        if not isinstance(cell_type_summary, dict) or cell_type_summary.get("embedding_rows") != 14:
            errors.append("cell_type_llm_summary.embedding_rows!=14")
    elif "cell_type_llm" in text:
        errors.append("contains old cell_type_llm key")
    if manifest.get("graph_feature_mode") != expected_graph_mode:
        errors.append(f"graph_feature_mode={manifest.get('graph_feature_mode')!r}")
    if errors:
        raise SystemExit(f"{manifest_path}: " + "; ".join(errors))
print(" ".join(missing))
PY
}

run_fold_script() {
  local prefix="$1"
  local stage="$2"
  local config="$3"
  local exp_suffix="$4"
  local script_path="$5"
  local fold_suffix="$6"
  local expected_graph_mode="$7"
  local requested_folds="$8"
  shift 8
  local run_prefix="${prefix}_${stage}_${config}"
  local exp_prefix="${run_prefix}_${exp_suffix}"
  local time_summary="${LOG_DIR}/${run_prefix}_runtime_summary.tsv"
  local run_folds="${requested_folds}"
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    run_folds="$(missing_folds "${exp_prefix}" "${fold_suffix}" "${expected_graph_mode}" "${requested_folds}")"
    if [[ -z "${run_folds}" ]]; then
      echo "[skip] stage=${stage} config=${config} exp=${exp_suffix}; folds already complete"
      return
    fi
  fi
  set_config_env "${config}"
  echo "[run] prefix=${prefix} stage=${stage} config=${config} exp=${exp_suffix} folds=${run_folds}"
  env "${COMMON_ENV[@]}" "$@" "${CONFIG_ENV[@]}" \
    "FOLDS=${run_folds}" \
    "EXP_PREFIX=${exp_prefix}" \
    "TIME_SUMMARY_PATH=${time_summary}" \
    bash "${script_path}"
}

DOUBLE_FIXED_ENV=(
  "PAIR_FUSION_MODE=dual"
  "PAIR_TYPE_FEATURES=1"
  "MSE_INACTIVE_LABEL_WEIGHT=0.2"
  "USE_DDI=1"
  "GRAPH_PAIR_ADD_SCALE=0.5"
)

default_stage_configs() {
  local stage="$1"
  case "${stage}" in
    stage1)
      echo "base mse050 mse075 mse100 mse050_drop010 mse050_drop020 mse075_drop010 mse100_drop010 mse050_lr1e4 mse050_lr3e4 mse050_warmdecay mse050_pre1 mse050_pre2 mse050_target_pdi mse050_target_ppi"
      ;;
    stage2)
      echo "base covdrop010 drop010 drop020 bs128 lr1e4 lr3e4 ctrl_drop010 covunk_cell covdrop010_drop010 covdrop010_drop020 covdrop010_bs128 covdrop010_lr1e4 covdrop010_lr3e4"
      ;;
    stage3)
      echo "base covdrop010 drop010 drop020 bs128 lr1e4 lr3e4 ctrl_drop010 covunk_celltype covdrop010_drop010 covdrop010_drop020 covdrop010_bs128 covdrop010_lr1e4 covdrop010_lr3e4"
      ;;
    stage4)
      echo "base drop010 drop020 bs128 lr1e4 lr3e4 dbl_mse010 dbl_mse050 rank005 drop020_lr1e4 drop020_lr3e4 drop020_bs128 drop020_mseinactive010 drop020_mseinactive050"
      ;;
    *)
      echo "[error] unknown stage: ${stage}" >&2
      exit 2
      ;;
  esac
}

explicit_stage_configs() {
  local stage="$1"
  local env_name
  env_name="$(echo "${stage}" | tr '[:lower:]' '[:upper:]')_CONFIGS"
  echo "${!env_name-}"
}

promoted_stage_configs() {
  local stage="$1"
  "${PYTHON_BIN}" "${TUNING_REPORT_SCRIPT}" \
    --base-prefix "${SCREEN_PREFIX}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --stage "${stage}" \
    --min-folds "${SCREEN_MIN_FOLDS}" \
    --emit-promoted-configs
}

selected_stage_config() {
  local stage="$1"
  local env_name="SELECTED_$(echo "${stage}" | tr '[:lower:]' '[:upper:]')_CONFIG"
  local explicit="${!env_name-}"
  if [[ -n "${explicit}" ]]; then
    echo "${explicit}"
    return
  fi
  "${PYTHON_BIN}" "${TUNING_REPORT_SCRIPT}" \
    --base-prefix "${FULL_PREFIX}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --stage "${stage}" \
    --min-folds "${FULL_MIN_FOLDS}" \
    --emit-selected-config
}

stage_configs_for_mode() {
  local stage="$1"
  local mode="$2"
  local explicit
  explicit="$(explicit_stage_configs "${stage}")"
  if [[ -n "${explicit}" ]]; then
    echo "${explicit}"
    return
  fi
  if [[ "${mode}" == "full" ]]; then
    promoted_stage_configs "${stage}"
    return
  fi
  default_stage_configs "${stage}"
}

run_stage() {
  local prefix="$1"
  local stage="$2"
  local folds="$3"
  local configs="$4"
  local config
  for config in ${configs}; do
    case "${stage}" in
      stage1)
        run_fold_script "${prefix}" "${stage}" "${config}" exp01_single_pert_stratified_5fold scripts/exp_01_single_pert_stratified_5fold.sh single_pert_stratified_fold real "${folds}"
        run_fold_script "${prefix}" "${stage}" "${config}" exp04_single_no_mse_5fold scripts/exp_04_single_no_mse_5fold.sh single_no_mse_fold real "${folds}"
        run_fold_script "${prefix}" "${stage}" "${config}" exp05_single_no_graph_5fold scripts/exp_05_single_no_pdi_5fold.sh single_no_graph_fold zero "${folds}"
        ;;
      stage2)
        run_fold_script "${prefix}" "${stage}" "${config}" exp03_single_cell_5fold scripts/exp_03_single_cell_5fold.sh single_cell_fold real "${folds}"
        ;;
      stage3)
        run_fold_script "${prefix}" "${stage}" "${config}" exp02_single_cell_type_5fold scripts/exp_02_single_cell_type_5fold.sh single_cell_type_fold real "${folds}"
        ;;
      stage4)
        run_fold_script "${prefix}" "${stage}" "${config}" exp06_double_pert_pair_5fold scripts/exp_06_double_pert_pair_5fold.sh double_pert_pair_fold real "${folds}" "${DOUBLE_FIXED_ENV[@]}"
        ;;
      *)
        echo "[error] unknown stage: ${stage}" >&2
        exit 2
        ;;
    esac
  done
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
    --format markdown | tee "${report_path}"
  echo "[report] ${report_path}"
}

run_screen() {
  local stage
  echo "[settings] SCREEN_PREFIX=${SCREEN_PREFIX}; SCREEN_FOLDS=${SCREEN_FOLDS}; STAGES=${STAGES}"
  for stage in ${STAGES}; do
    run_stage "${SCREEN_PREFIX}" "${stage}" "${SCREEN_FOLDS}" "$(stage_configs_for_mode "${stage}" screen)"
  done
  if [[ "${RUN_TUNING_REPORT}" == "1" ]]; then
    run_tuning_report "${SCREEN_PREFIX}" "${SCREEN_MIN_FOLDS}"
  fi
}

run_full() {
  local stage
  echo "[settings] SCREEN_PREFIX=${SCREEN_PREFIX}; FULL_PREFIX=${FULL_PREFIX}; FULL_FOLDS=${FULL_FOLDS}; STAGES=${STAGES}"
  for stage in ${STAGES}; do
    run_stage "${FULL_PREFIX}" "${stage}" "${FULL_FOLDS}" "$(stage_configs_for_mode "${stage}" full)"
  done
  if [[ "${RUN_TUNING_REPORT}" == "1" ]]; then
    run_tuning_report "${FULL_PREFIX}" "${FULL_MIN_FOLDS}"
  fi
}

run_selected_exp() {
  local exp_suffix="$1"
  local script_path="$2"
  local config="$3"
  shift 3
  local exp_prefix="${SELECTED_PREFIX}_${exp_suffix}"
  set_config_env "${config}"
  echo "[run] selected exp=${exp_suffix} config=${config} folds=${FINAL_FOLDS}"
  env "${COMMON_ENV[@]}" "$@" "${CONFIG_ENV[@]}" \
    "FOLDS=${FINAL_FOLDS}" \
    "EXP_PREFIX=${exp_prefix}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${SELECTED_PREFIX}_runtime_summary.tsv" \
    "REFERENCE_EPOCH_AGG=mean" \
    "REFERENCE_EPOCH_ROUNDING=nearest" \
    "REFERENCE_EPOCH_MIN_COUNT=5" \
    bash "${script_path}"
}

run_selected() {
  local stage1_config
  local stage2_config
  local stage3_config
  local stage4_config
  stage1_config="$(selected_stage_config stage1)"
  stage2_config="$(selected_stage_config stage2)"
  stage3_config="$(selected_stage_config stage3)"
  stage4_config="$(selected_stage_config stage4)"

  echo "[settings] SELECTED_PREFIX=${SELECTED_PREFIX}; FULL_PREFIX=${FULL_PREFIX}"
  echo "[selected] stage1=${stage1_config}; stage2_exp03=${stage2_config}; stage3_exp02=${stage3_config}; stage4_exp06=${stage4_config}"

  run_selected_exp exp01_single_pert_stratified_5fold scripts/exp_01_single_pert_stratified_5fold.sh "${stage1_config}"
  run_selected_exp exp04_single_no_mse_5fold scripts/exp_04_single_no_mse_5fold.sh "${stage1_config}"
  run_selected_exp exp05_single_no_graph_5fold scripts/exp_05_single_no_pdi_5fold.sh "${stage1_config}"
  run_selected_exp exp02_single_cell_type_5fold scripts/exp_02_single_cell_type_5fold.sh "${stage3_config}"
  run_selected_exp exp03_single_cell_5fold scripts/exp_03_single_cell_5fold.sh "${stage2_config}"
  run_selected_exp exp06_double_pert_pair_5fold scripts/exp_06_double_pert_pair_5fold.sh "${stage4_config}" "${DOUBLE_FIXED_ENV[@]}"
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
      --csv-out "${OUTPUT_DIR}/${SELECTED_PREFIX}_cell_drug_dose_time_eval.csv" \
      --json-out "${OUTPUT_DIR}/${SELECTED_PREFIX}_cell_drug_dose_time_eval.json" \
      --format markdown | tee "${report_markdown}"
    echo "[report] ${report_markdown}"
  fi
}

echo "[settings] RUN_MODE=${RUN_MODE}; GPU_IDS=${GPU_IDS}; CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
"${PYTHON_BIN}" -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py "${TUNING_REPORT_SCRIPT}"
bash -n scripts/ptv3_experiment_common.sh
validate_inputs

case "${RUN_MODE}" in
  screen)
    run_screen
    ;;
  full)
    run_full
    ;;
  selected)
    run_selected
    ;;
  all)
    run_screen
    run_full
    run_selected
    ;;
  *)
    echo "[error] unknown RUN_MODE=${RUN_MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

echo "[done] corrected Cell LLM clip10 parameter-search runner finished; RUN_MODE=${RUN_MODE}"
