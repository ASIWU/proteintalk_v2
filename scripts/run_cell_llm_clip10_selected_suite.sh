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
BASE_PREFIX="${BASE_PREFIX:-20260604_cell_llm_dose_clip10_selected_v1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${BASE_PREFIX}_runtime_summary.tsv}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz}"
RUN_REPORT="${RUN_REPORT:-1}"
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
  "CELL_LLM_MODE=frozen"
  "CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
  "CELL_LLM_FUSION_MODE=${CELL_LLM_FUSION_MODE:-covariate}"
  "CELL_LLM_CONDITION_SCALE=${CELL_LLM_CONDITION_SCALE:-0.0}"
  "CELL_LLM_LOGIT_SCALE=${CELL_LLM_LOGIT_SCALE:-0.0}"
  "CELL_LLM_DROPOUT=${CELL_LLM_DROPOUT:-0.0}"
  "GPU_IDS=${GPU_IDS}"
  "FOLDS=${FOLDS}"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=${LOGGER_BACKEND:-none}"
  "LOG_TO_WANDB=${LOG_TO_WANDB:-0}"
  "PROGRESS_BAR=${PROGRESS_BAR:-0}"
  "RUN_DATA_VALIDATION=${RUN_DATA_VALIDATION:-0}"
  "RUN_INFERENCE=1"
  "RUN_PREFLIGHT=${RUN_PREFLIGHT:-1}"
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
  "GRAPH_FEATURE_MODE=real"
  "GRAPH_STRUCTURAL_RP=1"
  "GRAPH_DRUG_CONCAT=1"
  "GRAPH_LOGIT_SCALE=2.0"
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
  "CKPT_DIR=${CKPT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}"
)

SINGLE_MSE050=(
  "LEARNING_RATE=2e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.15"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.50"
)

SINGLE_COVDROP010=(
  "LEARNING_RATE=2e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.15"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.25"
  "COVARIATE_UNK_DROPOUT=0.10"
)

DOUBLE_DROP020=(
  "LEARNING_RATE=2e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.20"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.25"
  "MSE_INACTIVE_LABEL_WEIGHT=0.20"
  "PAIR_FUSION_MODE=dual"
  "PAIR_TYPE_FEATURES=1"
  "USE_DDI=1"
  "GRAPH_PAIR_ADD_SCALE=0.5"
)

run_exp() {
  local exp_suffix="$1"
  local script_path="$2"
  local config_name="$3"
  shift 3
  local exp_prefix="${BASE_PREFIX}_${exp_suffix}"
  echo "[run] ${exp_suffix} config=${config_name} folds=${FOLDS}"
  env "${COMMON_ENV[@]}" "$@" "EXP_PREFIX=${exp_prefix}" bash "${script_path}"
}

echo "[settings] BASE_PREFIX=${BASE_PREFIX}; FOLDS=${FOLDS}; GPU_IDS=${GPU_IDS}; CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
"${PYTHON_BIN}" -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py utils/11_build_cell_llm_embeddings.py
bash -n scripts/ptv3_experiment_common.sh
validate_inputs

run_exp exp01_single_pert_stratified_5fold scripts/exp_01_single_pert_stratified_5fold.sh mse050 "${SINGLE_MSE050[@]}"
run_exp exp04_single_no_mse_5fold scripts/exp_04_single_no_mse_5fold.sh mse050 "${SINGLE_MSE050[@]}"
run_exp exp05_single_no_graph_5fold scripts/exp_05_single_no_pdi_5fold.sh mse050 "${SINGLE_MSE050[@]}"
run_exp exp02_single_cell_type_5fold scripts/exp_02_single_cell_type_5fold.sh covdrop010 "${SINGLE_COVDROP010[@]}"
run_exp exp03_single_cell_5fold scripts/exp_03_single_cell_5fold.sh covdrop010 "${SINGLE_COVDROP010[@]}"
run_exp exp06_double_pert_pair_5fold scripts/exp_06_double_pert_pair_5fold.sh drop020 "${DOUBLE_DROP020[@]}"
run_exp exp07_extra_single_all_train_infer scripts/exp_07_extra_single_all_train_infer.sh mse050 \
  "${SINGLE_MSE050[@]}" \
  "REFERENCE_5FOLD_CKPT_PATH=${CKPT_DIR}/${BASE_PREFIX}_exp01_single_pert_stratified_5fold"
run_exp exp08_extra_double_all_train_infer scripts/exp_08_extra_double_all_train_infer.sh drop020 \
  "${DOUBLE_DROP020[@]}" \
  "REFERENCE_5FOLD_CKPT_PATH=${CKPT_DIR}/${BASE_PREFIX}_exp06_double_pert_pair_5fold"

if [[ "${RUN_REPORT}" == "1" ]]; then
  report_markdown="${LOG_DIR}/${BASE_PREFIX}_cell_drug_dose_time_eval.md"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/report_cell_drug_time_eval.py \
    --prefix "${BASE_PREFIX}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --materialize-fold-predictions \
    --device "${REPORT_DEVICE}" \
    --infer-batch-size "${REPORT_INFER_BATCH_SIZE}" \
    --num-workers "${REPORT_NUM_WORKERS}" \
    --csv-out "${OUTPUT_DIR}/${BASE_PREFIX}_cell_drug_dose_time_eval.csv" \
    --json-out "${OUTPUT_DIR}/${BASE_PREFIX}_cell_drug_dose_time_eval.json" \
    --format markdown | tee "${report_markdown}"
  echo "[report] ${report_markdown}"
fi

echo "[done] corrected Cell LLM clip10 selected suite completed for BASE_PREFIX=${BASE_PREFIX}"
