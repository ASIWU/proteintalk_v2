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

REQUESTED_BASE_PREFIX="${BASE_PREFIX:-20260610_ptv1_cell_celltype_llm_exp13_posweight_v1_mse050_target_pdi}"
POSITIVE_WEIGHT_GRID="${POSITIVE_WEIGHT_GRID:-posw0p5:0.5 posw1:1 posw10:10 posw20:20 posw50:50 posw100:100 posw200:200 posw500:500 posw_negpos:neg/pos}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv1_cell_celltype_posweight}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz}"
RUN_EXP11="${RUN_EXP11:-1}"
RUN_EXP13_VALID="${RUN_EXP13_VALID:-1}"
RUN_EXP13_ORACLE="${RUN_EXP13_ORACLE:-1}"
RUN_EXP11_ORACLE_AUDIT="${RUN_EXP11_ORACLE_AUDIT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
STATIC_CHECKS="${STATIC_CHECKS:-1}"
RUN_DATA_VALIDATION="${RUN_DATA_VALIDATION:-1}"
REPORT_AFTER="${REPORT_AFTER:-1}"
REPORT_STRICT="${REPORT_STRICT:-1}"
EXPECTED_EXP11_TEST_ROWS="${EXPECTED_EXP11_TEST_ROWS:-799}"
EXPECTED_EXTRA_ROWS="${EXPECTED_EXTRA_ROWS:-218}"
EXP11_SAVE_TOP_K="${EXP11_SAVE_TOP_K:--1}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

CANDIDATES=()
REQUESTED_WEIGHTS=()

read_positive_weight_grid() {
  local item
  local candidate
  local requested
  for item in ${POSITIVE_WEIGHT_GRID}; do
    if [[ "${item}" != *:* ]]; then
      echo "[error] invalid POSITIVE_WEIGHT_GRID item: ${item}; expected candidate:requested" >&2
      exit 2
    fi
    candidate="${item%%:*}"
    requested="${item#*:}"
    CANDIDATES+=("${candidate}")
    REQUESTED_WEIGHTS+=("${requested}")
  done
  if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
    echo "[error] POSITIVE_WEIGHT_GRID is empty" >&2
    exit 2
  fi
}

resolve_neg_pos_weight() {
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import pickle

import pandas as pd

from dataset.training_ready_fast_dataset import encode_response_label

task_dir = Path("data/training_ready/ptv1/tasks/ptv1_aivc")
split_dir = Path("data/training_ready/ptv1/splits/ptv1_aivc")
df = pd.read_parquet(task_dir / "feature_table.parquet")
with (split_dir / "train_indices_fixed_experiment_type.pkl").open("rb") as handle:
    indices = pickle.load(handle)
positive = 0
negative = 0
missing = 0
for idx in indices:
    label, missing_mask = encode_response_label(df.at[int(idx), "PRISM1st_label_total"])
    if float(missing_mask) >= 0.5:
        missing += 1
    elif int(label) == 1:
        positive += 1
    else:
        negative += 1
if positive <= 0 or negative <= 0:
    raise SystemExit(f"cannot resolve neg/pos positive weight: positive={positive}, negative={negative}, missing={missing}")
print(f"{negative}\t{positive}\t{negative / positive:.10f}")
PY
}

prefix_complete() {
  local prefix="$1"
  CANDIDATE_LIST="${CANDIDATES[*]}" \
  CHECK_PREFIX="${prefix}" \
  CKPT_DIR="${CKPT_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  EXPECTED_EXP11_TEST_ROWS="${EXPECTED_EXP11_TEST_ROWS}" \
  EXPECTED_EXTRA_ROWS="${EXPECTED_EXTRA_ROWS}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import math
import os
import re
from pathlib import Path

prefix = os.environ["CHECK_PREFIX"]
candidates = os.environ["CANDIDATE_LIST"].split()
ckpt_root = Path(os.environ["CKPT_DIR"])
output_root = Path(os.environ["OUTPUT_DIR"])
expected_exp11 = int(os.environ["EXPECTED_EXP11_TEST_ROWS"])
expected_extra = int(os.environ["EXPECTED_EXTRA_ROWS"])

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def completed_train(name: str) -> bool:
    path = ckpt_root / name / "run_manifest.json"
    if not path.exists():
        return False
    return load_json(path).get("run_status") == "fit_completed"

def completed_infer(name: str, task: str, expected: int, split_strategy: str) -> bool:
    root = output_root / name / task
    manifest_path = root / "run_manifest.json"
    metrics_path = root / "metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        return False
    manifest = load_json(manifest_path)
    return (
        manifest.get("dataset_group") == "ptv1"
        and manifest.get("task_name") == task
        and manifest.get("split_strategy") == split_strategy
        and int(manifest.get("n_predictions") or -1) == expected
    )

def best_oracle_epoch(candidate: str) -> int | None:
    run_prefix = f"{prefix}_{candidate}"
    pattern = re.compile(rf"^{re.escape(run_prefix)}_extra_oracle_epoch([0-9]+)_from_exp11$")
    best_epoch = None
    best_score = -math.inf
    for metrics_path in output_root.glob(f"{run_prefix}_extra_oracle_epoch*_from_exp11/ptv1_extra_singledrug/metrics.json"):
        match = pattern.match(metrics_path.parent.parent.name)
        if not match:
            continue
        metrics = load_json(metrics_path).get("task", {})
        try:
            score = float(metrics.get("auprc"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = int(match.group(1))
    return best_epoch

for candidate in candidates:
    run_prefix = f"{prefix}_{candidate}"
    if not completed_train(f"{run_prefix}_ptv1_random_split"):
        raise SystemExit(1)
    if not completed_infer(f"{run_prefix}_extra_valid_from_exp11", "ptv1_extra_singledrug", expected_extra, "test_only"):
        raise SystemExit(1)
    epoch = best_oracle_epoch(candidate)
    if epoch is None:
        raise SystemExit(1)
    if not completed_infer(
        f"{run_prefix}_exp11_test_oracle_epoch{epoch}_from_exp11",
        "ptv1_aivc",
        expected_exp11,
        "fixed_experiment_type",
    ):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

versioned_prefix() {
  local requested="$1"
  local offset="$2"
  "${PYTHON_BIN}" - "${requested}" "${offset}" <<'PY'
import re
import sys

requested = sys.argv[1]
offset = int(sys.argv[2])
match = re.search(r"_v([0-9]+)_", requested)
if match:
    version = int(match.group(1)) + offset
    print(requested[: match.start()] + f"_v{version}_" + requested[match.end() :])
else:
    print(f"{requested}_v{offset + 1}")
PY
}

choose_base_prefix() {
  local requested="$1"
  local offset=0
  local candidate="${requested}"
  while prefix_complete "${candidate}"; do
    offset=$((offset + 1))
    candidate="$(versioned_prefix "${requested}" "${offset}")"
  done
  printf "%s\n" "${candidate}"
}

read_positive_weight_grid

NEG_POS_SUMMARY="$(resolve_neg_pos_weight)"
IFS=$'\t' read -r NEG_POS_NEG NEG_POS_POS NEG_POS_WEIGHT <<< "${NEG_POS_SUMMARY}"

BASE_PREFIX="$(choose_base_prefix "${REQUESTED_BASE_PREFIX}")"
if [[ "${BASE_PREFIX}" != "${REQUESTED_BASE_PREFIX}" ]]; then
  echo "[prefix] requested prefix has complete artifacts; using ${BASE_PREFIX}"
fi

TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${BASE_PREFIX}_runtime_summary.tsv}"
METADATA_TSV="${METADATA_TSV:-${LOG_DIR}/${BASE_PREFIX}_positive_weight_candidates.tsv}"
REPORT_MD_PATH="${REPORT_MD_PATH:-docs/2026-06-10_ptv1_exp13_positive_weight_tuning_report.md}"
REPORT_TSV_PATH="${REPORT_TSV_PATH:-outputs/${BASE_PREFIX}_positive_weight_tuning_report.tsv}"

resolved_positive_weight() {
  local requested="$1"
  if [[ "${requested}" == "neg/pos" ]]; then
    printf "%s\n" "${NEG_POS_WEIGHT}"
  else
    printf "%s\n" "${requested}"
  fi
}

write_candidate_metadata() {
  mkdir -p "$(dirname "${METADATA_TSV}")"
  printf "candidate\trequested_positive_weight\tresolved_positive_weight\tneg_count\tpos_count\tneg_pos_ratio\n" > "${METADATA_TSV}"
  local idx
  local resolved
  for idx in "${!CANDIDATES[@]}"; do
    resolved="$(resolved_positive_weight "${REQUESTED_WEIGHTS[$idx]}")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${CANDIDATES[$idx]}" \
      "${REQUESTED_WEIGHTS[$idx]}" \
      "${resolved}" \
      "${NEG_POS_NEG}" \
      "${NEG_POS_POS}" \
      "${NEG_POS_WEIGHT}" >> "${METADATA_TSV}"
  done
}

init_time_summary() {
  mkdir -p "$(dirname "${TIME_SUMMARY_PATH}")"
  if [[ ! -f "${TIME_SUMMARY_PATH}" ]]; then
    printf "kind\texperiment\ttask_name\tsplit_strategy\tsplit_name\tstatus\tstart_utc\tend_utc\tduration_sec\tartifact\n" > "${TIME_SUMMARY_PATH}"
  fi
}

utc_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

record_time() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${TIME_SUMMARY_PATH}"
}

COMMON_ENV=(
  "GPU_IDS=${GPU_IDS}"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=none"
  "LOG_TO_WANDB=0"
  "PROGRESS_BAR=${PROGRESS_BAR:-0}"
  "RUN_DATA_VALIDATION=0"
  "RUN_INFERENCE=1"
  "RUN_PREFLIGHT=0"
  "MODEL_TYPE=fast_delta"
  "HIDDEN_DIM=512"
  "EXPRESSION_LATENT_DIM=768"
  "COVARIATE_EMBEDDING_DIM=96"
  "BATCH_SIZE=${BATCH_SIZE:-256}"
  "INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-256}"
  "PRECISION=${PRECISION:-bf16-mixed}"
  "NUM_WORKERS=${NUM_WORKERS:-4}"
  "MAX_EPOCHS=${MAX_EPOCHS}"
  "SAVE_LAST_CKPT=1"
  "BEST_CKPT_METRIC=valid_auprc"
  "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-1.0}"
  "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-1.0}"
  "LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES:-1.0}"
  "INFER_LIMIT_BATCHES=${INFER_LIMIT_BATCHES:-}"
  "INFER_DEVICE=${INFER_DEVICE:-cuda:0}"
  "GRAPH_FEATURE_MODE=real"
  "GRAPH_STRUCTURAL_RP=1"
  "GRAPH_DRUG_CONCAT=1"
  "GRAPH_LOGIT_SCALE=2.0"
  "GRAPH_CACHE_DIR=${GRAPH_CACHE_DIR}"
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
  "PROTEIN_CONCAT_SCORE_MODE=multiply"
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
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
  "CKPT_DIR=${CKPT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "ALLOW_EXISTING_RUN=0"
)

CONFIG_ENV=()
set_config_env() {
  local resolved_weight="$1"
  CONFIG_ENV=(
    "LEARNING_RATE=2e-4"
    "BATCH_SIZE=${BATCH_SIZE:-256}"
    "DROPOUT=0.15"
    "WEIGHT_DECAY=1e-4"
    "MSE_WEIGHT=0.50"
    "MSE_TARGET_MODE=pdi"
    "POSITIVE_WEIGHT=${resolved_weight}"
    "POSITIVE_LABEL_SAMPLING_WEIGHT=1.0"
    "FOCAL_LOSS=0"
    "LABEL_SMOOTHING=0.0"
    "GRAPH_FEATURE_MODE=real"
    "GRAPH_STRUCTURAL_RP=1"
    "GRAPH_DRUG_CONCAT=1"
    "GRAPH_LOGIT_SCALE=2.0"
    "PROTEIN_CONCAT_MODE=pcep"
    "PROTEIN_CONCAT_SCORE_MODE=multiply"
    "TARGET_EXPRESSION_MODE=off"
    "CONTROL_EXPRESSION_DROPOUT=0.0"
    "BATCH_COV_LIST="
  )
}

train_completed() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
manifest = Path(sys.argv[1]) / "run_manifest.json"
if not manifest.exists():
    raise SystemExit(1)
with manifest.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("run_status") == "fit_completed" else 1)' "${CKPT_DIR}/${exp_name}"
}

infer_completed() {
  local exp_name="$1"
  local task_name="$2"
  local expected_rows="$3"
  local expected_split="$4"
  if [[ "${SKIP_COMPLETED}" != "1" ]]; then
    return 1
  fi
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
root = Path(sys.argv[1]) / sys.argv[2]
expected_rows = int(sys.argv[3])
expected_split = sys.argv[4]
manifest_path = root / "run_manifest.json"
metrics_path = root / "metrics.json"
if not manifest_path.exists() or not metrics_path.exists():
    raise SystemExit(1)
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("dataset_group") != "ptv1" or manifest.get("task_name") != sys.argv[2]:
    raise SystemExit(1)
if manifest.get("split_strategy") != expected_split:
    raise SystemExit(1)
if int(manifest.get("n_predictions") or -1) != expected_rows:
    raise SystemExit(1)
raise SystemExit(0)' "${OUTPUT_DIR}/${exp_name}" "${task_name}" "${expected_rows}" "${expected_split}"
}

ensure_clean_output_path() {
  local path="$1"
  if [[ -e "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[error] inference output directory already exists and is incomplete: ${path}" >&2
    echo "[error] remove it deliberately or choose a new BASE_PREFIX" >&2
    exit 1
  fi
}

validate_required_artifacts() {
  "${PYTHON_BIN}" utils/ptv1/03_validate_ptv1_training_ready.py
  "${PYTHON_BIN}" utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only --output "${CELL_LLM_EMBEDDING_PATH}"
  "${PYTHON_BIN}" utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py --validate-only --output "${CELL_TYPE_LLM_EMBEDDING_PATH}"
}

run_static_checks() {
  "${PYTHON_BIN}" -m py_compile \
    train.py \
    infer.py \
    scripts/ptv1/report_ptv1_exp13_positive_weight_tune.py \
    utils/ptv1/07_build_ptv1_cell_llm_embeddings.py \
    utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py
  bash -n \
    scripts/ptv1/exp_11_ptv1_random_split.sh \
    scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh \
    scripts/ptv1/run_ptv1_exp13_positive_weight_tune.sh \
    scripts/ptv1/ptv1_experiment_common.sh \
    scripts/ptv3_experiment_common.sh
}

run_exp11() {
  local candidate="$1"
  local requested="$2"
  local resolved="$3"
  local run_prefix="${BASE_PREFIX}_${candidate}"
  local exp_name="${run_prefix}_ptv1_random_split"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && train_completed "${exp_name}"; then
    echo "[skip] exp11 completed: ${exp_name}"
    return
  fi
  set_config_env "${resolved}"
  echo "[run] exp11 candidate=${candidate} requested=${requested} resolved=${resolved}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "SAVE_TOP_K=${EXP11_SAVE_TOP_K}" \
    "EXP_PREFIX=${run_prefix}" \
    "TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}" \
    bash scripts/ptv1/exp_11_ptv1_random_split.sh
}

run_exp13_valid() {
  local candidate="$1"
  local requested="$2"
  local resolved="$3"
  local run_prefix="${BASE_PREFIX}_${candidate}"
  local exp11_name="${run_prefix}_ptv1_random_split"
  local exp13_name="${run_prefix}_extra_valid_from_exp11"
  if infer_completed "${exp13_name}" "ptv1_extra_singledrug" "${EXPECTED_EXTRA_ROWS}" "test_only"; then
    echo "[skip] exp13 valid completed: ${exp13_name}"
    return
  fi
  if ! train_completed "${exp11_name}"; then
    echo "[warn] exp13 valid skipped because exp11 is incomplete: ${exp11_name}" >&2
    return
  fi
  set_config_env "${resolved}"
  echo "[run] exp13 valid candidate=${candidate} requested=${requested} resolved=${resolved}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "SAVE_TOP_K=${EXP11_SAVE_TOP_K}" \
    "EXP_PREFIX=${run_prefix}" \
    "EXP11_EXP_NAME=${exp11_name}" \
    "EXP13_EXP_NAME=${exp13_name}" \
    "TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}" \
    bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh direct
}

ckpt_epoch() {
  local ckpt="$1"
  local name
  name="$(basename "${ckpt}")"
  if [[ "${name}" =~ epoch=([0-9]+) ]]; then
    printf "%s\n" "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

run_exp13_oracle() {
  local candidate="$1"
  local requested="$2"
  local resolved="$3"
  local run_prefix="${BASE_PREFIX}_${candidate}"
  local exp11_name="${run_prefix}_ptv1_random_split"
  local exp11_dir="${CKPT_DIR}/${exp11_name}"
  if ! train_completed "${exp11_name}"; then
    echo "[warn] exp13 oracle skipped because exp11 is incomplete: ${exp11_name}" >&2
    return
  fi
  set_config_env "${resolved}"
  local found=0
  while IFS= read -r ckpt; do
    [[ -n "${ckpt}" ]] || continue
    found=1
    local epoch
    epoch="$(ckpt_epoch "${ckpt}")"
    local exp13_name="${run_prefix}_extra_oracle_epoch${epoch}_from_exp11"
    if infer_completed "${exp13_name}" "ptv1_extra_singledrug" "${EXPECTED_EXTRA_ROWS}" "test_only"; then
      echo "[skip] exp13 oracle completed: ${exp13_name}"
      continue
    fi
    echo "[run] exp13 oracle candidate=${candidate} requested=${requested} resolved=${resolved} epoch=${epoch}"
    env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
      "SAVE_TOP_K=${EXP11_SAVE_TOP_K}" \
      "EXP_PREFIX=${run_prefix}" \
      "EXP11_EXP_NAME=${exp11_name}" \
      "EXP11_CKPT_PATH=${ckpt}" \
      "EXP13_EXP_NAME=${exp13_name}" \
      "TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}" \
      bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh direct
  done < <(find "${exp11_dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort -V)
  if [[ "${found}" == "0" ]]; then
    echo "[warn] no saved epoch checkpoints found for oracle scan: ${exp11_dir}" >&2
  fi
}

best_oracle_for_candidate() {
  local candidate="$1"
  local run_prefix="${BASE_PREFIX}_${candidate}"
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${run_prefix}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
run_prefix = sys.argv[2]
pattern = re.compile(rf"^{re.escape(run_prefix)}_extra_oracle_epoch([0-9]+)_from_exp11$")
best = None
for metrics_path in output_root.glob(f"{run_prefix}_extra_oracle_epoch*_from_exp11/ptv1_extra_singledrug/metrics.json"):
    match = pattern.match(metrics_path.parent.parent.name)
    if not match:
        continue
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle).get("task", {})
    try:
        score = float(metrics.get("auprc"))
    except (TypeError, ValueError):
        continue
    if not math.isfinite(score):
        continue
    manifest_path = metrics_path.parent / "run_manifest.json"
    if not manifest_path.exists():
        continue
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    checkpoint = manifest.get("checkpoint_path")
    if not checkpoint or not Path(checkpoint).exists():
        continue
    epoch = int(match.group(1))
    row = (score, epoch, checkpoint)
    if best is None or row[0] > best[0]:
        best = row
if best is None:
    raise SystemExit(1)
print(f"{best[1]}\t{best[2]}\t{best[0]:.10f}")
PY
}

run_exp11_oracle_audit() {
  local candidate="$1"
  local requested="$2"
  local resolved="$3"
  local run_prefix="${BASE_PREFIX}_${candidate}"
  local best_line
  if ! best_line="$(best_oracle_for_candidate "${candidate}")"; then
    echo "[warn] exp11 oracle audit skipped because no completed oracle metrics exist for candidate=${candidate}" >&2
    return
  fi
  local epoch
  local checkpoint
  local oracle_auprc
  IFS=$'\t' read -r epoch checkpoint oracle_auprc <<< "${best_line}"
  local exp_name="${run_prefix}_exp11_test_oracle_epoch${epoch}_from_exp11"
  if infer_completed "${exp_name}" "ptv1_aivc" "${EXPECTED_EXP11_TEST_ROWS}" "fixed_experiment_type"; then
    echo "[skip] exp11 oracle audit completed: ${exp_name}"
    return
  fi
  set_config_env "${resolved}"
  local output_path="${OUTPUT_DIR}/${exp_name}/ptv1_aivc"
  ensure_clean_output_path "${output_path}"
  echo "[run] exp11 oracle audit candidate=${candidate} requested=${requested} resolved=${resolved} epoch=${epoch} oracle_auprc=${oracle_auprc}"
  local start_utc
  local end_utc
  local start_sec
  local end_sec
  local status
  start_utc="$(utc_now)"
  start_sec="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u infer.py \
    --dataset-group ptv1 \
    --model-type fast_delta \
    --task-name ptv1_aivc \
    --split-strategy fixed_experiment_type \
    --split-name test \
    --task-head response \
    --checkpoint-path "${checkpoint}" \
    --output-dir "${output_path}" \
    --batch-size "${INFER_BATCH_SIZE:-256}" \
    --hidden-dim 512 \
    --expression-latent-dim 768 \
    --covariate-embedding-dim 96 \
    --dropout 0.15 \
    --target-protein-max-length 32 \
    --graph-feature-mode real \
    --graph-feature-dim 128 \
    --graph-feature-seed 17 \
    --graph-cache-dir "${GRAPH_CACHE_DIR}" \
    --graph-layers 2 \
    --graph-init-scale 0.1 \
    --graph-logit-scale 2.0 \
    --graph-structural-rp \
    --graph-drug-concat \
    --protein-concat-mode pcep \
    --protein-concat-topk 512 \
    --protein-concat-score-mode multiply \
    --use-dose-covariate \
    --dose-covariate-fields pert_dose1 pert_dose2 \
    --cell-llm-mode frozen \
    --cell-llm-embedding-path "${CELL_LLM_EMBEDDING_PATH}" \
    --cell-llm-fusion-mode "${CELL_LLM_FUSION_MODE:-covariate}" \
    --cell-llm-condition-scale "${CELL_LLM_CONDITION_SCALE:-0.0}" \
    --cell-llm-logit-scale "${CELL_LLM_LOGIT_SCALE:-0.0}" \
    --cell-llm-dropout "${CELL_LLM_DROPOUT:-0.0}" \
    --cell-type-llm-mode frozen \
    --cell-type-llm-embedding-path "${CELL_TYPE_LLM_EMBEDDING_PATH}" \
    --cell-type-llm-fusion-mode "${CELL_TYPE_LLM_FUSION_MODE:-covariate}" \
    --cell-type-llm-condition-scale "${CELL_TYPE_LLM_CONDITION_SCALE:-0.0}" \
    --cell-type-llm-logit-scale "${CELL_TYPE_LLM_LOGIT_SCALE:-0.0}" \
    --cell-type-llm-dropout "${CELL_TYPE_LLM_DROPOUT:-0.0}" \
    --device "${INFER_DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-4}"
  status="$?"
  set -e
  end_utc="$(utc_now)"
  end_sec="$(date +%s)"
  record_time \
    "infer" \
    "${exp_name}" \
    "ptv1_aivc" \
    "fixed_experiment_type" \
    "test" \
    "${status}" \
    "${start_utc}" \
    "${end_utc}" \
    "$((end_sec - start_sec))" \
    "${output_path}"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

run_reports() {
  mkdir -p "$(dirname "${REPORT_MD_PATH}")" "$(dirname "${REPORT_TSV_PATH}")"
  local strict_args=()
  if [[ "${REPORT_STRICT}" == "1" ]]; then
    strict_args=(--strict)
  fi
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_exp13_positive_weight_tune.py \
    --base-prefix "${BASE_PREFIX}" \
    --metadata-tsv "${METADATA_TSV}" \
    --expected-exp11-test-rows "${EXPECTED_EXP11_TEST_ROWS}" \
    --expected-extra-rows "${EXPECTED_EXTRA_ROWS}" \
    --format markdown \
    "${strict_args[@]}" \
    > "${REPORT_MD_PATH}"
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_exp13_positive_weight_tune.py \
    --base-prefix "${BASE_PREFIX}" \
    --metadata-tsv "${METADATA_TSV}" \
    --expected-exp11-test-rows "${EXPECTED_EXP11_TEST_ROWS}" \
    --expected-extra-rows "${EXPECTED_EXTRA_ROWS}" \
    --format tsv \
    "${strict_args[@]}" \
    > "${REPORT_TSV_PATH}"
  echo "[report] ${REPORT_MD_PATH}"
  echo "[report] ${REPORT_TSV_PATH}"
}

write_candidate_metadata
init_time_summary

echo "[settings] BASE_PREFIX=${BASE_PREFIX}; GPU_IDS=${GPU_IDS}"
echo "[settings] candidates=${CANDIDATES[*]}"
echo "[settings] neg_pos_train_counts=neg:${NEG_POS_NEG} pos:${NEG_POS_POS} ratio:${NEG_POS_WEIGHT}"
echo "[settings] fixed_config=MSE_WEIGHT=0.50 MSE_TARGET_MODE=pdi LR=2e-4 DROPOUT=0.15 SAVE_TOP_K=${EXP11_SAVE_TOP_K} MAX_EPOCHS=${MAX_EPOCHS}"
echo "[settings] graph=real structural_rp=1 drug_concat=1 logit_scale=2.0"
echo "[settings] CELL_LLM_MODE=frozen; CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
echo "[settings] CELL_TYPE_LLM_MODE=frozen; CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH}"
echo "[settings] report_md=${REPORT_MD_PATH}; report_tsv=${REPORT_TSV_PATH}; metadata=${METADATA_TSV}"

if [[ "${STATIC_CHECKS}" == "1" ]]; then
  run_static_checks
fi

if [[ "${RUN_DATA_VALIDATION}" == "1" ]]; then
  validate_required_artifacts
fi

for idx in "${!CANDIDATES[@]}"; do
  candidate="${CANDIDATES[$idx]}"
  requested="${REQUESTED_WEIGHTS[$idx]}"
  resolved="$(resolved_positive_weight "${requested}")"
  if [[ "${RUN_EXP11}" == "1" ]]; then
    run_exp11 "${candidate}" "${requested}" "${resolved}"
  fi
  if [[ "${RUN_EXP13_VALID}" == "1" ]]; then
    run_exp13_valid "${candidate}" "${requested}" "${resolved}"
  fi
  if [[ "${RUN_EXP13_ORACLE}" == "1" ]]; then
    run_exp13_oracle "${candidate}" "${requested}" "${resolved}"
  fi
  if [[ "${RUN_EXP11_ORACLE_AUDIT}" == "1" ]]; then
    run_exp11_oracle_audit "${candidate}" "${requested}" "${resolved}"
  fi
done

if [[ "${REPORT_AFTER}" == "1" ]]; then
  run_reports
fi

echo "[done] PTV1 exp13 positive-weight tuning completed for BASE_PREFIX=${BASE_PREFIX}"
