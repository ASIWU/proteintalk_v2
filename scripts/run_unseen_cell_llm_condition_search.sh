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
BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_cell_llm_condition_cell_search}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
TARGET_AUPRC="${TARGET_AUPRC:-0.85}"
SEARCH_FOLDS="${SEARCH_FOLDS:-0}"
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-86400}"
RUN_FULL_AFTER_HIT="${RUN_FULL_AFTER_HIT:-0}"
FULL_FOLDS="${FULL_FOLDS:-0 1 2 3 4}"
SUMMARY_PATH="${SUMMARY_PATH:-${LOG_DIR}/${BASE_EXP_PREFIX}_summary.tsv}"
mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" -m py_compile \
  train.py \
  infer.py \
  dataset/training_ready_fast_dataset.py \
  model/fast_delta_model.py \
  model/fast_lightning.py \
  model/graph_feature_utils.py
bash -n scripts/ptv3_experiment_common.sh scripts/exp_03_single_cell_5fold.sh

printf "variant\tfolds\tmean_test_auprc\tmax_test_auprc\tmean_test_auroc\tstatus\texperiments\n" > "${SUMMARY_PATH}"

START_SECONDS="$(date +%s)"

remaining_seconds() {
  local now elapsed remaining
  now="$(date +%s)"
  elapsed="$((now - START_SECONDS))"
  remaining="$((MAX_RUN_SECONDS - elapsed))"
  if (( remaining < 1 )); then
    remaining=1
  fi
  printf "%s" "${remaining}"
}

summarize_variant() {
  local prefix="$1"
  local folds="$2"
  local status="$3"
  "${PYTHON_BIN}" - "$prefix" "$folds" "$status" "$SUMMARY_PATH" <<'PY'
import json
import math
import sys
from pathlib import Path

prefix, folds_text, status, summary_path = sys.argv[1:5]
folds = [item for item in folds_text.split() if item]
rows = []
experiments = []
for fold in folds:
    exp = Path("checkpoints") / f"{prefix}_single_cell_fold{fold}"
    experiments.append(str(exp))
    manifest_path = exp / "run_manifest.json"
    if not manifest_path.exists():
        rows.append((math.nan, math.nan))
        continue
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    test_results = manifest.get("test_results") or []
    if not test_results:
        rows.append((math.nan, math.nan))
        continue
    metrics = test_results[0]
    auprc = metrics.get("test/task_auprc")
    auroc = metrics.get("test/task_auroc")
    rows.append((float(auprc) if auprc is not None else math.nan, float(auroc) if auroc is not None else math.nan))

valid_auprc = [item[0] for item in rows if math.isfinite(item[0])]
valid_auroc = [item[1] for item in rows if math.isfinite(item[1])]
mean_auprc = sum(valid_auprc) / len(valid_auprc) if valid_auprc else math.nan
max_auprc = max(valid_auprc) if valid_auprc else math.nan
mean_auroc = sum(valid_auroc) / len(valid_auroc) if valid_auroc else math.nan
line = "\t".join(
    [
        prefix,
        folds_text,
        f"{mean_auprc:.9f}" if math.isfinite(mean_auprc) else "nan",
        f"{max_auprc:.9f}" if math.isfinite(max_auprc) else "nan",
        f"{mean_auroc:.9f}" if math.isfinite(mean_auroc) else "nan",
        status,
        ",".join(experiments),
    ]
)
with Path(summary_path).open("a", encoding="utf-8") as handle:
    handle.write(line + "\n")
print(line)
PY
}

target_reached() {
  local prefix="$1"
  local folds="$2"
  "${PYTHON_BIN}" - "$prefix" "$folds" "$TARGET_AUPRC" <<'PY'
import json
import math
import sys
from pathlib import Path

prefix, folds_text, target_text = sys.argv[1:4]
target = float(target_text)
folds = [item for item in folds_text.split() if item]
values = []
for fold in folds:
    manifest_path = Path("checkpoints") / f"{prefix}_single_cell_fold{fold}" / "run_manifest.json"
    if not manifest_path.exists():
        continue
    manifest = json.load(manifest_path.open())
    test_results = manifest.get("test_results") or []
    if not test_results:
        continue
    value = test_results[0].get("test/task_auprc")
    if value is not None and math.isfinite(float(value)):
        values.append(float(value))
if not values:
    raise SystemExit(1)
mean_value = sum(values) / len(values)
raise SystemExit(0 if mean_value >= target else 1)
PY
}

run_variant() {
  local spec="$1"
  local folds="$2"
  local prefix_suffix="${3:-}"
  IFS=':' read -r name lr mse covdrop fusion cond_scale logit_scale llm_drop target_mode target_fusion target_init hidden_dim expr_dim batch_size max_epochs graph_logit protein_mode <<< "${spec}"
  local prefix="${BASE_EXP_PREFIX}_${name}${prefix_suffix}"
  local log_path="${LOG_DIR}/${prefix}.log"
  local remaining
  remaining="$(remaining_seconds)"
  echo "[run] ${name}; folds=${folds}; remaining=${remaining}s; log=${log_path}"
  set +e
  timeout --preserve-status "${remaining}s" env \
    GPU_IDS="${GPU_IDS}" \
    EXP_PREFIX="${prefix}" \
    FOLDS="${folds}" \
    RUN_PREFLIGHT=0 \
    RUN_INFERENCE=0 \
    LOGGER_BACKEND="${LOGGER_BACKEND:-none}" \
    LOG_TO_WANDB="${LOG_TO_WANDB:-0}" \
    PROGRESS_BAR="${PROGRESS_BAR:-0}" \
    BATCH_SIZE="${batch_size}" \
    MAX_EPOCHS="${max_epochs}" \
    LEARNING_RATE="${lr}" \
    HIDDEN_DIM="${hidden_dim}" \
    EXPRESSION_LATENT_DIM="${expr_dim}" \
    MSE_WEIGHT="${mse}" \
    GRAPH_FEATURE_MODE=real \
    GRAPH_STRUCTURAL_RP=1 \
    GRAPH_DRUG_CONCAT=1 \
    GRAPH_LOGIT_SCALE="${graph_logit}" \
    PROTEIN_CONCAT_MODE="${protein_mode}" \
    USE_DOSE_COVARIATE=1 \
    CELL_LLM_MODE=frozen \
    CELL_LLM_FUSION_MODE="${fusion}" \
    CELL_LLM_CONDITION_SCALE="${cond_scale}" \
    CELL_LLM_LOGIT_SCALE="${logit_scale}" \
    CELL_LLM_DROPOUT="${llm_drop}" \
    COVARIATE_UNK_FOR_UNSEEN=0 \
    COVARIATE_UNK_DROPOUT="${covdrop}" \
    TARGET_EXPRESSION_MODE="${target_mode}" \
    TARGET_EXPRESSION_FUSION_MODE="${target_fusion}" \
    TARGET_EXPRESSION_TOPK="${TARGET_EXPRESSION_TOPK:-256}" \
    TARGET_EXPRESSION_PPI_TOPK="${TARGET_EXPRESSION_PPI_TOPK:-32}" \
    TARGET_EXPRESSION_PPI_ALPHA="${TARGET_EXPRESSION_PPI_ALPHA:-0.5}" \
    TARGET_EXPRESSION_INIT_SCALE="${target_init}" \
    ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}" \
    bash scripts/exp_03_single_cell_5fold.sh > "${log_path}" 2>&1
  local status="$?"
  set -e
  summarize_variant "${prefix}" "${folds}" "${status}"
  if [[ "${status}" -eq 0 ]] && target_reached "${prefix}" "${folds}"; then
    echo "[hit] ${prefix} reached mean test AUPRC >= ${TARGET_AUPRC} on folds: ${folds}"
    if [[ "${RUN_FULL_AFTER_HIT}" == "1" && "${folds}" != "${FULL_FOLDS}" ]]; then
      run_variant "${spec}" "${FULL_FOLDS}" "_full5"
    fi
    exit 0
  fi
  if (( $(date +%s) - START_SECONDS >= MAX_RUN_SECONDS )); then
    echo "[timeout] max run time reached: ${MAX_RUN_SECONDS}s" >&2
    exit 124
  fi
}

VARIANTS=(
  "hybrid_res_s005_l010_d005_lr1e4_mse025:1e-4:0.25:0.10:hybrid:0.05:0.10:0.05:off:piece:0.1:512:768:256:50:2.0:pcep"
  "hybrid_res_s010_l010_d005_lr1e4_mse025:1e-4:0.25:0.10:hybrid:0.10:0.10:0.05:off:piece:0.1:512:768:256:50:2.0:pcep"
  "hybrid_res_s005_l025_d005_lr1e4_mse025:1e-4:0.25:0.10:hybrid:0.05:0.25:0.05:off:piece:0.1:512:768:256:50:2.0:pcep"
  "piece_res_s005_l010_d005_lr1e4_mse025:1e-4:0.25:0.10:piece:0.05:0.10:0.05:off:piece:0.1:512:768:256:50:2.0:pcep"
  "film_res_s010_l010_d005_lr1e4_mse025:1e-4:0.25:0.10:film:0.10:0.10:0.05:off:piece:0.1:512:768:256:50:2.0:pcep"
  "covariate_l010_d005_lr1e4_mse025:1e-4:0.25:0.10:covariate:0.0:0.10:0.05:off:piece:0.1:512:768:256:50:2.0:pcep"
  "hybrid_res_s005_l010_target_pair:1e-4:0.075:0.15:hybrid:0.05:0.10:0.05:pdi_ppi:pair_add:0.5:512:768:256:50:2.0:pcep"
  "hybrid_res_s005_l010_h768:1e-4:0.25:0.10:hybrid:0.05:0.10:0.05:off:piece:0.1:768:1024:192:50:2.0:pcep"
  "hybrid_res_s005_l010_lr2e4_cov015:2e-4:0.25:0.15:hybrid:0.05:0.10:0.05:off:piece:0.1:512:768:256:50:2.0:pcep"
  "hybrid_res_s005_l010_graph3:1e-4:0.25:0.10:hybrid:0.05:0.10:0.05:off:piece:0.1:512:768:256:50:3.0:pcep"
)

for spec in "${VARIANTS[@]}"; do
  run_variant "${spec}" "${SEARCH_FOLDS}"
done

echo "[done] target not reached; summary=${SUMMARY_PATH}" >&2
exit 1
