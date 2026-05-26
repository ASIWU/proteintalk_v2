#!/usr/bin/env bash
# bash scripts/0526_baseline4_task_specific_2gpu_parallel.sh
# EXP_PREFIX=20260526_final_task_specific GPU_IDS=0,1 bash scripts/0526_baseline4_task_specific_2gpu_parallel.sh

set -euo pipefail

# 2026-05-26 full-suite launcher with task-specific fast_delta fusion defaults.
# The maintained scheduler honors GPU_IDS, so this wrapper constrains it to two
# single-GPU workers while keeping the same exp_01 to exp_08 execution flow.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_PREFIX="${EXP_PREFIX:-20260526_final_task_specific}"
export GPU_IDS="${GPU_IDS:-0,1}"
export DEVICES="${DEVICES:-1}"
export RUN_EXTRA_TASKS="${RUN_EXTRA_TASKS:-1}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != 2 )); then
  echo "[error] this launcher expects exactly two comma-separated GPU ids; got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi
if [[ "${DEVICES}" != "1" ]]; then
  echo "[error] this launcher runs two one-GPU workers; set DEVICES=1 or leave it unset" >&2
  exit 2
fi

export SINGLE_PAIR_FUSION_MODE="${SINGLE_PAIR_FUSION_MODE:-symmetric}"
export SINGLE_PAIR_TYPE_FEATURES="${SINGLE_PAIR_TYPE_FEATURES:-0}"
export SINGLE_MSE_INACTIVE_LABEL_WEIGHT="${SINGLE_MSE_INACTIVE_LABEL_WEIGHT:-1.0}"
export SINGLE_USE_DDI="${SINGLE_USE_DDI:-0}"
export SINGLE_GRAPH_PAIR_ADD_SCALE="${SINGLE_GRAPH_PAIR_ADD_SCALE:-0.0}"

export DOUBLE_PAIR_FUSION_MODE="${DOUBLE_PAIR_FUSION_MODE:-dual}"
export DOUBLE_PAIR_TYPE_FEATURES="${DOUBLE_PAIR_TYPE_FEATURES:-1}"
export DOUBLE_MSE_INACTIVE_LABEL_WEIGHT="${DOUBLE_MSE_INACTIVE_LABEL_WEIGHT:-0.2}"
export DOUBLE_USE_DDI="${DOUBLE_USE_DDI:-1}"
export DOUBLE_GRAPH_PAIR_ADD_SCALE="${DOUBLE_GRAPH_PAIR_ADD_SCALE:-0.5}"

echo "[launcher] EXP_PREFIX=${EXP_PREFIX}; GPU_IDS=${GPU_IDS}; DEVICES=${DEVICES}; RUN_EXTRA_TASKS=${RUN_EXTRA_TASKS}"
exec bash "${SCRIPT_DIR}/0521_baseline4_8gpu_parallel.sh" "$@"
