#!/usr/bin/env bash
# EXP_PREFIX=20260521_final_task_specific bash scripts/0521_baseline4_task_specific_8gpu_parallel.sh

set -euo pipefail

# Final baseline4 rerun launcher with task-specific fast_delta fusion defaults.
# It reuses the maintained 8-GPU scheduler and pins single/double configs before launch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

exec bash "${SCRIPT_DIR}/0521_baseline4_8gpu_parallel.sh" "$@"
