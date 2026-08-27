#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_SET_NAME="exp35_update0821_target_epoch2"
export EXP_PREFIX="${EXP_PREFIX:-20260821_exp35_update0821_target_epoch2}"
export OUTPUT_DATE="${OUTPUT_DATE:-2026-08-21}"
export TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-/tmp/proteintalk_exp35_update0821_target_runtime}"
export REQUIRED_TMUX_TARGET="gpu1:0"
export EXP34_TMUX_TARGET="${EXP35_TMUX_TARGET:-gpu1:0}"
export EXP34_BUILDER_SCRIPT="utils/35_build_update0821_target_training_ready.py"
export EXP34_BUILD_SUMMARY_REL="ptv3/exp35_update0821_target_build_summary.json"
export EXP34_TASKS_STRING="ptv3_exp35_update0821_manual_target"

exec bash "${SCRIPT_DIR}/exp_34_update0819_ood_epoch2_infer.sh"
