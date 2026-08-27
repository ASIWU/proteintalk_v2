#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="exp_04_v2_single_no_mse_global_meanstd_random_expression_5fold"
CONTROL_EXPRESSION_MODE="${CONTROL_EXPRESSION_MODE:-random_saved}"
# Default selected by the 2026-07-09 completed unseen-drug 5-fold run.
RANDOM_CONTROL_EXPRESSION_PATH="${RANDOM_CONTROL_EXPRESSION_PATH:-data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_global_normal_clip_seed42.npy}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

if [[ ! -f "${RANDOM_CONTROL_EXPRESSION_PATH}" ]]; then
  echo "[error] missing random control expression artifact: ${RANDOM_CONTROL_EXPRESSION_PATH}" >&2
  echo "[error] generate it first with scripts/generate_random_control_proteome.py" >&2
  exit 1
fi

ptv3_print_settings "exp_04_v2 single-drug no-MSE global mean/std random control-expression on pert_stratified 5-fold"
ptv3_run_preflight

for fold in "${FOLD_LIST[@]}"; do
  ptv3_train "${EXP_PREFIX}_exp04_v2_random_no_mse_fold${fold}" \
    ptv3_main_singledrug "pert_stratified_5fold_fold${fold}" response \
    --no-mse-loss
done

ptv3_done
