#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="double_pert_pair_5fold"

# Double-drug default from the 2026-05-21 data-composition iteration.
# Single-drug scripts keep the shared baseline4 defaults.
PAIR_FUSION_MODE="${PAIR_FUSION_MODE:-dual}"
PAIR_TYPE_FEATURES="${PAIR_TYPE_FEATURES:-1}"
MSE_INACTIVE_LABEL_WEIGHT="${MSE_INACTIVE_LABEL_WEIGHT:-0.2}"
USE_DDI="${USE_DDI:-1}"
GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.5}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

ptv3_print_settings "Baseline4 double-drug 5-fold split on canonical pert pair"
ptv3_run_preflight

for fold in "${FOLD_LIST[@]}"; do
  ptv3_train "${EXP_PREFIX}_double_pert_pair_fold${fold}" \
    ptv3_main_doubledrug "pert_id_5fold_fold${fold}" synergy
done

ptv3_done
