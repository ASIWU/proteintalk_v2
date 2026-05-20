#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="double_pert_pair_5fold"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

ptv3_print_settings "Double-drug 5-fold split on canonical pert pair"
ptv3_run_preflight

for fold in "${FOLD_LIST[@]}"; do
  ptv3_train "${EXP_PREFIX}_double_pert_pair_fold${fold}" \
    ptv3_main_doubledrug "pert_id_5fold_fold${fold}" synergy
done

ptv3_done
