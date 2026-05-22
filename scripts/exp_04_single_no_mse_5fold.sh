#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="single_no_mse_5fold"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

ptv3_print_settings "Baseline4 single-drug w/o MSE ablation on pert_stratified 5-fold"
ptv3_run_preflight

for fold in "${FOLD_LIST[@]}"; do
  ptv3_train "${EXP_PREFIX}_single_no_mse_fold${fold}" \
    ptv3_main_singledrug "pert_stratified_5fold_fold${fold}" response \
    --no-mse-loss
done

ptv3_done
