#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="single_pert_stratified_5fold"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

ptv3_print_settings "Single-drug 5-fold split on stratified pert_id"
ptv3_run_preflight

for fold in "${FOLD_LIST[@]}"; do
  ptv3_train "${EXP_PREFIX}_single_pert_stratified_fold${fold}" \
    ptv3_main_singledrug "pert_stratified_5fold_fold${fold}" response
done

ptv3_done
