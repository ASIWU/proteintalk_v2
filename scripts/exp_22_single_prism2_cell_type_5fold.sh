#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="single_prism2_cell_type_5fold"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

ptv3_print_settings "Baseline4 PRISM2 single-drug 5-fold split on cell_type"
ptv3_run_preflight

for fold in "${FOLD_LIST[@]}"; do
  ptv3_train "${EXP_PREFIX}_single_prism2_cell_type_fold${fold}" \
    ptv3_main_singledrug_prism2 "cell_type_5fold_fold${fold}" response \
    --effective-key1 PRISM2nd_label_total
done

ptv3_done
