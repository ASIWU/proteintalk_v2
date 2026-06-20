#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="exp12_ptv1_unseen_drug_5fold"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv1_experiment_common.sh"

ptv1_print_settings "PTV1 unseen-drug pert_id 5-fold"
ptv1_run_preflight

for fold in "${FOLD_LIST[@]}"; do
  ptv1_train "${EXP_PREFIX}_ptv1_unseen_drug_fold${fold}" \
    ptv1_aivc "pert_id_5fold_fold${fold}" response
done

ptv1_done
