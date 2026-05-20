#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name.  The readable scripts are:
#   scripts/exp_01_single_pert_stratified_5fold.sh
#   scripts/exp_02_single_cell_type_5fold.sh
#   scripts/exp_03_single_cell_5fold.sh
#   scripts/exp_04_single_no_mse_5fold.sh
#   scripts/exp_05_single_no_pdi_5fold.sh
#   scripts/exp_06_double_pert_pair_5fold.sh
#   scripts/exp_07_extra_single_all_train_infer.sh
#   scripts/exp_08_extra_double_all_train_infer.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/run_all_required_ptv3_experiments.sh
