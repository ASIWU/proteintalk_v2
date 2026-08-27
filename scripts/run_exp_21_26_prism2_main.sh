#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready_prism2_main}"
export EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_prism2_exp21_26}"
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

bash scripts/exp_21_single_prism2_pert_stratified_5fold.sh

export RUN_PREFLIGHT=0
bash scripts/exp_22_single_prism2_cell_type_5fold.sh
bash scripts/exp_23_single_prism2_cell_5fold.sh
bash scripts/exp_24_single_prism2_no_mse_5fold.sh
bash scripts/exp_25_single_prism2_no_pdi_5fold.sh
bash scripts/exp_26_double_prism2aux_pert_pair_5fold.sh
