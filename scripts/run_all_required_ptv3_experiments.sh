#!/usr/bin/env bash
set -euo pipefail

# Optional convenience launcher.  Each line calls one small experiment script.
# To understand or run one experiment only, open the corresponding exp_*.sh file.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_ptv3_required}"
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

bash scripts/exp_01_single_pert_stratified_5fold.sh

export RUN_PREFLIGHT=0
bash scripts/exp_02_single_cell_type_5fold.sh
bash scripts/exp_03_single_cell_5fold.sh
bash scripts/exp_04_single_no_mse_5fold.sh
bash scripts/exp_05_single_no_pdi_5fold.sh
bash scripts/exp_06_double_pert_pair_5fold.sh
bash scripts/exp_07_extra_single_all_train_infer.sh
bash scripts/exp_08_extra_double_all_train_infer.sh
