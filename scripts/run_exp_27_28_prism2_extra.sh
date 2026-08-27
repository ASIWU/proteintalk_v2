#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready_prism2_main}"
export EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_prism2_exp27_28}"
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

bash scripts/exp_27_extra_single_prism2_all_train_infer.sh

export RUN_PREFLIGHT=0
bash scripts/exp_28_extra_double_prism2aux_all_train_infer.sh
