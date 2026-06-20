#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="exp11_ptv1_random_split"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv1_experiment_common.sh"

ptv1_print_settings "PTV1 random split from fixed experiment_type artifact"
ptv1_run_preflight

ptv1_train "${EXP_PREFIX}_ptv1_random_split" \
  ptv1_aivc fixed_experiment_type response

ptv1_done
