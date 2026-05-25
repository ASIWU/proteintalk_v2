#!/usr/bin/env bash

# Shared mechanics for the small PTV3 experiment scripts.
# User-facing experiment choices should live in exp_*.sh, not here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${REPO_ROOT}/scripts/wandb_env.local}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${WANDB_ENV_FILE}"
  set +a
fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://100.96.30.112:8080}"
export WANDB_API_KEY="${WANDB_API_KEY:-local-f401d2b9276fb4a6dd1db4c1efee0512723ce6fb}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DEVICES="${DEVICES:-1}"
STRATEGY="${STRATEGY:-auto}"
PRECISION="${PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MODEL_TYPE="${MODEL_TYPE:-fast_delta}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
EXPRESSION_LATENT_DIM="${EXPRESSION_LATENT_DIM:-512}"
COVARIATE_EMBEDDING_DIM="${COVARIATE_EMBEDDING_DIM:-64}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.15}"
CONTROL_LAYERS="${CONTROL_LAYERS:-2}"
FUSION_LAYERS="${FUSION_LAYERS:-3}"
TARGET_LAYERS="${TARGET_LAYERS:-2}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"
SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-0.25}"
BCE_WEIGHT="${BCE_WEIGHT:-}"
POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-none}"
FOCAL_LOSS="${FOCAL_LOSS:-0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
MSE_INACTIVE_LABEL_WEIGHT="${MSE_INACTIVE_LABEL_WEIGHT:-1.0}"
MSE_GENE_SUBSAMPLE="${MSE_GENE_SUBSAMPLE:-0}"
MSE_GENE_WEIGHT_MODE="${MSE_GENE_WEIGHT_MODE:-off}"
MSE_GENE_WEIGHT_TOPK="${MSE_GENE_WEIGHT_TOPK:-4096}"
MSE_GENE_WEIGHT_SCALE="${MSE_GENE_WEIGHT_SCALE:-2.0}"
ACTIVE_LABEL_SAMPLING_WEIGHT="${ACTIVE_LABEL_SAMPLING_WEIGHT:-1.0}"
POSITIVE_LABEL_SAMPLING_WEIGHT="${POSITIVE_LABEL_SAMPLING_WEIGHT:-1.0}"
INACTIVE_LABEL_TRAIN_RATIO="${INACTIVE_LABEL_TRAIN_RATIO:--1.0}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1.0}"
INFER_LIMIT_BATCHES="${INFER_LIMIT_BATCHES:-}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_${EXPERIMENT_SET_NAME:-ptv3}}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
INFER_DEVICE="${INFER_DEVICE:-cuda:0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
SAVE_EVERY_N_TRAIN_STEPS="${SAVE_EVERY_N_TRAIN_STEPS:-}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"
SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-}"
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}"
REFERENCE_5FOLD_CKPT_PATH="${REFERENCE_5FOLD_CKPT_PATH:-}"
REFERENCE_EPOCH_AGG="${REFERENCE_EPOCH_AGG:-mean}"
REFERENCE_EPOCH_ROUNDING="${REFERENCE_EPOCH_ROUNDING:-nearest}"
REFERENCE_EPOCH_MIN_COUNT="${REFERENCE_EPOCH_MIN_COUNT:-5}"
REFERENCE_REQUIRE_TEST_COMPLETED="${REFERENCE_REQUIRE_TEST_COMPLETED:-1}"
REFERENCE_SPLIT_STRATEGY_REGEX="${REFERENCE_SPLIT_STRATEGY_REGEX:-}"
REFERENCE_ALLOW_MIXED_CONFIG="${REFERENCE_ALLOW_MIXED_CONFIG:-0}"
REFERENCE_ALLOW_DUPLICATE_SPLITS="${REFERENCE_ALLOW_DUPLICATE_SPLITS:-0}"
USER_SET_MONITOR="${MONITOR+x}"
MONITOR="${MONITOR:-}"
if [[ -n "${SAVE_EVERY_N_TRAIN_STEPS}" && -z "${USER_SET_MONITOR}" ]]; then
  MONITOR="none"
fi
MONITOR_MODE="${MONITOR_MODE:-}"
LOGGER_BACKEND="${LOGGER_BACKEND:-wandb}"
LOG_TO_WANDB="${LOG_TO_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-aivc_proteintalk}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
ALLOW_NONFINITE_MONITOR="${ALLOW_NONFINITE_MONITOR:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_DATA_VALIDATION="${RUN_DATA_VALIDATION:-0}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${EXP_PREFIX}_runtime_summary.tsv}"
TARGET_PROTEIN_MAX_LENGTH="${TARGET_PROTEIN_MAX_LENGTH:-32}"
GRAPH_FEATURE_MODE="${GRAPH_FEATURE_MODE:-real}"
GRAPH_FEATURE_DIM="${GRAPH_FEATURE_DIM:-128}"
GRAPH_FEATURE_SEED="${GRAPH_FEATURE_SEED:-17}"
GRAPH_STRUCTURAL_RP="${GRAPH_STRUCTURAL_RP:-1}"
GRAPH_MULTIHOP="${GRAPH_MULTIHOP:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache}"
GRAPH_LAYERS="${GRAPH_LAYERS:-2}"
GRAPH_INIT_SCALE="${GRAPH_INIT_SCALE:-0.1}"
GRAPH_DRUG_CONCAT="${GRAPH_DRUG_CONCAT:-1}"
GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.0}"
GRAPH_LOGIT_SCALE="${GRAPH_LOGIT_SCALE:-2.0}"
GRAPH_JUMP_FUSION="${GRAPH_JUMP_FUSION:-concat}"
GRAPH_JUMP_GATE="${GRAPH_JUMP_GATE:-softmax}"
GRAPH_JUMP_TEMPERATURE="${GRAPH_JUMP_TEMPERATURE:-1.0}"
PAIR_FUSION_MODE="${PAIR_FUSION_MODE:-symmetric}"
PAIR_TYPE_FEATURES="${PAIR_TYPE_FEATURES:-0}"
CELL_PAIR_FILM_SCALE="${CELL_PAIR_FILM_SCALE:-0.0}"
TARGET_EXPRESSION_MODE="${TARGET_EXPRESSION_MODE:-off}"
TARGET_EXPRESSION_DIM="${TARGET_EXPRESSION_DIM:-64}"
TARGET_EXPRESSION_TOPK="${TARGET_EXPRESSION_TOPK:-256}"
TARGET_EXPRESSION_PPI_TOPK="${TARGET_EXPRESSION_PPI_TOPK:-32}"
TARGET_EXPRESSION_PPI_ALPHA="${TARGET_EXPRESSION_PPI_ALPHA:-0.5}"
TARGET_EXPRESSION_INIT_SCALE="${TARGET_EXPRESSION_INIT_SCALE:-0.1}"
TARGET_EXPRESSION_SEED="${TARGET_EXPRESSION_SEED:-29}"
TARGET_EXPRESSION_FUSION_MODE="${TARGET_EXPRESSION_FUSION_MODE:-piece}"
TARGET_EXPRESSION_CHUNK_SIZE="${TARGET_EXPRESSION_CHUNK_SIZE:-64}"
TARGET_EXPRESSION_CACHE_DIR="${TARGET_EXPRESSION_CACHE_DIR:-}"
FORCE_TARGET_EXPRESSION_CACHE_REBUILD="${FORCE_TARGET_EXPRESSION_CACHE_REBUILD:-0}"
PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}"
PROTEIN_CONCAT_DIM="${PROTEIN_CONCAT_DIM:-64}"
PROTEIN_CONCAT_TOPK="${PROTEIN_CONCAT_TOPK:-512}"
PROTEIN_CONCAT_INIT_SCALE="${PROTEIN_CONCAT_INIT_SCALE:-0.1}"
PROTEIN_CONCAT_SEED="${PROTEIN_CONCAT_SEED:-23}"
PROTEIN_CONCAT_SCORE_MODE="${PROTEIN_CONCAT_SCORE_MODE:-multiply}"
PROTEIN_CONCAT_EXPR_SCALE="${PROTEIN_CONCAT_EXPR_SCALE:-1.0}"
CONTROL_LOGIT_SCALE="${CONTROL_LOGIT_SCALE:-0.0}"
PAIR_LOGIT_SCALE="${PAIR_LOGIT_SCALE:-0.0}"
PAIR_LOGIT_GATE="${PAIR_LOGIT_GATE:-0}"
TARGET_LOGIT_SCALE="${TARGET_LOGIT_SCALE:-0.0}"
COVARIATE_LOGIT_SCALE="${COVARIATE_LOGIT_SCALE:-0.0}"
AUX_COVARIATE_LOSS_FIELDS="${AUX_COVARIATE_LOSS_FIELDS:-}"
AUX_COVARIATE_LOSS_WEIGHT="${AUX_COVARIATE_LOSS_WEIGHT:-0.0}"
AUX_COVARIATE_LOSS_LABEL_SMOOTHING="${AUX_COVARIATE_LOSS_LABEL_SMOOTHING:-0.0}"
AUX_COVARIATE_CONTRASTIVE_FIELDS="${AUX_COVARIATE_CONTRASTIVE_FIELDS:-}"
AUX_COVARIATE_CONTRASTIVE_WEIGHT="${AUX_COVARIATE_CONTRASTIVE_WEIGHT:-0.0}"
AUX_COVARIATE_CONTRASTIVE_TEMPERATURE="${AUX_COVARIATE_CONTRASTIVE_TEMPERATURE:-0.2}"
RANKING_LOSS_WEIGHT="${RANKING_LOSS_WEIGHT:-0.0}"
RANKING_LOSS_MARGIN="${RANKING_LOSS_MARGIN:-0.0}"
RANKING_LOSS_GROUP_FIELD="${RANKING_LOSS_GROUP_FIELD:-Cell}"
CELL_PRIOR_MODE="${CELL_PRIOR_MODE:-off}"
CELL_PRIOR_K="${CELL_PRIOR_K:-8}"
CELL_PRIOR_TEMPERATURE="${CELL_PRIOR_TEMPERATURE:-0.2}"
CELL_PRIOR_CHUNK_SIZE="${CELL_PRIOR_CHUNK_SIZE:-512}"
CELL_PRIOR_LOGIT_SCALE="${CELL_PRIOR_LOGIT_SCALE:-0.0}"
CELL_PRIOR_FIXED_LOGIT_SCALE="${CELL_PRIOR_FIXED_LOGIT_SCALE:-0.0}"
COVARIATE_UNK_FOR_UNSEEN="${COVARIATE_UNK_FOR_UNSEEN:-0}"
COVARIATE_UNK_FIELDS="${COVARIATE_UNK_FIELDS:-}"
COVARIATE_UNK_DROPOUT="${COVARIATE_UNK_DROPOUT:-0.0}"
BATCH_COV_LIST="${BATCH_COV_LIST:-}"
USE_DDI="${USE_DDI:-0}"
PIN_MEMORY="${PIN_MEMORY:-1}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
PROGRESS_BAR="${PROGRESS_BAR:-1}"
export PTV_PROGRESS_BAR="${PROGRESS_BAR}"

read -r -a FOLD_LIST <<< "${FOLDS}"

COMMON_TRAIN_ARGS=(
  --dataset-group ptv3
  --model-type "${MODEL_TYPE}"
  --batch-size "${BATCH_SIZE}"
  --max-epochs "${MAX_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --expression-latent-dim "${EXPRESSION_LATENT_DIM}"
  --covariate-embedding-dim "${COVARIATE_EMBEDDING_DIM}"
  --num-heads "${NUM_HEADS}"
  --num-layers "${NUM_LAYERS}"
  --dropout "${DROPOUT}"
  --control-layers "${CONTROL_LAYERS}"
  --fusion-layers "${FUSION_LAYERS}"
  --target-layers "${TARGET_LAYERS}"
  --mse-weight "${MSE_WEIGHT}"
  --mse-inactive-label-weight "${MSE_INACTIVE_LABEL_WEIGHT}"
  --mse-gene-subsample "${MSE_GENE_SUBSAMPLE}"
  --mse-gene-weight-mode "${MSE_GENE_WEIGHT_MODE}"
  --mse-gene-weight-topk "${MSE_GENE_WEIGHT_TOPK}"
  --mse-gene-weight-scale "${MSE_GENE_WEIGHT_SCALE}"
  --active-label-sampling-weight "${ACTIVE_LABEL_SAMPLING_WEIGHT}"
  --positive-label-sampling-weight "${POSITIVE_LABEL_SAMPLING_WEIGHT}"
  --inactive-label-train-ratio "${INACTIVE_LABEL_TRAIN_RATIO}"
  --label-smoothing "${LABEL_SMOOTHING}"
  --optimizer-name "${OPTIMIZER_NAME}"
  --accelerator gpu
  --devices "${DEVICES}"
  --strategy "${STRATEGY}"
  --precision "${PRECISION}"
  --num-workers "${NUM_WORKERS}"
  --gradient-clip-val "${GRADIENT_CLIP_VAL}"
  --checkpoint-dir "${CKPT_DIR}"
  --log-dir "${LOG_DIR}"
  --save-every-n-epochs "${SAVE_EVERY_N_EPOCHS}"
  --save-top-k "${SAVE_TOP_K}"
  --best-ckpt-metric "${BEST_CKPT_METRIC}"
  --logger-backend "${LOGGER_BACKEND}"
  --wandb-project "${WANDB_PROJECT}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS}"
  --check-val-every-n-epoch "${CHECK_VAL_EVERY_N_EPOCH}"
  --limit-train-batches "${LIMIT_TRAIN_BATCHES}"
  --limit-val-batches "${LIMIT_VAL_BATCHES}"
  --limit-test-batches "${LIMIT_TEST_BATCHES}"
  --target-protein-max-length "${TARGET_PROTEIN_MAX_LENGTH}"
  --graph-feature-mode "${GRAPH_FEATURE_MODE}"
  --graph-feature-dim "${GRAPH_FEATURE_DIM}"
  --graph-feature-seed "${GRAPH_FEATURE_SEED}"
  --graph-cache-dir "${GRAPH_CACHE_DIR}"
  --graph-layers "${GRAPH_LAYERS}"
  --graph-init-scale "${GRAPH_INIT_SCALE}"
  --graph-pair-add-scale "${GRAPH_PAIR_ADD_SCALE}"
  --graph-logit-scale "${GRAPH_LOGIT_SCALE}"
  --graph-jump-fusion "${GRAPH_JUMP_FUSION}"
  --graph-jump-gate "${GRAPH_JUMP_GATE}"
  --graph-jump-temperature "${GRAPH_JUMP_TEMPERATURE}"
  --pair-fusion-mode "${PAIR_FUSION_MODE}"
  --cell-pair-film-scale "${CELL_PAIR_FILM_SCALE}"
  --target-expression-mode "${TARGET_EXPRESSION_MODE}"
  --target-expression-dim "${TARGET_EXPRESSION_DIM}"
  --target-expression-topk "${TARGET_EXPRESSION_TOPK}"
  --target-expression-ppi-topk "${TARGET_EXPRESSION_PPI_TOPK}"
  --target-expression-ppi-alpha "${TARGET_EXPRESSION_PPI_ALPHA}"
  --target-expression-init-scale "${TARGET_EXPRESSION_INIT_SCALE}"
  --target-expression-seed "${TARGET_EXPRESSION_SEED}"
  --target-expression-fusion-mode "${TARGET_EXPRESSION_FUSION_MODE}"
  --target-expression-chunk-size "${TARGET_EXPRESSION_CHUNK_SIZE}"
  --protein-concat-mode "${PROTEIN_CONCAT_MODE}"
  --protein-concat-dim "${PROTEIN_CONCAT_DIM}"
  --protein-concat-topk "${PROTEIN_CONCAT_TOPK}"
  --protein-concat-init-scale "${PROTEIN_CONCAT_INIT_SCALE}"
  --protein-concat-seed "${PROTEIN_CONCAT_SEED}"
  --protein-concat-score-mode "${PROTEIN_CONCAT_SCORE_MODE}"
  --protein-concat-expr-scale "${PROTEIN_CONCAT_EXPR_SCALE}"
  --control-logit-scale "${CONTROL_LOGIT_SCALE}"
  --pair-logit-scale "${PAIR_LOGIT_SCALE}"
  --target-logit-scale "${TARGET_LOGIT_SCALE}"
  --covariate-logit-scale "${COVARIATE_LOGIT_SCALE}"
  --aux-covariate-loss-weight "${AUX_COVARIATE_LOSS_WEIGHT}"
  --aux-covariate-loss-label-smoothing "${AUX_COVARIATE_LOSS_LABEL_SMOOTHING}"
  --aux-covariate-contrastive-weight "${AUX_COVARIATE_CONTRASTIVE_WEIGHT}"
  --aux-covariate-contrastive-temperature "${AUX_COVARIATE_CONTRASTIVE_TEMPERATURE}"
  --ranking-loss-weight "${RANKING_LOSS_WEIGHT}"
  --ranking-loss-margin "${RANKING_LOSS_MARGIN}"
  --ranking-loss-group-field "${RANKING_LOSS_GROUP_FIELD}"
  --cell-prior-mode "${CELL_PRIOR_MODE}"
  --cell-prior-k "${CELL_PRIOR_K}"
  --cell-prior-temperature "${CELL_PRIOR_TEMPERATURE}"
  --cell-prior-chunk-size "${CELL_PRIOR_CHUNK_SIZE}"
  --cell-prior-logit-scale "${CELL_PRIOR_LOGIT_SCALE}"
  --cell-prior-fixed-logit-scale "${CELL_PRIOR_FIXED_LOGIT_SCALE}"
  --covariate-unk-dropout "${COVARIATE_UNK_DROPOUT}"
)

if [[ -n "${SAVE_EVERY_N_TRAIN_STEPS}" ]]; then
  COMMON_TRAIN_ARGS+=(--save-every-n-train-steps "${SAVE_EVERY_N_TRAIN_STEPS}")
fi
if [[ "${SAVE_LAST_CKPT}" != "1" ]]; then
  COMMON_TRAIN_ARGS+=(--no-save-last-ckpt)
fi
if [[ -n "${CHECKPOINT_FILENAME}" ]]; then
  COMMON_TRAIN_ARGS+=(--checkpoint-filename "${CHECKPOINT_FILENAME}")
fi
if [[ -n "${MONITOR}" ]]; then
  COMMON_TRAIN_ARGS+=(--monitor "${MONITOR}")
fi
if [[ -n "${MONITOR_MODE}" ]]; then
  COMMON_TRAIN_ARGS+=(--monitor-mode "${MONITOR_MODE}")
fi
if [[ -n "${SCHEDULER_NAME}" ]]; then
  COMMON_TRAIN_ARGS+=(--scheduler-name "${SCHEDULER_NAME}")
fi
if [[ -n "${BCE_WEIGHT}" ]]; then
  COMMON_TRAIN_ARGS+=(--bce-weight "${BCE_WEIGHT}")
fi
if [[ -n "${POSITIVE_WEIGHT}" ]]; then
  COMMON_TRAIN_ARGS+=(--positive-weight "${POSITIVE_WEIGHT}")
fi
if [[ "${GRAPH_STRUCTURAL_RP}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--graph-structural-rp)
fi
if [[ "${GRAPH_MULTIHOP}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--graph-multihop)
fi
if [[ "${GRAPH_DRUG_CONCAT}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--graph-drug-concat)
fi
if [[ "${PAIR_TYPE_FEATURES}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--pair-type-features)
fi
if [[ "${PAIR_LOGIT_GATE}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--pair-logit-gate)
fi
if [[ -n "${TARGET_EXPRESSION_CACHE_DIR}" ]]; then
  COMMON_TRAIN_ARGS+=(--target-expression-cache-dir "${TARGET_EXPRESSION_CACHE_DIR}")
fi
if [[ "${FORCE_TARGET_EXPRESSION_CACHE_REBUILD}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--force-target-expression-cache-rebuild)
fi
if [[ "${USE_DDI}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--use-ddi)
fi
if [[ "${PIN_MEMORY}" != "1" ]]; then
  COMMON_TRAIN_ARGS+=(--no-pin-memory)
fi
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--compile-model)
fi
if [[ "${FOCAL_LOSS}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--focal-loss)
fi
if [[ "${LOG_TO_WANDB}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--log-to-wandb)
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-mode "${WANDB_MODE}")
fi
if [[ "${WANDB_LOG_MODEL}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-log-model)
fi
if [[ "${ALLOW_NONFINITE_MONITOR}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--allow-nonfinite-monitor)
fi
if [[ -n "${WANDB_TAGS}" ]]; then
  read -r -a WANDB_TAG_LIST <<< "${WANDB_TAGS}"
  COMMON_TRAIN_ARGS+=(--wandb-tags "${WANDB_TAG_LIST[@]}")
fi
if [[ "${COVARIATE_UNK_FOR_UNSEEN}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--covariate-unk-for-unseen)
fi
if [[ -n "${COVARIATE_UNK_FIELDS}" ]]; then
  read -r -a COVARIATE_UNK_FIELD_LIST <<< "${COVARIATE_UNK_FIELDS}"
  COMMON_TRAIN_ARGS+=(--covariate-unk-fields "${COVARIATE_UNK_FIELD_LIST[@]}")
fi
if [[ -n "${AUX_COVARIATE_LOSS_FIELDS}" ]]; then
  read -r -a AUX_COVARIATE_LOSS_FIELD_LIST <<< "${AUX_COVARIATE_LOSS_FIELDS}"
  COMMON_TRAIN_ARGS+=(--aux-covariate-loss-fields "${AUX_COVARIATE_LOSS_FIELD_LIST[@]}")
fi
if [[ -n "${AUX_COVARIATE_CONTRASTIVE_FIELDS}" ]]; then
  read -r -a AUX_COVARIATE_CONTRASTIVE_FIELD_LIST <<< "${AUX_COVARIATE_CONTRASTIVE_FIELDS}"
  COMMON_TRAIN_ARGS+=(--aux-covariate-contrastive-fields "${AUX_COVARIATE_CONTRASTIVE_FIELD_LIST[@]}")
fi
if [[ -n "${BATCH_COV_LIST}" ]]; then
  if [[ "${BATCH_COV_LIST}" == "__none__" ]]; then
    COMMON_TRAIN_ARGS+=(--batch-cov-list)
  else
    read -r -a BATCH_COV_FIELD_LIST <<< "${BATCH_COV_LIST}"
    COMMON_TRAIN_ARGS+=(--batch-cov-list "${BATCH_COV_FIELD_LIST[@]}")
  fi
fi

COMMON_INFER_ARGS=(
  --dataset-group ptv3
  --model-type "${MODEL_TYPE}"
  --batch-size "${INFER_BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --expression-latent-dim "${EXPRESSION_LATENT_DIM}"
  --covariate-embedding-dim "${COVARIATE_EMBEDDING_DIM}"
  --num-heads "${NUM_HEADS}"
  --num-layers "${NUM_LAYERS}"
  --dropout "${DROPOUT}"
  --control-layers "${CONTROL_LAYERS}"
  --fusion-layers "${FUSION_LAYERS}"
  --target-layers "${TARGET_LAYERS}"
  --target-protein-max-length "${TARGET_PROTEIN_MAX_LENGTH}"
  --graph-feature-mode "${GRAPH_FEATURE_MODE}"
  --graph-feature-dim "${GRAPH_FEATURE_DIM}"
  --graph-feature-seed "${GRAPH_FEATURE_SEED}"
  --graph-cache-dir "${GRAPH_CACHE_DIR}"
  --graph-layers "${GRAPH_LAYERS}"
  --graph-init-scale "${GRAPH_INIT_SCALE}"
  --graph-pair-add-scale "${GRAPH_PAIR_ADD_SCALE}"
  --graph-logit-scale "${GRAPH_LOGIT_SCALE}"
  --graph-jump-fusion "${GRAPH_JUMP_FUSION}"
  --graph-jump-gate "${GRAPH_JUMP_GATE}"
  --graph-jump-temperature "${GRAPH_JUMP_TEMPERATURE}"
  --pair-fusion-mode "${PAIR_FUSION_MODE}"
  --cell-pair-film-scale "${CELL_PAIR_FILM_SCALE}"
  --target-expression-mode "${TARGET_EXPRESSION_MODE}"
  --target-expression-dim "${TARGET_EXPRESSION_DIM}"
  --target-expression-topk "${TARGET_EXPRESSION_TOPK}"
  --target-expression-ppi-topk "${TARGET_EXPRESSION_PPI_TOPK}"
  --target-expression-ppi-alpha "${TARGET_EXPRESSION_PPI_ALPHA}"
  --target-expression-init-scale "${TARGET_EXPRESSION_INIT_SCALE}"
  --target-expression-seed "${TARGET_EXPRESSION_SEED}"
  --target-expression-fusion-mode "${TARGET_EXPRESSION_FUSION_MODE}"
  --target-expression-chunk-size "${TARGET_EXPRESSION_CHUNK_SIZE}"
  --protein-concat-mode "${PROTEIN_CONCAT_MODE}"
  --protein-concat-dim "${PROTEIN_CONCAT_DIM}"
  --protein-concat-topk "${PROTEIN_CONCAT_TOPK}"
  --protein-concat-init-scale "${PROTEIN_CONCAT_INIT_SCALE}"
  --protein-concat-seed "${PROTEIN_CONCAT_SEED}"
  --protein-concat-score-mode "${PROTEIN_CONCAT_SCORE_MODE}"
  --protein-concat-expr-scale "${PROTEIN_CONCAT_EXPR_SCALE}"
  --control-logit-scale "${CONTROL_LOGIT_SCALE}"
  --pair-logit-scale "${PAIR_LOGIT_SCALE}"
  --target-logit-scale "${TARGET_LOGIT_SCALE}"
  --covariate-logit-scale "${COVARIATE_LOGIT_SCALE}"
)
if [[ "${GRAPH_STRUCTURAL_RP}" == "1" ]]; then
  COMMON_INFER_ARGS+=(--graph-structural-rp)
fi
if [[ "${GRAPH_MULTIHOP}" == "1" ]]; then
  COMMON_INFER_ARGS+=(--graph-multihop)
fi
if [[ "${GRAPH_DRUG_CONCAT}" == "1" ]]; then
  COMMON_INFER_ARGS+=(--graph-drug-concat)
fi
if [[ "${PAIR_TYPE_FEATURES}" == "1" ]]; then
  COMMON_INFER_ARGS+=(--pair-type-features)
fi
if [[ -n "${TARGET_EXPRESSION_CACHE_DIR}" ]]; then
  COMMON_INFER_ARGS+=(--target-expression-cache-dir "${TARGET_EXPRESSION_CACHE_DIR}")
fi
if [[ "${FORCE_TARGET_EXPRESSION_CACHE_REBUILD}" == "1" ]]; then
  COMMON_INFER_ARGS+=(--force-target-expression-cache-rebuild)
fi
if [[ "${USE_DDI}" == "1" ]]; then
  COMMON_INFER_ARGS+=(--use-ddi)
fi

ptv3_utc_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

ptv3_init_time_summary() {
  mkdir -p "$(dirname "${TIME_SUMMARY_PATH}")"
  if [[ ! -f "${TIME_SUMMARY_PATH}" ]]; then
    printf "kind\texperiment\ttask_name\tsplit_strategy\tsplit_name\tstatus\tstart_utc\tend_utc\tduration_sec\tartifact\n" > "${TIME_SUMMARY_PATH}"
  fi
}

ptv3_record_time() {
  local kind="$1"
  local experiment="$2"
  local task_name="$3"
  local split_strategy="$4"
  local split_name="$5"
  local status="$6"
  local start_utc="$7"
  local end_utc="$8"
  local duration_sec="$9"
  local artifact="${10}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${kind}" \
    "${experiment}" \
    "${task_name}" \
    "${split_strategy}" \
    "${split_name}" \
    "${status}" \
    "${start_utc}" \
    "${end_utc}" \
    "${duration_sec}" \
    "${artifact}" >> "${TIME_SUMMARY_PATH}"
}

ptv3_ensure_clean_path() {
  local path="$1"
  local kind="$2"
  if [[ "${ALLOW_EXISTING_RUN}" == "1" ]]; then
    return
  fi
  if [[ -e "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[error] ${kind} already exists and is not empty: ${path}" >&2
    echo "[error] choose a new EXP_PREFIX or set ALLOW_EXISTING_RUN=1 intentionally" >&2
    exit 1
  fi
}

ptv3_print_settings() {
  local title="$1"
  ptv3_init_time_summary
  echo "[experiment] ${title}"
  echo "[settings] EXP_PREFIX=${EXP_PREFIX}"
  echo "[settings] FOLDS=${FOLDS}"
  echo "[settings] MODEL_TYPE=${MODEL_TYPE}; MAX_EPOCHS=${MAX_EPOCHS}"
  echo "[settings] BATCH_SIZE=${BATCH_SIZE}; DEVICES=${DEVICES}; STRATEGY=${STRATEGY}; GPU_IDS=${GPU_IDS}"
  echo "[settings] LEARNING_RATE=${LEARNING_RATE}; OPTIMIZER_NAME=${OPTIMIZER_NAME}; SCHEDULER_NAME=${SCHEDULER_NAME:-none}"
  echo "[settings] GRAPH_FEATURE_MODE=${GRAPH_FEATURE_MODE}; GRAPH_STRUCTURAL_RP=${GRAPH_STRUCTURAL_RP}; GRAPH_DRUG_CONCAT=${GRAPH_DRUG_CONCAT}; GRAPH_PAIR_ADD_SCALE=${GRAPH_PAIR_ADD_SCALE}; GRAPH_LOGIT_SCALE=${GRAPH_LOGIT_SCALE}; USE_DDI=${USE_DDI}; PROTEIN_CONCAT_MODE=${PROTEIN_CONCAT_MODE}"
  echo "[settings] PROTEIN_CONCAT_SCORE_MODE=${PROTEIN_CONCAT_SCORE_MODE}; PROTEIN_CONCAT_EXPR_SCALE=${PROTEIN_CONCAT_EXPR_SCALE}; PROTEIN_CONCAT_TOPK=${PROTEIN_CONCAT_TOPK}"
  echo "[settings] CONTROL_LOGIT_SCALE=${CONTROL_LOGIT_SCALE}; PAIR_LOGIT_SCALE=${PAIR_LOGIT_SCALE}; PAIR_LOGIT_GATE=${PAIR_LOGIT_GATE}; TARGET_LOGIT_SCALE=${TARGET_LOGIT_SCALE}; COVARIATE_LOGIT_SCALE=${COVARIATE_LOGIT_SCALE}"
  echo "[settings] AUX_COVARIATE_LOSS_FIELDS=${AUX_COVARIATE_LOSS_FIELDS:-none}; AUX_COVARIATE_LOSS_WEIGHT=${AUX_COVARIATE_LOSS_WEIGHT}; AUX_COVARIATE_LOSS_LABEL_SMOOTHING=${AUX_COVARIATE_LOSS_LABEL_SMOOTHING}"
  echo "[settings] AUX_COVARIATE_CONTRASTIVE_FIELDS=${AUX_COVARIATE_CONTRASTIVE_FIELDS:-none}; AUX_COVARIATE_CONTRASTIVE_WEIGHT=${AUX_COVARIATE_CONTRASTIVE_WEIGHT}; AUX_COVARIATE_CONTRASTIVE_TEMPERATURE=${AUX_COVARIATE_CONTRASTIVE_TEMPERATURE}"
  echo "[settings] CELL_PAIR_FILM_SCALE=${CELL_PAIR_FILM_SCALE}; TARGET_EXPRESSION_MODE=${TARGET_EXPRESSION_MODE}; TARGET_EXPRESSION_FUSION_MODE=${TARGET_EXPRESSION_FUSION_MODE}; TARGET_EXPRESSION_TOPK=${TARGET_EXPRESSION_TOPK}; TARGET_EXPRESSION_PPI_TOPK=${TARGET_EXPRESSION_PPI_TOPK}; TARGET_EXPRESSION_PPI_ALPHA=${TARGET_EXPRESSION_PPI_ALPHA}"
  echo "[settings] RANKING_LOSS_WEIGHT=${RANKING_LOSS_WEIGHT}; RANKING_LOSS_GROUP_FIELD=${RANKING_LOSS_GROUP_FIELD}; CELL_PRIOR_MODE=${CELL_PRIOR_MODE}; CELL_PRIOR_LOGIT_SCALE=${CELL_PRIOR_LOGIT_SCALE}; CELL_PRIOR_FIXED_LOGIT_SCALE=${CELL_PRIOR_FIXED_LOGIT_SCALE}"
  echo "[settings] MSE_GENE_WEIGHT_MODE=${MSE_GENE_WEIGHT_MODE}; MSE_GENE_WEIGHT_TOPK=${MSE_GENE_WEIGHT_TOPK}; MSE_GENE_WEIGHT_SCALE=${MSE_GENE_WEIGHT_SCALE}"
  echo "[settings] COVARIATE_UNK_FOR_UNSEEN=${COVARIATE_UNK_FOR_UNSEEN}; COVARIATE_UNK_FIELDS=${COVARIATE_UNK_FIELDS:-all-if-enabled}; COVARIATE_UNK_DROPOUT=${COVARIATE_UNK_DROPOUT}"
  echo "[settings] BATCH_COV_LIST=${BATCH_COV_LIST:-train.py-default}"
  echo "[settings] PAIR_FUSION_MODE=${PAIR_FUSION_MODE}; PAIR_TYPE_FEATURES=${PAIR_TYPE_FEATURES}; MSE_INACTIVE_LABEL_WEIGHT=${MSE_INACTIVE_LABEL_WEIGHT}; ACTIVE_LABEL_SAMPLING_WEIGHT=${ACTIVE_LABEL_SAMPLING_WEIGHT}; POSITIVE_LABEL_SAMPLING_WEIGHT=${POSITIVE_LABEL_SAMPLING_WEIGHT}; INACTIVE_LABEL_TRAIN_RATIO=${INACTIVE_LABEL_TRAIN_RATIO}"
  echo "[settings] LOGGER_BACKEND=${LOGGER_BACKEND}; LOG_TO_WANDB=${LOG_TO_WANDB}; WANDB_PROJECT=${WANDB_PROJECT}"
  echo "[settings] LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS}; CHECK_VAL_EVERY_N_EPOCH=${CHECK_VAL_EVERY_N_EPOCH}"
  echo "[settings] PROGRESS_BAR=${PROGRESS_BAR}"
  echo "[settings] BEST_CKPT_METRIC=${BEST_CKPT_METRIC}; MONITOR=${MONITOR:-auto}; MONITOR_MODE=${MONITOR_MODE:-auto}; ALLOW_NONFINITE_MONITOR=${ALLOW_NONFINITE_MONITOR}"
  echo "[settings] REFERENCE_5FOLD_CKPT_PATH=${REFERENCE_5FOLD_CKPT_PATH:-none}; REFERENCE_EPOCH_AGG=${REFERENCE_EPOCH_AGG}; REFERENCE_EPOCH_ROUNDING=${REFERENCE_EPOCH_ROUNDING}; REFERENCE_EPOCH_MIN_COUNT=${REFERENCE_EPOCH_MIN_COUNT}"
  echo "[settings] REFERENCE_REQUIRE_TEST_COMPLETED=${REFERENCE_REQUIRE_TEST_COMPLETED}; REFERENCE_SPLIT_STRATEGY_REGEX=${REFERENCE_SPLIT_STRATEGY_REGEX:-script-default}; REFERENCE_ALLOW_MIXED_CONFIG=${REFERENCE_ALLOW_MIXED_CONFIG}; REFERENCE_ALLOW_DUPLICATE_SPLITS=${REFERENCE_ALLOW_DUPLICATE_SPLITS}"
  echo "[settings] SAVE_EVERY_N_EPOCHS=${SAVE_EVERY_N_EPOCHS}; SAVE_EVERY_N_TRAIN_STEPS=${SAVE_EVERY_N_TRAIN_STEPS:-none}"
  echo "[settings] RUN_PREFLIGHT=${RUN_PREFLIGHT}; RUN_DATA_VALIDATION=${RUN_DATA_VALIDATION}; RUN_INFERENCE=${RUN_INFERENCE}"
  echo "[settings] TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}"
}

ptv3_run_preflight() {
  if [[ "${RUN_PREFLIGHT}" != "1" ]]; then
    return
  fi
  "${PYTHON_BIN}" -m py_compile \
    train.py \
    infer.py \
    dataset/training_ready_dataset.py \
    dataset/training_ready_fast_dataset.py \
    model/fast_delta_model.py \
    model/fast_lightning.py \
    model/graph_feature_utils.py \
    model/training_ready_models.py \
    model/training_ready_lightning.py \
    scripts/check_wandb_auth.py \
    scripts/select_reference_epoch.py
  if [[ "${RUN_DATA_VALIDATION}" == "1" ]]; then
    "${PYTHON_BIN}" utils/01_validate_standardized_outputs.py
    "${PYTHON_BIN}" utils/03_validate_training_ready_outputs.py
  fi
  "${PYTHON_BIN}" scripts/check_wandb_auth.py \
    --logger-backend "${LOGGER_BACKEND}" \
    --log-to-wandb "${LOG_TO_WANDB}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-env-file "${WANDB_ENV_FILE}"
}

ptv3_train() {
  local exp_name="$1"
  local task_name="$2"
  local split_strategy="$3"
  local task_head="$4"
  shift 4

  ptv3_ensure_clean_path "${CKPT_DIR}/${exp_name}" "checkpoint directory"
  ptv3_ensure_clean_path "${LOG_DIR}/${exp_name}" "log directory"

  local start_utc
  local end_utc
  local start_sec
  local end_sec
  local status
  start_utc="$(ptv3_utc_now)"
  start_sec="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u train.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --experiment-name "${exp_name}" \
    --task-name "${task_name}" \
    --split-strategy "${split_strategy}" \
    --task-head "${task_head}" \
    "$@"
  status="$?"
  set -e
  end_utc="$(ptv3_utc_now)"
  end_sec="$(date +%s)"

  ptv3_record_time \
    "train" \
    "${exp_name}" \
    "${task_name}" \
    "${split_strategy}" \
    "-" \
    "${status}" \
    "${start_utc}" \
    "${end_utc}" \
    "$((end_sec - start_sec))" \
    "${CKPT_DIR}/${exp_name}"

  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

ptv3_best_checkpoint() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
exp = Path(sys.argv[1])
manifest_path = exp / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing run manifest: {manifest_path}")
manifest = json.load(manifest_path.open())
if manifest.get("run_status") != "fit_completed":
    raise SystemExit(f"run is not fit_completed: {manifest_path}")
checkpoint = manifest.get("best_model_path") or str(exp / "last.ckpt")
if not Path(checkpoint).exists():
    raise SystemExit(f"checkpoint does not exist: {checkpoint}")
print(checkpoint)' "${CKPT_DIR}/${exp_name}"
}

ptv3_last_checkpoint() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import sys; from pathlib import Path
checkpoint = Path(sys.argv[1]) / "last.ckpt"
if not checkpoint.exists():
    raise SystemExit(f"checkpoint does not exist: {checkpoint}")
print(checkpoint)' "${CKPT_DIR}/${exp_name}"
}

ptv3_reference_epoch() {
  local reference_path="$1"
  local task_name="$2"
  local task_head="$3"
  local split_strategy_regex="$4"
  local summary_json="$5"
  local -a reference_args=(
    "${reference_path}"
    --task-name "${task_name}"
    --expect-task-head "${task_head}"
    --expect-model-type "${MODEL_TYPE}"
    --expect-dataset-group ptv3
    --method "${REFERENCE_EPOCH_AGG}"
    --rounding "${REFERENCE_EPOCH_ROUNDING}"
    --min-count "${REFERENCE_EPOCH_MIN_COUNT}"
    --summary-json "${summary_json}"
  )
  if [[ -n "${split_strategy_regex}" ]]; then
    reference_args+=(--split-strategy-regex "${split_strategy_regex}")
  fi
  if [[ "${REFERENCE_REQUIRE_TEST_COMPLETED}" == "1" ]]; then
    reference_args+=(--require-test-completed)
  fi
  if [[ "${REFERENCE_ALLOW_MIXED_CONFIG}" == "1" ]]; then
    reference_args+=(--allow-mixed-reference-config)
  fi
  if [[ "${REFERENCE_ALLOW_DUPLICATE_SPLITS}" == "1" ]]; then
    reference_args+=(--allow-duplicate-split-strategies)
  fi
  "${PYTHON_BIN}" scripts/select_reference_epoch.py \
    "${reference_args[@]}"
}

ptv3_record_reference_epoch_policy() {
  local exp_name="$1"
  local reference_path="$2"
  local reference_task_name="$3"
  local reference_epoch="$4"
  local applied_max_epochs="$5"
  local reference_split_strategy_regex="$6"
  local reference_summary_json="$7"
  "${PYTHON_BIN}" -c 'import json, sys; from datetime import datetime, timezone; from pathlib import Path
run_dir = Path(sys.argv[1])
manifest_path = run_dir / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing run manifest: {manifest_path}")
selected_checkpoint_path = run_dir / "last.ckpt"
if not selected_checkpoint_path.exists():
    raise SystemExit(f"reference epoch policy requires last.ckpt, but it does not exist: {selected_checkpoint_path}")
reference_summary_path = Path(sys.argv[9])
reference_summary = None
if str(reference_summary_path):
    if not reference_summary_path.exists():
        raise SystemExit(f"missing reference epoch summary: {reference_summary_path}")
    with reference_summary_path.open("r", encoding="utf-8") as handle:
        reference_summary = json.load(handle)
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest["reference_epoch_policy"] = {
    "enabled": True,
    "reference_path": sys.argv[2],
    "reference_task_name": sys.argv[3],
    "selected_epoch": int(sys.argv[4]),
    "applied_max_epochs": int(sys.argv[5]),
    "aggregation": sys.argv[6],
    "rounding": sys.argv[7],
    "min_count": int(sys.argv[8]),
    "require_test_completed": sys.argv[10] == "1",
    "split_strategy_regex": sys.argv[11] or None,
    "allow_mixed_reference_config": sys.argv[12] == "1",
    "allow_duplicate_split_strategies": sys.argv[13] == "1",
    "checkpoint_policy": "fixed_reference_epoch_last_ckpt",
    "selected_checkpoint_path": str(selected_checkpoint_path.resolve()),
    "reference_summary": reference_summary,
    "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
' "${CKPT_DIR}/${exp_name}" \
    "${reference_path}" \
    "${reference_task_name}" \
    "${reference_epoch}" \
    "${applied_max_epochs}" \
    "${REFERENCE_EPOCH_AGG}" \
    "${REFERENCE_EPOCH_ROUNDING}" \
    "${REFERENCE_EPOCH_MIN_COUNT}" \
    "${reference_summary_json}" \
    "${REFERENCE_REQUIRE_TEST_COMPLETED}" \
    "${reference_split_strategy_regex}" \
    "${REFERENCE_ALLOW_MIXED_CONFIG}" \
    "${REFERENCE_ALLOW_DUPLICATE_SPLITS}"
}

ptv3_infer() {
  local checkpoint_path="$1"
  local task_name="$2"
  local task_head="$3"
  local exp_name="$4"
  local -a limit_args=()
  if [[ -n "${INFER_LIMIT_BATCHES}" ]]; then
    limit_args=(--limit-batches "${INFER_LIMIT_BATCHES}")
  fi

  ptv3_ensure_clean_path "${OUTPUT_DIR}/${exp_name}/${task_name}" "inference output directory"

  local start_utc
  local end_utc
  local start_sec
  local end_sec
  local status
  start_utc="$(ptv3_utc_now)"
  start_sec="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u infer.py \
    "${COMMON_INFER_ARGS[@]}" \
    --task-name "${task_name}" \
    --split-strategy test_only \
    --split-name test \
    --task-head "${task_head}" \
    --checkpoint-path "${checkpoint_path}" \
    --output-dir "${OUTPUT_DIR}/${exp_name}/${task_name}" \
    --batch-size "${INFER_BATCH_SIZE}" \
    --device "${INFER_DEVICE}" \
    "${limit_args[@]}"
  status="$?"
  set -e
  end_utc="$(ptv3_utc_now)"
  end_sec="$(date +%s)"

  ptv3_record_time \
    "infer" \
    "${exp_name}" \
    "${task_name}" \
    "test_only" \
    "test" \
    "${status}" \
    "${start_utc}" \
    "${end_utc}" \
    "$((end_sec - start_sec))" \
    "${OUTPUT_DIR}/${exp_name}/${task_name}"

  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

ptv3_done() {
  echo "[done] runtime summary: ${TIME_SUMMARY_PATH}"
}
