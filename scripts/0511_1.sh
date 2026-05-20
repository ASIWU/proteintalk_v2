source /mnt/shared-storage-user/wuhao/miniconda3/bin/activate 
conda activate flow_v2
cd /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2
export HF_DATASETS_CACHE=/home/wuhao/beam_wuhao/cache
export HF_HUB_CACHE=/mnt/shared-storage-user/beam/wuhao/hf_cache
export WANDB_CACHE_DIR=/mnt/shared-storage-user/beam/wuhao/wandb_cache
export WANDB_ARTIFACT_CACHE=10GB
export http_proxy=http://wuhao:Za8ZkuZapFh3v2KJf5ytMIbmcu0tyYHmuAqE9QzkUxX1Zwif4GQU9IiT9BNf@proxy.h.pjlab.org.cn:23128
export https_proxy=http://wuhao:Za8ZkuZapFh3v2KJf5ytMIbmcu0tyYHmuAqE9QzkUxX1Zwif4GQU9IiT9BNf@proxy.h.pjlab.org.cn:23128
export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"
# change the base url when restart the server
export WANDB_BASE_URL="http://100.96.30.112:8080"
export WANDB_API_KEY="local-f401d2b9276fb4a6dd1db4c1efee0512723ce6fb"

export NUM_WORKERS=16

EXP_PREFIX="${EXP_PREFIX:-20260512_single_pert_stratified_5fold}" \
LOG_TO_WANDB="${LOG_TO_WANDB:-1}" \
FOLDS="${FOLDS:-0 1 2 3 4}" \
MAX_EPOCHS="${MAX_EPOCHS:-100}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}" \
BATCH_SIZE="${BATCH_SIZE:-128}" \
bash scripts/exp_01_single_pert_stratified_5fold.sh

EXP_PREFIX="${EXP_PREFIX:-20260512_extra_single_all_train_infer}" \
LOG_TO_WANDB="${LOG_TO_WANDB:-1}" \
RUN_INFERENCE="${RUN_INFERENCE:-1}" \
MAX_EPOCHS="${MAX_EPOCHS:-100}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}" \
BATCH_SIZE="${BATCH_SIZE:-128}" \
REFERENCE_5FOLD_CKPT_PATH="${REFERENCE_5FOLD_CKPT_PATH:-checkpoints/20260512_single_pert_stratified_5fold}" \
REFERENCE_EPOCH_AGG="${REFERENCE_EPOCH_AGG:-median}" \
REFERENCE_EPOCH_MIN_COUNT="${REFERENCE_EPOCH_MIN_COUNT:-5}" \
SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}" \
SCHEDULER_NAME="${SCHEDULER_NAME:-}" \
bash scripts/exp_07_extra_single_all_train_infer.sh
