#!/bin/bash
# ============================================================================
# OPHSD USPTO-50K (reaction classification) training — Draft-Verify harness
# ============================================================================
# Same Draft-Verify harness as ``train_ophsd_lawbench.sh`` but with English
# prompts and the 10-class USPTO label space.  Memory-bank entries store
# reaction SMILES as ``fact`` for cosine retrieval.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-8B \
#   TRAIN_FILE=/path/to/uspto/train.parquet \
#   VAL_FILE=/path/to/uspto/val_uspto.parquet \
#   HARNESS_TRAIN_DATA=/path/to/uspto/data_train_10k.json \
#   HARNESS_VAL_DATA=/path/to/uspto/val_uspto.parquet \
#   bash scripts/train_ophsd_uspto.sh
#
# Required env vars:
#   MODEL_PATH, TRAIN_FILE, VAL_FILE,
#   HARNESS_TRAIN_DATA, HARNESS_VAL_DATA  (see comments above for formats)
# ============================================================================

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"
HARNESS_ROOT="${REPO_ROOT}"

ulimit -n 65535
export PYTHONPATH="${SRC_ROOT}:${HARNESS_ROOT}:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

# ============================================================================
# Configuration
# ============================================================================

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TRAIN_FILE:?TRAIN_FILE must point to a verl train parquet}"
: "${VAL_FILE:?VAL_FILE must point to a verl val parquet}"
: "${HARNESS_TRAIN_DATA:?HARNESS_TRAIN_DATA must point to the uspto train JSON}"
: "${HARNESS_VAL_DATA:?HARNESS_VAL_DATA must point to the uspto val JSON or parquet}"

MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
learning_rate=${LEARNING_RATE:-1e-6}
total_training_steps=${TOTAL_TRAINING_STEPS:-150}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-10}
resume_mode=${RESUME_MODE:-auto}
resume_args=(trainer.resume_mode="$resume_mode")
[ -n "${RESUME_FROM_PATH:-}" ] && resume_args+=(trainer.resume_from_path="$RESUME_FROM_PATH")

val_n=${VAL_N:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
rollout_n=${ROLLOUT_N:-1}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}

opsd_loss_type=${OPSD_LOSS_TYPE:-reverse_kl}
opsd_chunk_size=${OPSD_CHUNK_SIZE:-256}
opsd_max_length=${OPSD_MAX_LENGTH:-8192}

harness_port=${HARNESS_PORT:-8000}
harness_workers=${HARNESS_WORKERS:-100}
harness_val_workers=${HARNESS_VAL_WORKERS:-100}
harness_val_n=${HARNESS_VAL_N:-1}
draft_k=${DRAFT_K:-5}
confirm_k=${CONFIRM_K:-5}
challenge_k=${CHALLENGE_K:-5}
cold_start_threshold=${COLD_START_THRESHOLD:-10}
embedding_model=${EMBEDDING_MODEL:-BAAI/bge-small-zh-v1.5}

ENABLE_THINKING=${ENABLE_THINKING:-True}
val_temperature=${VAL_TEMPERATURE:-0.6}
val_top_p=${VAL_TOP_P:-0.8}
val_top_k=${VAL_TOP_K:-20}
val_before_train=${VAL_BEFORE_TRAIN:-False}

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ============================================================================
# Build experiment name
# ============================================================================

MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')
if [ "$ENABLE_THINKING" = "True" ]; then THINK_TAG="thinking"; else THINK_TAG="nothink"; fi
EXP_NAME=${MODEL_NAME_SAFE}-OPHSD-uspto-${opsd_loss_type}-${THINK_TAG}-lr${learning_rate}-bs${train_batch_size}

OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

echo "=== OPHSD USPTO Training ==="
echo "MODEL_PATH:           $MODEL_PATH"
echo "TRAIN_FILE:           $TRAIN_FILE"
echo "VAL_FILE:             $VAL_FILE"
echo "HARNESS_TRAIN_DATA:   $HARNESS_TRAIN_DATA"
echo "HARNESS_VAL_DATA:     $HARNESS_VAL_DATA"
echo "EMBEDDING_MODEL:      $embedding_model"
echo "EXP_NAME:             $EXP_NAME"
echo "============================"

unset RANK WORLD_SIZE NODE_RANK NNODES MASTER_ADDR MASTER_PORT 2>/dev/null || true
export MASTER_PORT=$(shuf -i 29500-39999 -n 1)

python -m ophsd.main_ophsd \
    --config-path "${SRC_ROOT}/ophsd/config" \
    --config-name ophsd_trainer \
    benchmark=uspto \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_FILE']" \
    data.return_raw_chat=True \
    data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_util \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    opsd.loss_type=${opsd_loss_type} \
    opsd.chunk_size=${opsd_chunk_size} \
    opsd.max_length=${opsd_max_length} \
    harness.port=${harness_port} \
    harness.model_name=${MODEL_NAME} \
    harness.model_path=${MODEL_PATH} \
    harness.workers=${harness_workers} \
    harness.val_workers=${harness_val_workers} \
    harness.val_n=${harness_val_n} \
    harness.uspto.train_data_path=${HARNESS_TRAIN_DATA} \
    harness.uspto.val_data_path=${HARNESS_VAL_DATA} \
    harness.uspto.embeddings_path=${HARNESS_TRAIN_EMBEDDINGS:-} \
    harness.uspto.val_embeddings_path=${HARNESS_VAL_EMBEDDINGS:-} \
    harness.uspto.embedding_model=${embedding_model} \
    harness.uspto.draft_k=${draft_k} \
    harness.uspto.confirm_k=${confirm_k} \
    harness.uspto.challenge_k=${challenge_k} \
    harness.uspto.cold_start_threshold=${cold_start_threshold} \
    reward.custom_reward_function.path="${SRC_ROOT}/rewards/uspto_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=ophsd \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    +trainer.validation_data_dir=$output_dir \
    trainer.val_before_train=${val_before_train} \
    trainer.log_val_generations=10 \
    trainer.save_freq=$save_freq \
    "${resume_args[@]}" \
    trainer.test_freq=$test_freq \
    trainer.total_training_steps=$total_training_steps

echo "=== OPHSD uspto training completed ==="
echo "Checkpoints saved to: $output_dir"
