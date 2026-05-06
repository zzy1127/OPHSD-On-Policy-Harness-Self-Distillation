#!/bin/bash
# ============================================================================
# OPHSD math training (Plan-Solve harness as the privileged-context teacher)
# ============================================================================
# The student p_S(·|x) is also the teacher: a Plan-Solve harness orchestrates
# several calls to the *current* student via the colocated vLLM server,
# producing z(x); the teacher prediction p_S(·|z(x)) is then distilled back
# into the bare-prompt student.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-8B  TRAIN_FILE=/path/to/train.parquet \
#   VAL_FILE=/path/to/val.parquet bash scripts/train_ophsd_math.sh
#
# Required env vars:
#   MODEL_PATH               — student/teacher checkpoint
#   TRAIN_FILE               — verl-format parquet (DeepMath training prompts)
#   VAL_FILE                 — verl-format parquet (math validation prompts)
#
# Useful overrides:
#   TOTAL_TRAINING_STEPS    — default 150
#   TRAIN_BATCH_SIZE        — default 64
#   VAL_N                   — pass@N for validation, default 8
#   OPSD_LOSS_TYPE          — reverse_kl | forward_kl | jsd, default reverse_kl
#   OPSD_MAX_LENGTH         — max seq len for teacher+student, default 12288
#   HARNESS_USE_GT          — true | false, default true
#   ENABLE_THINKING         — True | False (Qwen3 thinking mode), default True
#   RESUME_MODE             — auto | disable | resume_path, default auto
#   RESUME_FROM_PATH        — global_step_* dir to resume from (optional)
# ============================================================================

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"            # ophsd_train/
SRC_ROOT="${REPO_ROOT}/src"
HARNESS_ROOT="${REPO_ROOT}"                            # contains harnesses/

ulimit -n 65535

export PYTHONPATH="${SRC_ROOT}:${HARNESS_ROOT}:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

echo "PYTHONPATH: $PYTHONPATH"

# ============================================================================
# Configuration
# ============================================================================

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TRAIN_FILE:?TRAIN_FILE must point to a verl train parquet}"
: "${VAL_FILE:?VAL_FILE must point to a verl val parquet}"
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
if [ -n "${RESUME_FROM_PATH:-}" ]; then
  resume_args+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

val_n=${VAL_N:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
rollout_n=${ROLLOUT_N:-1}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}

opsd_loss_type=${OPSD_LOSS_TYPE:-reverse_kl}
opsd_chunk_size=${OPSD_CHUNK_SIZE:-256}
opsd_max_length=${OPSD_MAX_LENGTH:-12288}
opsd_sample_ratio=${OPSD_SAMPLE_RATIO:-1.0}
opsd_entropy_alpha=${OPSD_ENTROPY_ALPHA:-2.0}
opsd_jsd_gamma=${OPSD_JSD_GAMMA:-0.0}
opsd_jsd_topk=${OPSD_JSD_TOPK:-0}

# Harness (math / Plan-Solve)
harness_port=${HARNESS_PORT:-8000}
harness_workers=${HARNESS_WORKERS:-100}
harness_val_workers=${HARNESS_VAL_WORKERS:-100}
harness_val_n=${HARNESS_VAL_N:-1}
harness_solve_max_tokens=${HARNESS_SOLVE_MAX_TOKENS:-8192}
harness_plan_max_tokens=${HARNESS_PLAN_MAX_TOKENS:-4096}
harness_use_gt=${HARNESS_USE_GT:-true}

ENABLE_THINKING=${ENABLE_THINKING:-True}
if [ "$ENABLE_THINKING" = "True" ]; then
    val_temperature=${VAL_TEMPERATURE:-0.6}
else
    val_temperature=${VAL_TEMPERATURE:-0.7}
fi
val_top_p=${VAL_TOP_P:-0.8}
val_top_k=${VAL_TOP_K:-20}
val_before_train=${VAL_BEFORE_TRAIN:-False}

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# ============================================================================
# Build experiment name
# ============================================================================

MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')
if [ "$ENABLE_THINKING" = "True" ]; then THINK_TAG="thinking"; else THINK_TAG="nothink"; fi
GT_TAG=""
if [ "$harness_use_gt" = "false" ] || [ "$harness_use_gt" = "False" ]; then GT_TAG="-nogt"; fi

EXP_NAME=${MODEL_NAME_SAFE}-OPHSD-math-${opsd_loss_type}-${THINK_TAG}-lr${learning_rate}-bs${train_batch_size}${GT_TAG}

OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

echo "=== OPHSD Math Training ==="
echo "MODEL_PATH:           $MODEL_PATH"
echo "TRAIN_FILE:           $TRAIN_FILE"
echo "VAL_FILE:             $VAL_FILE"
echo "train_batch_size:     $train_batch_size"
echo "learning_rate:        $learning_rate"
echo "total_training_steps: $total_training_steps"
echo "max_response_length:  $max_response_length"
echo "opsd_loss_type:       $opsd_loss_type"
echo "harness_use_gt:       $harness_use_gt"
echo "val_n (pass@N):       $val_n"
echo "EXP_NAME:             $EXP_NAME"
echo "output_dir:           $output_dir"
echo "==========================="

unset RANK WORLD_SIZE NODE_RANK NNODES MASTER_ADDR MASTER_PORT 2>/dev/null || true
export MASTER_PORT=$(shuf -i 29500-39999 -n 1)

python -m ophsd.main_ophsd \
    --config-path "${SRC_ROOT}/ophsd/config" \
    --config-name ophsd_trainer \
    benchmark=math \
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
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    opsd.loss_type=${opsd_loss_type} \
    opsd.chunk_size=${opsd_chunk_size} \
    opsd.max_length=${opsd_max_length} \
    opsd.sample_ratio=${opsd_sample_ratio} \
    opsd.entropy_alpha=${opsd_entropy_alpha} \
    opsd.jsd_gamma=${opsd_jsd_gamma} \
    opsd.jsd_topk=${opsd_jsd_topk} \
    harness.port=${harness_port} \
    harness.model_name=${MODEL_NAME} \
    harness.model_path=${MODEL_PATH} \
    harness.workers=${harness_workers} \
    harness.val_workers=${harness_val_workers} \
    harness.val_n=${harness_val_n} \
    harness.math.val_data_path=$VAL_FILE \
    harness.math.solve_max_tokens=${harness_solve_max_tokens} \
    harness.math.plan_max_tokens=${harness_plan_max_tokens} \
    harness.math.use_gt=${harness_use_gt} \
    reward.custom_reward_function.path="${SRC_ROOT}/rewards/math_reward.py" \
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

echo ""
echo "=== OPHSD math training completed ==="
echo "Model: $MODEL_NAME"
echo "Checkpoints saved to: $output_dir"
