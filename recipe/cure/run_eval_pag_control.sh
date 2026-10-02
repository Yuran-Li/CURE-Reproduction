#!/usr/bin/env bash
# Single-solution eval on the same sets PAG evaluates.
# Decoding matches PAG val_kwargs (n=8, T=0.6, top_p=0.95).
# This is the first-solution accuracy. It is not PAG's turn-2 accuracy
# and it is not CURE's critique-then-regenerate test-time loop.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAG_ROOT="${PAG_ROOT:-/data/yuranli/LLM/2026.04/github_references/Policy-As-GenVerifier}"

# shellcheck disable=SC1091
source "${CONDA_ROOT:-/data/yuranli/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-cure}"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/data/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/yuranli/hf-cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/yuranli/hf-cache/hub}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"

MODEL_PATH="${MODEL_PATH:-/data/yuranli/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
# Must be a global_step_* directory, not the experiment root.
RESUME_PATH="${RESUME_PATH:-$REPO_ROOT/checkpoints/cure/qwen1p5b_cure/global_step_560}"
TRAIN_DATASET="${TRAIN_DATASET:-$PAG_ROOT/datasets/math7500.parquet}"
VAL_N="${VAL_N:-8}"

math500="$PAG_ROOT/datasets/math500.parquet"
aime2024="$PAG_ROOT/datasets/aime2024.parquet"
aime2025="$PAG_ROOT/datasets/aime2025.parquet"
minervamath="$PAG_ROOT/datasets/minervamath.parquet"
VAL_FILES="${VAL_FILES:-['$math500','$aime2024','$aime2025','$minervamath']}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_cure_qwen1p5b}"

python3 -m verl.trainer.main_ppo \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_PATH" \
    algorithm.adv_estimator=cure \
    algorithm.use_kl_in_reward=False \
    data.train_files="['$TRAIN_DATASET']" \
    data.val_files="$VAL_FILES" \
    data.filter_overlong_prompts=True \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.7}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    custom_reward_function.path="${SCRIPT_DIR}/reward_score.py" \
    custom_reward_function.name=reward_func \
    trainer.logger="['console']" \
    trainer.project_name=cure \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$REPO_ROOT/checkpoints/cure/$EXPERIMENT_NAME" \
    trainer.log_val_generations=2 \
    +trainer.critique=False \
    +trainer.refine=False \
    "$@"
