#!/usr/bin/env bash
# CURE as a controlled comparison against the local PAG run
# qwen1p5b_pag from Qwen2.5-1.5B-Instruct (outputs/2026-07-20/15-01-31).
# The initial checkpoint is the base Instruct model, not the SFT checkpoint.
#
# Locked to PAG:
#   initial checkpoint, math7500 train set, math500 val set,
#   answer reward (recipe/cure/reward_score.py == PAG math_verify),
#   prompts per step, optimizer steps, decode limits for the answer,
#   val decode (n=8, T=0.6, top_p=0.95).
# Left as CURE:
#   adv_estimator=cure, critique + guided regeneration, experience replay.
#
# Budget (drop_last, 7500 // 512 = 14 steps/epoch, 40 epochs):
#   optimizer steps          560          (same as PAG)
#   prompts seen             560 * 512 = 286720
#   PAG rollouts / step      512 * 4 answers + 512 * 4 verifies
#                            + revisions only when GenRM says wrong
#                            = 4096 to 6144 generations
#   CURE rollouts / step     512 * 4 solves
#                            + 1024 critiques
#                            + 1024 * 4 guided regenerations
#                            = 7168 generations
#   CURE is about 1.2x-1.8x PAG inference per step. That gap is the
#   critique/refine structure, not a larger prompt budget. Log both.
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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
N_GPUS="${N_GPUS:-8}"

MODEL_PATH="${MODEL_PATH:-/data/yuranli/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
TRAIN_DATASET="${TRAIN_DATASET:-$PAG_ROOT/datasets/math7500.parquet}"
VAL_FILES="${VAL_FILES:-['$PAG_ROOT/datasets/math500.parquet']}"

PROJECT_NAME="${PROJECT_NAME:-PAG}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen1p5b_cure}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"

# Matched to PAG qwen1p5b_pag (base Instruct, math7500, 40 epochs).
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-40}"
N_SOLVE="${N_SOLVE:-4}"
# Official CURE ratio: critique_batch_size / (train_batch * n) = 256 / 512 = 1/2.
CRITIQUE_BATCH_SIZE="${CRITIQUE_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * N_SOLVE / 2))}"
N_CRITIQUE="${N_CRITIQUE:-1}"
N_REFINE="${N_REFINE:-4}"
# Answer length matches PAG. Prompt length is longer because a critique
# prompt contains the question and the full solution.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

# USE_WANDB=1 writes the same step dict to wandb.
# val/multiturn/* is math500, n=8. Each sample is generate, then a free critique
# (verify), then one regen unless the critique says the answer is correct.
# Those keys are not logged on the train batch. grpo/solve_* is.
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  LOGGER="['console','wandb']"
else
  LOGGER="['console']"
fi

echo "[cure-pag-control] MODEL_PATH=$MODEL_PATH"
echo "[cure-pag-control] TRAIN_DATASET=$TRAIN_DATASET"
echo "[cure-pag-control] VAL_FILES=$VAL_FILES"
echo "[cure-pag-control] batch=$TRAIN_BATCH_SIZE epochs=$TOTAL_EPOCHS solve_n=$N_SOLVE critique_batch=$CRITIQUE_BATCH_SIZE refine_n=$N_REFINE"
echo "[cure-pag-control] planned optimizer steps=$(( (7500 / TRAIN_BATCH_SIZE) * TOTAL_EPOCHS )) prompts_seen=$(( (7500 / TRAIN_BATCH_SIZE) * TOTAL_EPOCHS * TRAIN_BATCH_SIZE ))"
echo "[cure-pag-control] planned generations/step=$(( TRAIN_BATCH_SIZE * N_SOLVE + CRITIQUE_BATCH_SIZE * N_CRITIQUE + CRITIQUE_BATCH_SIZE * N_REFINE ))"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=cure \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="['$TRAIN_DATASET']" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation=error \
    +data.critique_batch_size="$CRITIQUE_BATCH_SIZE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.6}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="$N_SOLVE" \
    +actor_rollout_ref.rollout.critique_kwargs.n="$N_CRITIQUE" \
    +actor_rollout_ref.rollout.refine_kwargs.n="$N_REFINE" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    custom_reward_function.path="${SCRIPT_DIR}/reward_score.py" \
    custom_reward_function.name=reward_func \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.val_before_train=True \
    trainer.log_val_generations=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=8 \
    trainer.default_local_dir="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.test_freq=25 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.resume_mode=auto \
    +trainer.critique=True \
    +trainer.critique_focus_on_failed_solutions=True \
    +trainer.critique_interval_start=0 \
    +trainer.critique_interval_end=0.9 \
    +trainer.critique_train_discriminability=True \
    +trainer.critique_force_error_conclusion=True \
    +trainer.critique_format_reward=True \
    +trainer.refine=True \
    +trainer.refine_only_wrong_original=True \
    +trainer.disable_refine_training=False \
    +trainer.experience_replay=True \
    +trainer.disable_solving_training=False \
    "$@"
