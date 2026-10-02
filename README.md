# CURE: Critique-Driven Unified Reinforcement Learning for Test-Time Self-Improvement

Official repository for the ACL 2026 paper **CURE: Critique-Driven Unified Reinforcement Learning for Test-Time Self-Improvement**.

## Method Overview

![CURE method overview](assets/method_overview.png)

CURE is a unified RLVR framework for training a single policy to solve, self-verify, critique, and re-explore. Instead of relying on external teacher feedback at test time, CURE teaches the model to generate high-level strategic hints and use them to restart reasoning from a fresh context. This critique-driven loop helps reduce anchoring on initial incorrect solutions and enables iterative test-time self-improvement for reasoning tasks.

## Install

CURE is based on `verl` 0.5.0. You can install the runtime by following the
[official verl 0.5.x documentation](https://verl.readthedocs.io/en/v0.5.x/), or
use the dependency snapshot in this repository.

```bash
conda create -n cure python=3.11 -y
conda activate cure
pip install -r requirements.txt
pip install -e . --no-deps
```

Install cluster-specific CUDA/PyTorch packages according to your hardware and
the verl documentation.

## Data Prep

CURE uses training data from a subset of
[`zwhe99/DeepMath-103K`](https://huggingface.co/datasets/zwhe99/DeepMath-103K).
Following the paper, remove examples whose answers are binary and randomly
downsample the remaining pool to 76,800 examples.

Next, prepare the dataset for training with verl:

```bash
python -m recipe.cure.scripts.prepare_data \
  --input-path data/DeepMath-103K/train.parquet \
  --output-path data/DeepMath-103K/train_76k8.parquet
```

Use `--input-path` and `--output-path` to point the script to your local
preprocessed DeepMath-103K parquet files.

## Controlled comparison with PAG

This checkout can train CURE against the local PAG run `qwen1p5b_pag`.
The initial checkpoint is Qwen2.5-1.5B-Instruct, not the SFT checkpoint.
Only the training algorithm changes. The comparison is not the CURE paper's
DeepMath-76.8k / Qwen2.5-7B setting.

| Variable | PAG `qwen1p5b_pag` | CURE `qwen1p5b_cure` |
| --- | --- | --- |
| Base model | Qwen2.5-1.5B-Instruct | same |
| Initial checkpoint | `models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | same |
| Training set | `datasets/math7500.parquet` (7500) | same file, prompts not rewritten |
| Answer reward | `math_verify`, last 300 chars, `boxed_match_priority=0`, math-verify 0.9.0 | `recipe/cure/reward_score.py` |
| Prompts / step | 512 (`drop_last`, 14 steps/epoch) | 512 |
| Optimizer steps | 40 epochs = 560 | 40 epochs = 560 |
| Prompts seen | 286720 | 286720 |
| Generations / step | 4096 answers+verifies, plus a revision when GenRM says wrong (4096–6144) | 2048 solves + 1024 critiques + 4096 guided = 7168 |
| Evaluation files | MATH-500 during training; AIME 2024/2025 and Minerva in the standalone eval | same files, same verifier |
| Val decode | n=8, temperature 0.6, top_p 0.95 | same, but CURE's trainer scores the first solution only |
| Algorithm | PAG (GAE, GenRM verify, selective revision) | CURE (`adv_estimator=cure`, critique + guided regeneration) |

The `cure` conda env is already installed at `/data/yuranli/envs/cure`.
Do not train in the `PAG` env: both packages are named `verl`, and this env
points at the CURE tree.

| Component | Installed |
| --- | --- |
| Python | 3.11.16 |
| torch | 2.6.0+cu124 |
| vLLM | 0.8.5.post1 |
| transformers | 4.54.0 |
| flash-attn | 2.7.4.post1 (cu124 / torch 2.6 wheel; CURE's pin is 2.7.3) |
| math-verify | 0.9.0, with `latex2sympy2_extended==1.11.0` (same checker as PAG) |

Difficulty bins on this base model and `math7500` are already measured
(K=4, temperature 1.0, max 2048 tokens, shared answer reward).
Full dump: `outputs/pass_rate/math7500_qwen1p5b_base.json`.

| Bin | Count | Probability |
| --- | --- | --- |
| Zero-pass | 2588 | 0.345 |
| Partial-pass | 3536 | 0.471 |
| Full-pass | 1376 | 0.183 |

Mean accuracy is 0.424. Zero + partial is 0.816, so the critique pool is not
empty and the set is not saturated. Report these three probabilities with the
comparison. The generation gap (7168 vs 4096–6144) is the method. Report
prompts seen and rollout count; one epoch of each method is not a matched budget.

### Start training

From the CURE repo, on 8 GPUs:

```bash
cd /data/yuranli/LLM/2026.04/github_references/CURE
bash recipe/cure/run_pag_control.sh
```

The script activates `cure`, sets `PYTHONNOUSERSITE=1` and `VLLM_USE_V1=0`,
and writes checkpoints to `checkpoints/cure/qwen1p5b_cure`. WandB is on by
default; set `USE_WANDB=0` for console only. Hydra overrides can be appended
after the script name.

`verl/trainer/config/data/legacy_data.yaml` was missing from this checkout.
It is restored from verl 0.5.0; without it Hydra exits before any GPU work
with `Could not find 'data/legacy_data'`.

### Eval after a checkpoint exists

`RESUME_PATH` must be a `global_step_*` directory. This scores the first
solution only. PAG's `final_acc` includes the revision turn, so the two
numbers are not the same metric.

```bash
cd /data/yuranli/LLM/2026.04/github_references/CURE
RESUME_PATH=checkpoints/cure/qwen1p5b_cure/global_step_560 \
  bash recipe/cure/run_eval_pag_control.sh
```

The paper entry point is unchanged: `bash recipe/cure/run_qwen2.5-7b_math.sh`.

## Run Main Experiment

```bash
bash recipe/cure/run_qwen2.5-7b_math.sh
```

The script runs on 4 GPUs by
default. The most commonly changed environment variables are:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` | HF model id or local checkpoint path |
| `DATA_HOME` | `$PWD/data` | Directory containing `DeepMath-103K/train_76k8.parquet` and validation files |
| `CHECKPOINT_DIR` | `$PWD/checkpoints` | Base checkpoint output directory |
| `WANDB_API_KEY` | unset | Set in your shell before running if using W&B |

You can also override verl/Hydra arguments at the end of the command.

## Citation

```bibtex
@inproceedings{chen-etal-2026-cure,
  title     = {{CURE}: Critique-Driven Unified Reinforcement Learning for Test-Time Self-Improvement},
  author    = {Guirong Chen and Shuqi Ye and Wenkai Yang and Shiqi Shen and Guangyao Shen and Yankai Lin},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2026.acl-long.1321/}
}
```

## Acknowledgements

This project is built upon [verl](https://github.com/volcengine/verl). We sincerely thank the verl team and community for their great work! 😀

## License

Apache-2.0. See `LICENSE`; upstream verl attribution is included in `NOTICE`.

