#!/usr/bin/env bash
# Dedicated CURE env. Do not install this verl into the PAG env.
# Torch/CUDA follow the local PAG machine (CUDA 12.4). vLLM and the
# flash-attn wheel are installed after torch so the ABI matches.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/data/yuranli/miniconda3}"
ENV_NAME="${CONDA_ENV:-cure}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.11 -y
fi
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

python -m pip install -U pip
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install "vllm==0.8.5.post1"

# Match PAG's answer checker. CURE's requirements.txt pins 0.8.0; do not use it.
python -m pip install "math-verify==0.9.0"

python -m pip install \
  accelerate==1.6.0 \
  codetiming==1.4.0 \
  datasets==3.5.0 \
  dill==0.3.8 \
  fastapi==0.115.12 \
  hydra-core==1.3.2 \
  latex2sympy2_extended==1.11.0 \
  numpy==1.26.4 \
  packaging==24.2 \
  pandas==2.2.3 \
  peft==0.15.1 \
  pyarrow==19.0.1 \
  pybind11==2.13.6 \
  pylatexenc==2.10 \
  "ray[default]==2.47.0" \
  tensordict==0.6.2 \
  torchdata==0.11.0 \
  transformers==4.54.0 \
  uvicorn==0.34.0 \
  wandb==0.20.1

# flash-attn: prefer the prebuilt cu124 / torch2.6 / cxx11abiFALSE wheel.
# CURE pins 2.7.3; 2.7.4.post1 is the wheel that matches this torch ABI.
PY_TAG="$(python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
if ! python -c "import flash_attn" >/dev/null 2>&1; then
  wget -nv -O "/tmp/${WHL}" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${WHL}"
  python -m pip install "/tmp/${WHL}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python -m pip install -e "${REPO_ROOT}" --no-deps

python - << 'PY'
import torch, vllm, transformers, flash_attn, math_verify
print("python", __import__("sys").version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("math_verify", getattr(math_verify, "__version__", "ok"))
PY
echo "SETUP_DONE env=${ENV_NAME}"
