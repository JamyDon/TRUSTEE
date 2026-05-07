#!/usr/bin/env bash
set -e
pip install --upgrade pip
export MAX_JOBS=32

pip install --no-cache-dir "vllm==0.12.0" "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" "tensordict==0.10.0" torchdata

pip install "transformers[hf_xet]>=4.57.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=22.0.0" pandas \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard

pip install "nvidia-ml-py>=13.590.44" "fastapi[standard]>=0.125.0" "pydantic>=2.12.5" "grpcio>=1.76.0"

pip install flash-attn==2.8.1 --no-build-isolation --no-cache-dir
pip install flashinfer-python==0.5.3 --no-build-isolation --no-cache-dir

pip install opencv-python opencv-fixer
pip install uvloop==0.21.0
pip install numpy==1.26.4

cd verl/
pip install -e .
