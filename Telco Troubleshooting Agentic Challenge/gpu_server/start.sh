#!/bin/bash
set -e

MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
PORT=30000

echo "--- GPU SERVER STARTUP ---"

# 1. Install Dependencies
echo "[1/3] Installing system dependencies..."
apt-get update
apt-get install -y libnuma1 protobuf-compiler build-essential pkg-config curl

echo "[2/3] Installing SGLang..."
pip install -U pip setuptools wheel
pip install -U "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python"

# 2. Final Prep and Launch
POD_ID=$(curl -s --max-time 2 "http://169.254.254.254/v1/id" || echo "YOUR_POD_ID")

echo ""
echo "[3/3] Starting SGLang Server..."
echo "    ENDPOINT URL: https://${POD_ID}-${PORT}.proxy.runpod.net/v1"
echo "    Wait for 'Server is ready' log below."
echo "------------------------------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp-size 2 \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
