#!/bin/bash
# ============================================================
# SMOKE TEST — validates the full pipeline on 5 train scenarios
# Run this BEFORE the full inference to catch config errors.
# Assumes SGLang is already running at AGENT_MODEL_URL.
# ============================================================

set -e

[ -f .env ] && export $(grep -v '^#' .env | xargs)

if [ -z "$AGENT_MODEL_URL" ]; then
    echo "ERROR: AGENT_MODEL_URL not set. Copy .env.example to .env and fill in your RunPod URL."
    exit 1
fi

echo "=== Verifying LLM endpoint: $AGENT_MODEL_URL ==="
curl -sf "$AGENT_MODEL_URL/models" > /dev/null \
    || { echo "ERROR: LLM server not reachable at $AGENT_MODEL_URL"; exit 1; }
echo "  LLM server OK"

echo "=== Starting tool server (train split) ==="
DATA_SPLIT=train python server.py &
SERVER_PID=$!
trap "echo 'Stopping tool server...'; kill $SERVER_PID 2>/dev/null" EXIT
sleep 5

curl -sf http://localhost:7860/health > /dev/null \
    || { echo "ERROR: Tool server not responding on :7860"; exit 1; }
echo "  Tool server OK"

echo "=== Running 5 train scenarios ==="
mkdir -p results_smoke logs
python main.py \
    --server_url   http://localhost:7860 \
    --max_samples  5 \
    --max_iterations 4 \
    --save_dir     ./results_smoke \
    --log_file     ./logs/smoke.log \
    --verbose

echo ""
echo "=== Validation report ==="
python scripts/validate.py \
    --results results_smoke/result.csv \
    --data    data/Phase_1/train.json \
    --top_errors 5

echo ""
echo "Smoke test complete. Check IoU above before running full inference."
