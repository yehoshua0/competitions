#!/bin/bash
# ============================================================
# CPU AGENT RUNNER — one-click full inference
# Assumes SGLang is already running at AGENT_MODEL_URL.
# Usage:
#   ./run.sh              — test split, 500 scenarios
#   MAX_SAMPLES=20 DATA_SPLIT=train ./run.sh  — quick train run
# ============================================================

set -e

[ -f .env ] && export $(grep -v '^#' .env | xargs)

SPLIT="${DATA_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
MAX_WORKERS="${MAX_WORKERS:-4}"
SAVE_DIR="./results"

if [ -z "$AGENT_MODEL_URL" ]; then
    echo "ERROR: AGENT_MODEL_URL not set."
    echo "  1. Start SGLang on RunPod (see README.md)"
    echo "  2. Copy the printed URL into .env as AGENT_MODEL_URL"
    echo "  3. Re-run this script"
    exit 1
fi

# ── Verify endpoints ───────────────────────────────────────
echo "=== Checking LLM server: $AGENT_MODEL_URL ==="
curl -sf "$AGENT_MODEL_URL/models" > /dev/null \
    || { echo "ERROR: LLM server not reachable. Is SGLang running on RunPod?"; exit 1; }
echo "  OK"

# ── Start tool server ──────────────────────────────────────
echo "=== Starting tool server (DATA_SPLIT=$SPLIT) ==="
DATA_SPLIT=$SPLIT python server.py &
SERVER_PID=$!
trap "echo 'Stopping tool server...'; kill $SERVER_PID 2>/dev/null" EXIT
sleep 5

curl -sf http://localhost:7860/health > /dev/null \
    || { echo "ERROR: Tool server not responding on :7860"; exit 1; }
echo "  OK"

# ── Run agent ──────────────────────────────────────────────
echo ""
echo "=== Running agent ($MAX_SAMPLES scenarios, split=$SPLIT, workers=$MAX_WORKERS) ==="
mkdir -p "$SAVE_DIR" logs
python main.py \
    --server_url    http://localhost:7860 \
    --max_samples   "$MAX_SAMPLES" \
    --max_iterations 4 \
    --max_workers   "$MAX_WORKERS" \
    --save_dir      "$SAVE_DIR" \
    --log_file      ./logs/run.log \
    --verbose 2>&1 | tee logs/run_stdout.log

# ── Format submission ──────────────────────────────────────
echo ""
echo "=== Formatting submission ==="
python scripts/format_submission.py \
    --predictions "$SAVE_DIR/result.csv" \
    --sample      data/SampleSubmission.csv \
    --output      "$SAVE_DIR/submission.csv"

echo ""
echo "======================================================"
echo "  Done. Upload to Zindi: $SAVE_DIR/submission.csv"
echo "======================================================"
