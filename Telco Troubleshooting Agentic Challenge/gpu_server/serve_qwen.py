"""
Modal serving script for Qwen/Qwen3.5-35B-A3B (MoE) via vLLM.

Exposes a fully OpenAI-compatible /v1 API including:
  - Tool / function calling  (required by both Track A and Track B agents)
  - Reasoning / thinking tokens  (surfaces as reasoning_content in responses)
  - Streaming

Compatible with main.py's OpenAI client as-is.

Usage:
    # 1. Download weights (once)
    modal run serve_qwen.py

    # 2a. Dev mode — live reload, URL printed to terminal
    modal serve serve_qwen.py

    # 2b. Persistent deployment
    modal deploy serve_qwen.py

    # 3. Point your agent at the Modal endpoint
    PROVIDER_API_URL="https://<workspace>--qwen-serve-serve.modal.run/v1" \
    PROVIDER_API_KEY="dummy" \
    python main.py ...
"""

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "Qwen/Qwen3.5-35B-A3B"   # official HF repo
MODEL_DIR  = "/models/qwen35-a3b"
SERVED_AS  = "Qwen/Qwen3.5-35B-A3B"   # model name the API will answer to

# Qwen3.5-35B-A3B: ~70 GB in bf16 (MoE stores all expert weights)
# 2x A100-80GB  → tp=2, 35 GB/GPU → comfortable
# 1x H100-80GB  → tp=1 is *too tight*; use H100:2 or quantize first
GPU_CONFIG          = "A100-80GB:2"
TENSOR_PARALLEL     = 2
MAX_MODEL_LEN       = 32768
GPU_MEM_UTILIZATION = 0.92

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "hf-transfer",
    )
    .pip_install(
        # Qwen3.5 requires vLLM nightly (qwen3/qwen3_coder parsers not in stable release)
        "vllm",
        extra_options="--extra-index-url https://wheels.vllm.ai/nightly",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Use FlashAttention-2 via Triton (ships with vLLM ≥0.8, no extra install)
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        }
    )
)

model_volume = modal.Volume.from_name("qwen35-a3b-weights", create_if_missing=True)

app = modal.App("qwen-serve", image=vllm_image)


# ---------------------------------------------------------------------------
# Weight download  (run once before serving)
# ---------------------------------------------------------------------------
@app.function(
    volumes={MODEL_DIR: model_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model():
    import os
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],          # safetensors only
        token=os.environ.get("HF_TOKEN"),
    )
    model_volume.commit()
    print(f"Done — weights in {MODEL_DIR}")


# ---------------------------------------------------------------------------
# vLLM OpenAI-compatible API server
#
# modal.web_server() exposes whatever listens on PORT 8000 as a public HTTPS
# endpoint. vLLM's built-in API server handles:
#   - /v1/chat/completions  (with tool_calls + reasoning_content)
#   - /v1/models
#   - streaming (SSE)
#   - concurrent requests (via vLLM's own scheduler)
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU_CONFIG,
    volumes={MODEL_DIR: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,   # keep 1 container alive permanently — avoids cold-start loop
)
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=900)   # CUDA graph compilation can take 10+ min
def serve():
    import subprocess

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        # Model
        "--model",                MODEL_DIR,
        "--served-model-name",    SERVED_AS,
        "--tokenizer",            MODEL_DIR,
        "--trust-remote-code",

        # Precision + parallelism
        "--dtype",                "bfloat16",
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--gpu-memory-utilization", str(GPU_MEM_UTILIZATION),
        "--max-model-len",        str(MAX_MODEL_LEN),

        # Throughput tuning
        "--enable-chunked-prefill",
        "--max-num-seqs",         "32",

        # Only compile CUDA graphs for small batch sizes (agent sends 1-2 requests at a time).
        # This cuts cold-start compilation time from ~10 min down to ~1 min.
        "--cudagraph-capture-sizes", "1", "2", "4", "8", "16",

        # ── Tool calling ─────────────────────────────────────────────────
        # qwen3_coder is the correct parser for Qwen3.5 tool-call format.
        "--enable-auto-tool-choice",
        "--tool-call-parser",     "qwen3_coder",

        # ── Reasoning / thinking tokens ──────────────────────────────────
        # --reasoning-parser alone enables reasoning; no --enable-reasoning flag.
        # vLLM strips <think>…</think> from content → reasoning_content field.
        "--reasoning-parser",     "qwen3",

        # Server
        "--port",  "8000",
        "--host",  "0.0.0.0",
    ]

    print("Starting vLLM server:", " ".join(cmd))
    subprocess.Popen(cmd)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(download: bool = True):
    if download:
        print("Downloading model weights (this may take ~20 min on first run)...")
        download_model.remote()

    print()
    print("Weights ready. Start the server with:")
    print("  modal serve serve_qwen.py    # dev mode")
    print("  modal deploy serve_qwen.py   # persistent")
    print()
    print("Then set in your .env:")
    print('  PROVIDER_API_URL="https://<workspace>--qwen-serve-serve.modal.run/v1"')
    print('  PROVIDER_API_KEY="dummy"')
    print('  MODEL="Qwen/Qwen3.5-35B-A3B"')
