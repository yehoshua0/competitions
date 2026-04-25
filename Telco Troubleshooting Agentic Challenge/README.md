# Track A: Telco Troubleshooting and Optimization Agentic Challenge

## Competition Description

A **€40,000** global AI competition hosted on [Zindi](https://zindi.africa/competitions/telco-troubleshooting-agentic-challenge) challenging participants to build intelligent agents capable of diagnosing and troubleshooting wireless network faults in realistic 5G telecom environments.

Agents interact with a domain-specific **Agent Tool Server** (FastAPI simulation sandbox) that exposes structured tool APIs — covering cell info, RSRP/SINR measurements, KPIs, antenna geometry, signaling logs, and more. The agent must call these tools, reason over the returned data, and identify the root cause(s) from a set of labeled options.

**Base model:** Qwen3.5-35B-A3B — mandatory, may be fine-tuned (LoRA, full fine-tuning) but cannot be replaced or scaled to a different parameter count.

### Tracks

| Track | Domain | Training Set | Test Set | Answer Format |
|-------|--------|-------------|----------|---------------|
| **A (this repo)** | Wireless / 5G | 2,000 questions + answers | 500 questions | `C3` or `C5\|C7` (ascending) |
| B | IP Networks | 50 questions | — | Open-ended exact match |

Only one track prize per person/team — winning Track A disqualifies from Track B.

### Scoring

**Score = IoU accuracy × time discount**

- **IoU accuracy** = intersection(predicted, ground_truth) / union(predicted, ground_truth)
- **Time discount:** < 5 min → 100%, 5–10 min → 80%, 10–15 min → 60%, > 15 min → 0%

Leaderboard: 30% public / 70% private split. Tiebreaker: earliest submission timestamp.

### Prizes (Track A)

| Place | Prize | Extras |
|-------|-------|--------|
| 1st | €12,500 | Leader pass + up to $3,500 travel to MWC Shanghai 2026 |
| 2nd | €5,000 | — |
| 3rd | €2,500 | — |

Plus 10,000 Zindi points distributed across top finishers.

### Phase Timeline

| Phase | Dates | Action | Submissions |
|-------|-------|--------|-------------|
| Phase 1 | Apr 3 – May 4 | Practice; unlimited `result.csv` submissions | Unlimited (max 1,000 API calls/day) |
| Phase 2 | May 4 – May 18 | Elimination; top 30 advance | Up to 3 result submissions |
| Phase 3 | May 18 – May 29 | Final; submit `main.py` + fine-tuned weights | 1 attempt, run by organizer |

Winners announced: **May 29, 2026**.

---

## Architecture

> **Thesis:** The metric is IoU, not accuracy. Every spurious Cx in the prediction expands the denominator. Every missed Cx shrinks the numerator. The winning move is a tightly-constrained context: a static expert prompt encoding 5G fault knowledge, a parallel tool prefetch that surfaces the signal *before* the LLM reasons, and a short ReAct loop (≤4 iterations) that confirms rather than discovers.
>
> **Compliance note:** Per host clarification (21 Apr 2026), solutions must work exclusively through server tools — no direct access to raw scenario JSON. The `misc/` folder contains earlier exploratory code (rule engine, RAG at-inference, layer1 prompt builder) that reads raw data and is **invalid at inference**. It is kept as reference only.
>
> **Reference:** `misc/Qwen3-32BAI-Telco-main/` is the winning solution from the previous edition of this competition (same organizer, same tool-server format). Consult it for prompt design, rule logic, and submission patterns when stuck.

### Pipeline

```
  OFFLINE (built once from train.json, shipped with submission)
  ┌─────────────────────────────────────────────────────────────┐
  │  RAG index  (signaling pattern + metric similarity)          │
  │  → up to 3 few-shot examples injected into system prompt    │
  └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Static System Prompt  (src/system_prompt.py)                │
  │  · 7 canonical 5G fault patterns with diagnostic criteria   │
  │  · Per-pattern: key signals, confirming tools, Cx mapping   │
  │  · Closes option set: "answer is one or more of C1–C16"    │
  │  · Cached by SGLang RadixAttention → ~3× throughput gain   │
  └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Parallel Tool Prefetch  (before LLM loop)                   │
  │  · get_signaling_plane_event_log                             │
  │  · get_kpi_data                                              │
  │  · get_config_data + get_cell_info                          │
  │  3–4 calls dispatched concurrently → ≤2 s                  │
  └─────────────────────────────┬───────────────────────────────┘
                                 │ prefetch results → tool messages
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Qwen3.5-35B-A3B ReAct Loop  (max 4 iterations)              │
  │  SGLang on RunPod A40 — AWQ INT4, ~120 tok/s                │
  │  Follow-up tool calls for targeted confirmation only        │
  │  → \boxed{C3}  or  \boxed{C5|C7}                           │
  └─────────────────────────────────────────────────────────────┘
```

### Training data landscape (2000 scenarios)

| Signaling pattern | Count |    % | Likely fault class |
|-------------------|------:|-----:|-------------------|
| a3_same           |   985 | 49.2 | Antenna tilt / PDCCH / transport (serving cell issue) |
| a3_diff           |   542 | 27.1 | HO threshold / missing neighbor (inter-cell issue) |
| a2_a5             |   245 | 12.2 | Coverage + inter-freq gap |
| a2_only           |   142 |  7.1 | Weak coverage |
| none              |    86 |  4.3 | Config-only — use KPI / antenna geometry |

### Infrastructure (updated)

| Component | Spec | Est. cost |
|---|---|---:|
| GPU pod | **2× NVIDIA A100 80GB** on Runpod (Secure Cloud) | **~$2.98/hr** |
| Model | Qwen/Qwen3.5-35B-A3B (served with TP=2) | — |
| Serving | **SGLang (GitHub main)** + OpenAI-compatible endpoints | — |
| Public access | Runpod **HTTPS proxy URL**: `https://<POD_ID>-30000.proxy.runpod.net/v1/...` | — |
| Notes | 1× A100 80GB can be fragile; **2× A100** is the smooth “always starts” config | — |

> Why 2× A100? In practice the model + runtime allocations (KV cache, mamba cache, CUDA graphs, etc.) can be tight on 1×80GB. With 2×80GB (tensor parallel), startup is stable and throughput is better.

---

### GPU setup (Runpod — run once per session)

#### Pod creation (recommended config)

1. Go to https://console.runpod.io/deploy and deploy a **GPU Pod** (Secure Cloud).
2. Choose **A100 80GB** and set:
   - **GPU count:** `2`
   - **Image / Template:** a Runpod PyTorch image (example: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
   - **Container disk:** `120 GB`
   - **Volume disk:** `300 GB`
   - **Expose ports:** `30000/http` (OpenAI API), plus optional `22/tcp` and `8888/http`

3. Wait until the pod is **Running**.
4. In the pod page, open **Connect**:
   - Copy the **HTTP endpoint** corresponding to `30000/http`.
   - It will look like:
     ```
     https://<POD_ID>-30000.proxy.runpod.net
     ```

---

#### One-time install requirements (inside the pod)

SGLang main requires a few build dependencies.

```bash
apt-get update
apt-get install -y \
  libnuma1 \
  protobuf-compiler \
  build-essential \
  pkg-config \
  curl
```

Install Rust (needed because SGLang main builds a Rust component):

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source /root/.cargo/env
```

Upgrade pip tooling:

```bash
python -m pip install -U pip setuptools wheel
```

---

#### Install SGLang (GitHub main) + launch Qwen3.5 with tool calling

Install SGLang main (pip 26+ syntax):

```bash
pip install -U "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python"
```

Launch the server (TP=2, Qwen tool + reasoning parsers):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sglang serve \
  --model-path Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 2 \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

This exposes OpenAI-compatible endpoints at:

- `http://localhost:30000/v1/chat/completions` (inside pod)
- `https://<POD_ID>-30000.proxy.runpod.net/v1/chat/completions` (public via Runpod proxy)

---

#### Smoke test on the GPU pod (curl)

Health:

```bash
curl -s http://127.0.0.1:30000/health
```

Tool calling (must return `tool_calls`, not only JSON in `content`):

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3.5-35B-A3B",
    "messages":[{"role":"user","content":"Use the calculator tool: what is 2+2?"}],
    "tools":[{"type":"function","function":{"name":"calculator","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],
    "tool_choice":"required",
    "temperature":0.0,
    "max_completion_tokens":64
  }' | jq '.choices[0].message'
```

---

#### What to copy back to CPU machine

Set the **base URL** (must end with `/v1`) in your `.env`:

```dotenv
AGENT_MODEL_URL=https://<POD_ID>-30000.proxy.runpod.net/v1
AGENT_MODEL_NAME=Qwen/Qwen3.5-35B-A3B
AGENT_API_KEY=dummy
```

Your CPU agent will then call:

`https://<POD_ID>-30000.proxy.runpod.net/v1/chat/completions`
```

---

### CPU setup (VS Code / local machine)

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure `.env`**

```bash
cp .env.example .env
# Edit .env — paste the RunPod URL from the GPU step:
# AGENT_MODEL_URL=https://<POD_ID>-30000.proxy.runpod.net
```

---

### Smoke test (always run before full inference)

Validates the full pipeline end-to-end on 5 training scenarios with known ground truth.

```bash
bash smoke_test.sh
```

**What to check in the output:**
- `LLM server OK` and `Tool server OK` (both endpoints reachable)
- Mean IoU > 0 (model is producing answers, not blanks)
- Each scenario wall-time < 5 minutes (time discount is 100% under 5 min)
- `\boxed{...}` appears in the verbose log (extraction working)

---

### Full inference run

Once the smoke test passes:

```bash
bash run.sh
# Output: results/submission.csv  (ready to upload to Zindi)
```

For a quick train-split run (has ground truth for measuring IoU):

```bash
MAX_SAMPLES=20 DATA_SPLIT=train bash run.sh
python scripts/validate.py --results results/result.csv --data data/Phase_1/train.json
```

Expected wall time: ~500 scenarios × ~120 s ≈ 17 hours at BF16 on an A100.  
`results/result.csv` is checkpointed every 10 scenarios — a crash is recoverable.

---

### Phase 3 submission zip

The organizer requires: `readme.md` + `main.py` + weights. The GPU and CPU code must be delivered separately.

**Zip contents:**

```
submission.zip
├── README.md              ← this file
├── gpu_server/
│   ├── start.sh           ← one-click GPU server
│   └── requirements.txt   ← GPU deps
├── main.py                ← agent (CPU)
├── server.py              ← tool server (CPU, provided by organizer — included for testing)
├── _types.py
├── logger.py
├── utils.py
├── misc/utils/__init__.py
├── scripts/
│   ├── format_submission.py
│   └── validate.py
├── data/Phase_1/train.json
├── data/SampleSubmission.csv
├── requirements.txt       ← CPU deps
├── .env.example
├── run.sh                 ← one-click CPU runner
└── smoke_test.sh          ← pre-launch validation
```

**Build the zip (from repo root):**

```bash
zip -r submission.zip \
    README.md gpu_server/ main.py server.py _types.py logger.py utils.py \
    misc/utils/ scripts/ data/Phase_1/train.json data/SampleSubmission.csv \
    requirements.txt .env.example run.sh smoke_test.sh \
    --exclude "**/__pycache__/*" "**/*.pyc" "**/.env"
```

**Organizer instructions (paste into their terminal):**

```bash
# GPU machine — run once:
bash gpu_server/start.sh
# → prints AGENT_MODEL_URL=https://...

# CPU machine — set URL then run:
echo "AGENT_MODEL_URL=https://<POD_ID>-30000.proxy.runpod.net" > .env
bash smoke_test.sh   # verify
bash run.sh          # generate results/submission.csv
```

**Before submitting:**
- [ ] Replace `server.py` in the zip with the organizer's final version if updated
- [ ] Include fine-tuned weights path in `gpu_server/start.sh` if LoRA adapter is used
- [ ] Run `bash smoke_test.sh` from a clean clone to confirm nothing is missing

