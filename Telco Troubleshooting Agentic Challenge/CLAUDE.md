# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview

Track A of a global AI Agent competition for wireless network troubleshooting and optimization. Agents must diagnose 5G network issues (single or multiple root causes) by calling a simulation server's tool APIs. Answers are formatted as `C3` (single) or `C5|C7` (multiple, ascending order), extracted from `\boxed{}` notation.

**Base model:** Qwen3.5-35B-A3B — must be **self-hosted** (e.g. RunPod + SGLang). External commercial APIs (OpenRouter, OpenAI, Anthropic) are **forbidden** by competition rules.

**Scoring:** IoU accuracy × time discount (100% < 5 min, 80% for 5–10 min, 60% for 10–15 min, 0% > 15 min).

## Running the Agent

```bash
# 1. Copy environment template and fill in your RunPod SGLang URL
cp .env.example .env
# Edit .env: set AGENT_MODEL_URL=https://<your-runpod-pod>.proxy.runpod.net/v1

# 2. One-click full run (starts server, runs agent, formats submission)
./run.sh

# Quick partial run on train split
MAX_SAMPLES=20 DATA_SPLIT=train ./run.sh

# Manual run (useful for debugging)
DATA_SPLIT=train python server.py &   # start tool server
python main.py --server_url http://localhost:7860 --max_samples 20 --verbose

# Offline validation against train.json ground truth
python scripts/validate.py --results results/result.csv --data data/Phase_1/train.json
```

**Required env vars** (set in `.env` or shell):
- `AGENT_MODEL_URL` — SGLang `/v1` base URL on RunPod
- `AGENT_API_KEY` — API key (any string works for self-hosted SGLang)

## Architecture

```
server.py  ←→  main.py
   │               │
FastAPI app    Environment + AgentsRunner
(tools/sim)    │
               ├── src/prefetch.py       parallel tool prefetch
               ├── src/options_parser.py option → {action, cells}
               └── src/system_prompt.py  static expert prompt
```

**`server.py`** — Read-only FastAPI server provided by organizers. Must not be modified. Loads scenarios from `data/Phase_1/{test,train}.json`, routes context via `X-Scenario-Id` header, exposes tool endpoints and `/scenario/all`.

**`main.py`** — The participant-modifiable agent. Key classes:
- `Environment` — discovers tools from `/tools`, maps function names to HTTP endpoints via `endpoint_mapper`, executes tool calls as GET requests with `X-Scenario-Id` header.
- `AgentsRunner` — runs the ReAct loop. `run()` handles one scenario (prefetch → build context → ≤4 LLM turns → format fallback → canonicalize answer). `benchmark()` iterates all scenarios and writes `results/result.csv`.

**`src/prefetch.py`** — `build_prefetch_bundle()` runs three parallel stages using `ThreadPoolExecutor` to gather tool evidence before the first LLM call: (1) throughput logs + config + KPI + all PCIs; (2) signaling, serving cell signals, neighbor PCIs at the worst-throughput timestamp; (3) cell info and neighbor RSRP for the serving and top-3 neighbor PCIs.

**`src/options_parser.py`** — Parses the scenario's option list (no LLM, no data access) into a structured dict: action type (decrease_a3_offset, set_pdcch_2sym, etc.), cell IDs (regex `\d+_\d+`), and `by_action`/`by_cell` indices.

**`src/system_prompt.py`** — Static expert prompt (~600 tokens). Contains the 7 canonical fault→action patterns (P1–P7), the diagnostic protocol, and cell attribution rules. Static so SGLang's RadixAttention can KV-cache it across all 500 scenarios.

**`_types.py`** — Pydantic models: `Scenario`, `Task`, `ScenarioData` (pipe-delimited CSV strings), `ToolCall`, `AgentResult`.

**`utils.py`** — `extract_answer_all` (parses last `\boxed{}`), `compute_score` (Jaccard IoU), time-filtering helpers, `load_scenarios`.

**`scripts/validate.py`** — Offline IoU analysis against `train.json`. Reports per-bucket IoU by cardinality (single/multi), signaling pattern (a3_diff, a3_same, a2_only, a2_a5, rrc_only), and Cx option presence. Use this to identify the worst-performing bucket and direct improvements.

**`scripts/format_submission.py`** — Aligns `result.csv` to the `SampleSubmission.csv` schema for Zindi upload.

**`misc/`** — Reference/experimental code (rule engine, RAG, layer-1 approaches). Not imported by `main.py`.

## Key Implementation Details

- **Run loop cap:** `max_iterations` is hard-capped to `min(self.max_iterations, 4)` in `run()` regardless of the CLI flag, because the prefetch already handles most evidence-gathering.
- **Answer format:** `_canonicalize_answer_from_sources()` extracts `C\d+` IDs from the LLM output, filters to valid option IDs, deduplicates, and sorts ascending. A format-fallback turn is injected when no `\boxed{}` is found after the ReAct loop.
- **Tool discovery:** Fetched dynamically from `/tools` at runtime — do not hardcode tool schemas.
- **Scenario routing:** Every tool call passes `scenario_id` via `X-Scenario-Id` header.
- **Data format:** All scenario data is pipe-delimited (`|`) CSV embedded in JSON. `_parse_pipe_table()` in `prefetch.py` deserializes these.
- **`server.py` must not be modified** — it is replaced by the organizer in Phase 3.
- **Accessing raw scenario data directly** (bypassing tools) is explicitly **forbidden** by competition rules.

## Data

- `data/Phase_1/train.json` — 2000 scenarios with questions + answers (for local validation)
- `data/Phase_1/test.json` — 500 scenarios with questions, no answers (submit these)
- `data/SampleSubmission.csv` — reference format for Zindi upload

Output: `results/result.csv` (columns: `scenario_id`, `answers`), `results/submission.csv` (Zindi-ready).

## Infrastructure (self-hosted model)

The model is served via SGLang on RunPod (2× A100 80GB):

```bash
# On RunPod pod
sglang serve \
  --model-path Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 --port 30000 \
  --tp-size 2 --disable-custom-all-reduce
```

The RunPod proxy URL becomes `AGENT_MODEL_URL` in `.env`.

## Phase Timeline

| Phase | Dates | Action |
|-------|-------|--------|
| Phase 1 | Apr 3 – May 4 | Submit `result.csv` (unlimited) |
| Phase 2 | May 4–18 | Submit `result.csv` (3 attempts) |
| Phase 3 | May 18–29 | Submit `main.py` + fine-tuned weights (1 attempt) |
