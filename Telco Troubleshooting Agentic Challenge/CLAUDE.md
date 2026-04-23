# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview

Track A of a global AI Agent competition for wireless network troubleshooting and optimization. Agents must diagnose 5G network issues (single or multiple root causes) by calling a simulation server's tool APIs. Answers are formatted as `C3` (single) or `C5|C7` (multiple, ascending order), extracted from `\boxed{}` notation.

**Base model:** Qwen3.5-35B-A3B (via OpenRouter or local deployment). Can be fine-tuned but not replaced.

**Scoring:** IoU accuracy × time discount (100% < 5 min, 80% for 5–10 min, 60% for 10–15 min, 0% > 15 min).

## Running the Agent

```bash
# 1. Start the tool server (must be running before main.py)
python server.py

# 2. Run the agent against the test set
python main.py \
  --server_url http://localhost:7860 \
  --model_url https://openrouter.ai/api/v1 \
  --model_name qwen/qwen3.5-35b-a3b \
  --max_samples 500 \
  --max_iterations 10 \
  --save_dir ./results \
  --verbose

# Use train split instead of test
DATA_SPLIT=train python server.py
```

Set `AGENT_API_KEY` env var (or edit `main.py` line 27) with your OpenRouter key.

## Architecture

```
server.py  ←→  main.py
   │               │
FastAPI app    AgentsRunner + Environment
(tools/sim)    (OpenAI-compatible client)
```

**`server.py`** — Read-only FastAPI server. Loads scenarios from `data/Phase_1/{test,train}.json`. Routes scenario context via `X-Scenario-Id` request header. Exposes tool endpoints (cell info, RSRP/SINR, KPIs, antenna geometry, etc.) and `/scenario/all` for scenario enumeration.

**`main.py`** — The participant-modifiable agent file. Two classes:
- `Environment`: discovers tools from `/tools`, maps function names to HTTP endpoints, executes tool calls by forwarding them as GET requests with `X-Scenario-Id` header.
- `AgentsRunner`: runs the ReAct-style tool-use loop. `run()` handles a single scenario; `benchmark()` iterates all scenarios and writes `results/result.csv`.

**`_types.py`** — Pydantic models for the full data schema: `Scenario`, `Task`, `ScenarioData` (pipe-delimited CSVs), `ToolCall`, `AgentResult`.

**`utils.py`** — Answer extraction (`extract_answer_all` parses `\boxed{}` from LLM output), `compute_score` (IOU), time-filtering helpers for dataframes, scenario loader.

## Key Implementation Details

- **Answer format:** The LLM must output `\boxed{C3}` or `\boxed{C5|C7}`. `extract_answer_all` grabs the last `\boxed{}` match. `free_mode` (default on via `--free_mode` being `store_false`) adds a follow-up prompt if no boxed answer is found.
- **Tool discovery:** Tools are fetched dynamically at runtime from `/tools` — do not hardcode tool schemas.
- **Scenario routing:** Every tool call includes `scenario_id` via `X-Scenario-Id` header; the server uses this to return data for the correct scenario.
- **Data format:** All scenario data (config, user-plane, signaling, traffic, MR) is stored as pipe-delimited (`|`) CSV strings embedded in the JSON.
- **`server.py` must not be modified** — it is replaced by the organizer in Phase 3.

## Data

- `data/Phase_1/train.json` — 2000 scenarios with questions + answers
- `data/Phase_1/test.json` — 500 scenarios with questions, no answers
- `examples/traces.json` — example agent traces

Output: `results/result.csv` with columns `scenario_id`, `answers`.

## Phase Timeline

| Phase | Dates | Action |
|-------|-------|--------|
| Phase 1 | Apr 3 – May 4 | Submit `result.csv` (unlimited) |
| Phase 2 | May 4–18 | Submit `result.csv` (3 attempts) |
| Phase 3 | May 18–29 | Submit `main.py` + fine-tuned weights (1 attempt) |
