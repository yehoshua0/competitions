# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview

**Telco Troubleshooting and Optimization Agentic Challenge** — €40,000 prize pool, two tracks.

- **Track A (Wireless):** Agent diagnoses 5G network faults (single or multi-root-cause) by calling a simulation server's HTTP tool APIs. Answers formatted as `C3` (single) or `C5|C7` (multiple, ascending). Extracted from `\boxed{}` notation.
- **Track B (IP Networks):** Agent troubleshoots IP network issues via CLI interactions for Huawei/Cisco/H3C devices. Exact-match scoring.

**Base model (both tracks):** Qwen3.5-35B-A3B. Fine-tuning (LoRA, full) is allowed; swapping to a different architecture is not.

**Scoring (Phase 3 only applies the time discount):**
- Score = IoU accuracy × time discount
- < 5 min → 100% | 5–10 min → 80% | 10–15 min → 60% | > 15 min → 0%

**Phase timeline:**

| Phase | Dates | Details |
|-------|-------|---------|
| Phase 1 | Apr 3 – May 4 | Unlimited submissions; 1,000 API calls/day |
| Phase 2 | May 4–18 | Top 30 advance; 3 submissions max |
| Phase 3 | May 18–29 | 1 attempt; submit `main.py` + optional fine-tuned weights |

## Track A Architecture

```
server.py  ←→  main.py
   │               │
FastAPI app    Environment + AgentsRunner
(tools/sim)    (OpenAI-compatible client)
```

**`server.py`** (read-only, replaced by organizer in Phase 3) — FastAPI app that loads scenarios from `data/Phase_1/{test,train}.json`. Routes scenario context via `X-Scenario-Id` request header. Exposes tool endpoints (cell info, RSRP/SINR, KPIs, antenna geometry, etc.) and `/scenario/all`.

**`main.py`** (participant-modifiable) — Two classes:
- `Environment`: discovers tools from `/tools`, maps function names to HTTP endpoints via `endpoint_mapper`, executes tool calls as GET requests with `X-Scenario-Id` header.
- `AgentsRunner`: ReAct-style tool-use loop. `run()` handles a single scenario; `benchmark()` iterates all scenarios and writes `results/result.csv`. `free_mode` (default on via `--free_mode` using `store_false`) adds a follow-up prompt if no boxed answer is found.

**`_types.py`** — Pydantic models: `Scenario`, `Task`, `ScenarioData` (pipe-delimited CSVs), `ToolCall`, `AgentResult`.

**`utils.py`** — Answer extraction (`extract_answer_all` parses `\boxed{}`, takes last match), `compute_score` (IoU), time-filtering helpers, scenario loader.

**`logger.py`** — Standard Python logger with stdout + stderr handlers and optional file output.

## Running Track A

```bash
cd "Track A"

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

# Phase 3: local GPU deployment (vLLM or SGLang, OpenAI-compatible)
# AGENT_API_KEY="dummy" works for local servers that ignore Authorization header
```

Set `AGENT_API_KEY` env var (or edit `main.py` line 27) with your OpenRouter key for Phase 1/2 cloud testing.

## Key Implementation Constraints

- **All data access must go through the tool server HTTP API** — never read `data/Phase_1/test.json` directly. In Phase 3, the JSON files will not exist; only the server API will be available.
- **Tool schemas must be discovered dynamically** at runtime from `/tools` — do not hardcode them.
- **`server.py` must not be modified** — it is replaced by the organizer in Phase 3.
- **No external telecom datasets** — only the provided `data/` files.
- **Open-source tools only** — no AutoML, no paid-only services.
- **Phase 1 patterns do not generalize** — the organizers explicitly warned that Phase 1 can be solved by pattern matching. Solutions must reason from actual network signal data to survive Phase 2/3.

## Track A Data

- `data/Phase_1/train.json` — 2000 scenarios with questions + answers
- `data/Phase_1/test.json` — 500 scenarios with questions, no answers
- `examples/traces.json` — example agent traces
- Output: `results/result.csv` with columns `scenario_id`, `answers`
- All scenario data (config, user-plane, signaling, traffic, MR) is stored as pipe-delimited (`|`) CSV strings embedded in the JSON.

## Track B Structure

```
Track B/
├── agent/
│   ├── evaluate_openclaw.py         # Evaluation runner
│   ├── openclaw_config/             # Agent identity/tool/soul config (AGENTS.md, TOOLS.md, etc.)
│   ├── skills/                      # Skill modules (adv_tunnel, l2_link, l3_route, infra_maintenance)
│   └── requirements.txt
├── data/Phase_1/test.json           # 50 IP network questions (Phase 1)
└── devices_outputs/                 # Pre-captured CLI output files per scenario/device
```

Track B uses an OpenCLAW-based agent that troubleshoots Huawei/Cisco/H3C IP network devices. Skills cover L2 links, L3 routing, advanced tunnels, and infrastructure maintenance. Scoring is exact match (not IoU).

## Submission

- `submission/Phase_1/result.csv` — current submission file
- `submission/Phase_1/submission_example.csv` — format reference
- Format: `scenario_id`, `answers` columns; multi-answer separated by `|` in ascending order (e.g., `C5|C7`)

## Leaderboard Context

| Date | Track | Config | Score |
|------|-------|--------|-------|
| 2026-04-17 | A | LAYER 0 only (rule engine, no LLM) | 0.464 |

LAYER 0 covers ~60–70% of scenarios with deterministic rules. Remaining scenarios returned empty answers (no LLM fallback). LAYER 1 (LLM) is the next improvement target.
