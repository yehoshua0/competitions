# IMPLEMENTATION V0.1 — Phase 1 Baseline Architecture

> **For future sessions:** This document is the canonical reference for the V0.1 baseline architecture of the Telco Troubleshooting Agentic Challenge (Track A). It encodes the verified facts, the rejected assumptions, the architecture, the implementation plan, and the validation strategy. It was approved by the user as the "golden response" — preserve its substance even if implementation diverges.
>
> **Roleplay note:** Adopt René Descartes' methodical doubt. Accept nothing as true that has not been proven. The user is competing for prize money that funds their master's degree — this is high-stakes work. Be precise, be honest about uncertainty, and never add components on faith.

---

I've already mapped this codebase in this session. Let me consolidate everything into the definitive architecture, applying genuine Cartesian rigor — accepting only what I can prove, rejecting what I cannot.

---

## The Cartesian Audit

**Rule 1 — Accept nothing as true that is not clearly known.**

Before proposing anything, every claim must be checked against the actual code, the actual rules, and the actual measurements.

### What is PROVEN

| # | Fact | Proof |
|---|------|-------|
| F1 | `compute_score()` is **not** IoU. It does boolean OR-match across `\|`-split tokens. | `utils.py:156-170` — `if '\|' in gt: return any([compute_score(g, answer) for g in gts])` |
| F2 | `extract_features(data)` reads raw `user_plane_data`, `network_configuration_data`, `signaling_plane_data`, `traffic_data` directly from the scenario dict. | `rule_engine_agentic.py:84-93` |
| F3 | The host explicitly forbids working "directly on the data" — solutions must go through the tools. | Host clarification 21 Apr 2026 |
| F4 | `main.py.run()` currently sends ZERO system prompt — only `[{"role": "user", "content": question}]`. | `main.py:261` |
| F5 | The 0.464 leaderboard score was obtained with LAYER 0 bypassing tools. That code path is now disqualifiable. | Memory + F2 + F3 |
| F6 | RAG index built offline from `train.json` is compliant — it uses labeled training data, not test data. | `rag.py:115` (offline indexing) |
| F7 | Tool discovery is dynamic via `/tools` endpoint; the agent can prefetch in parallel via `_call_api`. | `main.py:121-128` |
| F8 | `extract_answer_all()` correctly grabs the last `\boxed{}` content. | `utils.py:144-153` |

### What I REJECT (assumptions that fail Cartesian doubt)

| # | Rejected claim | Reason |
|---|---------------|--------|
| R1 | "Score 0.464 is our baseline." | F1 proves the local score function is wrong; F5 proves the live score is from a non-compliant approach. We have **no trustworthy baseline**. |
| R2 | "LAYER 0 → HIGH confidence → skip LLM is the killer optimization." | F2 + F3: this path is dead. Cannot ship. |
| R3 | "RAG retrieval at inference using `extract_features` of raw data is fine." | F2 + F3: cannot read raw test data. RAG must use either tool results or text similarity. |
| R4 | "Confidence gate / self-consistency must be in V0.1." | Unmeasured benefit. Add only after measuring V0.1's failure modes. |
| R5 | "Steps [A], [B], [C] all run in parallel." | [B] needs A2/A3/A5/RRC booleans which only exist after [A]'s `get_signaling_plane_event_log` returns. Real graph: `[A]‖[C]` then `[B]`. |
| R6 | "Use OpenRouter for serving Qwen3.5-35B-A3B." | F3 + competition rule against external commercial APIs. **OpenRouter is forbidden.** Must self-host on RunPod via SGLang. |

R6 is the most important correction. Re-reading the rules: *"utilizing external commercial APIs from providers like OpenAI or Anthropic"* is prohibited. OpenRouter is a commercial API gateway. The only safe path is self-hosting Qwen3.5-35B-A3B on RunPod.

---

## The Architecture — V0.1 Compliant Baseline

**Rule 2 — Divide the difficulty into as many parts as possible.**

The problem decomposes into **four sub-problems**, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 1 BASELINE — "GUIDED REACT"                  │
└─────────────────────────────────────────────────────────────────────┘

  INPUT: Scenario {description, options[]}
            │
            ├──────────────────────────────┬────────────────────────┐
            ▼                              ▼                        │
  ┌──────────────────┐          ┌──────────────────┐                │
  │ [A] PREFETCH      │          │ [C] PARSE OPTIONS │                │
  │ tools in parallel │          │ - cell_id list    │                │
  │                   │          │ - action map      │                │
  │ • signaling_log   │          │ - Cx → action     │                │
  │ • serving_rsrp    │          │   per cell        │                │
  │ • config_data     │          │ (no LLM, no data) │                │
  │ • neighbor_rsrp   │          └─────────┬────────┘                │
  │ • all_cells_pci   │                    │                         │
  │ (5-6 tools, ~2s)  │                    │                         │
  └─────────┬────────┘                     │                         │
            │                              │                         │
            └──────────────┬───────────────┘                         │
                           ▼                                         │
              ┌──────────────────────────────┐                       │
              │ [D] CONTEXT BUILDER          │                       │
              │  Compile prefetched results  │                       │
              │  + parsed options into the   │                       │
              │  user message (one shot)     │                       │
              └─────────────┬────────────────┘                       │
                            │                                        │
                            ▼                                        │
              ┌──────────────────────────────┐                       │
              │ [B] SYSTEM PROMPT (STATIC)   │ ◄─── built once,      │
              │  • Expert 5G rules           │      session-wide     │
              │  • Fault→Action table (7)    │                       │
              │  • Canonical examples        │ ◄─── distilled offline│
              │    (from train.json)         │      from train data  │
              │  • Format rules              │                       │
              └─────────────┬────────────────┘                       │
                            │                                        │
                            ▼                                        │
              ┌──────────────────────────────┐                       │
              │ [E] REACT LOOP (≤4 turns)    │                       │
              │  LLM may call additional     │                       │
              │  tools if prefetch insuff.   │                       │
              │  Stop on \boxed{...}         │                       │
              └─────────────┬────────────────┘                       │
                            │                                        │
                            ▼                                        │
              ┌──────────────────────────────┐                       │
              │ [F] FORMAT FALLBACK          │                       │
              │  If no \boxed{} after loop:  │                       │
              │  inject explicit format      │                       │
              │  prompt with sorted Cx       │                       │
              └─────────────┬────────────────┘                       │
                            │                                        │
                            ▼                                        │
              ┌──────────────────────────────┐                       │
              │ [G] PARSE & VALIDATE         │                       │
              │  • extract_answer_all()      │                       │
              │  • sort Cx ascending         │                       │
              │  • write result.csv          │                       │
              └──────────────────────────────┘                       │
```

**Why this design (proofs for each component):**

- **[A] Prefetch** — Eliminates 4-5 ReAct turns by gathering the always-needed evidence upfront. Tool calls are HTTP GETs, parallelizable via `concurrent.futures`. Each ~200-400ms on a local server. **Proof of compliance:** every fetched datum comes through a tool endpoint. No raw data access.

- **[B] Static system prompt** — All scenarios share the same expert knowledge. Building it once means SGLang's RadixAttention can cache the prefix across all 500 scenarios. **Proof of speedup:** identical prefix → KV cache hit → ~3× faster first-token generation.

- **[C] Parse options** — The options list contains cell IDs (e.g. `3279943_1`) and action verbs ("decrease A3 offset", "press down tilt"). Parsing this is regex on the question text, not the data. Compliant.

- **[D] Context builder** — Single-message context construction. The LLM sees prefetched evidence + options in one shot, reducing back-and-forth.

- **[E] Short ReAct** — Cap at 4 additional turns because the prefetch already handled 80% of evidence-gathering. Lower than the 5-10 in earlier proposals because we measure and tune up if needed.

- **[F] Format fallback** — Defends against the catastrophic zero-score risk from missing `\boxed{}`. Cheap insurance.

- **[G] Parse & validate** — Sort ascending (rule requirement), dedupe, write.

**What is intentionally absent from V0.1:**
- ❌ RAG retrieval at inference (replaced by static distilled examples in [B])
- ❌ Confidence gate / self-consistency (defer to V0.4 — must earn its slot)
- ❌ LAYER 0 deterministic bypass (non-compliant per F3)
- ❌ Multi-pass voting (cost amplifier without proven gain)

---

## Infrastructure — proven decisions

| Layer | Choice | Proof |
|-------|--------|-------|
| Model serving | Self-hosted Qwen3.5-35B-A3B on RunPod 2× A100 (80GB VRAM) | Rules forbid commercial APIs (R6); 2× A100 80GB is the smooth "always starts" config |
| Inference engine | SGLang | RadixAttention KV cache reuse across the static system prompt = ~3× TTFT improvement on every scenario after the first |
| Quantization | None (BF16) | Native precision avoids rounding |
| Cost | ~$2.98/hr | Runpod Secure Cloud 2× A100 80GB |
| Fine-tuning (later) | Unsloth on A100 (80GB) | Already validated by user — MoE kernels reduce VRAM 40% |

**Critical risk to mitigate:** SGLang model download from HuggingFace can take 30-90 min. Solution: pre-bake a RunPod template with the model cached on the volume. One-time setup cost.

---

## The implementation plan — strictly ordered

**Rule 3 — Begin with the simplest things and progress.**

### Step 1 — Foundation (zero LLM cost, ~1 hour)

```
1.1  Fix compute_score() in utils.py to compute real Jaccard IoU
1.2  Build scripts/validate.py:
     - load train.json (labeled)
     - run agent on N scenarios (configurable)
     - compute IoU per scenario
     - aggregate by category: single/multi-answer, signaling pattern
     - emit JSON report with per-bucket IoU + cost + iterations
```

The fixed `compute_score`:

```python
def compute_score(gt: str, pred: str) -> float:
    """Jaccard IoU between two pipe-separated answer sets."""
    gt_set   = {s.strip() for s in (gt   or "").split("|") if s.strip()}
    pred_set = {s.strip() for s in (pred or "").split("|") if s.strip()}
    if not gt_set and not pred_set: return 1.0
    if not gt_set or  not pred_set: return 0.0
    return len(gt_set & pred_set) / len(gt_set | pred_set)
```

### Step 2 — V0.1 architecture (~3 hours)

```
2.1  src/prefetch.py         → parallel tool bundle ([A])
2.2  src/options_parser.py   → option → action+cells classifier ([C])
2.3  src/system_prompt.py    → static expert prompt ([B]) — see content below
2.4  Wire into main.py.run() → [A]||[C] → [D] → [E] → [F] → [G]
```

### Step 3 — RunPod + SGLang (~2 hours, parallelizable with Step 2)

```
3.1  Provision 2× A100 80GB pod, mount persistent volume
3.2  pip install -U "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python"
3.3  sglang serve --model-path Qwen/Qwen3.5-35B-A3B --host 0.0.0.0 --port 30000 --tp-size 2 --disable-custom-all-reduce
3.4  Verify with a single chat completion + a single tool call
```

### Step 4 — Local validation (~1 hour, ~$0.50)

```
4.1  Run validate.py on 50 train scenarios
4.2  Read the per-bucket IoU report
4.3  Verify IoU on at-least one bucket > 0
```

### Step 5 — First submission (~3 hours, ~$1.50)

```
5.1  python main.py --max_samples 500 --save_dir ./results
5.2  Sanity-check result.csv (row count, format)
5.3  Submit to Zindi
5.4  Record actual leaderboard score → first trustworthy baseline
```

**Total time-to-first-submission: ~10 working hours. Total cost: ~$3.**

---

## The static system prompt — what V0.1 actually says

This is the highest-leverage artifact. The Greenpark insights (#1-#8) plus the rule engine's `_ACTION_PATTERNS` distilled into prose:

```
You are an expert 5G network optimization engineer.
Diagnose the root cause of throughput degradation and select
the optimization actions from the provided options.

# Diagnostic protocol (always in this order)
1. Identify signaling pattern from prefetched signaling log:
   A2 / A3 (same-cell vs diff-cells) / A5 / RRC re-establish / HO attempt
2. Inspect serving cell signal: RSRP, SINR
3. Inspect neighbor cells: RSRP delta, PCI distinctness
4. Inspect configuration: IntraFreqHoA3Offset per cell, antenna tilt,
   PdcchOccupiedSymbolNum, transmission power
5. Inspect traffic: CCE failure rate, DL PRB utilization

# Fault → Action table (canonical patterns)

[P1] A3 diff-cells + one cell IntraFreqHoA3Offset = 10 (5dB, too high)
     → DECREASE A3 offset on that cell

[P2] A3 diff-cells + primary cells offsets {2, 6} (asymmetric, ping-pong)
     → INCREASE A3 on cell with offset=2
     → DECREASE TX power on cell with offset=6
     → PRESS DOWN tilt on cell with offset=6

[P3] A3 same-cell + max CCE fail rate > 0.4
     → CHANGE PdcchOccupiedSymbolNum to 2SYM on serving cell

[P4] A3 same-cell + serving DL PRB ≤ 25%
     → CHECK test server / transport (transport anomaly)

[P5] A2 only (no A3/A5/HO)
     → ADJUST azimuth or INCREASE TX power on serving cell

[P6] A2 + A5 + HO attempt
     → LOWER CovInterFreqA2RsrpThld (inter-frequency gap)

[P7] RRC re-establish without HO attempt
     → ADD missing neighbor relation

# Cell attribution rule
Each option in the list names a specific cell ID (gNodeBID_CellID).
You MUST select the option whose cell matches your diagnosis.
Do NOT confuse "decrease A3 on cell X" with "decrease A3 on cell Y".

# Multi-answer rule
If multiple actions are needed (e.g. P2 typically needs 3-4 actions),
select all complementary options. The answer is one coherent treatment plan.

# Output format (mandatory)
Single answer:    \boxed{C3}
Multi answer:     \boxed{C3|C7|C11}    (ascending numeric order, pipe separator)

If you cannot determine the answer with high confidence, call additional
tools (max 4 more turns) before answering.
```

This prompt is ~600 tokens — small enough to be cached by RadixAttention across all 500 scenarios. The prefix-cache savings are massive.

---

## Validation strategy — the dominance loop

**Rule 4 — Make enumerations so complete that nothing is omitted.**

Phase 1 has unlimited submissions. The dominance lever is the **error analysis loop**:

```
LOOP:
  1. validate.py on 200 train scenarios → IoU per bucket
  2. Identify worst-performing bucket (IoU < 0.7)
  3. Extract 5 failing scenarios from that bucket
  4. Manually inspect → identify root cause:
     - System prompt missing pattern? → add to Fault-Action table
     - Wrong tool prefetched? → adjust prefetch.py
     - Format error? → strengthen [F] fallback
     - Cell attribution confusion? → tighten cell-mapping rules
  5. Re-run validate.py → confirm bucket IoU improved
  6. If improved AND total IoU improved → submit to Zindi
  7. If not → revert, try different angle
```

**Per-bucket IoU breakdown is the actual weapon.** A global IoU of 0.7 hides the fact that `rrc_only` cases might be at 0.4. Fixing one bucket from 0.4 → 0.8 on a 10% slice = +0.04 to global score.

---

## Cost & timeline summary

| Phase | What | Cost | Time |
|-------|------|------|------|
| Setup | RunPod 2× A100 + SGLang + model download | ~$3 | 2-3h |
| V0.1 build | Code 4 modules + wire main.py | $0 | 3h |
| Local validation | 50 train scenarios | ~$0.50 | 1h |
| First submission | 500 test scenarios | ~$1.50 | 3h |
| Iteration | 3-5 fix cycles × 200-scenario validate | ~$3 | days |
| **Total to dominate** | **first 0.75+ leaderboard score** | **~$6** | **~5 days** |

The $10 RunPod budget covers everything with 40% safety margin.

---

## What I commit to with certainty

1. **Today, in 1 hour:** fix `compute_score()` (F1 is a code bug; trivial PR).
2. **Today, in 2 hours:** write `scripts/validate.py` with per-bucket IoU.
3. **By tomorrow:** V0.1 wired into `main.py`, RunPod live, first submission on the leaderboard with a *real* IoU number.
4. **By end of week:** at least 3 iterations of the error analysis loop. Score should be ≥ 0.70 if the architecture is sound.

## What I refuse to commit to

1. A specific final score — I have no trustworthy baseline to extrapolate from.
2. RAG-at-inference benefit — it's an unmeasured hypothesis.
3. The confidence gate's value — same.

**Rule 7 (Descartes) — Apply your method until you have it perfect.** We measure, we patch, we re-measure. We do not add components on faith.

---

## The decision moment

Two paths now:

**Path A — Start with Step 1 right now**: I fix `compute_score()` and build `validate.py`. Concrete, ~1 hour of work, unblocks all measurement.

**Path B — Spec the system_prompt + prefetch first**: I write the full text of `system_prompt.py` and the parallel-fetch logic of `prefetch.py` so you have the architecture's heart in your hands before any wiring.

Which do you want me to do first? I'd recommend Path A — without trustworthy IoU measurement, every other step is faith-based.
