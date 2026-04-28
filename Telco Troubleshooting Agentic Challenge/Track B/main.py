#!/usr/bin/env python3
"""Track B — IP Network Troubleshooting Agent (standalone, no OpenClaw dependency)."""

import argparse
import csv
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_SERVER_URL = "http://127.0.0.1:7860"
DEFAULT_MODEL_URL  = os.environ.get("PROVIDER_API_URL", "https://llm.wavespeed.ai/v1")
DEFAULT_MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen3.5-35B-A3B")
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_CONCURRENCY    = int(os.environ.get("MAX_WORKERS", 2))

BASE_DIR    = Path(__file__).parent
RESULT_DIR  = BASE_DIR / "results"
EXAMPLES_DIR = BASE_DIR / "examples"

# ── Proxy bypass (tool server is local; LLM calls go out normally) ────────────
_NO_PROXY = os.environ.get("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", _NO_PROXY)
os.environ.setdefault("no_proxy", _NO_PROXY)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trackb")

# ── Tool schema ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": (
                "Execute a read-only CLI command on a network device and return its output. "
                "Use Huawei 'display ...' commands for Huawei devices, "
                "Cisco 'show ...' commands for Cisco devices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "Exact device name as shown in the topology (e.g. 'Alpha-Center-01').",
                    },
                    "command": {
                        "type": "string",
                        "description": "CLI command to execute (e.g. 'display lldp neighbor brief').",
                    },
                },
                "required": ["device_name", "command"],
            },
        },
    }
]

SYSTEM_PROMPT = """\
You are NetOps-Agent, an expert IP network operations engineer for an O&M competition.
You diagnose network issues and answer topology/routing questions by running CLI commands on real devices.

## Environment
- Devices are Huawei VRP unless stated otherwise (use `display` commands).
- One device named PSS is Cisco IOS (use `show` commands).
- All commands must be exact — wrong syntax returns a CLI error, not data.
- You have one tool: execute_command(device_name, command).
- Call tools ONE AT A TIME (sequential, never concurrent within a question).
- Time limit: 10 minutes per question — be efficient, avoid redundant queries.

## Command reference (Huawei — use for all devices except PSS)
Topology / neighbors:
  display lldp neighbor brief          <- primary topology discovery
  display interface brief              <- interface UP/DOWN status
  display interface description        <- remote node/port in description field
  display arp                          <- IP<->port mapping (overrides description if conflict)

Routing:
  display ip routing-table             <- IPv4 routes and next-hops
  display ip interface brief           <- interface IP addresses
  display ospf peer                    <- OSPF neighbor state
  display ospf routing                 <- OSPF-learned routes
  display bgp vpnv4 all routing-table  <- BGP VPNv4 routes
  display bgp evpn all routing-table   <- BGP EVPN routes

L2:
  display vlan                         <- VLAN membership
  display stp brief                    <- STP port roles
  display mac-address                  <- MAC table
  display eth-trunk <id>               <- Eth-Trunk/LAG details

Advanced:
  display vxlan tunnel                 <- VXLAN tunnel status
  display bfd session all              <- BFD session state
  display vrrp verbose                 <- VRRP state
  display ip pool                      <- DHCP pool status

## Strategy by question type

### Topology reconstruction (find UP links for a node)
1. Run `display lldp neighbor brief` on the target device — direct LLDP neighbor discovery (preferred).
2. If LLDP is unavailable or empty, fall back:
   a. Run `display interface description` — get remote node/port from the description field.
   b. Verify with `display arp` on both local and remote device.
   c. If description and ARP disagree, ARP wins.
3. Only include interfaces whose physical state is UP.
4. If the target device's own commands are inaccessible, query surrounding nodes and find links pointing to the target.

### Path query (trace route from A to B)
1. On the source device, run `display ip routing-table` — find the route and next-hop for the destination IP/prefix.
2. Move to the next-hop device, repeat until you reach the destination device.
3. Output all devices in traversal order.

### Fault localization
1. Check interface status (`display interface brief`).
2. Check routing table and OSPF/BGP neighbor state on affected devices.
3. Identify the failing component (interface down, missing route, protocol session down, etc.).

## Output rules — CRITICAL
- Output ONLY the final answer. No reasoning, no explanation, no markdown.
- Follow the EXACT format specified in the question.
- Topology links: LocalNode(LocalPort)->RemoteNode(RemotePort), one per line, no trailing blank line.
- Path: Node1->Node2->Node3 on one line, no spaces around ->.
- Do not add any introductory or closing words.
"""


# ── Qwen XML tool-call fallback ───────────────────────────────────────────────

def _extract_xml_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse Qwen-style embedded tool calls from assistant content or reasoning."""
    results: List[Tuple[str, Dict[str, Any]]] = []

    # Format A — XML parameters
    xml_block = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
    param_block = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    for tm in xml_block.finditer(text):
        fn_name = tm.group(1)
        params: Dict[str, Any] = {}
        for pm in param_block.finditer(tm.group(2)):
            val = pm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            params[pm.group(1)] = val
        results.append((fn_name, params))

    # Format B — JSON body
    if not results:
        json_block = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for jm in json_block.finditer(text):
            try:
                obj = json.loads(jm.group(1))
                fn_name = obj.get("name") or obj.get("function")
                args = obj.get("arguments", obj.get("parameters", {}))
                if fn_name and isinstance(args, dict):
                    results.append((fn_name, args))
            except (json.JSONDecodeError, KeyError):
                pass

    return results


# ── Tool server call ─────────────────────────────────────────────────────────

_tool_session = requests.Session()
_tool_session.trust_env = False


def call_tool_server(server_url: str, device_name: str, command: str, question_number: int) -> str:
    url = f"{server_url.rstrip('/')}/api/agent/execute"
    try:
        r = _tool_session.post(
            url,
            json={
                "device_name": device_name,
                "command": command.strip(),
                "question_number": question_number,
            },
            timeout=30,
        )
        data = r.json()
        result = data.get("result", "")
        if result:
            return result
        if "error" in data:
            return f"Error: {data['error']}"
        return json.dumps(data)
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to tool server. Is server.py running on port 7860?"
    except Exception as e:
        return f"Error: {e}"


# ── Context management ────────────────────────────────────────────────────────

_TOOL_RESULT_CHAR_LIMIT = 3000   # truncate individual tool outputs beyond this
_CONTEXT_CHAR_BUDGET    = 80_000 # ~25K tokens; trim oldest exchanges above this

def _trim_messages(messages: list) -> list:
    """
    Prevent context overflow by:
    1. Hard-capping each tool result at _TOOL_RESULT_CHAR_LIMIT chars.
    2. Dropping the oldest tool-call + tool-result pairs when the total
       serialized length exceeds _CONTEXT_CHAR_BUDGET.
    System prompt (idx 0) and user question (idx 1) are never dropped.
    """
    # Step 1 — truncate oversized tool results in-place
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > _TOOL_RESULT_CHAR_LIMIT:
                msg["content"] = content[:_TOOL_RESULT_CHAR_LIMIT] + "\n[...truncated]"

    # Step 2 — drop oldest assistant+tool pairs while budget exceeded
    while len(json.dumps(messages)) > _CONTEXT_CHAR_BUDGET and len(messages) > 4:
        # messages[0]=system, messages[1]=user — skip them
        # messages[2] should be the oldest assistant turn; drop it + its tool reply
        drop = 1 if messages[2].get("role") == "assistant" else 0
        messages.pop(2)                     # assistant turn (with tool_calls)
        if len(messages) > 2 and messages[2].get("role") == "tool":
            messages.pop(2)                 # corresponding tool result

    return messages


def _force_final_answer(client: OpenAI, model_name: str, messages: list) -> str:
    """One extra call with tool_choice=none to pull out a stuck final answer."""
    try:
        prompt = messages + [{
            "role": "user",
            "content": (
                "Based on your investigation above, provide your final answer now "
                "in the exact required format. No tool calls, just the answer."
            ),
        }]
        resp = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            temperature=0,
            max_tokens=512,
            tool_choice="none",
        )
        content   = (resp.choices[0].message.content or "").strip()
        reasoning = (getattr(resp.choices[0].message, "reasoning_content", "") or "").strip()
        return content or reasoning
    except Exception:
        return ""


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_question(
    client: OpenAI,
    server_url: str,
    question_id: int,
    question_text: str,
    model_name: str,
    max_iterations: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the ReAct agent loop for one question.
    Returns dict with keys: answer, completion (trace text), tool_calls, num_tool_calls.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]

    trace_lines: List[str] = [f"== Q{question_id} Trace ==\n"]
    call_count = 0
    last_content = ""

    for iteration in range(max_iterations):
        messages = _trim_messages(messages)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=4096,
        )
        msg = response.choices[0].message
        raw_content = msg.content or ""
        reasoning   = getattr(msg, "reasoning_content", "") or ""
        actual_tcs  = msg.tool_calls or []

        if raw_content:
            last_content = raw_content

        # Detect XML tool calls embedded in content/reasoning (Qwen quirk)
        fake_tcs: List[dict] = []
        if not actual_tcs:
            for fn_name, fn_args in _extract_xml_tool_calls(raw_content + "\n" + reasoning):
                fake_tcs.append({
                    "id": f"call_xml_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": json.dumps(fn_args)},
                })

        calls_to_run = actual_tcs or fake_tcs

        # Build assistant turn
        assistant_turn: dict = {"role": "assistant", "content": raw_content}
        if calls_to_run:
            assistant_turn["tool_calls"] = (
                [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in calls_to_run
                ]
                if actual_tcs
                else calls_to_run  # already dicts for fake_tcs
            )
        messages.append(assistant_turn)

        # No tool calls → final answer
        if not calls_to_run:
            answer = raw_content.strip() or reasoning.strip()
            if not answer:
                # Model finished but returned empty content — force it to write the answer
                answer = _force_final_answer(client, model_name, messages)
                if verbose and answer:
                    log.debug(f"  [{question_id}] Forced final answer: {answer[:80]!r}")
            if reasoning:
                trace_lines.append(f"[Reasoning]\n{reasoning}\n")
            if verbose:
                log.debug(f"  [{question_id}] Final answer after {iteration+1} iter(s): {answer[:80]!r}")
            return {
                "answer": answer,
                "completion": "\n".join(trace_lines),
                "num_tool_calls": call_count,
            }

        # Execute each tool call sequentially
        for tc in calls_to_run:
            if hasattr(tc, "function"):
                tc_id, fn_name, fn_args_str = tc.id, tc.function.name, tc.function.arguments
            else:
                tc_id, fn_name, fn_args_str = tc["id"], tc["function"]["name"], tc["function"]["arguments"]

            if fn_name != "execute_command":
                result_text = f"Error: unknown tool '{fn_name}'"
            else:
                try:
                    args    = json.loads(fn_args_str)
                    device  = args.get("device_name", "").strip()
                    command = args.get("command", "").strip()
                except (json.JSONDecodeError, KeyError):
                    device, command = "", ""
                    result_text = "Error: invalid tool arguments JSON"
                else:
                    call_count += 1
                    if verbose:
                        log.debug(f"  [{question_id}] call#{call_count}: {device!r} | {command!r}")
                    result_text = call_tool_server(server_url, device, command, question_id)
                    if verbose:
                        log.debug(f"  [{question_id}] result: {result_text[:150]!r}")

                    # Append to trace
                    trace_lines.append(f"[Step {call_count}] {device} > {command}")
                    trace_lines.append(result_text.strip())
                    trace_lines.append("")

            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})

    log.warning(f"  [{question_id}] Hit max_iterations={max_iterations}")
    answer = last_content.strip()
    if not answer:
        answer = _force_final_answer(client, model_name, messages)
    return {
        "answer": answer,
        "completion": "\n".join(trace_lines) + "\n[max iterations reached]",
        "num_tool_calls": call_count,
    }


# ── Result I/O ────────────────────────────────────────────────────────────────

def init_csv(csv_path: Path):
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["id", "prediction"])


def append_csv(csv_path: Path, question_id: int, answer: str):
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([question_id, answer])


def load_questions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": item["task"]["id"], "question": item["task"]["question"]} for item in data]


def _write_traces(traces_path: Path, traces: list):
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)


def _write_submission(result_csv: Path, submission_csv: Path, questions_json: Path):
    """Merge Track B predictions into the Zindi submission template."""
    # Build int id → scenario_id map from the questions JSON
    with open(questions_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    id_to_scenario = {item["task"]["id"]: item["scenario_id"] for item in raw}

    results_df  = pd.read_csv(result_csv)
    template_path = (
        Path(__file__).parent / ".." / "submission" / "Phase_1" / "result.csv"
    )
    template_df = pd.read_csv(template_path)

    # Map integer id → scenario_id → prediction
    results_df["scenario_id"] = results_df["id"].map(id_to_scenario)
    answer_map = dict(zip(results_df["scenario_id"], results_df["prediction"].fillna("")))

    id_col = template_df.columns[0]
    template_df["Track B"] = template_df[id_col].map(answer_map).fillna(
        template_df["Track B"].fillna("")
    )
    template_df.rename(columns={id_col: "ID"}, inplace=True)

    template_df.to_csv(submission_csv, index=False, encoding="utf-8", lineterminator="\n")
    log.info(f"Filled {len(answer_map)} Track B predictions into {submission_csv}")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(args):
    RESULT_DIR.mkdir(exist_ok=True)
    EXAMPLES_DIR.mkdir(exist_ok=True)

    csv_path        = RESULT_DIR / "result.csv"
    submission_path = RESULT_DIR / "submission.csv"
    traces_path     = EXAMPLES_DIR / "real_traces.json"

    init_csv(csv_path)

    client = OpenAI(
        api_key=os.environ.get("PROVIDER_API_KEY", "dummy"),
        base_url=args.model_url,
    )

    questions = load_questions(args.input)

    if args.questions:
        ids = {int(x.strip()) for x in args.questions.split(",")}
        questions = [q for q in questions if q["id"] in ids]

    if args.max_samples and args.max_samples < len(questions):
        questions = questions[: args.max_samples]

    log.info(
        f"Track B agent | model={args.model_name} | questions={len(questions)} "
        f"| concurrency={args.concurrency} | max_iter={args.max_iterations}"
    )

    all_traces: list = []
    lock = Lock()

    def process(q: dict) -> tuple:
        t0 = time.time()
        result = run_question(
            client=client,
            server_url=args.server_url,
            question_id=q["id"],
            question_text=q["question"],
            model_name=args.model_name,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        answer = result["answer"]

        with lock:
            append_csv(csv_path, q["id"], answer)
            all_traces.append({
                "question_id": q["id"],
                "question": q["question"],
                "completion": result["completion"],
                "prediction": answer,
                "num_tool_calls": result["num_tool_calls"],
                "elapsed_s": round(elapsed, 1),
            })
            _write_traces(traces_path, all_traces)

        log.info(f"  [#{q['id']:02d}] {elapsed:.1f}s | {result['num_tool_calls']} calls — {answer[:100]!r}")
        return q["id"], answer

    if args.concurrency <= 1:
        for i, q in enumerate(questions, 1):
            log.info(f"[{i}/{len(questions)}] Question #{q['id']} ...")
            process(q)
    else:
        log.info(f"Concurrent mode: up to {args.concurrency} questions in parallel.")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(process, q): q for q in questions}
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    log.error(f"  [#{q['id']}] thread error: {e}")

    # Write final submission.csv merged into Zindi template
    _write_submission(csv_path, submission_path, Path(args.input))

    log.info(f"Results    → {csv_path}")
    log.info(f"Submission → {submission_path}")
    log.info(f"Traces     → {traces_path}  ({len(all_traces)} entries)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Track B IP Network Troubleshooting Agent")
    parser.add_argument(
        "--server_url", default=DEFAULT_SERVER_URL,
        help=f"Tool server base URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--model_url", default=DEFAULT_MODEL_URL,
        help="LLM API base URL (default from PROVIDER_API_URL env)",
    )
    parser.add_argument(
        "--model_name", default=DEFAULT_MODEL_NAME,
        help="Model name (default from MODEL env)",
    )
    parser.add_argument(
        "--input", "-i", default="data/Phase_1/test.json",
        help="Path to questions JSON (default: data/Phase_1/test.json)",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Max tool-call iterations per question (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of questions to run (for quick testing)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="Max parallel questions (competition limit: 2)",
    )
    parser.add_argument(
        "--questions", type=str, default=None,
        help="Comma-separated question IDs to run (e.g. '1,2,5')",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging (shows each tool call and result)",
    )
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    benchmark(args)


if __name__ == "__main__":
    main()
