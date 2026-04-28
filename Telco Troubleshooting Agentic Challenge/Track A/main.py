#!/usr/bin/env python
# -*-coding:utf-8 -*-

# Import dependencies
import argparse
import json
import logging
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd
import httpx
import requests
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from _types import ToolCall, ToolFunction
from logger import init_logger
from utils import (
    print_model_response,
    print_tool_call,
    print_tool_result,
    extract_answer,
    extract_answer_all,
    extract_cx_fallback,
    compute_score,
)
from rule_engine import diagnose, build_layer0_system_prompt
from rag import build_rag_index, format_rag_context, RagIndex
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from .env file (e.g., API keys)
PROVIDER_API_KEY = os.environ.get("PROVIDER_API_KEY", "dummy")

SYSTEM_PROMPT = """\
CRITICAL: Every response MUST end with \\boxed{CX} or \\boxed{CX|CY|...} (options in ascending order).
Never end a response without it. This is mandatory, no exceptions.

You are an expert 5G network engineer performing drive test analysis and network optimization.
Your task is to diagnose the root cause of user throughput degradation and select the correct
optimization action(s) from the provided options.

## Investigation Protocol

Call tools in this order -- stop as soon as you have enough evidence:

1. get_throughput_logs() -- Find the timestamp with the LOWEST throughput (the worst degradation point).
2. get_serving_cell_pci(time=T) -- ONE call only, at the single timestamp with the lowest throughput.
3. get_cell_info(pci=P) -- Full config: A3 offset, tilt, power, neighbor list, PdcchOccupiedSymbolNum.
4. get_serving_cell_rsrp(time=T) + get_serving_cell_sinr(time=T) -- Signal quality at degradation.
5. get_neighboring_cells_pci(time=T) -- Neighbor PCIs visible at that moment.
6. get_neighboring_cell_rsrp(time=T, pci=N) -- For each neighbor, compare RSRP vs. serving.
7. get_signaling_plane_event_log() -- Only if handover event details are still needed.

## Critical Mapping Rule

Option labels use the format gNodeBID_CellID (e.g., "3225568_1" = gNodeB 3225568, Cell 1).
get_cell_info(pci=X) returns "gNodeB ID" and "Cell ID" -- use these to link a PCI to an option label.

## A3 Handover Condition

Handover fires when:  Neighbor_RSRP > Serving_RSRP + A3Offset_dBm + A3Hyst_dBm
Where:  A3Offset_dBm = IntraFreqHoA3Offset [0.5dB] / 2
        A3Hyst_dBm   = IntraFreqHoA3Hyst [0.5dB]   / 2
Always compute this explicitly using the values from get_cell_info.

## Root Cause Patterns

Throughput drops, PCI unchanged, strong neighbor exists, A3 threshold NOT exceeded
  -> Late handover, A3 offset too high
  -> Fix: Decrease A3 Offset for the serving cell

Throughput drops, PCI unchanged, strong neighbor in measurements but NOT in cell neighbor list
  -> Missing neighbor relation
  -> Fix: Add neighbor relationship

PCI switches rapidly back and forth between two cells
  -> Ping-pong, A3 offset too low
  -> Fix: Increase A3 Offset for both involved cells

Good RSRP + good SINR + very low throughput + high CCE fail rate
  -> PDCCH congestion
  -> Fix: Modify PdcchOccupiedSymbolNum to 2SYM for affected cell

Good RSRP + good SINR + low throughput despite full resource blocks
  -> Transmission or backhaul fault
  -> Fix: "Check test server and transmission issues"

RSRP consistently < -100 dBm
  -> Weak coverage
  -> Fix: Increase TX power, lift tilt, or decrease inter-freq threshold (CovInterFreqA2/A5RsrpThld)

Poor SINR (-5 to +5 dB) despite adequate RSRP, many neighbors at similar signal levels
  -> Overlap / interference
  -> Fix: Decrease TX power or press down tilt for dominant interferer

## Answer Format

Single answer:   \\boxed{C9}
Multiple answers: \\boxed{C2|C8|C11}  (ascending order, 2-4 options)
REMINDER: \\boxed{} is MANDATORY. Always write it as the very last thing in your response.\
"""


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _build_tool_guidance(features: Dict[str, Any]) -> str:
    """Focused tool-call suggestion based on detected signaling pattern (for LAYER 1 prompt injection)."""
    if not features:
        return ""
    lines = ["", "# Suggested Investigation Order (based on pre-analysis)"]
    a3_diff = features.get("has_a3_diff_cells", False)
    a3_same = features.get("has_a3_same_cell", False)
    has_a2  = features.get("has_a2", False)
    has_a5  = features.get("has_a5", False)
    has_rrc = features.get("has_rrc", False)
    has_ho  = features.get("has_ho_attempt", False)
    if a3_diff:
        lines += [
            "1. get_signaling_plane_event_log — confirm NREventA3 content (ServCellPCI vs NCellPCI)",
            "2. get_config_data — check IntraFreqHoA3Offset for primary cells",
            "3. get_serving_cell_rsrp + get_neighboring_cell_rsrp — verify RSRP delta",
        ]
    elif a3_same:
        lines += [
            "1. get_signaling_plane_event_log — confirm NREventA3 same-cell content",
            "2. get_kpi_data or get_rbs_allocated_to_user — check CCE fail rate and PRB utilization",
            "3. get_config_data — check PdcchOccupiedSymbolNum setting",
        ]
    elif has_a2 and has_a5:
        lines += [
            "1. get_signaling_plane_event_log — confirm A2/A5 thresholds",
            "2. get_serving_cell_rsrp + get_neighboring_cell_rsrp — find inter-freq coverage gap",
            "3. get_config_data — check CovInterFreqA2RsrpThld",
        ]
    elif has_a2 and not has_a5 and not has_ho:
        lines += [
            "1. get_serving_cell_rsrp + get_serving_cell_sinr — confirm weak/interference on serving",
            "2. get_gnodeb_location + get_user_location — check coverage geometry",
            "3. get_config_data — check azimuth and power settings",
        ]
    elif has_rrc and not has_ho:
        lines += [
            "1. get_signaling_plane_event_log — confirm RRC re-establishment without HO",
            "2. get_neighboring_cells_pci — check if strong neighbor is missing from config",
            "3. get_config_data — verify neighbor relations",
        ]
    else:
        lines += [
            "1. get_signaling_plane_event_log — determine failure type",
            "2. get_serving_cell_rsrp + get_serving_cell_sinr — check signal quality",
            "3. get_config_data — check antenna and A3 parameters",
        ]
    return "\n".join(lines)


def _filter_tools_for_diagnosis(
    all_tool_defs: List[Dict[str, Any]],
    features: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only the tools relevant to the detected signaling pattern.

    Cuts the tool list from ~20 to 5-7, reducing prompt size and model confusion.
    Falls back to the full list when features are unavailable.
    """
    if not features or not all_tool_defs:
        return all_tool_defs

    a3_diff = features.get("has_a3_diff_cells", False)
    a3_same = features.get("has_a3_same_cell", False)
    has_a2  = features.get("has_a2", False)
    has_a5  = features.get("has_a5", False)
    has_rrc = features.get("has_rrc", False)
    has_ho  = features.get("has_ho_attempt", False)

    if a3_diff:
        keep = {
            "get_signaling_plane_event_log", "get_cell_info",
            "get_serving_cell_rsrp", "get_neighboring_cells_pci",
            "get_neighboring_cell_rsrp", "get_throughput_logs",
        }
    elif a3_same:
        keep = {
            "get_signaling_plane_event_log", "get_kpi_data",
            "get_cell_info", "get_rbs_allocated_to_user",
            "get_throughput_logs",
        }
    elif has_a2 and has_a5:
        keep = {
            "get_signaling_plane_event_log", "get_serving_cell_rsrp",
            "get_neighboring_cell_rsrp", "get_cell_info",
            "get_neighboring_cells_pci",
        }
    elif has_a2 and not has_ho:
        keep = {
            "get_serving_cell_rsrp", "get_serving_cell_sinr",
            "get_gnodeb_location", "get_user_location",
            "get_cell_info",
        }
    elif has_rrc and not has_ho:
        keep = {
            "get_signaling_plane_event_log", "get_neighboring_cells_pci",
            "get_cell_info", "get_serving_cell_rsrp",
        }
    else:
        keep = {
            "get_signaling_plane_event_log", "get_serving_cell_rsrp",
            "get_serving_cell_sinr", "get_cell_info",
            "get_neighboring_cells_pci", "get_neighboring_cell_rsrp",
            "get_kpi_data",
        }

    filtered = [
        t for t in all_tool_defs
        if (t.get("function", {}).get("name") or t.get("name", "")) in keep
    ]
    return filtered if filtered else all_tool_defs


def _extract_xml_tool_calls(content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse Qwen-style embedded tool calls that leak into assistant content.

    Handles two formats the model occasionally emits instead of proper tool_calls:

    Format A (XML):
        <tool_call>
        <function=get_cell_info>
        <parameter=pci>468</parameter>
        </function>
        </tool_call>

    Format B (JSON):
        <tool_call>{"name": "get_cell_info", "arguments": {"pci": 468}}</tool_call>
    """
    results: List[Tuple[str, Dict[str, Any]]] = []

    # Format A
    xml_block = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
    param_block = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    for tm in xml_block.finditer(content):
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

    # Format B (only if Format A found nothing)
    if not results:
        json_block = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for jm in json_block.finditer(content):
            try:
                obj = json.loads(jm.group(1))
                fn_name = obj.get("name") or obj.get("function")
                args = obj.get("arguments", obj.get("parameters", {}))
                if fn_name and isinstance(args, dict):
                    results.append((fn_name, args))
            except (json.JSONDecodeError, KeyError):
                pass

    return results


# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

class Environment:
    """
    Responsible for:
    - discovering tool descriptors from FastAPI `/tools`
    - executing tool calls requested by the LLM
    - applying per-scenario context via X-Scenario-Id header
    """

    # server endpoints are different from agent tools. Agent only has access to tools exposed via /tools endpoint 
    endpoint_mapper = {
        "get_all_scenario": "/scenario/all",
        "get_config_data": "/config-data",
        "get_user_plane_data": "/user-plane-data",
        "get_throughput_logs": "/throughput-logs",
        "get_cell_info": "/cell-info",
        "get_gnodeb_location": "/gnodeb-location",
        "get_user_location": "/user-location",
        "get_serving_cell_pci": "/serving-cell-pci",
        "get_serving_cell_rsrp": "/serving-cell-rsrp",
        "get_serving_cell_sinr": "/serving-cell-sinr",
        "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
        "get_neighboring_cells_pci": "/neighboring-cells-pci",
        "get_neighboring_cell_rsrp": "/neighboring-cell-rsrp",
        "get_signaling_plane_event_log": "/signaling-plane-event-log",
        "get_all_cells_pci": "/all-cells-pci",
        "get_available_tools": "/tools",
        "health": "/health",
        "judge_mainlobe_or_not": "/judge_mainlobe",
        "calculate_horizontal_angle": "/calculate_horizontal_angle",
        "calculate_tilt_angle": "/calculate_tilt_angle",
        "calculate_pathloss": "/calculate_pathloss",
        "calculate_overlap_ratio": "/calculate_overlap_ratio",
        "get_kpi_data": "/get_kpi_data",
        "get_mr_data": "/get_mr_data",
        "optimize_antenna_gain": "/optimize_antenna_gain"
    }

    def __init__(self, server_url: str, verbose: bool = False, log_file: Optional[str] = None, timeout: float = 15.0,
                 logger: logging.Logger = None):
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.timeout = timeout  # in seconds
        self.logger = logger if logger is not None else init_logger()

    def _headers(self, scenario_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if scenario_id:
            headers["X-Scenario-Id"] = scenario_id
        return headers

    def _call_api(
            self,
            function_name: str,
            scenario_id: Optional[str] = None,
            **params: Any,
    ) -> Dict[str, Any]:
        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}

        url = f"{self.server_url}{endpoint}"
        headers = self._headers(scenario_id=scenario_id)

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params}")
            data = resp.json()
            return data
        except requests.exceptions.HTTPError:
            # FastAPI error responses often include {"detail": "..."}
            try:
                detail = resp.json().get("detail", str(resp.text))
            except Exception:
                detail = str(resp.text)
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params} -> HTTPError: {detail}")
            return {"error": detail}
        except Exception as e:
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params} -> ERROR: {e}")
            return {"error": str(e)}

    def get_tools(self) -> List[Dict[str, Any]]:
        """Fetch OpenAI-like tool descriptors from /tools."""
        tools = self._call_api("get_available_tools")
        if isinstance(tools, dict) and "error" in tools:
            return []
        if not isinstance(tools, list):
            return []
        return tools

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Fetch all scenarios available."""
        scenarios = self._call_api("get_all_scenario")
        if isinstance(scenarios, dict) and "error" in scenarios:
            return []
        if not isinstance(scenarios, list):
            return []
        return scenarios

    # Fields from get_cell_info that matter for diagnosis — everything else is stripped
    _CELL_INFO_KEEP = {
        "gNodeB ID", "Cell ID", "PCI",
        "Transmission Power", "Mechanical Downtilt", "Digital Tilt",
        "IntraFreqHoA3Offset [0.5dB]", "IntraFreqHoA3Hyst [0.5dB]", "IntraFreqHoA3TimeToTrig",
        "PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", "PdcchOccupiedSymbolNum",
        "InterFreqHoEventType",
        "CovInterFreqA2RsrpThld [dBm]", "InterFreqA2Hyst [0.5dB]",
        "CovInterFreqA5RsrpThld1 [dBm]", "CovInterFreqA5RsrpThld2 [dBm]",
    }

    def _filter_result(self, function_name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        if function_name == "get_cell_info" and isinstance(raw, dict):
            return {k: v for k, v in raw.items() if k in self._CELL_INFO_KEEP}
        return raw

    def execute(self, tool_call: ToolCall, scenario_id: Optional[str] = None) -> str:
        """
        Execute a single OpenAI tool_call and return a JSON string for the tool message.
        """
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = self._call_api(function_name=function_name, scenario_id=scenario_id, **arguments)
            result = self._filter_result(function_name, result)
            return json.dumps(result, ensure_ascii=False)

        except json.JSONDecodeError:
            error_msg = f"Tool parameter parsing failed: {tool_call.function.arguments}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        except Exception as e:
            error_msg = f"Tool invocation execution failed: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)


# ------------------------------------------------------------------------------
# LLM Agent Runner
# ------------------------------------------------------------------------------

class AgentsRunner:
    """
    Owns:
    - OpenAI client
    - solve() loop (tool calling)
    - benchmark() across scenarios and attempts
    """

    def __init__(
            self,
            environment: Environment,
            model_url: str,
            model_name: str,
            model_provider: Optional[str] = None,
            max_tokens: int = 4000,
            max_retries: int = 3,
            max_iterations: int = 20,
            temperature: float = 0.1,
            verbose: bool = False,
            no_think: bool = False,
            logger: logging.Logger = None
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose
        self.no_think = no_think
        self.logger = logger if logger is not None else init_logger()
        self.running_metrics = {}
        self._tool_defs_cache: Optional[List[Dict[str, Any]]] = None

        self.client = OpenAI(
            base_url=model_url,
            api_key=PROVIDER_API_KEY,
            http_client=httpx.Client(verify=False, timeout=httpx.Timeout(15.0, read=90.0)),
        )

        # Build RAG index once from training data (0 API cost, pure in-memory similarity)
        self.rag_index: Optional[RagIndex] = build_rag_index("data/Phase_1/train.json")

    def _get_tools(self) -> List[Dict[str, Any]]:
        if self._tool_defs_cache is None:
            self._tool_defs_cache = self.environment.get_tools()
        return self._tool_defs_cache

    def _call_model(self, messages: List[Dict[str, Any]], functions: List[Dict[str, Any]], **kwargs):
        base_wait_time = 1.0

        call_kwargs = {
            "model": f"{self.model_provider}/{self.model_name}" if self.model_provider else self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs
        }

        if functions:
            call_kwargs["tools"] = functions
            call_kwargs["tool_choice"] = "auto"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**call_kwargs)
                return response.choices[0].message

            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                if self.verbose:
                    self.logger.error(traceback.format_exc())

                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500 and exc.status_code != 429:
                    if self.verbose:
                        self.logger.info("Non-retriable exception: %s", exc)
                    return None

                if attempt == self.max_retries:
                    if self.verbose:
                        self.logger.info("Final failure after %s attempts: %s", self.max_retries, exc)
                    return None

                wait = base_wait_time * (2 ** (attempt - 1))
                if self.verbose:
                    self.logger.info("Retry %s/%s after %.1fs due to: %s", attempt, self.max_retries, wait, exc)
                time.sleep(wait)

            except Exception as exc:
                if self.verbose:
                    self.logger.info("Unhandled exception: %s", exc)
                return None

        return None

    def run(self, scenario: Dict[str, Any], tool_defs: Optional[List[Dict[str, Any]]] = None, free_mode: bool = False, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {})

        root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in task.get("options", [])])

        tool_defs = tool_defs if tool_defs is not None else self._get_tools()
        if not tool_defs:
            return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available"}

        question = task.get("description", "") + f"\nOptions:\n{root_causes}"

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        num_tool_calls = 0
        list_tool_calls = []
        status = None
        reason = None
        last_msg = None
        _tool_cache: Dict[str, str] = {}

        for i in range(self.max_iterations):
            self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 1} conversation, calling tools:")

            is_last_iter = (i == self.max_iterations - 1)
            # On the final iteration, force a synthesis-only round: no tools offered,
            # and an explicit user nudge so the model writes \boxed{} instead of calling again.
            if is_last_iter and num_tool_calls > 0:
                messages.append({
                    "role": "user",
                    "content": (
                        "You have collected enough network data. Synthesize your findings and "
                        "write your final diagnosis answer in \\boxed{} notation now. "
                        "No more tool calls."
                    ),
                })
            iter_tools = [] if is_last_iter else tool_defs

            msg = self._call_model(messages, functions=iter_tools)
            if msg is None:
                continue

            last_msg = msg
            raw = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", "") or ""
            actual_tool_calls = msg.tool_calls or []

            # Detect XML tool calls embedded in content OR reasoning_content
            # (provider quirk: Qwen sometimes emits <tool_call> XML instead of proper tool_calls)
            # Skip on last iteration — we offered no tools, so any XML tool calls are ignored
            # to ensure the synthesis round always yields a text answer.
            fake_tcs: List[ToolCall] = []
            if not actual_tool_calls and not is_last_iter:
                for fn_name, fn_args in _extract_xml_tool_calls(raw + "\n" + reasoning):
                    fake_tcs.append(ToolCall(
                        id=f"call_xml_{uuid.uuid4().hex[:16]}",
                        type="function",
                        function=ToolFunction(name=fn_name, arguments=json.dumps(fn_args)),
                    ))

            # Append assistant message — include fake tool_calls so tool-result IDs stay consistent
            effective_tcs = actual_tool_calls or fake_tcs or None
            messages.append({"role": "assistant", "content": raw, "tool_calls": effective_tcs})

            if self.verbose:
                print_model_response(msg, logger=self.logger, minimize=False)

            calls_to_run = actual_tool_calls or fake_tcs

            if calls_to_run:
                num_tool_calls += len(calls_to_run)
                for j, tool_call in enumerate(calls_to_run):
                    if self.verbose:
                        print_tool_call(tool_call, logger=self.logger)

                    _cache_key = f"{tool_call.function.name}::{tool_call.function.arguments}"
                    if _cache_key in _tool_cache:
                        tool_result = _tool_cache[_cache_key]
                    else:
                        tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)
                        _tool_cache[_cache_key] = tool_result
                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})

                    if self.verbose:
                        print_tool_result(tool_result, logger=self.logger)

                    has_failed = "error" in tool_result
                    list_tool_calls.append({
                        "function_name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "turn": i + 1,
                        "has_failed": has_failed,
                        "order": j + 1,
                        "results": tool_result,
                    })

            else:
                # No tool calls at all — check for final answer or nudge
                if extract_answer(raw) or extract_answer(reasoning):
                    status = "solved"
                    break
                elif is_last_iter:
                    # Final synthesis round: accept the response even without \boxed{};
                    # _solve_scenario will apply extract_cx_fallback on the returned answer.
                    status = "solved"
                    break
                else:
                    # No answer yet (empty output or mid-reasoning pause) → nudge to conclude
                    messages.append({
                        "role": "user",
                        "content": (
                            "Based on the network data you have collected, analyze the root cause "
                            "and give your final answer in \\boxed{} notation. Do not call any more tools."
                        ),
                    })

        if status is None:
            status = "unresolved"
            reason = "The maximum number of iterations has been reached."

        # Optional final constraint prompt
        if free_mode:
            current_answer = (getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content", "")) if last_msg else ""
            current_traces = getattr(last_msg, "reasoning_content", "") if last_msg else ""
            agent_answer = extract_answer(current_answer) or extract_answer(current_traces)
            if agent_answer == "":
                self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 2} conversation, answer question:")
                status = "solved"

                if 'Select the most appropriate optimization solution' in question:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FINAL ANSWER REQUIRED.\n"
                                "Respond with ONLY the boxed answer — no explanation, no reasoning, no extra text.\n"
                                "Format: \\boxed{CX}  (example: \\boxed{C3})\n\n"
                                f"Available options:\n{root_causes}\n"
                                "Your answer (ONLY \\boxed{CX}):"
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FINAL ANSWER REQUIRED.\n"
                                "Respond with ONLY the boxed answer — no explanation, no reasoning, no extra text.\n"
                                "Format: \\boxed{CX|CY|CZ}  (example: \\boxed{C3|C5} or \\boxed{C7|C11|C14})\n\n"
                                f"Available options:\n{root_causes}\n"
                                "Your answer (ONLY \\boxed{CX|CY|...}):"
                            ),
                        }
                    )

                msg2 = self._call_model(messages, functions=[])
                if msg2 is not None:
                    last_msg = msg2

        return {
            "scenario_id": scenario_id,
            "num_iterations": (i + 1),
            "tool_calls": list_tool_calls,
            "num_tool_calls": num_tool_calls,
            "status": status,
            "traces": getattr(last_msg, "reasoning_content", "") if last_msg else "",
            "answer": getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content","") if last_msg else "",
            "messages": messages,
            "reason": reason,
        }

    def _solve_scenario(
            self,
            scenario: Dict[str, Any],
            tool_defs: List[Dict[str, Any]],
            num_attempts: int,
            free_mode: bool,
            diagnosis=None,   # pre-computed by benchmark(); avoids re-running LAYER 0
    ) -> Dict[str, Any]:
        scenario_id = scenario.get("scenario_id")
        start_time = time.time()

        # ── LAYER 0: deterministic rule engine (0 LLM calls if high confidence) ──
        # diagnosis is pre-computed by benchmark() for all scenarios before the worker
        # pool starts. We only re-run it here if it wasn't supplied (e.g. direct call).
        if diagnosis is None:
            try:
                diagnosis = diagnose(scenario)
                pink, reset = "\033[95m", "\033[0m"
                self.logger.info(
                    f"{pink}[LAYER 0] {scenario_id}: confidence={diagnosis.confidence} — {diagnosis.rule_desc}{reset}"
                )
            except Exception as exc:
                self.logger.warning(f"[LAYER 0] {scenario_id}: exception — {exc}")

        if diagnosis is not None and diagnosis.confidence == "high":
            answer_str = "|".join(diagnosis.answer_ids)
            latency = round(time.time() - start_time, 2)
            pink, reset = "\033[95m", "\033[0m"
            self.logger.info(
                f"{pink}[Scenario {scenario_id}] LAYER0 answer={answer_str} gt={scenario.get('answer')}{reset}"
            )
            return {
                "scenario_id": scenario_id,
                "answer": answer_str,
                "answers": [answer_str],
                "ground_truth": scenario.get("answer"),
                "accuracy": float(compute_score(answer_str, scenario.get("answer", ""))),
                "latency": latency,
                "num_iterations": 0,
                "num_tool_calls": 0,
                "tool_calls": [],
                "response": f"\\boxed{{{answer_str}}}",
                "traces": diagnosis.rule_desc,
                "free_mode": free_mode,
                "layer": "LAYER0",
            }

        # ── LAYER 1: LLM agent — build augmented system prompt ───────────────────
        features = diagnosis.features if diagnosis is not None else {}

        if diagnosis is not None:
            layer0_ctx = build_layer0_system_prompt(diagnosis)
            tool_guidance = _build_tool_guidance(features)
            augmented_prompt = SYSTEM_PROMPT + "\n\n" + layer0_ctx
            if tool_guidance:
                augmented_prompt += "\n" + tool_guidance
            # RAG: inject top-3 similar training cases as few-shot context
            if self.rag_index is not None and features:
                similar = self.rag_index.find_similar(features, k=3)
                rag_ctx = format_rag_context(similar)
                if rag_ctx:
                    augmented_prompt += "\n\n" + rag_ctx
        else:
            augmented_prompt = SYSTEM_PROMPT

        # /no_think: disable extended thinking for speed (WaveSpeed-specific prefix)
        if self.no_think:
            augmented_prompt = "/no_think\n" + augmented_prompt

        # Tool filtering: show only the 5-7 tools relevant to the detected pattern
        focused_tools = _filter_tools_for_diagnosis(tool_defs, features)

        valid_ids = [o["id"] for o in scenario.get("task", {}).get("options", [])]
        n_success = 0.0
        agent_answers: List[str] = []
        sample_response: Dict[str, Any] = {}

        for attempt in range(num_attempts):
            self.logger.info(f"[Scenario {scenario_id}] attempt {attempt + 1}/{num_attempts}")
            response = self.run(scenario=scenario, tool_defs=focused_tools, free_mode=free_mode, system_prompt=augmented_prompt)
            sample_response = response

            if response.get("status") == "solved":
                agent_answer = (
                    extract_answer_all(response.get("answer", ""))
                    or extract_answer_all(response.get("traces", ""))
                )
                # Fallback 1: extract last C\d+ from free text (model said answer w/o \boxed{})
                if not agent_answer:
                    combined = (response.get("answer", "") or "") + " " + (response.get("traces", "") or "")
                    agent_answer = extract_cx_fallback(combined, valid_ids)
                    if agent_answer:
                        self.logger.info(f"[Scenario {scenario_id}] Cx-fallback extracted: {agent_answer}")

                ground_truth = scenario.get("answer")
                n_success += compute_score(agent_answer, ground_truth)
                agent_answers.append(agent_answer)
                pink, reset = "\033[95m", "\033[0m"
                self.logger.info(f"{pink}[Scenario {scenario_id}] answer={agent_answer} gt={ground_truth}{reset}")

        # Fallback 2: if LAYER 1 returned no answer, use LAYER 0 low-confidence guess
        final_answer = agent_answers[0] if agent_answers else ""
        if not final_answer and diagnosis is not None and diagnosis.answer_ids:
            final_answer = "|".join(diagnosis.answer_ids)
            self.logger.info(f"[Scenario {scenario_id}] LAYER0-fallback: {final_answer} (conf={diagnosis.confidence})")

        acc = n_success / float(num_attempts)
        latency = round((time.time() - start_time) / float(num_attempts), 2)

        return {
            "scenario_id": scenario_id,
            "answer": final_answer,
            "answers": agent_answers if agent_answers else ([final_answer] if final_answer else []),
            "ground_truth": scenario.get("answer"),
            "accuracy": acc,
            "latency": latency,
            "num_iterations": sample_response.get("num_iterations", 0),
            "num_tool_calls": sample_response.get("num_tool_calls", 0),
            "tool_calls": sample_response.get("tool_calls", []),
            "response": sample_response.get("answer", ""),
            "traces": sample_response.get("traces", ""),
            "free_mode": free_mode,
            "layer": "LAYER1",
        }

    def _format_submission(self, save_dir: str) -> None:
        """Merge Track A answers into the Zindi submission template."""
        result_path = os.path.join(save_dir, "result.csv")
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "submission", "Phase_1", "result.csv"
        )

        if not os.path.exists(result_path):
            self.logger.warning("No result.csv found — skipping submission format.")
            return
        if not os.path.exists(template_path):
            self.logger.warning(f"Submission template not found at {template_path} — skipping.")
            return

        results_df = pd.read_csv(result_path)
        template_df = pd.read_csv(template_path)

        id_col = template_df.columns[0]  # "scenario_id" or "ID"
        answer_map = dict(zip(results_df["scenario_id"], results_df["answers"].fillna("")))
        template_df.rename(columns={id_col: "ID"}, inplace=True)
        template_df["Track A"] = template_df["ID"].map(answer_map).fillna(template_df["Track A"].fillna(""))

        out_path = os.path.join(save_dir, "submission.csv")
        template_df.to_csv(out_path, index=False)
        self.logger.info(f"Submission file written to {out_path}")

    def benchmark(
            self,
            num_attempts: int,
            save_dir: str,
            save_freq: int = 10,
            max_samples: int = None,
            max_workers: int = 1,
            free_mode: bool = False,
            checkpoint_file: Optional[str] = None,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        # Fetch tools once — shared across all scenarios (read-only)
        tool_defs = self._get_tools()

        scenarios = self.environment.get_scenarios()
        if max_samples is not None:
            scenarios = scenarios[:min(max_samples, len(scenarios))]

        # Checkpoint: resume from previous partial run
        if checkpoint_file is None:
            checkpoint_file = os.path.join(save_dir, "checkpoint.json")

        completed_ids: set = set()
        save_result: List[Dict[str, Any]] = []

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            all_prev = ckpt.get("results", [])
            # Only scenarios with a non-empty answer are truly done — blanks are retried
            save_result = [r for r in all_prev if r.get("answer", "").strip()]
            completed_ids = {r["scenario_id"] for r in save_result}
            n_blank = len(all_prev) - len(save_result)
            self.logger.info(
                f"Resuming: {len(completed_ids)} with answers, "
                f"{n_blank} blank answer(s) will be retried."
            )

        all_pending = [s for s in scenarios if s.get("scenario_id") not in completed_ids]
        self.logger.info(f"Scenarios to process: {len(all_pending)} / {len(scenarios)}")

        lock = Lock()

        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        os.makedirs(examples_dir, exist_ok=True)
        traces_path = os.path.join(examples_dir, "real_traces.json")

        def _flush(force: bool = False):
            with lock:
                if save_result:
                    df = pd.DataFrame([{"scenario_id": r["scenario_id"], "answers": r["answer"]} for r in save_result])
                    df.to_csv(os.path.join(save_dir, "result.csv"), index=False)
                    with open(traces_path, "w", encoding="utf-8") as tf:
                        json.dump(save_result, tf, ensure_ascii=False, indent=2)
                completed = [r["scenario_id"] for r in save_result]
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({"completed": completed, "results": save_result}, f)

        # ── BATCH 0: pre-classify all pending with LAYER 0 (~1s, no API calls) ──
        # High-confidence scenarios get instant answers; only low/none go to LAYER 1.
        pink, reset = "\033[95m", "\033[0m"
        self.logger.info(f"{pink}[BATCH 0] Pre-classifying {len(all_pending)} scenarios...{reset}")

        layer1_queue: List[Dict[str, Any]] = []      # (scenario, diagnosis) pairs
        layer1_diagnoses: Dict[str, Any] = {}        # scenario_id -> diagnosis

        for s in all_pending:
            sid = s.get("scenario_id")
            try:
                d = diagnose(s)
            except Exception as exc:
                self.logger.warning(f"[LAYER 0] {sid}: exception — {exc}")
                d = None

            if d is not None and d.confidence == "high":
                answer_str = "|".join(d.answer_ids)
                self.logger.info(f"{pink}[LAYER 0] {sid}: {answer_str} (gt={s.get('answer')}){reset}")
                save_result.append({
                    "scenario_id": sid,
                    "answer": answer_str,
                    "answers": [answer_str],
                    "ground_truth": s.get("answer"),
                    "accuracy": float(compute_score(answer_str, s.get("answer", ""))),
                    "latency": 0.0,
                    "num_iterations": 0,
                    "num_tool_calls": 0,
                    "tool_calls": [],
                    "response": f"\\boxed{{{answer_str}}}",
                    "traces": d.rule_desc,
                    "free_mode": free_mode,
                    "layer": "LAYER0",
                })
            else:
                layer1_queue.append(s)
                layer1_diagnoses[sid] = d  # may be None

        _flush(force=True)
        self.logger.info(
            f"{pink}[BATCH 0] Done — {len(save_result) - len([r for r in save_result if r.get('layer') != 'LAYER0'])} already in checkpoint + "
            f"{len(save_result)} LAYER0 instant | {len(layer1_queue)} -> LAYER 1{reset}"
        )

        # ── BATCH 1: run LAYER 1 scenarios through the LLM worker pool ──
        pending = layer1_queue

        def _on_done(result: Dict[str, Any], idx: int, pbar: tqdm):
            with lock:
                save_result.append(result)
            pbar.update(1)
            if (idx + 1) % save_freq == 0 or (idx + 1) == len(pending):
                _flush()

        with tqdm(total=len(pending), desc="LAYER 1", unit="scenario") as pbar:
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._solve_scenario, s, tool_defs, num_attempts, free_mode,
                            layer1_diagnoses.get(s.get("scenario_id"))
                        ): (i, s)
                        for i, s in enumerate(pending)
                    }
                    for future in as_completed(futures):
                        idx, scenario = futures[future]
                        try:
                            result = future.result()
                            _on_done(result, idx, pbar)
                        except Exception as exc:
                            sid = scenario.get("scenario_id")
                            self.logger.error(f"[Scenario {sid}] failed: {exc}")
                            pbar.update(1)
            else:
                for idx, scenario in enumerate(pending):
                    result = self._solve_scenario(
                        scenario, tool_defs, num_attempts, free_mode,
                        layer1_diagnoses.get(scenario.get("scenario_id"))
                    )
                    _on_done(result, idx, pbar)

        _flush(force=True)
        self._format_submission(save_dir)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agents benchmarking")
    parser.add_argument("--server_url", type=str, default=os.getenv("SERVER_URL", "http://localhost:7860"))
    parser.add_argument("--model_url", type=str, default=os.getenv("PROVIDER_API_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL", "Qwen/Qwen3.5-35B-A3B"))
    parser.add_argument("--model_provider", type=str, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500) # full inference (uses test split by default)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=int(os.getenv("MAX_WORKERS", 1)))
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./logs/log.log")
    parser.add_argument("--no_free_mode", action="store_false", dest="free_mode")
    parser.add_argument("--no_think", action="store_true", help="Prepend /no_think to disable extended thinking (WaveSpeed)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger = init_logger(log_file=args.log_file)

    env = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    runner = AgentsRunner(
        environment=env,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        verbose=args.verbose,
        no_think=args.no_think,
        logger=logger,
    )

    runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        max_workers=args.max_workers,
        free_mode=args.free_mode,
    )
