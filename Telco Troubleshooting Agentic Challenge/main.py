#!/usr/bin/env python
# -*-coding:utf-8 -*-

import argparse
import csv
import io
import json
import logging
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import pandas as pd
import httpx
import requests
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from _types import ToolCall
from logger import init_logger
from src.options_parser import parse_options
from src.prefetch import build_prefetch_bundle
from src.system_prompt import get_system_prompt
from misc.utils import (
    print_model_response,
    print_tool_call,
    print_tool_result,
    extract_answer,
    extract_answer_all,
    compute_score,
)

os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")


BOXED_ID_RE = re.compile(r"C\d+", re.IGNORECASE)


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
                self.logger.info(f"[Tools API] scenario_id={scenario_id} GET {url} params={params}")
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

    def execute(self, tool_call: ToolCall, scenario_id: Optional[str] = None) -> str:
        """
        Execute a single OpenAI tool_call and return a JSON string for the tool message.
        """
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = self._call_api(function_name=function_name, scenario_id=scenario_id, **arguments)
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
# Prompt / Answer helpers
# ------------------------------------------------------------------------------

def _parse_pipe_table(text: str) -> List[Dict[str, str]]:
    if not text or not isinstance(text, str):
        return []
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    reader = csv.DictReader(io.StringIO(cleaned), delimiter="|")
    return [dict(row) for row in reader]


def _format_option_summary(parsed_options: Dict[str, Any]) -> str:
    lines: List[str] = []
    for item in parsed_options.get("parsed_options", []):
        cells = ", ".join(item.get("cell_ids") or []) or "none"
        lines.append(f"- {item['id']}: action={item['action']}; cells={cells}; label={item['label']}")
    return "\n".join(lines)


def _format_action_groups(parsed_options: Dict[str, Any]) -> str:
    lines: List[str] = []
    for action, option_ids in sorted(parsed_options.get("by_action", {}).items()):
        lines.append(f"- {action}: {', '.join(option_ids)}")
    return "\n".join(lines)


def _format_prefetch_summary(prefetch_bundle: Dict[str, Any]) -> str:
    summary = prefetch_bundle.get("summary", {})
    if summary.get("error"):
        return f"Prefetch error: {summary['error']}"

    serving_config = summary.get("serving_config") or {}
    serving_kpi = summary.get("serving_kpi") or {}
    signaling_flags = summary.get("signaling_flags") or {}
    neighbor_cells = summary.get("neighbor_cells") or []

    neighbor_lines = []
    for neighbor in neighbor_cells:
        neighbor_lines.append(
            f"- PCI {neighbor.get('pci')}: neighbor_rsrp={neighbor.get('rsrp_dbm')}"
        )

    lines = [
        f"Focus timestamp: {summary.get('focus_time')}",
        f"Worst throughput: {summary.get('focus_throughput_mbps')} Mbps",
        f"Serving PCI: {summary.get('serving_pci')}",
        f"Serving RSRP: {summary.get('serving_rsrp_dbm')} dBm",
        f"Serving SINR: {summary.get('serving_sinr_db')} dB",
        f"Allocated RBs: {summary.get('serving_rbs')}",
        (
            "Signaling flags: "
            f"A2={signaling_flags.get('a2_count', 0)}, "
            f"A3={signaling_flags.get('a3_count', 0)}, "
            f"A5={signaling_flags.get('a5_count', 0)}, "
            f"RRC_reest={signaling_flags.get('rrc_reest_count', 0)}, "
            f"handover={signaling_flags.get('handover_count', 0)}"
        ),
        (
            "Serving config: "
            f"cell={serving_config.get('gNodeB ID')}_{serving_config.get('Cell ID')}, "
            f"A3Offset={serving_config.get('IntraFreqHoA3Offset [0.5dB]')}, "
            f"A3Hyst={serving_config.get('IntraFreqHoA3Hyst [0.5dB]')}, "
            f"TxPower={serving_config.get('Transmission Power')}, "
            f"PDCCH={serving_config.get('PdcchOccupiedSymbolNum')}, "
            f"Azimuth={serving_config.get('Mechanical Azimuth')}, "
            f"Downtilt={serving_config.get('Mechanical Downtilt')}"
        ),
        (
            "Serving KPI: "
            f"DL_PRB={serving_kpi.get('Downlink PRB utilization(%)')}, "
            f"DL_CCE_util={serving_kpi.get('Downlink CCE utilization(%)')}, "
            f"DL_CCE_success={serving_kpi.get('Downlink CCE Allocation Success Rate(%)')}, "
            f"DL_user_tp={serving_kpi.get('User Downlink Throughput(Mbps)')}"
        ),
        "Neighbor summary:",
    ]
    lines.extend(neighbor_lines or ["- none"])
    return "\n".join(lines)


def _build_initial_user_message(task: Dict[str, Any], parsed_options: Dict[str, Any], prefetch_bundle: Dict[str, Any]) -> str:
    option_text = "\n".join(
        f"{item['id']}: {item['label']}" for item in task.get("options", [])
    )
    return (
        "Scenario description:\n"
        f"{task.get('description', '').strip()}\n\n"
        "Prefetched evidence:\n"
        f"{_format_prefetch_summary(prefetch_bundle)}\n\n"
        "Parsed option catalogue:\n"
        f"{_format_option_summary(parsed_options)}\n\n"
        "Options grouped by action:\n"
        f"{_format_action_groups(parsed_options)}\n\n"
        "Original options:\n"
        f"{option_text}\n\n"
        "Use the prefetched evidence first. If an ambiguity remains, call additional tools before answering. "
        "Return only one boxed answer in the final response."
    )


def _canonicalize_answer(text: str, valid_option_ids: Optional[List[str]] = None) -> str:
    extracted = extract_answer_all(text or "")
    source = extracted or (text or "")
    option_ids = [match.upper() for match in BOXED_ID_RE.findall(source)]
    if valid_option_ids:
        valid_set = {item.upper() for item in valid_option_ids}
        option_ids = [item for item in option_ids if item in valid_set]
    unique_sorted = sorted(set(option_ids), key=lambda value: int(value[1:]))
    if not unique_sorted:
        return ""
    return f"\\boxed{{{'|'.join(unique_sorted)}}}"


def _canonicalize_answer_from_sources(valid_option_ids: Optional[List[str]], *sources: Any) -> str:
    for source in sources:
        normalized = _canonicalize_answer(source if isinstance(source, str) else str(source or ""), valid_option_ids)
        if normalized:
            return normalized
    return ""


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
            max_tokens: int = 16000,
            max_retries: int = 3,
            max_iterations: int = 20,
            verbose: bool = False,
            logger: logging.Logger = None
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logger if logger is not None else init_logger()
        self.running_metrics = {}

        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            # http_client=httpx.Client(verify=False),
            http_client=httpx.Client(timeout=180.0),
        )

    def _call_model(self, messages: List[Dict[str, Any]], functions: List[Dict[str, Any]], **kwargs):
        base_wait_time = 1.0

        call_kwargs = {
            "model": f"{self.model_provider}/{self.model_name}" if self.model_provider else self.model_name,
            "messages": messages,
            # "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_tokens,
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

    # def run(self, scenario: Dict[str, Any], free_mode: bool = False) -> Dict[str, Any]:
    #     scenario_id = scenario.get("scenario_id")
    #     task = scenario.get("task", {})

    #     root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in task.get("options", [])])

    #     # tools from server
    #     tool_defs = self.environment.get_tools()
    #     if not tool_defs:
    #         return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available"}

    #     question = task.get("description", "") + f"\nOptions:\n{root_causes}"

    #     messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]

    #     num_tool_calls = 0
    #     list_tool_calls = []
    #     status = None
    #     reason = None
    #     last_msg = None

    #     for i in range(self.max_iterations):
    #         self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 1} conversation, calling tools:")

    #         msg = self._call_model(messages, functions=tool_defs)
    #         if msg is None:
    #             continue

    #         last_msg = msg
    #         messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

    #         if self.verbose:
    #             print_model_response(msg, logger=self.logger, minimize=False)

    #         # tool calls
    #         if msg.tool_calls:
    #             num_tool_calls += len(msg.tool_calls)

    #             for j, tool_call in enumerate(msg.tool_calls):
    #                 if self.verbose:
    #                     print_tool_call(tool_call, logger=self.logger)

    #                 tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)

    #                 messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})

    #                 if self.verbose:
    #                     print_tool_result(tool_result, logger=self.logger)

    #                 has_failed = "error" in tool_result
    #                 list_tool_calls.append(
    #                     {
    #                         "function_name": tool_call.function.name,
    #                         "arguments": tool_call.function.arguments,
    #                         "turn": i + 1,
    #                         "has_failed": has_failed,
    #                         "order": j + 1,
    #                         "results": tool_result
    #                     }
    #                 )

    #         # final answer
    #         # elif msg.content or msg.reasoning_content:
    #         elif msg.content:
    #             status = "solved"
    #             break

    #         else:
    #             status = "unresolved"
    #             reason = "Unable to answer this question."
    #             break

    #     if status is None:
    #         status = "unresolved"
    #         reason = "The maximum number of iterations has been reached."

    #     # Optional final constraint prompt
    #     if free_mode:
    #         current_answer = getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content",
    #                                                                      "") if last_msg else "",
    #         current_traces = getattr(last_msg, "reasoning_content", "") if last_msg else ""
    #         agent_answer = extract_answer(current_answer) or extract_answer(current_traces)
    #         if agent_answer == "":
    #             self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 2} conversation, answer question:")
    #             status = "solved"

    #             if 'Select the most appropriate optimization solution' in question:
    #                 messages.append(
    #                     {
    #                         "role": "user",
    #                         "content": (
    #                             "This is a single-answer question. Select the most appropriate optimization solution and enclose its number in \\boxed{{}} "
    #                             f"in the final answer. For example, \\boxed{{C3}} \nPotential root causes:\n{root_causes}\n"
    #                         ),
    #                     }
    #                 )
    #             else:
    #                 messages.append(
    #                     {
    #                         "role": "user",
    #                         "content": (
    #                             "This is a multiple-answer question. Select two to four possible optimization solutions and enclose their numbers in \\boxed{{}} "
    #                             f"in the final answer. For example,  \\boxed{{C3|C5}} or \\boxed{{C7|C11}}. \nPotential root causes:\n{root_causes}\n"
    #                         ),
    #                     }
    #                 )


    #             msg2 = self._call_model(messages, functions=[])
    #             if msg2 is not None:
    #                 last_msg = msg2

    #     return {
    #         "scenario_id": scenario_id,
    #         "num_iterations": (i + 1),
    #         "tool_calls": list_tool_calls,
    #         "num_tool_calls": num_tool_calls,
    #         "status": status,
    #         "traces": getattr(last_msg, "reasoning_content", "") if last_msg else "",
    #         "answer": getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content","") if last_msg else "",
    #         "messages": messages,
    #         "reason": reason,
    #     }

    def run(self, scenario: Dict[str, Any], free_mode: bool = False,
            tool_defs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {})
        parsed_options = parse_options(task.get("options", []))
        prefetch_bundle = build_prefetch_bundle(self.environment, scenario_id=scenario_id)

        # tools from server — accept pre-fetched to avoid a per-scenario HTTP call
        if tool_defs is None:
            tool_defs = self.environment.get_tools()
        if not tool_defs:
            return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available"}
        question = _build_initial_user_message(task, parsed_options, prefetch_bundle)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": question},
        ]

        num_tool_calls = 0
        list_tool_calls = []
        status = None
        reason = None
        last_msg = None

        max_iterations = min(self.max_iterations, 4)
        for i in range(max_iterations):
            self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 1} conversation, calling tools:")

            msg = self._call_model(messages, functions=tool_defs)
            if msg is None:
                continue

            last_msg = msg

            # ---- store assistant message correctly (do NOT force empty content) ----
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if getattr(msg, "content", None) is not None:
                assistant_msg["content"] = msg.content
            if getattr(msg, "tool_calls", None):
                assistant_msg["tool_calls"] = msg.tool_calls

            messages.append(assistant_msg)

            if self.verbose:
                print_model_response(msg, logger=self.logger, minimize=False)

            # ---- tool calls: execute and then immediately continue (call model again) ----
            if getattr(msg, "tool_calls", None):
                num_tool_calls += len(msg.tool_calls)

                for j, tool_call in enumerate(msg.tool_calls):
                    if self.verbose:
                        print_tool_call(tool_call, logger=self.logger)

                    tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)

                    # IMPORTANT: tool message must include tool_call_id
                    messages.append(
                        {"role": "tool", "content": tool_result, "tool_call_id": tool_call.id}
                    )

                    if self.verbose:
                        print_tool_result(tool_result, logger=self.logger)

                    has_failed = "error" in tool_result
                    list_tool_calls.append(
                        {
                            "function_name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "turn": i + 1,
                            "has_failed": has_failed,
                            "order": j + 1,
                            "results": tool_result,
                        }
                    )

                # Now that tool outputs are appended, ask the model again next loop
                continue

            # ---- no tool calls: this should be the final answer attempt ----
            content = getattr(msg, "content", None) or ""
            if content.strip() != "":
                status = "solved"
                break

            # If content empty and no tools => unresolved
            status = "unresolved"
            reason = "Model returned empty response without tool calls."
            break

        if status is None:
            status = "unresolved"
            reason = "The maximum number of iterations has been reached."

        # ---- Force a boxed final answer if extraction fails (competition format) ----
        # This is crucial: many models answer in prose unless forced.
        if status == "solved":
            raw = (getattr(last_msg, "content", None) or "")
            extracted = extract_answer_all(raw)

            if extracted == "":
                self.logger.info(f"\n[Scenario: {scenario_id}] Final formatting turn (force \\boxed{{}}):")

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You MUST answer with ONLY one of these exact formats:\n"
                            "- \\boxed{C3}\n"
                            "- \\boxed{C5|C7}\n"
                            "Rules:\n"
                            f"1) Use only IDs from the provided Options: {', '.join(parsed_options.get('valid_option_ids', []))}.\n"
                            "2) Sort ascending.\n"
                            "3) No words, no punctuation, no newlines, only the box.\n"
                        )
                    }
                )

                msg2 = self._call_model(messages, functions=[])
                if msg2 is not None:
                    last_msg = msg2
                    # store it too (optional but keeps trace)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": getattr(msg2, "content", None),
                        }
                    )

        normalized_answer = _canonicalize_answer_from_sources(
            parsed_options.get("valid_option_ids"),
            (getattr(last_msg, "content", "") or "") if last_msg else "",
            getattr(last_msg, "reasoning_content", "") if last_msg else "",
        )

        return {
            "scenario_id": scenario_id,
            "num_iterations": (i + 1),
            "tool_calls": list_tool_calls,
            "num_tool_calls": num_tool_calls,
            "status": status,
            "traces": getattr(last_msg, "reasoning_content", "") if last_msg else "",
            "answer": normalized_answer or ((getattr(last_msg, "content", "") or "") if last_msg else ""),
            "messages": messages,
            "reason": reason,
        }


    def benchmark(
            self,
            num_attempts: int,
            save_dir: str,
            save_freq: int = 10,
            max_samples: int = None,
            free_mode: bool = False,
            max_workers: int = 1,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        scenarios = self.environment.get_scenarios()
        if max_samples is not None:
            scenarios = scenarios[:min(max_samples, len(scenarios))]

        # Fetch tool definitions once — reused across all scenarios
        tool_defs = self.environment.get_tools()
        if not tool_defs:
            self.logger.error("No tools available from server — aborting benchmark.")
            return

        n_total = len(scenarios)
        # Ordered slots so result.csv rows stay in original scenario order
        result_slots: List[Optional[Dict[str, Any]]] = [None] * n_total
        lock = threading.Lock()
        completed_count = 0

        def _process(idx: int, scenario: Dict[str, Any]) -> None:
            nonlocal completed_count
            scenario_id = scenario.get("scenario_id")
            start_time = time.time()

            n_success = 0.0
            agent_answers: List[str] = []
            sample_response: Optional[Dict[str, Any]] = {}

            for attempt in range(num_attempts):
                self.logger.info(f"[Scenario {scenario_id}] attempt {attempt + 1}/{num_attempts}")

                response = self.run(scenario=scenario, free_mode=free_mode, tool_defs=tool_defs)
                sample_response = response

                agent_answer = (
                    extract_answer_all(response.get("answer", ""))
                    or extract_answer_all(response.get("traces", ""))
                )

                if response.get("status") == "solved" and agent_answer:
                    ground_truth = scenario.get("answer")
                    n_success += compute_score(agent_answer, ground_truth)
                    agent_answers.append(agent_answer)
                    pink = "\033[95m"
                    reset = "\033[0m"
                    self.logger.info(f"{pink}\n[Scenario: {scenario_id}] Agent's answer is {agent_answer}, ground truth is {ground_truth}{reset}.")

                elif response.get("status") == "solved":
                    self.logger.warning(
                        "[Scenario: %s] Model returned solved status but no parseable answer. Raw answer=%r traces=%r",
                        scenario_id,
                        response.get("answer", ""),
                        response.get("traces", ""),
                    )

            acc = n_success / float(num_attempts)
            latency = round((time.time() - start_time) / float(num_attempts), 2)
            submission_answer = agent_answers[0] if agent_answers else (
                extract_answer_all((sample_response or {}).get("answer", ""))
                or extract_answer_all((sample_response or {}).get("traces", ""))
                or "C16"
            )

            slot = {
                "scenario_id": scenario_id,
                "free_mode": free_mode,
                "response": (sample_response or {}).get("answer", ""),
                "traces": (sample_response or {}).get("traces", ""),
                "num_iterations": (sample_response or {}).get("num_iterations", 0),
                "num_tool_calls": (sample_response or {}).get("num_tool_calls", 0),
                "tool_calls": (sample_response or {}).get("tool_calls", []),
                "answers": agent_answers,
                "ground_truth": scenario.get("answer"),
                "accuracy": acc,
                "latency": latency,
                "_submission_answer": submission_answer,
            }

            with lock:
                result_slots[idx] = slot
                completed_count += 1
                # Periodic checkpoint: write all completed results in original order
                if completed_count % save_freq == 0 or completed_count == n_total:
                    valid = [s for s in result_slots if s is not None]
                    df = pd.DataFrame([{"scenario_id": s["scenario_id"], "answers": s["_submission_answer"]} for s in valid])
                    df.to_csv(os.path.join(save_dir, "result.csv"), index=False)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process, idx, sc): idx for idx, sc in enumerate(scenarios)}
            with tqdm(as_completed(futures), total=n_total, unit="scenario",
                      desc=f"Benchmarking (workers={max_workers})") as pbar:
                for future in pbar:
                    try:
                        future.result()
                        with lock:
                            done = sum(1 for s in result_slots if s is not None)
                            answered = sum(1 for s in result_slots if s is not None and s["_submission_answer"])
                        pbar.set_postfix(answered=answered, empty=done - answered)
                    except Exception as exc:
                        idx = futures[future]
                        self.logger.error(f"Scenario index {idx} raised an exception: {exc}", exc_info=True)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    # _model_url  = os.environ.get("AGENT_MODEL_URL",  "http://localhost:30000/v1")
    _model_url  = os.environ.get("AGENT_MODEL_URL",  "https://rbb6b3g72lf89t-30000.proxy.runpod.net/v1")
    _model_name = os.environ.get("AGENT_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B")
    _max_workers = int(os.environ.get("AGENT_MAX_WORKERS", "1"))

    parser = argparse.ArgumentParser(description="Agents benchmarking")
    parser.add_argument("--server_url",    type=str,  default="http://localhost:7860")
    parser.add_argument("--model_url",     type=str,  default=_model_url)
    parser.add_argument("--model_name",    type=str,  default=_model_name)
    parser.add_argument("--model_provider",type=str,  default=None)
    parser.add_argument("--max_workers",   type=int,  default=_max_workers,
                        help="Parallel scenario workers (AGENT_MAX_WORKERS env var)")
    parser.add_argument("--num_attempts",  type=int,  default=1)
    parser.add_argument("--max_samples",   type=int,  default=130)
    parser.add_argument("--save_freq",     type=int,  default=10)
    parser.add_argument("--max_tokens",    type=int,  default=16000)
    parser.add_argument("--max_iterations",type=int,  default=10)
    parser.add_argument("--save_dir",      type=str,  default="./results")
    parser.add_argument("--log_file",      type=str,  default="./logs/log.log")
    parser.add_argument("--free_mode",     action="store_false")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    logger = init_logger(log_file=args.log_file)

    Environment = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    runner = AgentsRunner(
        environment=Environment,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        logger=logger
    )

    runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        free_mode=args.free_mode,
        max_workers=args.max_workers,
    )
