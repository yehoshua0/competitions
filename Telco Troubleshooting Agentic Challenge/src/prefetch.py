#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_pipe_table(text: str) -> List[Dict[str, str]]:
    if not text or not isinstance(text, str):
        return []
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    reader = csv.DictReader(io.StringIO(cleaned), delimiter="|")
    return [dict(row) for row in reader]


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_parallel(environment: Any, scenario_id: str, calls: Iterable[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(environment._call_api, function_name=name, scenario_id=scenario_id, **params): (name, params)
            for name, params in calls
        }
        for future in as_completed(future_map):
            name, params = future_map[future]
            key = name if not params else f"{name}:{params}"
            try:
                results[key] = {
                    "tool": name,
                    "params": params,
                    "result": future.result(),
                }
            except Exception as exc:
                results[key] = {
                    "tool": name,
                    "params": params,
                    "result": {"error": str(exc)},
                }
    return results


def _select_focus_point(throughput_result: Dict[str, Any]) -> Dict[str, Any]:
    logs_text = throughput_result.get("Logs", "")
    rows = _parse_pipe_table(logs_text)
    min_row = None
    min_value = None

    for row in rows:
        value = _safe_float(row.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]"))
        if value is None:
            continue
        if min_value is None or value < min_value:
            min_value = value
            min_row = row

    timestamps = [row.get("Timestamp", "") for row in rows if row.get("Timestamp")]
    return {
        "rows": rows,
        "focus_time": (min_row or {}).get("Timestamp"),
        "focus_throughput_mbps": min_value,
        "first_time": timestamps[0] if timestamps else None,
        "last_time": timestamps[-1] if timestamps else None,
    }


def _summarize_signaling(signaling_text: str) -> Dict[str, Any]:
    lowered = (signaling_text or "").lower()
    return {
        "a2_count": lowered.count("a2"),
        "a3_count": lowered.count("a3"),
        "a5_count": lowered.count("a5"),
        "rrc_reest_count": lowered.count("re-est") + lowered.count("reest"),
        "handover_count": lowered.count("handover") + lowered.count("ho "),
    }


def _find_config_row(config_rows: List[Dict[str, str]], pci: Optional[int]) -> Optional[Dict[str, str]]:
    if pci is None:
        return None
    pci_text = str(pci)
    for row in config_rows:
        if str(row.get("PCI", "")).strip() == pci_text:
            return row
    return None


def _find_kpi_row(
    kpi_rows: List[Dict[str, str]],
    gnodeb_id: Optional[str],
    cell_id: Optional[str],
) -> Optional[Dict[str, str]]:
    if not gnodeb_id or not cell_id:
        return None
    for row in kpi_rows:
        if str(row.get("gNodeB_ID", "")).strip() == str(gnodeb_id) and str(row.get("Cell_ID", "")).strip() == str(cell_id):
            return row
    return None


def build_prefetch_bundle(environment: Any, scenario_id: str) -> Dict[str, Any]:
    stage_one = _run_parallel(
        environment,
        scenario_id,
        [
            ("get_throughput_logs", {}),
            ("get_config_data", {}),
            ("get_kpi_data", {}),
            ("get_all_cells_pci", {}),
        ],
    )

    throughput_result = stage_one.get("get_throughput_logs", {}).get("result", {})
    config_result = stage_one.get("get_config_data", {}).get("result", {})
    kpi_result = stage_one.get("get_kpi_data", {}).get("result", {})
    focus = _select_focus_point(throughput_result)

    focus_time = focus.get("focus_time")
    if not focus_time:
        return {
            "focus": focus,
            "summary": {"error": "Unable to determine focus timestamp from throughput logs."},
            "raw": stage_one,
        }

    stage_two = _run_parallel(
        environment,
        scenario_id,
        [
            ("get_signaling_plane_event_log", {"time": focus_time}),
            ("get_serving_cell_pci", {"time": focus_time}),
            ("get_serving_cell_rsrp", {"time": focus_time}),
            ("get_serving_cell_sinr", {"time": focus_time}),
            ("get_neighboring_cells_pci", {"time": focus_time}),
            ("get_rbs_allocated_to_user", {"time": focus_time}),
        ],
    )

    serving_pci = stage_two.get(f"get_serving_cell_pci:{{'time': '{focus_time}'}}", {}).get("result", {}).get("PCI")
    neighbor_pcis = stage_two.get(f"get_neighboring_cells_pci:{{'time': '{focus_time}'}}", {}).get("result", {}).get("PCIs", [])
    if not isinstance(neighbor_pcis, list):
        neighbor_pcis = []

    stage_three_calls: List[Tuple[str, Dict[str, Any]]] = []
    if serving_pci is not None:
        stage_three_calls.append(("get_cell_info", {"pci": int(serving_pci)}))
    for pci in neighbor_pcis[:3]:
        stage_three_calls.append(("get_neighboring_cell_rsrp", {"time": focus_time, "pci": int(pci)}))

    stage_three = _run_parallel(environment, scenario_id, stage_three_calls) if stage_three_calls else {}

    config_rows = _parse_pipe_table(config_result.get("Network Configuration Data", ""))
    kpi_rows = _parse_pipe_table(kpi_result.get("Traffic Data", ""))
    serving_cell_info = stage_three.get(f"get_cell_info:{{'pci': {serving_pci}}}", {}).get("result", {})

    serving_config = _find_config_row(config_rows, serving_pci)
    serving_kpi = _find_kpi_row(
        kpi_rows,
        serving_config.get("gNodeB ID") if serving_config else None,
        serving_config.get("Cell ID") if serving_config else None,
    )

    neighbor_summary: List[Dict[str, Any]] = []
    for pci in neighbor_pcis[:3]:
        rsrp_key = f"get_neighboring_cell_rsrp:{{'time': '{focus_time}', 'pci': {int(pci)}}}"
        rsrp_result = stage_three.get(rsrp_key, {}).get("result", {})
        neighbor_summary.append(
            {
                "pci": int(pci),
                "rsrp_dbm": rsrp_result.get("Filtered Tx BRSRP [dBm]"),
            }
        )

    raw = {}
    raw.update(stage_one)
    raw.update(stage_two)
    raw.update(stage_three)

    summary = {
        "focus_time": focus_time,
        "focus_throughput_mbps": focus.get("focus_throughput_mbps"),
        "throughput_window": {
            "start": focus.get("first_time"),
            "end": focus.get("last_time"),
            "samples": len(focus.get("rows", [])),
        },
        "serving_pci": serving_pci,
        "serving_rsrp_dbm": stage_two.get(f"get_serving_cell_rsrp:{{'time': '{focus_time}'}}", {}).get("result", {}).get("SS-RSRP (dBm)"),
        "serving_sinr_db": stage_two.get(f"get_serving_cell_sinr:{{'time': '{focus_time}'}}", {}).get("result", {}).get("SS-SINR (dB)"),
        "serving_rbs": stage_two.get(f"get_rbs_allocated_to_user:{{'time': '{focus_time}'}}", {}).get("result", {}).get("RBs"),
        "signaling_flags": _summarize_signaling(stage_two.get(f"get_signaling_plane_event_log:{{'time': '{focus_time}'}}", {}).get("result", "")),
        "serving_cell_info": serving_cell_info,
        "serving_config": serving_config,
        "serving_kpi": serving_kpi,
        "neighbor_cells": neighbor_summary,
    }

    return {
        "focus": focus,
        "summary": summary,
        "raw": raw,
    }
