#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, List


CELL_ID_RE = re.compile(r"\b\d+_\d+\b")
OPTION_ID_RE = re.compile(r"^C\d+$", re.IGNORECASE)
DEGREE_RE = re.compile(r"by\s+(-?\d+)\s+degrees", re.IGNORECASE)


def classify_option_label(label: str) -> str:
    text = (label or "").strip().lower()

    if "decrease a3 offset threshold" in text:
        return "decrease_a3_offset"
    if "increase a3 offset threshold" in text:
        return "increase_a3_offset"
    if "decrease covinterfreqa2rsrpthld" in text:
        return "decrease_interfreq_thresholds"
    if "modify pdcchoccupiedsymbolnum to 2sym" in text:
        return "set_pdcch_2sym"
    if "check test server and transmission issues" in text:
        return "check_transport"
    if "add neighbor relationship" in text:
        return "add_neighbor_relation"
    if "adjust the azimuth" in text:
        return "adjust_azimuth"
    if "press down the tilt angle" in text:
        return "press_down_tilt"
    if "lift the tilt angle" in text:
        return "lift_tilt"
    if "increase transmission power" in text:
        return "increase_tx_power"
    if "decrease transmission power" in text:
        return "decrease_tx_power"
    if "insufficient data" in text:
        return "insufficient_data"
    return "unknown"


def parse_options(options: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed_options: List[Dict[str, Any]] = []
    by_action: Dict[str, List[str]] = {}
    by_cell: Dict[str, List[str]] = {}
    valid_option_ids: List[str] = []

    for option in options or []:
        option_id = str(option.get("id", "")).strip().upper()
        label = str(option.get("label", "")).strip()
        if not OPTION_ID_RE.match(option_id):
            continue

        action = classify_option_label(label)
        cell_ids = CELL_ID_RE.findall(label)
        degree_match = DEGREE_RE.search(label)
        degrees = int(degree_match.group(1)) if degree_match else None

        parsed = {
            "id": option_id,
            "label": label,
            "action": action,
            "cell_ids": cell_ids,
            "primary_cell": cell_ids[0] if cell_ids else None,
            "degrees": degrees,
            "is_multi_cell": len(cell_ids) > 1,
        }
        parsed_options.append(parsed)
        valid_option_ids.append(option_id)
        by_action.setdefault(action, []).append(option_id)
        for cell_id in cell_ids:
            by_cell.setdefault(cell_id, []).append(option_id)

    return {
        "parsed_options": parsed_options,
        "valid_option_ids": sorted(valid_option_ids, key=lambda value: int(value[1:])),
        "cell_ids": sorted(by_cell.keys()),
        "by_action": by_action,
        "by_cell": by_cell,
    }
