"""LAYER 0 — Deterministic Rule Engine for Telco Troubleshooting Agentic Challenge.

High-confidence rules bypass the LLM entirely (0 API calls).
Low/none-confidence cases inject extracted features into the LLM system prompt.

Verified patterns (from training data analysis):
  P4a  -- A3 diff-cell event + one cell has IntraFreqHoA3Offset=10 -> dec_a3 on that cell
  P4b  -- A3 diff-cell event + primary cells have A3={2,6}          -> multi-action answer
  P6a  -- A3 same-cell event + max CCE fail rate > 0.4              -> pdcch (2SYM fix)
  P6b  -- A3 same-cell event + serving cell DL PRB <= 25%           -> transport check
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    confidence: str          # "high" | "low" | "none"
    answer_ids: List[str]    # e.g. ["C9"] or ["C2","C8","C11","C16"] — sorted by numeric suffix
    cause_types: List[str]   # e.g. ["dec_a3"]
    rule_desc: str           # human-readable description of the rule that fired
    features: Dict           # extracted numeric/boolean features (for LLM context injection)
    excluded_options: List[str]   # option IDs definitively ruled out
    candidate_options: List[str]  # remaining options the LLM should focus on
    tag: str                 # "single-answer" | "multiple-answer"


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def parse_pipe_data(data_str: Optional[str]) -> pd.DataFrame:
    """Parse a pipe-delimited CSV string into a DataFrame, stripping whitespace."""
    if not data_str or not data_str.strip():
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(data_str), sep="|", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        return df.map(lambda x: x.strip() if isinstance(x, str) else x)
    except Exception:
        try:
            # Fallback: applymap for older pandas
            df = pd.read_csv(StringIO(data_str), sep="|", dtype=str)
            df.columns = [c.strip() for c in df.columns]
            return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        except Exception:
            return pd.DataFrame()


def find_col(df: pd.DataFrame, includes: List[str], excludes: List[str] = []) -> Optional[str]:
    """Return first column name that contains all `includes` substrings and none of `excludes`."""
    for col in df.columns:
        if all(inc in col for inc in includes) and not any(exc in col for exc in excludes):
            return col
    return None


def _build_cell_id(row: Any) -> str:
    """Build 'gNodeBID_CellID' string from a net_df row."""
    gnb = str(row.get("gNodeB ID", "")).strip()
    cid = str(row.get("Cell ID", "")).strip()
    return f"{gnb}_{cid}"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all diagnostic features from the 5 scenario data tables.

    Returns a flat dict with keys for signal metrics, tilt angles, signaling
    booleans, traffic stats, and derived serving-cell info.
    """
    up_df   = parse_pipe_data(data.get("user_plane_data"))
    net_df  = parse_pipe_data(data.get("network_configuration_data"))
    sig_df  = parse_pipe_data(data.get("signaling_plane_data"))
    traf_df = parse_pipe_data(data.get("traffic_data"))

    feat: Dict[str, Any] = {
        # user_plane_data
        "min_rsrp": None, "avg_rsrp": None, "mean_sinr": None,
        "handovers": 0, "max_speed": 0.0, "avg_rb": None, "max_cce_fail": 0.0,
        "num_neighbors": 0, "has_pci_conflict": False,
        # network_configuration_data
        "max_tilt": 0.0, "min_tilt": 999.0, "total_tilt": 0.0,
        "a3_offsets": {},          # {cell_id: str}
        "pdcch_1sym_cell": None,
        # signaling_plane_data
        "has_rrc": False, "has_a2": False, "has_a5": False,
        "has_a3_diff_cells": False, "has_a3_same_cell": False,
        "has_ho_attempt": False,
        # traffic_data
        "serving_dl_prb": None,
        # derived
        "serving_cell_id": None,
    }

    # ── User plane ──────────────────────────────────────────────────────────
    if not up_df.empty:
        rsrp_col = find_col(up_df, ["Serving SS-RSRP"])
        sinr_col = find_col(up_df, ["Serving SS-SINR"])
        rb_col   = find_col(up_df, ["DL RB Num"])
        spd_col  = find_col(up_df, ["GPS Speed"])
        cce_col  = find_col(up_df, ["CCE Fail"])
        pci_col  = find_col(up_df, ["Serving PCI"])

        for attr, col in [("min_rsrp", rsrp_col), ("avg_rsrp", rsrp_col),
                          ("mean_sinr", sinr_col), ("avg_rb", rb_col)]:
            if col:
                vals = pd.to_numeric(up_df[col], errors="coerce").dropna()
                if not vals.empty:
                    feat[attr] = float(vals.min() if "min" in attr else vals.mean())

        if spd_col:
            vals = pd.to_numeric(up_df[spd_col], errors="coerce").dropna()
            if not vals.empty:
                feat["max_speed"] = float(vals.max())

        if cce_col:
            vals = pd.to_numeric(up_df[cce_col], errors="coerce").dropna()
            if not vals.empty:
                feat["max_cce_fail"] = float(vals.max())

        if pci_col:
            pcis = pd.to_numeric(up_df[pci_col], errors="coerce").dropna()
            feat["handovers"] = int((pcis.diff() != 0).sum())

            if not pcis.empty and not net_df.empty:
                serving_pci = str(int(Counter(pcis.tolist()).most_common(1)[0][0]))
                if "PCI" in net_df.columns:
                    match = net_df[net_df["PCI"] == serving_pci]
                    if not match.empty:
                        feat["serving_cell_id"] = _build_cell_id(match.iloc[0])

        # Unique neighbor PCIs
        neigh_pcis: Set[str] = set()
        for col in up_df.columns:
            if "Neighbor" in col and "PCI" in col:
                for v in up_df[col].dropna():
                    s = str(v).strip()
                    if s and s not in ("-", "nan", ""):
                        neigh_pcis.add(s)
        feat["num_neighbors"] = len(neigh_pcis)

        # PCI mod-30 conflict detection
        if pci_col and neigh_pcis:
            try:
                serving_pci_int = int(feat.get("serving_cell_id", "").split("_")[0])
            except Exception:
                serving_pci_int = None
            if serving_pci_int is not None:
                for np_str in neigh_pcis:
                    try:
                        if (serving_pci_int % 30) == (int(np_str) % 30):
                            feat["has_pci_conflict"] = True
                            break
                    except Exception:
                        pass

    # ── Network configuration ───────────────────────────────────────────────
    if not net_df.empty:
        md_col    = find_col(net_df, ["Mechanical Downtilt"])
        dt_col    = find_col(net_df, ["Digital Tilt"])
        a3_col    = find_col(net_df, ["IntraFreqHoA3Offset"])
        pdcch_col = find_col(net_df, ["PdcchOccupied"])

        tilts: List[float] = []
        for _, row in net_df.iterrows():
            md = 0.0
            dt = 0.0
            try:
                if md_col:
                    md = float(row[md_col])
            except Exception:
                pass
            try:
                if dt_col:
                    dt_raw = float(row[dt_col])
                    dt = 6.0 if dt_raw == 255.0 else dt_raw  # 255 is a vendor sentinel -> 6
            except Exception:
                pass
            tilts.append(md + dt)

            cell_id = _build_cell_id(row)
            if a3_col:
                feat["a3_offsets"][cell_id] = str(row.get(a3_col, "")).strip()

        if tilts:
            feat["max_tilt"]   = float(max(tilts))
            feat["min_tilt"]   = float(min(tilts))
            feat["total_tilt"] = float(sum(tilts))

        if pdcch_col:
            match = net_df[net_df[pdcch_col].str.strip() == "1SYM"]
            if not match.empty:
                feat["pdcch_1sym_cell"] = _build_cell_id(match.iloc[0])

    # ── Signaling plane ─────────────────────────────────────────────────────
    if not sig_df.empty and "Event Name" in sig_df.columns:
        events = sig_df["Event Name"].dropna().str.strip().tolist()

        feat["has_rrc"]        = any(e in ("NRRrcReestabAttempt", "NRRrcSetupRequest") for e in events)
        feat["has_a2"]         = any(e == "NREventA2" for e in events)   # exact match (not MeasConfig)
        feat["has_a5"]         = any(e == "NREventA5" for e in events)
        feat["has_ho_attempt"] = any(e == "NRHandoverAttempt" for e in events)

        content_col = find_col(sig_df, ["Event Content"])
        if content_col:
            a3_rows = sig_df[sig_df["Event Name"].str.strip() == "NREventA3"]
            for _, row in a3_rows.iterrows():
                content = str(row.get(content_col, ""))
                ms = re.search(r"ServCellPCI:(\d+)", content)
                mn = re.search(r"NCellPCI:(\d+)", content)
                if ms and mn:
                    if ms.group(1) == mn.group(1):
                        feat["has_a3_same_cell"] = True
                    else:
                        feat["has_a3_diff_cells"] = True

    # ── Traffic data — serving cell DL PRB ──────────────────────────────────
    scid = feat.get("serving_cell_id")
    if not traf_df.empty and scid:
        parts = scid.split("_")
        if len(parts) >= 2:
            gnb_id, cell_id = parts[0], parts[1]
            # traffic_data uses gNodeB_ID / Cell_ID (underscores)
            gnb_col  = find_col(traf_df, ["gNodeB"])
            cell_col = find_col(traf_df, ["Cell"])
            prb_col  = find_col(traf_df, ["Downlink PRB"])
            if gnb_col and cell_col and prb_col:
                mask = (traf_df[gnb_col].str.strip() == gnb_id) & \
                       (traf_df[cell_col].str.strip() == cell_id)
                match = traf_df[mask]
                if not match.empty:
                    try:
                        feat["serving_dl_prb"] = float(match.iloc[0][prb_col])
                    except Exception:
                        pass

    return feat


# ---------------------------------------------------------------------------
# Option -> action mapping
# ---------------------------------------------------------------------------

# Order matters: more specific keywords before generic ones
_ACTION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("dec_a3",        ["decrease", "a3 offset"]),
    ("inc_a3",        ["increase", "a3 offset"]),
    ("dec_power",     ["decrease", "transmission power"]),
    ("inc_power",     ["increase", "transmission power"]),
    ("press_tilt",    ["press down", "tilt"]),
    ("lift_tilt",     ["lift", "tilt"]),
    ("add_neighbor",  ["add neighbor"]),
    ("cov_interfreq", ["covinterfreqa2rsrpthld"]),
    ("pdcch",         ["pdcchoccupied"]),
    ("azimuth",       ["azimuth"]),
    ("transport",     ["check test server"]),
    ("insufficient",  ["insufficient data"]),
]


def _classify_label(label: str) -> str:
    lo = label.lower().replace(" ", "")
    for action, kws in _ACTION_PATTERNS:
        if all(kw.replace(" ", "") in lo for kw in kws):
            return action
    return "unknown"


def _extract_cell_ids_from_label(label: str) -> List[str]:
    """Extract cell IDs in format 'gNodeBID_CellID' (e.g. '3267220_2') from a label."""
    return re.findall(r"\d+_\d+", label)


def identify_primary_cells(options: List[Dict]) -> List[str]:
    """Return the 2 cell IDs most frequently mentioned across all option labels."""
    counter: Counter = Counter()
    for opt in options:
        for cid in _extract_cell_ids_from_label(opt.get("label", "")):
            counter[cid] += 1
    return [c for c, _ in counter.most_common(2)]


def map_options_to_actions(options: List[Dict]) -> Dict:
    """Build structured lookup dicts from an options list.

    Returns:
        by_action: {action_type: [{"id": "C9", "cells": ["3267220_2"]}]}
        by_cell:   {cell_id: {action_type: option_id}}
        all_ids:   ["C1", "C2", ...]
    """
    by_action: Dict[str, List[Dict]] = {}
    by_cell: Dict[str, Dict[str, str]] = {}

    for opt in options:
        oid   = opt["id"]
        label = opt.get("label", "")
        action = _classify_label(label)
        cells  = _extract_cell_ids_from_label(label)

        by_action.setdefault(action, []).append({"id": oid, "cells": cells})
        for cid in cells:
            by_cell.setdefault(cid, {})[action] = oid

    return {
        "by_action": by_action,
        "by_cell":   by_cell,
        "all_ids":   [o["id"] for o in options],
    }


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def _sort_ids(ids: List[str]) -> List[str]:
    """Sort option IDs by numeric suffix (C2, C8, C11 -> ['C2', 'C8', 'C11'])."""
    try:
        return sorted(ids, key=lambda x: int(re.search(r"\d+", x).group()))
    except Exception:
        return sorted(ids)


def _get_opt(by_cell: Dict, cell_id: str, action: str) -> Optional[str]:
    return by_cell.get(cell_id, {}).get(action)


def _first_opt(by_action: Dict, action: str) -> Optional[str]:
    entries = by_action.get(action, [])
    return entries[0]["id"] if entries else None


def apply_rules(
    features: Dict,
    option_map: Dict,
    primary_cells: List[str],
    tag: str,
) -> Tuple[List[str], str, List[str], str]:
    """Apply all rules in priority order.

    Returns (answer_ids, confidence, cause_types, rule_desc).
    answer_ids is empty when confidence is "none".
    """
    by_action = option_map.get("by_action", {})
    by_cell   = option_map.get("by_cell",   {})
    a3_offsets = features.get("a3_offsets", {})

    # ── P4a: A3 diff-cell event + one cell has A3 offset = 10 (= 5 dB) ────
    # Fix: decrease A3 offset on that cell to trigger earlier handover.
    if features.get("has_a3_diff_cells"):
        a3_10_cells = [cid for cid, v in a3_offsets.items() if v == "10"]
        # Prefer primary cells to avoid operating on background cells
        primary_a3_10 = [c for c in a3_10_cells if c in primary_cells]
        targets = primary_a3_10 if primary_a3_10 else a3_10_cells

        for cell in targets:
            opt = _get_opt(by_cell, cell, "dec_a3")
            if opt:
                return (
                    [opt], "high", ["dec_a3"],
                    f"P4a: cell {cell} A3=10 -> dec_a3 -> {opt}",
                )

        # ── P4b: A3 diff-cell event + primary cells have A3 offsets {2, 6} ─
        # Fix: dec_power + press_tilt + inc_a3 on A3=6 cell; inc_a3 on A3=2 cell.
        if len(primary_cells) >= 2:
            pc_a3 = {c: a3_offsets.get(c, "") for c in primary_cells}
            a3_values = set(pc_a3.values())

            if a3_values == {"2", "6"}:
                cell_6 = next(c for c in primary_cells if pc_a3[c] == "6")
                cell_2 = next(c for c in primary_cells if pc_a3[c] == "2")
                answers: List[str] = []
                for act in ("dec_power", "press_tilt", "inc_a3"):
                    o = _get_opt(by_cell, cell_6, act)
                    if o:
                        answers.append(o)
                o2 = _get_opt(by_cell, cell_2, "inc_a3")
                if o2:
                    answers.append(o2)
                if answers:
                    return (
                        _sort_ids(answers), "high",
                        ["dec_power", "press_tilt", "inc_a3", "inc_a3"],
                        f"P4b: A3={{2,6}} cells {cell_6},{cell_2} -> multi-action",
                    )

    # ── P6a: A3 same-cell event + high CCE fail rate -> PDCCH congestion ────
    # The CCE fail metric is measured on the SERVING CELL, so we fix the serving cell.
    if features.get("has_a3_same_cell") and features.get("max_cce_fail", 0.0) > 0.4:
        serving_cell = features.get("serving_cell_id")
        if serving_cell:
            opt = _get_opt(by_cell, serving_cell, "pdcch")
            if opt:
                return (
                    [opt], "high", ["pdcch"],
                    f"P6a: CCE_fail={features['max_cce_fail']:.2f}>0.4 -> pdcch {serving_cell} -> {opt}",
                )
        opt = _first_opt(by_action, "pdcch")
        if opt:
            return ([opt], "high", ["pdcch"], f"P6a: CCE_fail>0.4 -> pdcch (fallback) -> {opt}")

    # ── P6b: A3 same-cell event + low DL PRB -> transport/backhaul issue ────
    if features.get("has_a3_same_cell"):
        dl_prb = features.get("serving_dl_prb")
        if dl_prb is not None and dl_prb <= 25.0:
            opt = _first_opt(by_action, "transport")
            if opt:
                return (
                    [opt], "high", ["transport"],
                    f"P6b: DL_PRB={dl_prb:.1f}%<=25 -> transport -> {opt}",
                )

    # ── Layer B: reference threshold rules (informational, low confidence) ──
    # These mirror the 17 rules from the previous competition's winning solution.
    # In the new agentic format they yield LOW confidence so the LLM can verify.
    min_rsrp  = features.get("min_rsrp")  or -80.0
    handovers = features.get("handovers") or 0
    max_speed = features.get("max_speed") or 0.0
    avg_rb    = features.get("avg_rb")    or 200.0
    max_tilt  = features.get("max_tilt")  or 0.0
    min_tilt  = features.get("min_tilt")  or 0.0
    total_tilt = features.get("total_tilt") or 0.0
    num_neigh  = features.get("num_neighbors") or 0

    # Cause type -> action mapping for the new 22-option format
    cause_action = {
        "weak_coverage":   "lift_tilt",
        "neighbor_higher": "dec_a3",
        "overlap":         "press_tilt",
        "handover":        "inc_a3",
        "overshoot":       "dec_power",
        "high_speed":      "cov_interfreq",
        "low_rb":          "pdcch",
    }

    threshold_rules: List[Tuple[bool, str, str]] = [
        (num_neigh >= 3,                                  "overlap",         f"num_neighbors={num_neigh}>=3"),
        (handovers >= 3,                                  "handover",        f"handovers={handovers}>=3"),
        (handovers == 2,                                  "overshoot",       f"handovers=2"),
        (max_speed > 40,                                  "high_speed",      f"max_speed={max_speed:.1f}>40"),
        (avg_rb < 170,                                    "low_rb",          f"avg_rb={avg_rb:.1f}<170"),
        (max_tilt < 12,                                   "neighbor_higher", f"max_tilt={max_tilt:.1f}<12"),
        (total_tilt < 19,                                 "neighbor_higher", f"total_tilt={total_tilt:.1f}<19"),
        (min_tilt < 6,                                    "neighbor_higher", f"min_tilt={min_tilt:.1f}<6"),
        (min_tilt < 10 and min_rsrp > -89,               "neighbor_higher", f"min_tilt<10 rsrp>-89"),
        (min_rsrp < -90,                                  "weak_coverage",   f"min_rsrp={min_rsrp:.1f}<-90"),
        (max_tilt > 29,                                   "weak_coverage",   f"max_tilt={max_tilt:.1f}>29"),
        (total_tilt > 52,                                 "weak_coverage",   f"total_tilt={total_tilt:.1f}>52"),
        (min_tilt > 25,                                   "weak_coverage",   f"min_tilt={min_tilt:.1f}>25"),
        # Boundary cases
        (min_rsrp < -88.5 and total_tilt > 39 and max_tilt >= 22,
                                                          "weak_coverage",   "C1/C3 boundary lean-C1"),
        (min_tilt < 10 or total_tilt <= 35,              "neighbor_higher", "C1/C3 boundary lean-C3"),
    ]

    for condition, cause, desc in threshold_rules:
        if condition:
            action = cause_action.get(cause, "")
            if not action:
                continue
            # Try primary cells first, then any available option of this action type
            for cell in primary_cells:
                opt = _get_opt(by_cell, cell, action)
                if opt:
                    return (
                        [opt], "low", [cause],
                        f"LB({desc}) -> {action} on {cell} -> {opt}",
                    )
            opt = _first_opt(by_action, action)
            if opt:
                return ([opt], "low", [cause], f"LB({desc}) -> {action} -> {opt}")

    return ([], "none", [], "No rule matched — routing to LLM")


# ---------------------------------------------------------------------------
# LLM context injection
# ---------------------------------------------------------------------------

def build_layer0_system_prompt(diagnosis: DiagnosisResult) -> str:
    """Return a system prompt prefix injecting LAYER 0 pre-analysis for low/none confidence cases."""
    if diagnosis.confidence == "high":
        return ""

    f = diagnosis.features

    def fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = [
        "[LAYER 0 Pre-Analysis — use this context to guide your tool calls]",
        (f"Signal: min_rsrp={fmt(f.get('min_rsrp'))} dBm, "
         f"mean_sinr={fmt(f.get('mean_sinr'))} dB, "
         f"handovers={f.get('handovers')}, max_speed={fmt(f.get('max_speed'))} km/h, "
         f"avg_rb={fmt(f.get('avg_rb'))}"),
        (f"Tilt: max={fmt(f.get('max_tilt'))}°, min={fmt(f.get('min_tilt'))}°, "
         f"total={fmt(f.get('total_tilt'))}°"),
        (f"Signaling: rrc={f.get('has_rrc')}, a2={f.get('has_a2')}, a5={f.get('has_a5')}, "
         f"a3_diff={f.get('has_a3_diff_cells')}, a3_same={f.get('has_a3_same_cell')}, "
         f"ho_attempt={f.get('has_ho_attempt')}"),
        f"A3 offsets per cell: {f.get('a3_offsets')}",
        (f"Traffic: CCE_fail_max={fmt(f.get('max_cce_fail'))}, "
         f"serving_DL_PRB={fmt(f.get('serving_dl_prb'))}%"),
        f"Rule engine: {diagnosis.confidence} confidence — {diagnosis.rule_desc}",
    ]
    if diagnosis.excluded_options:
        lines.append(f"Excluded options: {', '.join(diagnosis.excluded_options)}")
    if diagnosis.candidate_options:
        lines.append(f"Focus on these candidates: {', '.join(diagnosis.candidate_options)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def diagnose(scenario: Any) -> DiagnosisResult:
    """Run LAYER 0 on a scenario dict or Pydantic Scenario object.

    Returns a DiagnosisResult. If confidence=="high", the caller can return
    the answer immediately without any LLM call.
    """
    # Normalize to plain dict
    if hasattr(scenario, "model_dump"):
        s = scenario.model_dump()
    elif hasattr(scenario, "dict"):
        s = scenario.dict()
    else:
        s = dict(scenario)

    task    = s.get("task", {})
    options: List[Dict] = task.get("options", [])
    tag: str = s.get("tag", "single-answer")
    data_raw = s.get("data", {})

    if hasattr(data_raw, "model_dump"):
        data: Dict = data_raw.model_dump()
    elif hasattr(data_raw, "dict"):
        data = data_raw.dict()
    else:
        data = data_raw if isinstance(data_raw, dict) else {}

    all_ids = [o["id"] for o in options]

    try:
        features      = extract_features(data)
        option_map    = map_options_to_actions(options)
        primary_cells = identify_primary_cells(options)
        answer_ids, confidence, cause_types, rule_desc = apply_rules(
            features, option_map, primary_cells, tag
        )
    except Exception as exc:
        logger.warning(f"[LAYER 0] Exception during diagnosis: {exc}", exc_info=True)
        features = {}
        answer_ids, confidence, cause_types, rule_desc = [], "none", [], f"Exception: {exc}"
        all_ids = [o["id"] for o in options]

    excluded   = [oid for oid in all_ids if oid not in answer_ids] if confidence == "high" else []
    candidates = answer_ids if confidence == "high" else [oid for oid in all_ids if oid not in excluded]

    return DiagnosisResult(
        confidence=confidence,
        answer_ids=answer_ids,
        cause_types=cause_types,
        rule_desc=rule_desc,
        features=features,
        excluded_options=excluded,
        candidate_options=candidates,
        tag=tag,
    )
