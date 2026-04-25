#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Offline validation for Track A — computes per-bucket IoU from a result.csv.

Usage:
    python scripts/validate.py \
        --results results/result.csv \
        --data data/Phase_1/train.json \
        [--top_errors N]

Buckets reported:
  - Overall
  - By answer cardinality: single | multi
  - By signaling pattern: a3_diff | a3_same | a2_only | a2_a5 | rrc_only | none
  - By ground-truth Cx option (which options appear in the GT)
  - Top-N worst scenarios (for manual inspection)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from collections import defaultdict
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_score


# ---------------------------------------------------------------------------
# Signaling pattern detection (reads raw train JSON data, offline-only)
# ---------------------------------------------------------------------------

def _parse_pipe(raw: Optional[str]) -> pd.DataFrame:
    if not raw or not raw.strip():
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(raw), sep="|", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        try:
            return df.map(lambda x: x.strip() if isinstance(x, str) else x)
        except AttributeError:
            return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    except Exception:
        return pd.DataFrame()


def detect_signaling_pattern(scenario: dict) -> str:
    """Classify the signaling signature from the raw scenario dict into one of:
    a3_diff, a3_same, a2_only, a2_a5, rrc_only, none

    Signal format (from empirical data inspection):
      Column 'Event Name' — values like NREventA3, NREventA2, NREventA5,
      NRRRCReestablishAttempt, NREventA2MeasConfig (config, not triggered), etc.
      Column 'Event Content' — semicolon-separated key:value pairs,
      e.g. 'ServCellPCI:536;NCellPCI:556;...'
    """
    raw = (scenario.get("data") or {}).get("signaling_plane_data") or ""
    sig_df = _parse_pipe(raw)

    if sig_df.empty or "Event Name" not in sig_df.columns:
        return "none"

    names = sig_df["Event Name"].fillna("").str.strip()

    # Only consider *triggered* events (no MeasConfig suffix)
    a3_df   = sig_df[names == "NREventA3"]
    has_a2  = (names == "NREventA2").any()
    has_a5  = (names == "NREventA5").any()
    has_rrc = names.str.contains("RRCReestablish", na=False).any()

    if not a3_df.empty:
        # Determine same-cell vs diff-cell via Event Content
        content_col = "Event Content" if "Event Content" in sig_df.columns else None
        if content_col:
            same_count = 0
            for content in a3_df[content_col].fillna(""):
                m_serv  = re.search(r"ServCellPCI:(\d+)", content)
                m_ncell = re.search(r"NCellPCI:(\d+)", content)
                if m_serv and m_ncell and m_serv.group(1) == m_ncell.group(1):
                    same_count += 1
            # "same" if every A3 row is same-cell; "diff" if any row is diff-cell
            return "a3_same" if same_count == len(a3_df) else "a3_diff"
        return "a3_diff"

    if has_a2 and has_a5:
        return "a2_a5"
    if has_a2:
        return "a2_only"
    if has_rrc:
        return "rrc_only"
    return "none"


# ---------------------------------------------------------------------------
# Cx option bucketing
# ---------------------------------------------------------------------------

def gt_options(gt: str) -> List[str]:
    return sorted(s.strip().upper() for s in gt.split("|") if s.strip())


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def validate(results_path: str, data_path: str, top_errors: int = 20) -> None:
    # Load ground truth
    with open(data_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    gt_map: Dict[str, str] = {s["scenario_id"]: s["answer"] for s in scenarios}
    scenario_map: Dict[str, dict] = {s["scenario_id"]: s for s in scenarios}

    # Load predictions
    preds_df = pd.read_csv(results_path, dtype=str).fillna("")
    if "scenario_id" not in preds_df.columns or "answers" not in preds_df.columns:
        print(f"ERROR: result.csv must have 'scenario_id' and 'answers' columns. Got: {list(preds_df.columns)}")
        sys.exit(1)

    rows = []
    for _, row in preds_df.iterrows():
        sid = str(row["scenario_id"]).strip()
        pred = str(row["answers"]).strip()
        gt = gt_map.get(sid, "")
        if not gt:
            continue  # skip test-only scenarios
        score = compute_score(gt, pred)
        pattern = detect_signaling_pattern(scenario_map.get(sid, {}))
        cardinality = "multi" if "|" in gt else "single"
        rows.append({
            "scenario_id": sid,
            "gt": gt,
            "pred": pred,
            "score": score,
            "pattern": pattern,
            "cardinality": cardinality,
        })

    if not rows:
        print("No matching scenarios found between result.csv and train.json.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    n_total = len(df)
    mean_iou = df["score"].mean()

    # --- Overall ---
    _header("Overall")
    print(f"  Scenarios evaluated : {n_total}")
    print(f"  Mean IoU            : {mean_iou:.4f}")
    empty_pred = (df["pred"] == "").sum()
    print(f"  Empty predictions   : {empty_pred} ({100*empty_pred/n_total:.1f}%)")

    # --- By answer cardinality ---
    _header("By answer cardinality (ground truth)")
    _bucket_table(df, "cardinality")

    # --- By signaling pattern ---
    _header("By signaling pattern")
    _bucket_table(df, "pattern")

    # --- By Cx option presence in GT ---
    _header("By ground-truth Cx option (IoU when that Cx appears in GT)")
    option_scores: Dict[str, List[float]] = defaultdict(list)
    for _, row in df.iterrows():
        for opt in gt_options(row["gt"]):
            option_scores[opt].append(row["score"])
    opt_rows = [(opt, len(v), sum(v)/len(v)) for opt, v in sorted(option_scores.items())]
    _print_table(["Option", "Count", "Mean IoU"], opt_rows, fmts=["{}", "{}", "{:.4f}"])

    # --- Top errors ---
    if top_errors > 0:
        _header(f"Top {top_errors} worst scenarios")
        worst = df.nsmallest(top_errors, "score")[["scenario_id", "gt", "pred", "score", "pattern"]]
        for _, r in worst.iterrows():
            print(f"  {r['scenario_id']:20s}  gt={r['gt']:15s}  pred={r['pred']:15s}  iou={r['score']:.3f}  [{r['pattern']}]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def _bucket_table(df: pd.DataFrame, col: str) -> None:
    groups = df.groupby(col)
    rows = []
    for name, grp in sorted(groups):
        rows.append((name, len(grp), grp["score"].mean()))
    _print_table([col, "Count", "Mean IoU"], rows, fmts=["{}", "{}", "{:.4f}"])


def _print_table(headers: List[str], rows: list, fmts: List[str]) -> None:
    # Compute column widths
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    header_str = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_str}")
    print(f"  {sep}")
    for row in rows:
        cells = [fmts[i].format(row[i]).ljust(widths[i]) for i in range(len(headers))]
        print(f"  {'  '.join(cells)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline IoU validation for Track A")
    parser.add_argument("--results", default="results/result.csv",
                        help="Path to result.csv from benchmark run")
    parser.add_argument("--data",    default="data/Phase_1/train.json",
                        help="Path to train.json (with ground-truth answers)")
    parser.add_argument("--top_errors", type=int, default=20,
                        help="Print top-N worst scenarios (0 to skip)")
    args = parser.parse_args()

    validate(args.results, args.data, args.top_errors)
