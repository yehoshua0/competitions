#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Format result.csv (Track A predictions) into the official submission format.

Usage:
    python scripts/format_submission.py \
        --predictions results/result.csv \
        --sample     data/SampleSubmission.csv \
        --output     results/submission.csv

Output columns: ID, Track A, Track B
  - Track A rows: filled with our sorted predictions
  - Track B rows: placeholder values copied verbatim from SampleSubmission.csv
  - Any Track A scenario not in predictions gets an empty cell (safe default)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sort_cx(answer: str) -> str:
    """Sort Cx codes numerically ascending and deduplicate. 'C3|C7|C3' → 'C3|C7'."""
    if not answer or not answer.strip():
        return ""
    parts = {s.strip().upper() for s in answer.split("|") if re.match(r"C\d+", s.strip(), re.IGNORECASE)}
    sorted_parts = sorted(parts, key=lambda x: int(re.search(r"\d+", x).group()))
    return "|".join(sorted_parts)


def format_submission(predictions_path: str, sample_path: str, output_path: str) -> None:
    sample = pd.read_csv(sample_path, dtype=str).fillna("")
    preds  = pd.read_csv(predictions_path, dtype=str).fillna("")

    if "ID" not in sample.columns or "Track A" not in sample.columns or "Track B" not in sample.columns:
        print(f"ERROR: SampleSubmission.csv must have columns ID, Track A, Track B. Got: {list(sample.columns)}")
        sys.exit(1)

    if "scenario_id" not in preds.columns or "answers" not in preds.columns:
        print(f"ERROR: result.csv must have columns scenario_id, answers. Got: {list(preds.columns)}")
        sys.exit(1)

    pred_map = {
        str(row["scenario_id"]).strip(): sort_cx(str(row["answers"]).strip())
        for _, row in preds.iterrows()
    }

    out_rows = []
    track_a_count = 0
    track_b_count = 0
    missing_preds = 0

    for _, row in sample.iterrows():
        sid        = str(row["ID"]).strip()
        sample_a   = str(row["Track A"]).strip()
        sample_b   = str(row["Track B"]).strip()

        if sample_b:
            # Track B row — preserve placeholder exactly
            out_rows.append({"ID": sid, "Track A": "", "Track B": sample_b})
            track_b_count += 1
        else:
            # Track A row — fill with our prediction
            our_answer = pred_map.get(sid, "")
            if not our_answer:
                missing_preds += 1
            out_rows.append({"ID": sid, "Track A": our_answer, "Track B": ""})
            track_a_count += 1

    out_df = pd.DataFrame(out_rows, columns=["ID", "Track A", "Track B"])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Written: {output_path}")
    print(f"  Track A rows  : {track_a_count}")
    print(f"  Track B rows  : {track_b_count}")
    print(f"  Total rows    : {len(out_rows)}")
    if missing_preds:
        print(f"  WARNING: {missing_preds} Track A scenarios had no prediction (empty answer submitted)")
    else:
        print(f"  All Track A scenarios have predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format result.csv into official submission format")
    parser.add_argument("--predictions", default="results/result.csv",
                        help="Path to result.csv from benchmark run (scenario_id, answers)")
    parser.add_argument("--sample",      default="data/SampleSubmission.csv",
                        help="Path to official SampleSubmission.csv")
    parser.add_argument("--output",      default="results/submission.csv",
                        help="Output path for the formatted submission")
    args = parser.parse_args()

    format_submission(args.predictions, args.sample, args.output)
