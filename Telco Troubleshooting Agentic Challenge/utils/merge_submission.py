#!/usr/bin/env python3
"""
Merge Track A and Track B predictions into the Zindi submission template.

Usage (from repo root or utils/):
    python utils/merge_submission.py
    python utils/merge_submission.py --track_a Track\ A/results/result.csv
    python utils/merge_submission.py --out submission/Phase_1/submission.csv

Outputs:
    submission/Phase_1/submission.csv   — ready to upload to Zindi

Sources (auto-detected if they exist, skipped with a warning if missing):
    Track A: Track A/results/result.csv          (columns: scenario_id, answers)
    Track B: Track B/results/result.csv          (columns: id, prediction)
             Track B/data/Phase_1/test.json      (id → scenario_id mapping)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ── Resolve repo root regardless of where the script is invoked from ──────────
UTILS_DIR  = Path(__file__).resolve().parent
REPO_ROOT  = UTILS_DIR.parent

TEMPLATE   = REPO_ROOT / "submission" / "Phase_1" / "result.csv"
DEFAULT_OUT = REPO_ROOT / "submission" / "Phase_1" / "submission.csv"

DEFAULT_TRACK_A_CSV  = REPO_ROOT / "Track A" / "results" / "result.csv"
DEFAULT_TRACK_B_CSV  = REPO_ROOT / "Track B" / "results" / "result.csv"
DEFAULT_TRACK_B_JSON = REPO_ROOT / "Track B" / "data" / "Phase_1" / "test.json"


def load_track_a(path: Path) -> dict:
    """Return {scenario_id: answer_string} from Track A result.csv."""
    df = pd.read_csv(path)
    return dict(zip(df["scenario_id"].astype(str), df["answers"].fillna("").astype(str)))


def load_track_b(csv_path: Path, json_path: Path) -> dict:
    """Return {scenario_id: answer_string} from Track B result.csv + test.json mapping."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    id_to_scenario = {item["task"]["id"]: item["scenario_id"] for item in raw}

    df = pd.read_csv(csv_path)
    df["scenario_id"] = df["id"].map(id_to_scenario)
    return dict(zip(df["scenario_id"].astype(str), df["prediction"].fillna("").astype(str)))


def merge(track_a_map: dict, track_b_map: dict, out_path: Path):
    template = pd.read_csv(TEMPLATE)
    id_col   = template.columns[0]          # "scenario_id" in result.csv

    if track_a_map:
        template["Track A"] = (
            template[id_col].map(track_a_map).fillna(template["Track A"].fillna(""))
        )
    if track_b_map:
        template["Track B"] = (
            template[id_col].map(track_b_map).fillna(template["Track B"].fillna(""))
        )

    # Zindi expects column name "ID"
    template.rename(columns={id_col: "ID"}, inplace=True)

    # Strip leading/trailing whitespace only — do NOT collapse newlines.
    # Track B fault-localization answers are legitimately multi-line
    # (one fault per line); pandas will quote them correctly in the CSV.
    def _clean(val):
        if not isinstance(val, str):
            return ""
        return val.strip().replace("\r\n", "\n").replace("\r", "\n")

    template["Track A"] = template["Track A"].fillna("").apply(_clean)
    template["Track B"] = template["Track B"].fillna("").apply(_clean)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use plain UTF-8 (no BOM) and Unix line endings to avoid parser errors on Zindi
    template.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return template


def main():
    parser = argparse.ArgumentParser(description="Merge Track A + B into Zindi submission CSV")
    parser.add_argument("--track_a", type=Path, default=DEFAULT_TRACK_A_CSV,
                        help="Track A result.csv (columns: scenario_id, answers)")
    parser.add_argument("--track_b", type=Path, default=DEFAULT_TRACK_B_CSV,
                        help="Track B result.csv (columns: id, prediction)")
    parser.add_argument("--track_b_json", type=Path, default=DEFAULT_TRACK_B_JSON,
                        help="Track B test.json for id→scenario_id mapping")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output submission.csv path")
    args = parser.parse_args()

    if not TEMPLATE.exists():
        sys.exit(f"Template not found: {TEMPLATE}")

    track_a_map: dict = {}
    if args.track_a.exists():
        track_a_map = load_track_a(args.track_a)
        print(f"[Track A] Loaded {len(track_a_map)} predictions from {args.track_a}")
    else:
        print(f"[Track A] Not found — skipping ({args.track_a})", file=sys.stderr)

    track_b_map: dict = {}
    if args.track_b.exists():
        if not args.track_b_json.exists():
            print(f"[Track B] JSON mapping not found — skipping ({args.track_b_json})", file=sys.stderr)
        else:
            track_b_map = load_track_b(args.track_b, args.track_b_json)
            print(f"[Track B] Loaded {len(track_b_map)} predictions from {args.track_b}")
    else:
        print(f"[Track B] Not found — skipping ({args.track_b})", file=sys.stderr)

    if not track_a_map and not track_b_map:
        sys.exit("No predictions found for either track. Nothing to do.")

    df = merge(track_a_map, track_b_map, args.out)

    filled_a = (df["Track A"].fillna("") != "").sum()
    filled_b = (df["Track B"].fillna("") != "").sum()
    print(f"\nSubmission written -> {args.out}")
    print(f"  Track A filled: {filled_a} / {len(df)}")
    print(f"  Track B filled: {filled_b} / {len(df)}")


if __name__ == "__main__":
    main()
