#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split Phase 2 test dataset into:
1) Network troubleshooting questions
2) General knowledge questions
"""

import argparse
from pathlib import Path
import pandas as pd


# =========================
# Configuration
# =========================
DEFAULT_INPUT_FILE = "phase_2_test.csv"
NETWORK_OUTPUT_FILE = "network_troubleshooting.csv"
GENERAL_OUTPUT_FILE = "general_knowledge.csv"

QUESTION_COL = "question"

# More comprehensive regex for 5G / Network troubleshooting
NETWORK_KEYWORDS_REGEX = (
    r"5G|throughput|drive-test|drive test|NR|handover|PCI|RSRP|RSRQ|SINR|"
    r"root cause|gNodeB|downtilt|beam"
)


# =========================
# Utility Functions
# =========================
def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV using python engine for better robustness with large text fields.
    """
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path, engine="python")
    print("Columns:", list(df.columns))
    print(df.head())
    return df


def split_by_domain(
    df: pd.DataFrame, question_col: str, regex: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into network troubleshooting and general knowledge
    based on regex matching.
    """
    mask = df[question_col].str.contains(regex, case=False, na=False)

    network_df = df[mask]
    general_df = df[~mask]

    print("\nSplit summary:")
    print(f"Network troubleshooting entries: {len(network_df)}")
    print(f"General knowledge entries: {len(general_df)}")

    return network_df, general_df


def save_datasets(
    network_df: pd.DataFrame,
    general_df: pd.DataFrame,
    output_path: Path,
):
    """
    Save split datasets to disk.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    network_path = output_path / NETWORK_OUTPUT_FILE
    general_path = output_path / GENERAL_OUTPUT_FILE

    network_df.to_csv(network_path, index=False)
    general_df.to_csv(general_path, index=False)

    print("\nSaved files:")
    print(f"- {network_path}")
    print(f"- {general_path}")


# =========================
# Main Pipeline
# =========================
def main(input_path: Path, output_path: Path, input_file: str):
    try:
        df = load_dataset(input_path / input_file)

        network_df, general_df = split_by_domain(
            df=df,
            question_col=QUESTION_COL,
            regex=NETWORK_KEYWORDS_REGEX,
        )

        save_datasets(
            network_df=network_df,
            general_df=general_df,
            output_path=output_path,
        )

        print("\n✅ Done. Phase 2 dataset successfully split.")
        print("Next step: Label the general knowledge questions.")

    except Exception as e:
        print(f"\n❌ Error while processing Phase 2 dataset:\n{e}")


# =========================
# CLI Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split Phase 2 test dataset into network troubleshooting and general knowledge"
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("./../data/raw/"),
        help="Path to input data directory",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./../data/processed/"),
        help="Path to output data directory",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Phase 2 test CSV filename",
    )

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.input_file)