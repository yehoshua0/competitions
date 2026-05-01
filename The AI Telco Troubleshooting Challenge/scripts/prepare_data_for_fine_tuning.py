#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data_for_fine_tuning.py

Concatenate multiple datasets into a single SFT-ready CSV file.
Supports optional extra columns such as reasoning_trace.

Required columns:
- ID
- question
- answer

Optional columns:
- reasoning_trace
- any future metadata columns
"""

import argparse
from pathlib import Path
import pandas as pd

# -----------------------------
# Core logic
# -----------------------------
def prepare_sft_data(
    input_files: list[Path],
    output_path: Path,
    extra_cols: list[str] | None = None,
    shuffle: bool = True,
    seed: int = 3407,
):
    """
    Load, concatenate, shuffle, and save SFT training data.

    Args:
        input_files: List of CSV paths to concatenate
        output_path: Output CSV path
        extra_cols: Optional list of additional columns to keep
        shuffle: Whether to shuffle rows before saving
        seed: Random seed for reproducibility
    """
    required_cols = ["ID", "question", "answer"]
    extra_cols = extra_cols or []

    dfs = []

    print("Loading input datasets...")
    for path in input_files:
        print(f"  - {path}")
        df = pd.read_csv(path, on_bad_lines="skip")

        # Validate required columns
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            raise ValueError(f"{path} is missing required columns: {missing_required}")

        # Determine columns to keep
        keep_cols = required_cols.copy()

        for col in extra_cols:
            if col in df.columns:
                keep_cols.append(col)
            else:
                print(f"⚠️  Warning: '{col}' not found in {path}, filling with empty values")
                df[col] = ""

        dfs.append(df[keep_cols])

    # Concatenate
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Total rows before shuffle: {len(full_df)}")

    # Shuffle
    if shuffle:
        full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print("✓ Dataset shuffled")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    full_df.to_csv(output_path, index=False)

    print("\n✅ SFT dataset saved")
    print(f"Path: {output_path}")
    print(f"Shape: {full_df.shape}")
    print(f"Columns: {full_df.columns.tolist()}")
    print("Done. You can now use this file for supervised fine-tuning. Upload to Google Drive for next steps.")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare concatenated dataset for supervised fine-tuning (SFT)."
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "./../data/processed/augmented_train_with_trace.csv",
            "./../data/processed/general_knowledge_labeled.csv",
        ],
        help="Input CSV files (space-separated)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./../data/processed/sft_data.csv",
        help="Output SFT CSV file",
    )

    parser.add_argument(
        "--extra-cols",
        nargs="*",
        default=[],
        help="Optional extra columns to include (e.g. reasoning_trace)",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before saving",
    )

    args = parser.parse_args()

    prepare_sft_data(
        input_files=[Path(p) for p in args.inputs],
        output_path=Path(args.output),
        extra_cols=args.extra_cols,
        shuffle=not args.no_shuffle,
    )