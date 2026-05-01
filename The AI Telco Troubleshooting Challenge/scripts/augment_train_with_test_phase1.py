#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment the original training dataset with Phase 1 test data and ground-truth labels.
Works anywhere with:
python augment.py --input-path data/raw --output-path data/processed
"""

from pathlib import Path
import argparse
import pandas as pd
from datasets import Dataset


# =========================
# Configuration
# =========================
TRAIN_FILE = "train.csv"
TEST_FILE = "phase_1_test.csv"
TRUTH_FILE = "phase_1_test_truth.csv"
OUTPUT_FILE = "augmented_train.csv"

ID_COL = "ID"
QUESTION_COL = "question"
ANSWER_COL = "answer"
TRUTH_MODEL_COL = "Qwen2.5-1.5B-Instruct"


# =========================
# Utility Functions
# =========================
def load_csv(path: Path) -> pd.DataFrame:
    print(f"\nLoading: {path}")
    df = pd.read_csv(path)
    print("Columns:", list(df.columns))
    print(df.head())
    return df


def process_truth_df(truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract base ID (remove _1/_2/etc.) and keep one label per ID.
    """
    df = truth_df.copy()
    df[ID_COL] = df[ID_COL].str.rsplit("_", n=1).str[0]

    df = (
        df[[ID_COL, TRUTH_MODEL_COL]]
        .drop_duplicates()
        .rename(columns={TRUTH_MODEL_COL: ANSWER_COL})
    )

    print("\nProcessed ground truth (unique IDs):")
    print(df.head())
    return df


def merge_test_with_truth(
    test_df: pd.DataFrame, truth_df: pd.DataFrame
) -> pd.DataFrame:
    df = pd.merge(test_df, truth_df, on=ID_COL, how="inner")
    print("\nMerged test + truth:")
    print(df.head())
    return df


def build_augmented_dataset(
    train_df: pd.DataFrame, augment_df: pd.DataFrame
) -> pd.DataFrame:
    required_cols = [ID_COL, QUESTION_COL, ANSWER_COL]

    full_df = pd.concat(
        [
            train_df[required_cols],
            augment_df[required_cols],
        ],
        ignore_index=True,
    )

    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\nFinal augmented dataset stats:")
    print("Shape:", full_df.shape)
    print(full_df.head())
    print(full_df.tail())

    return full_df


# =========================
# Main Pipeline
# =========================
def main(input_path: Path, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(input_path / TRAIN_FILE)
    test_df = load_csv(input_path / TEST_FILE)
    truth_df = load_csv(input_path / TRUTH_FILE)

    truth_processed = process_truth_df(truth_df)
    augmented_test_df = merge_test_with_truth(test_df, truth_processed)
    full_df = build_augmented_dataset(train_df, augmented_test_df)

    # Save CSV
    output_csv = output_path / OUTPUT_FILE
    full_df.to_csv(output_csv, index=False)
    print(f"\nSaved augmented CSV to: {output_csv}")

    # Convert to Hugging Face Dataset (optional but handy)
    hf_dataset = Dataset.from_pandas(full_df)
    print("\nHugging Face Dataset:")
    print(hf_dataset)

    print("\nDone ✅ Ready for upload to Google Drive.")
    print("Next step: Split Phase 2 test data into 5G troubleshooting vs general knowledge.")


# =========================
# CLI Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment training data with Phase 1 test data and ground truth"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("./../data/raw/"),
        help="Path to input CSV files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./../data/processed/"),
        help="Path to save augmented dataset",
    )

    args = parser.parse_args()
    main(args.input_path, args.output_path)