#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label general knowledge dataset (Phase 2) using Ollama.

Output:
- ID: question ID
- question: original question
- reasoning_trace: model's reasoning (gold think if available)
- answer: 1, 2, 3, or 4 (always one of the options)

Features:
- Structured system prompt for reasoning and mapping
- Handles numeric, logic, and general knowledge questions
- Fallback logic: \boxed{} extraction + numeric/text comparison
- Debug mode for first N questions
- tqdm progress bar
- Retry logic
- CLI input/output paths

Time:
- Total runtime depends on dataset size and model response time
For qwen2.5:7b-instruct, expect ~92 min 12.77 sec for 78 questions.
"""

import argparse
from pathlib import Path
import pandas as pd
import time
import re
import ollama
from tqdm import tqdm

# =========================
# Configuration
# =========================
DEFAULT_INPUT_FILE = "general_knowledge.csv"
DEFAULT_OUTPUT_FILE = "general_knowledge_labeled.csv"
DEBUG_SAMPLE_SIZE = 5
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
OLLAMA_MODEL = "qwen2.5:7b-instruct"

# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = """You are a precise reasoning assistant for general knowledge, logic, and math questions.
For every question, strictly follow these three steps:

1. <thinking>: Solve the problem step-by-step, explaining your reasoning clearly.
2. <mapping>: Compare your final conclusion to the provided options (1, 2, 3, or 4), and identify which option matches best.
3. <answer>: State only the option number, enclosed in \\boxed{}.

Example Output:
<thinking>The event probability is calculated as ...</thinking>
<mapping>The result corresponds to Option 2.</mapping>
\\boxed{2}

Important: Always select one of the provided options (1, 2, 3, or 4), even for non-numeric questions.
"""

# =========================
# Utility Functions
# =========================
def load_dataset(path: Path) -> pd.DataFrame:
    print(f"\nLoading dataset: {path}")
    df = pd.read_csv(path, engine="python")
    print("Columns:", list(df.columns))
    print(df.head())
    return df


def query_model(question: str, options_reminder: str, model: str) -> str:
    """
    Query Ollama model with retry logic. Returns raw response text.
    """
    user_prompt = f"{question}\n\nReminder: Your answer must be one of these options: {options_reminder}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"⏳ Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print("❌ Max retries reached. Skipping this question.")
                return ""


def parse_options(question: str) -> dict:
    """
    Extract options from question text.
    Returns dict: option_number -> option_text or numeric value
    """
    pattern = r"(\d):\s*(.+?)(?=\s+\d:|$)"
    matches = re.findall(pattern, question, flags=re.DOTALL)
    options = {}
    for num, val in matches:
        val_clean = val.strip()
        try:
            # Convert LaTeX fraction to float if numeric
            if r"\frac" in val_clean:
                frac_match = re.match(r"\\frac\{(\d+)\}\{(\d+)\}", val_clean)
                if frac_match:
                    val_clean = float(frac_match.group(1)) / float(frac_match.group(2))
                else:
                    val_clean = val_clean  # keep as string if not a simple fraction
            else:
                val_clean = float(val_clean)
        except Exception:
            pass  # keep as string if not numeric
        options[int(num)] = val_clean
    return options


def extract_answer(reasoning: str, question: str) -> str:
    """
    Extract answer 1-4 from reasoning trace.
    1️⃣ Try \boxed{N} first
    2️⃣ Fallback: numeric comparison for numbers, text matching for general knowledge
    """
    # Try \boxed{N} first
    match = re.search(r'\\boxed\{([1-4])\}', reasoning)
    if match:
        return match.group(1)

    # Fallback
    options = parse_options(question)
    reasoning_lower = reasoning.lower()

    # 1️⃣ Numeric matching
    numeric_options = {k: v for k, v in options.items() if isinstance(v, (int, float))}
    nums_in_reasoning = re.findall(r"[-+]?\d*\.?\d+", reasoning)
    for n in nums_in_reasoning[::-1]:  # start from last number
        try:
            val = float(n)
            if numeric_options:
                closest_option = min(numeric_options.items(), key=lambda x: abs(x[1] - val))
                return str(closest_option[0])
        except Exception:
            continue

    # 2️⃣ Text matching for non-numeric options
    for num, val in options.items():
        if isinstance(val, str) and val.lower() in reasoning_lower:
            return str(num)

    # 3️⃣ As last resort, pick Option 1
    return "1"


def label_dataset(df: pd.DataFrame, question_col: str, model: str, debug: bool = False) -> pd.DataFrame:
    reasoning_traces = []
    answers = []

    if debug:
        df = df.head(DEBUG_SAMPLE_SIZE)
        print(f"\nDEBUG MODE: Only labeling first {DEBUG_SAMPLE_SIZE} questions")

    start_total = time.time()
    for idx, question in enumerate(tqdm(df[question_col], desc="Labeling questions", unit="q")):
        start = time.time()
        options_reminder = ", ".join(str(i) for i in parse_options(question).keys())
        reasoning = query_model(question, options_reminder, model)
        answer = extract_answer(reasoning, question)
        reasoning_traces.append(reasoning)
        answers.append(answer)
        elapsed = time.time() - start

        if debug:
            print(f"\n--- DEBUG QUESTION {idx+1} ---")
            print(f"Question:\n{question}")
            print(f"Answer extracted: {answer}")
            print(f"Reasoning trace:\n{reasoning}")
            print(f"Labeled in {elapsed:.2f} sec")
            print("---------------------------")

    total_elapsed = time.time() - start_total
    minutes, seconds = divmod(total_elapsed, 60)
    print(f"\n✅ All questions labeled! Total time: {int(minutes)} min {seconds:.2f} sec")
    print('Next step: Enhance the augmented training dataset with gold reasoning traces.')

    df["reasoning_trace"] = reasoning_traces
    df["answer"] = answers
    return df


def save_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved labeled dataset to: {path}")


# =========================
# Main Pipeline
# =========================
def main(input_path: Path, output_path: Path, input_file: str, model: str, debug: bool = False):
    try:
        df = load_dataset(input_path / input_file)
        df_labeled = label_dataset(df, question_col="question", model=model, debug=debug)
        save_dataset(df_labeled, output_path / DEFAULT_OUTPUT_FILE)
    except Exception as e:
        print(f"\n❌ Error in labeling pipeline: {e}")


# =========================
# CLI Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label general knowledge dataset using Ollama with structured system prompt"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("./../data/processed/"),
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./../data/processed/"),
        help="Path to save labeled dataset",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Input CSV filename",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help="Ollama model to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (first 5 questions only, with detailed output)",
    )

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.input_file, args.model, debug=args.debug)