#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# There is exist a fast version of this script on Kaggle

"""
generate_reasoning_traces.py

Generate reasoning traces for a dataset with questions and ground truth answers.
The output CSV will contain: ID, question, answer, reasoning_trace.
Supports:
- Debug mode (process first 5 rows)
- Auto-resume (skip already labeled rows in working output)
- Progress tracking
- Optional validation
"""

import pandas as pd
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import ollama

SAVE_EVERY = 5  # save progress every N samples

# -----------------------------
# System prompt for reasoning
# -----------------------------
SYSTEM_PROMPT = """You are an expert 5G telecommunications network analyst specializing in root cause analysis (RCA) of throughput degradation issues.

You will be given:
1. A QUESTION containing a real 5G drive-test scenario with:
   - User plane data (throughput, RSRP, SINR, handovers, etc.)
   - Engineering parameters (cell configs, downtilt, azimuth, power, etc.)
   - Eight candidate root causes (C1-C8)
2. The CORRECT ANSWER (one of C1-C8)

Your task is to generate a structured reasoning trace that explains WHY the correct answer is right, following this exact 3-task structure:

**Task 1: Data Analysis**
Systematically analyze the provided data:
- Throughput patterns: When does throughput drop? What are the values?
- Serving cell analysis: PCI, RSRP, SINR, coverage distance, configuration
- Neighbor cell analysis: Top neighbors, their signal strength, throughput comparison
- Mobility patterns: Handovers, GPS speed, location changes
- Resource allocation: Scheduled RBs (Resource Blocks)
- Cell configurations: Downtilt angles, azimuth, beam scenarios, PCI mod 30 relationships
- Coverage topology: Distances, overlapping coverage, co-location

**Task 2: Root Cause Analysis**
Use systematic elimination or contradiction-based reasoning:
- For EACH of the 8 candidate root causes (C1-C8), evaluate whether it fits the observed data
- Rule out implausible causes by showing contradictions with the data
- For causes that cannot be ruled out, explain what evidence supports them
- Compare the strength of evidence for remaining candidates

Evaluation guidelines:
- C1 (Excessive downtilt): Check total downtilt angle vs. beam scenario vertical beamwidth
- C2 (Over-shooting): Check if serving cell distance exceeds 1km
- C3 (Higher neighbor throughput): Compare serving vs. neighbor cell throughput
- C4 (Overlapping coverage): Check for non-colocated co-frequency neighbors
- C5 (Frequent handovers): Count handover events
- C6 (PCI mod 30 conflict): Calculate PCI mod 30 for serving and neighbors
- C7 (High speed): Check if GPS speed exceeds 40 km/h
- C8 (Low RBs): Check if average scheduled RBs are below 160

**Task 3: Conclusion**
- Identify the most plausible root cause based on Task 2 analysis
- Explain why this cause best explains the observed throughput degradation
- State the final answer clearly (e.g., "C3")

**Summary**
Provide a concise 2-3 sentence summary of the root cause and key evidence.

IMPORTANT FORMATTING:
- Use clear section headers: "Task 1: Data Analysis", "Task 2: Root Cause Analysis", "Task 3: Conclusion", "Summary"
- Be specific with numbers and measurements from the data
- Show your reasoning step-by-step
- The final answer MUST match the CORRECT ANSWER provided
- Focus on explaining WHY the correct answer is right, not on discovering it

Your reasoning should be thorough, technical, and grounded in the actual measurements and configurations provided in the question.
"""

# -----------------------------
# Query model function
# -----------------------------
def query_model(question: str, answer: str, model: str = "qwen2.5:7b-instruct",
                retries: int = 3, wait_sec: int = 5) -> str:
    """
    Query Ollama model with structured prompt, return reasoning trace.
    """
    prompt = f"QUESTION: {question}\nCORRECT ANSWER: {answer}"
    
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "num_gpu": 999,  # use all available GPUs
                    "temperature": 0.2, # low temperature for focused reasoning
                    "num_predict": 1600, # allow longer responses
                    "num_ctx": 8192, # larger context window
                    "num_thread": 12, # increase threads for speed
                    "num_batch": 1024, # larger batch size
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait_sec)
    print(f"❌ Failed to get response for question: {question[:50]}...")
    return ""  # fallback empty reasoning trace

# -----------------------------
# Validate reasoning trace (optional)
# -----------------------------
def validate_reasoning_trace(trace: str, answer: str) -> bool:
    """
    Basic validation: check that trace contains Task1, Task2, Task3, Summary, and mentions answer
    """
    return all([
        "Task 1" in trace or "Data Analysis" in trace,
        "Task 2" in trace or "Root Cause Analysis" in trace,
        "Task 3" in trace or "Conclusion" in trace,
        "Summary" in trace,
        answer in trace,
        len(trace) > 200
    ])

# -----------------------------
# Main processing function
# -----------------------------
def generate_reasoning(input_path: Path, debug: bool = False, model: str = "qwen2.5:7b-instruct",
                       validate: bool = True):
    """
    Auto-resume processing on working output file, generate reasoning traces.
    """
    # Output working file
    output_path = input_path.parent / (input_path.stem + "_with_trace.csv")
    
    # Load working output if exists, else copy input
    if output_path.exists():
        print(f"Resuming from existing output: {output_path}")
        df = pd.read_csv(output_path, on_bad_lines='skip')
    else:
        df = pd.read_csv(input_path, on_bad_lines='skip')
        df = df.copy()
        if 'reasoning_trace' not in df.columns:
            df['reasoning_trace'] = ""
    
    # Auto-resume: only process rows with empty reasoning_trace
    pending_mask = df['reasoning_trace'].isna() | (df['reasoning_trace'].str.strip() == "")
    pending_df = df[pending_mask].copy()
    
    if debug:
        pending_df = pending_df.head(5)
        print("⚠️ DEBUG MODE: Only processing first 5 questions")
    
    print(f"Questions to process: {len(pending_df)}")
    
    valid_count, invalid_count = 0, 0
    
    processed_since_save = 0

    for idx, row in tqdm(
        pending_df.iterrows(),
        total=len(pending_df),
        desc="Generating reasoning"
    ):
        reasoning = query_model(row['question'], row['answer'], model=model)
        df.at[idx, 'reasoning_trace'] = f"<thinking>{reasoning}</thinking>"

        processed_since_save += 1

        if validate and reasoning:
            is_valid = validate_reasoning_trace(reasoning, row['answer'])
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                tqdm.write(f"⚠️ Row {idx} may be invalid. Consider checking manually.")

        # ─── Periodic checkpoint save ─────────────────────────
        if processed_since_save >= SAVE_EVERY:
            df.to_csv(output_path, index=False)
            tqdm.write(f"💾 Progress saved ({SAVE_EVERY} samples)")
            processed_since_save = 0
    
    # Save updated CSV
    df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Reasoning traces saved to: {output_path}")
    print(f"Total processed: {len(pending_df)}")
    if validate:
        print(f"Valid traces: {valid_count}")
        print(f"Invalid traces: {invalid_count}")
    print(f"{'='*60}")
    print("Done. Next: merge reasoning traces from augmented_train and general_knowledge for training.")
    
# -----------------------------
# CLI interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning traces for QA dataset.")
    parser.add_argument("--input", type=str, default="./../data/processed/augmented_train.csv",
                        help="Path to input CSV with question+answer")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (first 5 rows only)")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct", help="Ollama model name")
    parser.add_argument("--no-validate", action="store_true", help="Disable reasoning trace validation")
    args = parser.parse_args()
    
    generate_reasoning(
        Path(args.input),
        debug=args.debug,
        model=args.model,
        validate=not args.no_validate
    )