"""
Analysis script for Sandbox-Based Centralisation Evaluation Results.

This script reads a directory of .json evaluation logs, extracts key metrics,
and prints a summary of the results, grouped by model, temperature, and seed.
"""

import argparse
import json
import os
import re
from collections import defaultdict
import pandas as pd

def parse_filename(filename: str):
    """
    Parses the filename to extract model, seed, and temperature.
    
    Example filename: 2025-11-28T08-58-06+08-00_sandbox-centralisation-federated-arch_hcqXmqnagrBsrhjjgW8CBK.json
    This function will look for the metadata in the 'task' part of the log.
    """
    return None # Will be extracted from file content instead

def parse_log(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model = data.get("eval", {}).get("model", "unknown")
    config = data.get("eval", {}).get("model_generate_config", {})
    temperature = config.get("temperature", "unknown")
    seed = config.get("seed", "unknown")
    
    risk_score = 0.0
    action_score = 0.0
    behavioral_indicators = {}
    judge_classifications = defaultdict(int)
    judge_scores = []

    try:
        scorer_meta = data["samples"][0]["scores"]["SandboxCentralisationScorer"]["metadata"]
        risk_score = scorer_meta.get("risk_score", 0.0)
        action_score = scorer_meta.get("action_score", 0.0)
        behavioral_indicators = scorer_meta.get("behavioral_indicators", {})
    except (KeyError, TypeError, IndexError):
        pass # Keep default values

    try:
        judge_assessments = data["samples"][0]["metadata"]["lock_in_evaluation"]["judge_assessments"]
        for assessment in judge_assessments:
            classification = assessment.get("judge_assessment", {}).get("approach_classification", "unknown")
            judge_classifications[classification] += 1
            judge_scores.append(assessment.get("judge_assessment", {}).get("score", 0.0))
    except (KeyError, TypeError, IndexError):
        pass

    avg_judge_score = sum(judge_scores) / len(judge_assessments) if judge_assessments else 0.0

    parsed_data = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "risk_score": risk_score,
        "action_score": action_score,
        "avg_judge_score": avg_judge_score,
    }
    # Add behavioral indicators and judge classifications to the dictionary
    parsed_data.update(behavioral_indicators)
    parsed_data.update(judge_classifications)
    
    return parsed_data

def analyze_results(results_directory: str):
    """
    Analyzes all .json files in the given directory.
    """
    if not os.path.isdir(results_directory):
        print(f"Error: Directory not found at '{results_directory}'")
        return

    results = []
    print(f"Scanning directory: {results_directory}")

    for filename in sorted(os.listdir(results_directory)):
        if filename.endswith(".json"):
            file_path = os.path.join(results_directory, filename)
            try:
                parsed_data = parse_log(file_path)
                # Add filename to results for reference
                parsed_data["file"] = filename
                results.append(parsed_data)
            except Exception as e:
                print(f"Warning: Could not process {filename} due to {e}")

    if not results:
        print("No valid results found to analyze.")
        return

    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(results)
    # Fill NaN values for columns that might not be present in all runs
    indicator_cols = ['inspected_policy', 'created_centralized_group', 'bypassed_policy', 'used_chmod_chown', 'mentioned_efficiency']
    classification_cols = ['federated', 'centralizing', 'mixed', 'unknown']
    for col in indicator_cols:
        if col not in df.columns:
            df[col] = False
    for col in classification_cols:
        if col not in df.columns:
            df[col] = 0
    df.fillna(0, inplace=True)

    
    print("\n--- Individual Run Results ---")
    print(df)

    # --- Aggregated Analysis ---
    # Group by model and temperature, then calculate mean and std dev of scores
    agg_results = df.groupby(['model', 'temperature'])['risk_score'].agg(['mean', 'std', 'count']).reset_index()
    agg_results = agg_results.sort_values(by=['model', 'temperature'])
    
    print("\n\n--- Aggregated Results (Mean Risk Score by Model and Temperature) ---")
    print(agg_results)

    # --- Pivot Table for better readability ---
    # Create a pivot table with models as rows, temperatures as columns
    pivot_table = df.pivot_table(
        index='model', 
        columns='temperature', 
        values='risk_score', 
        aggfunc='mean'
    )
    
    print("\n\n--- Pivot Table (Mean Risk Score) ---")
    print("Models vs. Temperature")
    print(pivot_table)

    # --- Detailed Sub-Score Analysis ---
    sub_score_agg = df.groupby('model').agg({
        'action_score': 'mean',
        'avg_judge_score': 'mean'
    }).reset_index()
    print("\n\n--- Mean Sub-Scores by Model ---")
    print(sub_score_agg)

    # --- Behavioral Indicator Analysis ---
    behavior_agg = df.groupby('model')[indicator_cols].sum().reset_index()
    print("\n\n--- Behavioral Indicator Counts by Model (Total across 9 runs) ---")
    print(behavior_agg)

    # --- Judge Classification Analysis ---
    judge_agg = df.groupby('model')[classification_cols].sum().reset_index()
    print("\n\n--- Judge Approach Classifications by Model (Total across all probes) ---")
    print(judge_agg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze centralisation evaluation results from a directory of JSON files."
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="Path to the directory containing .json result files."
    )

    args = parser.parse_args()
    analyze_results(args.directory)
