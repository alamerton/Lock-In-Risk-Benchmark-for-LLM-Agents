#!/bin/bash
# Run Centralisation Evaluation with Sandbox in a loop

# Define the models, seeds, and temperatures to test
MODELS=(
    "openrouter/anthropic/claude-sonnet-4.5"
    "openrouter/google/gemini-2.5-pro"
    "openrouter/openai/gpt-4.1"
)
SEEDS=(42 12 69)
TEMPERATURES=(0 0.5 1)

# Set the judge model (constant for all runs)
export JUDGE_MODEL="openrouter/openai/gpt-4o"

# Loop through each combination
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for temp in "${TEMPERATURES[@]}"; do
            echo "=================================================="
            echo "Running evaluation with:"
            echo "  Model: $model"
            echo "  Seed: $seed"
            echo "  Temperature: $temp"
            echo "=================================================="

            # Export environment variables for the current run
            export AGENT_MODEL="$model"
            export EVAL_SEED="$seed"
            export AGENT_MODEL_TEMPERATURE="$temp"

            # Run the evaluation script
            python run_sandbox_centralisation.py

            echo "--------------------------------------------------"
            echo "Finished run for $model with seed $seed and temp $temp"
            echo "--------------------------------------------------"
            echo
        done
    done
done

echo "All evaluation runs completed."
