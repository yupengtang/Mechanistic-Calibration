#!/bin/bash
# Main Experiment Runner for BEAT-120
# Runs the core experimental harness as specified in DESIGN.md

set -e

# Set HuggingFace cache to project directory
export HF_HOME=/storage/project/r-pkastner3-0/ytang454/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p $HF_HOME

echo "======================================================"
echo "BEAT-120 Core Experiment"
echo "======================================================"

# Validate environment
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Create with: echo 'OPENROUTER_API_KEY=your-key' > .env"
    exit 1
fi

# Validate frozen artifacts
echo ""
echo "Validating frozen artifacts..."
echo "------------------------------------------------------"

required_files=(
    "frozen_artifacts/beat120_questions.jsonl"
    "frozen_artifacts/models.json"
    "frozen_artifacts/analysis_plan.md"
    "prompts/registry.json"
    "config.json"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing required file: $file"
        exit 1
    fi
done

echo "All required files present"

# Parse command-line arguments
DEVICE="${1:-cuda}"
MODE="${2:-core}"  # core or extended

echo ""
echo "Configuration:"
echo "------------------------------------------------------"
echo "  Device: $DEVICE"
echo "  Mode: $MODE"
echo "------------------------------------------------------"

# Run experiment
echo ""
echo "Running experiment..."
echo "------------------------------------------------------"

if [ "$MODE" = "extended" ]; then
    # Extended harness (more temperatures, more replicates)
    echo "Running EXTENDED harness..."
    python run_experiment.py \
        --config config.json \
        --questions frozen_artifacts/beat120_questions.jsonl \
        --output results/trials_extended.jsonl \
        --device "$DEVICE"
else
    # Core harness (T={0.0, 0.7}, R=3)
    echo "Running CORE harness..."
    python run_experiment.py \
        --config config.json \
        --questions frozen_artifacts/beat120_questions.jsonl \
        --output results/trials.jsonl \
        --device "$DEVICE"
fi

echo ""
echo "======================================================"
echo "Experiment Complete!"
echo "======================================================"
echo "Results saved to: results/"
echo ""
echo "Next steps:"
echo "  1. Analyze results: bash analyze_all.sh"
echo "  2. Generate figures: python generate_figures.py"
echo "======================================================"

