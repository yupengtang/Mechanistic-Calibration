#!/bin/bash
# One-click pipeline for BEAT-120 experiments

set -e

# Set HuggingFace cache to project directory
export HF_HOME=/storage/project/r-pkastner3-0/ytang454/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p $HF_HOME

echo "======================================================"
echo "BEAT-300 Experimental Pipeline"
echo "======================================================"

# Validate environment
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Create with: echo 'OPENROUTER_API_KEY=your-key' > .env"
    exit 1
fi

# Validate frozen artifacts
echo ""
echo "Step 1: Validating frozen artifacts..."
echo "------------------------------------------------------"

artifacts=(
    "frozen_artifacts/beat120_questions.jsonl"
    "frozen_artifacts/models.json"
    "frozen_artifacts/analysis_plan.md"
    "prompts/registry.json"
    "config.json"
)

for artifact in "${artifacts[@]}"; do
    if [ ! -f "$artifact" ]; then
        echo "ERROR: Missing $artifact"
        exit 1
    fi
done

echo "All frozen artifacts present"

# Run experiment
echo ""
echo "Step 2: Running experiment (core harness)..."
echo "------------------------------------------------------"

python run_experiment.py \
    --config config.json \
    --questions frozen_artifacts/beat120_questions.jsonl \
    --output results/trials.jsonl \
    --device cuda

# Analyze results
echo ""
echo "Step 3: Analyzing results..."
echo "------------------------------------------------------"

python analyze_results.py \
    --trials results/trials.jsonl \
    --output-dir results/analysis

# Summary
echo ""
echo "======================================================"
echo "Pipeline Complete!"
echo "======================================================"
echo "Results:"
echo "  • Raw trials: results/trials.jsonl"
echo "  • Analysis:   results/analysis/*.csv"
echo ""
echo "Next steps:"
echo "  • Review results/analysis/summary_by_condition.csv"
echo "  • Check results/analysis/ssi_behavioral.csv for SSI"
echo "  • Generate figures (see EXPERIMENTS.md)"
echo "======================================================"

