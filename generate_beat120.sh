#!/bin/bash
# Generate BEAT-120 Questions from Candidate Pool
# Implements cognitive boundary filtering (DESIGN.md Section 3.3)

set -e

source setup_env.sh

echo "=========================================="
echo "BEAT-120 Question Generation"
echo "=========================================="
echo ""

# Check prerequisites
if [ ! -f "data/candidate_pool.jsonl" ]; then
    echo "[ERROR] data/candidate_pool.jsonl not found"
    echo ""
    echo "Generate it first with:"
    echo "  python scripts/collect_candidates.py --output data/candidate_pool.jsonl"
    echo ""
    exit 1
fi

# Check if already exists
if [ -f "frozen_artifacts/beat120_questions.jsonl" ]; then
    echo "[WARNING] frozen_artifacts/beat120_questions.jsonl already exists"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Aborted"
        exit 0
    fi
fi

echo "[INFO] Starting boundary selection..."
echo ""
echo "Parameters:"
echo "  Screening models: Llama-3.1-70B-Instruct (local), gpt-4o-mini (API)"
echo "  Samples per candidate: 10"
echo "  Temperature: 0.7"
echo "  Target boundary: [0.4, 0.6]"
echo "  Target count: 120 (40 Medicine, 40 Science, 40 Law)"
echo ""

# Run selection
python -m src.beat120_builder \
    --candidates data/candidate_pool.jsonl \
    --output frozen_artifacts/beat120_questions.jsonl \
    --selection-log frozen_artifacts/selection_log.jsonl \
    --screening-models "meta-llama/Llama-3.1-70B-Instruct" "openai/gpt-4o-mini" \
    --num-samples 10 \
    --temperature 0.7 \
    --target-count 120 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[SUCCESS] BEAT-120 Generated"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  Questions: frozen_artifacts/beat120_questions.jsonl"
    echo "  Selection log: frozen_artifacts/selection_log.jsonl"
    echo ""
    echo "Next steps:"
    echo "  1. Run experiment: sbatch run_experiment.sbatch core"
    echo "  2. Or test mode: sbatch run_experiment.sbatch test"
else
    echo ""
    echo "[ERROR] BEAT-120 generation failed"
    exit 1
fi
