#!/bin/bash
#SBATCH --job-name=beat120_experiment
#SBATCH --account=gts-mlin91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err

# BEAT-120 Full Experiment on PACE Phoenix
# Runs core experimental protocol with local mechanistic probing

echo "=========================================="
echo "BEAT-120 Experiment (PACE Phoenix)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set environment (models + packages in project directory)
export HF_HOME=/storage/project/r-pkastner3-0/ytang454/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTHONUSERBASE=/storage/project/r-pkastner3-0/ytang454/python_packages
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

echo "[OK] HuggingFace cache: $HF_HOME"
echo "[OK] Python packages: $PYTHONUSERBASE"
echo ""

# Check environment
echo "Checking environment..."
python -c "
import torch
import transformers
print('[OK] torch:', torch.__version__)
print('[OK] transformers:', transformers.__version__)
print('[OK] CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[OK] GPU:', torch.cuda.get_device_name(0))
"

if [ $? -ne 0 ]; then
    echo "[ERROR] Environment check failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running BEAT-120 Core Experiment"
echo "=========================================="
echo ""

# Mode: core (T={0.0, 0.7}, R=3) or test (T={0.0}, R=1)
MODE="${1:-core}"

if [ "$MODE" = "test" ]; then
    echo "[MODE] Test mode (1 question, 1 replicate)"
    python run_experiment.py \
        --config config.json \
        --questions frozen_artifacts/beat120_questions_SAMPLE.jsonl \
        --output results/trials_test.jsonl \
        --device cuda \
        --test-mode
else
    echo "[MODE] Core experiment (full BEAT-120)"
    
    # Check if full questions file exists
    if [ ! -f "frozen_artifacts/beat120_questions.jsonl" ]; then
        echo "[ERROR] beat120_questions.jsonl not found"
        echo "[ERROR] You must generate it first using:"
        echo "        python -m src.beat120_builder --candidates data/candidate_pool.jsonl --output frozen_artifacts/beat120_questions.jsonl"
        exit 1
    fi
    
    python run_experiment.py \
        --config config.json \
        --questions frozen_artifacts/beat120_questions.jsonl \
        --output results/trials.jsonl \
        --device cuda
fi

exit_code=$?

echo ""
echo "=========================================="
echo "Experiment Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "[SUCCESS] Experiment completed successfully"
    echo ""
    echo "Results saved to: results/"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results:"
    echo "     bash analyze_all.sh"
    echo "  2. Generate figures:"
    echo "     python generate_figures.py"
else
    echo "[ERROR] Experiment failed"
    echo "Check logs/experiment_${SLURM_JOB_ID}.err for details"
fi

exit $exit_code
