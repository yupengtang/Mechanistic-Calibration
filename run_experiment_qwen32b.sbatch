#!/bin/bash
#SBATCH --job-name=beat300_qwen32b
#SBATCH --account=gts-mlin91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/experiment_qwen32b_%j.out
#SBATCH --error=logs/experiment_qwen32b_%j.err

# BEAT-300 Experiment for Qwen-32B (requires 2 GPUs for model parallelism)
# This script is specialized for running Qwen/Qwen2.5-32B-Instruct which needs multi-GPU

echo "=========================================="
echo "BEAT-300 Experiment: Qwen-32B (Multi-GPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x A100 (model parallelism)"
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

# Check GPU availability
echo "Checking GPUs..."
python -c "
import torch
import transformers
print('[OK] torch:', torch.__version__)
print('[OK] transformers:', transformers.__version__)
print('[OK] CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[OK] GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'[OK] GPU {i}:', torch.cuda.get_device_name(i))
"

if [ $? -ne 0 ]; then
    echo "[ERROR] Environment check failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running BEAT-300 for Qwen-32B"
echo "=========================================="
echo ""

# Mode: core (T={0.0, 0.7}, R=3) or test (T={0.0}, R=1)
MODE="${1:-core}"

if [ "$MODE" = "test" ]; then
    echo "[MODE] Test mode (1 question, 1 replicate)"
    python run_experiment.py \
        --config config.json \
        --questions frozen_artifacts/beat300_questions_SAMPLE.jsonl \
        --output results/trials_qwen32b_test.jsonl \
        --models Qwen/Qwen2.5-32B-Instruct \
        --device cuda \
        --test-mode
else
    echo "[MODE] Core experiment (full BEAT-300, Qwen-32B only)"
    
    # Check if full questions file exists
    if [ ! -f "frozen_artifacts/beat300_questions.jsonl" ]; then
        echo "[ERROR] beat300_questions.jsonl not found"
        echo "[ERROR] You must generate it first using:"
        echo "        python -m src.beat300_builder --candidates data/candidate_pool.jsonl --output frozen_artifacts/beat300_questions.jsonl"
        exit 1
    fi
    
    python run_experiment.py \
        --config config.json \
        --questions frozen_artifacts/beat300_questions.jsonl \
        --output results/trials_qwen32b.jsonl \
        --models Qwen/Qwen2.5-32B-Instruct \
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
    echo "[SUCCESS] Qwen-32B experiment completed successfully"
    echo ""
    echo "Results saved to: results/trials_qwen32b*.jsonl"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results:"
    echo "     python analyze_results.py results/trials_qwen32b.jsonl"
    echo "  2. Generate figures:"
    echo "     python generate_figures.py"
else
    echo "[ERROR] Experiment failed"
    echo "Check logs/experiment_qwen32b_${SLURM_JOB_ID}.err for details"
fi

exit $exit_code
