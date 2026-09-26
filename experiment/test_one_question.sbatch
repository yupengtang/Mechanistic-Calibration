#!/bin/bash
#SBATCH --job-name=beat300_test_1q
#SBATCH --account=gts-mlin91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_one_question_%j.out
#SBATCH --error=logs/test_one_question_%j.err

# Quick test: 1 question to verify environment setup

echo "=========================================="
echo "BEAT-300 Environment Test: 1 Question"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set HF_HOME and PYTHONUSERBASE at submission time to use shared storage.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"

echo "[OK] Environment configured"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs results

# Run test
python test_one_question.py

exit_code=$?

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "[SUCCESS] Environment test passed"
    echo ""
    echo "Your setup is ready!"
else
    echo "[ERROR] Test failed"
    echo "Check logs/test_one_question_${SLURM_JOB_ID}.err for details"
fi

exit $exit_code
