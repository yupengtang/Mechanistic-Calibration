#!/bin/bash
#SBATCH --job-name=download_beat120_models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=logs/download_models_%j.out
#SBATCH --error=logs/download_models_%j.err

# BEAT-120 Model Download on PACE Phoenix GPU Node
# This script downloads all 4 core models to project storage

echo "=========================================="
echo "BEAT-120 Model Download (PACE Phoenix)"
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

mkdir -p $HF_HOME
mkdir -p logs

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
echo "Starting Model Downloads"
echo "=========================================="
echo ""
echo "Models will be downloaded to: $HF_HOME/hub"
echo "Total size: ~234 GB"
echo ""

# Run automatic downloader
python auto_download_models.py <<EOF
y
EOF

exit_code=$?

echo ""
echo "=========================================="
echo "Download Job Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "[SUCCESS] All models downloaded"
    echo ""
    echo "Verify with: python test_model_loading.py"
    echo "Run experiment: python run_experiment.py --test-mode --device cuda"
else
    echo "[ERROR] Some downloads failed"
    echo "Check logs/download_models_${SLURM_JOB_ID}.err for details"
fi

exit $exit_code
