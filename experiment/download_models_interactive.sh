#!/bin/bash
# Interactive model download (for use in salloc session)

source setup_env.sh

echo "=========================================="
echo "Interactive Model Download"
echo "=========================================="
echo ""

# Check CUDA
python -c "
import torch
if torch.cuda.is_available():
    print('[OK] CUDA available')
    print('[OK] GPU:', torch.cuda.get_device_name(0))
else:
    print('[ERROR] CUDA not available!')
    print('[ERROR] You need to run this on a GPU node.')
    print('')
    print('Start a GPU session:')
    print('  salloc --gres=gpu:A100:1 --mem=100G --time=4:00:00')
    print('  Then run this script again.')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "=========================================="
echo "HuggingFace Login (for Llama models)"
echo "=========================================="
echo ""

if huggingface-cli whoami > /dev/null 2>&1; then
    echo "[OK] Already logged in to HuggingFace"
else
    echo "[INFO] Not logged in to HuggingFace"
    echo ""
    echo "Llama models require authentication."
    echo "  1. Get token: https://huggingface.co/settings/tokens"
    echo "  2. Accept license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    read -p "Login now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "[WARNING] Skipping login - Llama models will fail"
    fi
fi

echo ""
echo "=========================================="
echo "Starting Model Downloads"
echo "=========================================="
echo ""

python auto_download_models.py

echo ""
echo "[INFO] Download complete!"
echo ""
echo "Next steps:"
echo "  python test_model_loading.py"
echo "  python run_experiment.py --test-mode --device cuda"
