#!/bin/bash
# Wait for pip install to complete, then start model downloads

set -e

echo "=========================================="
echo "BEAT-120 Model Download Starter"
echo "=========================================="
echo ""

# Wait for pip install to complete
echo "Waiting for PyTorch installation to complete..."
echo "(This may take 5-10 minutes)"
echo ""

while ps -p 4001846 > /dev/null 2>&1; do
    sleep 10
    echo -n "."
done

echo ""
echo "[OK] PyTorch installation complete"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'[OK] torch {torch.__version__}'); print(f'[OK] CUDA: {torch.cuda.is_available()}')" || {
    echo "[ERROR] PyTorch installation failed"
    exit 1
}

python -c "import transformers; print(f'[OK] transformers {transformers.__version__}')" || {
    echo "[ERROR] transformers installation failed"
    exit 1
}

echo ""
echo "=========================================="
echo "Starting Model Downloads"
echo "=========================================="
echo ""

# Check HuggingFace authentication
echo "Checking HuggingFace authentication..."
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "[OK] Authenticated to HuggingFace"
else
    echo "[WARNING] Not authenticated to HuggingFace"
    echo ""
    echo "Llama models require authentication."
    echo "If you want to download Llama models:"
    echo "  1. Get token: https://huggingface.co/settings/tokens"
    echo "  2. Run: huggingface-cli login"
    echo "  3. Accept license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    read -p "Do you want to login now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    fi
fi

echo ""
echo "Starting automatic model download..."
echo ""

# Run automatic downloader
python auto_download_models.py

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
