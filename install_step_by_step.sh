#!/bin/bash
# Step-by-step installation to avoid memory issues

set -e

# Set HuggingFace cache to project directory
export HF_HOME=/storage/project/r-pkastner3-0/ytang454/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p $HF_HOME

echo "=========================================="
echo "BEAT-120 Step-by-Step Installation"
echo "=========================================="
echo ""
echo "[INFO] HuggingFace cache: $HF_HOME"
echo ""

# Step 1: Install torch
echo "Step 1/5: Installing PyTorch (this will take 5-10 minutes)..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
echo "[OK] PyTorch installed"
echo ""

# Step 2: Install torchvision (optional, but included for completeness)
echo "Step 2/5: Installing torchvision..."
pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu118
echo "[OK] torchvision installed"
echo ""

# Step 3: Install transformers
echo "Step 3/5: Installing transformers..."
pip install --no-cache-dir transformers
echo "[OK] transformers installed"
echo ""

# Step 4: Install accelerate
echo "Step 4/5: Installing accelerate..."
pip install --no-cache-dir accelerate
echo "[OK] accelerate installed"
echo ""

# Step 5: Install huggingface_hub
echo "Step 5/5: Installing huggingface_hub..."
pip install --no-cache-dir huggingface_hub
echo "[OK] huggingface_hub installed"
echo ""

# Verify
echo "=========================================="
echo "Verification"
echo "=========================================="
python -c "import torch; print('[OK] torch', torch.__version__)"
python -c "import torch; print('[OK] CUDA available' if torch.cuda.is_available() else '[WARNING] CUDA not available')"
python -c "import transformers; print('[OK] transformers', transformers.__version__)"
python -c "import accelerate; print('[OK] accelerate', accelerate.__version__)"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  python auto_download_models.py"
echo ""
