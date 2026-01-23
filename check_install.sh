#!/bin/bash
# Check installation and download progress

echo "Installation Status:"
echo "===================="
echo ""

# Check if pip is running
if ps -p 4018898 > /dev/null 2>&1; then
    echo "[RUNNING] PyTorch installation in progress..."
    echo "Checking progress:"
    tail -20 /storage/home/hcoda1/3/ytang454/.cursor/projects/storage-project-r-pkastner3-0-ytang454-Mechanistic-Calibration/terminals/19901.txt
else
    echo "[COMPLETE] Installation process finished"
    
    # Verify packages
    echo ""
    echo "Verifying packages:"
    python -c "import torch; print('[OK] torch', torch.__version__)" 2>&1 || echo "[MISSING] torch"
    python -c "import transformers; print('[OK] transformers', transformers.__version__)" 2>&1 || echo "[MISSING] transformers"
    python -c "import torch; print('[OK] CUDA available' if torch.cuda.is_available() else '[WARNING] CUDA not available')" 2>&1
fi

echo ""
