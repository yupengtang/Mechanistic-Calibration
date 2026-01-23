#!/bin/bash
# Download models for BEAT-120 mechanistic probing
# Run this on a machine with GPU and sufficient disk space

set -e

echo "=========================================="
echo "BEAT-120 Model Download Script"
echo "=========================================="
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || {
    echo "Warning: torch not installed or CUDA not available"
    echo "Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
}

echo ""
echo "=========================================="
echo "Step 1: Install HuggingFace Hub"
echo "=========================================="
pip install -q huggingface-hub transformers accelerate

echo ""
echo "=========================================="
echo "Step 2: Login to HuggingFace (if needed)"
echo "=========================================="
echo "Some models require authentication."
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Do you need to login to HuggingFace? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli login
fi

echo ""
echo "=========================================="
echo "Step 3: Download Core Models"
echo "=========================================="
echo ""

# Core models from DESIGN.md §6.1
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
)

echo "Models to download:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Calculate approximate disk space needed
echo "Estimated disk space needed:"
echo "  - Llama-3.1-8B-Instruct:  ~16 GB"
echo "  - Llama-3.1-70B-Instruct: ~140 GB"
echo "  - Qwen2.5-7B-Instruct:    ~14 GB"
echo "  - Qwen2.5-32B-Instruct:   ~64 GB"
echo "  Total: ~234 GB"
echo ""

read -p "Continue with download? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo ""
echo "Downloading models (this will take a while)..."
echo ""

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Downloading: $model"
    echo "----------------------------------------"
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = '$model'
print(f'Loading {model_id}...')

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f'✓ Tokenizer downloaded')

# Download model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='auto',
    low_cpu_mem_usage=True
)
print(f'✓ Model downloaded')

# Get model size
param_count = sum(p.numel() for p in model.parameters())
print(f'✓ Parameters: {param_count / 1e9:.1f}B')

print(f'✓ {model_id} ready!')
print()
" || {
        echo "Failed to download $model"
        echo "This might be due to:"
        echo "  1. Insufficient disk space"
        echo "  2. Missing HuggingFace authentication (for Llama models)"
        echo "  3. GPU memory issues"
        echo ""
        read -p "Continue with remaining models? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    }
done

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Models are cached in: ~/.cache/huggingface/hub/"
echo ""
echo "Next steps:"
echo "  1. Test setup: python verify_setup.py"
echo "  2. Quick test: python run_experiment.py --test-mode"
echo "  3. Full run: bash run_full_pipeline.sh"
echo ""
