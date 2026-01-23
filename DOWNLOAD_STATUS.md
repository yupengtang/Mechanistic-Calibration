# Model Download Status

## Current Process

The system is automatically:
1. Installing PyTorch with CUDA 11.8 support
2. Installing Transformers and related libraries  
3. Downloading all 4 core models sequentially

## Check Status

```bash
# Check installation progress
bash check_install.sh

# Once installed, check download progress
python auto_download_models.py
```

## Manual Steps Required

### For Llama Models (Required)

Llama models require HuggingFace authentication:

```bash
# 1. Login to HuggingFace
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens

# 2. Accept licenses (click "Agree" on these pages)
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
```

## Models Being Downloaded

### Core Set (Main Experiments)
- [  ] Qwen/Qwen2.5-7B-Instruct (~14 GB) - No auth needed
- [  ] meta-llama/Llama-3.1-8B-Instruct (~16 GB) - Auth required
- [  ] Qwen/Qwen2.5-32B-Instruct (~64 GB) - No auth needed  
- [  ] meta-llama/Llama-3.1-70B-Instruct (~140 GB) - Auth required

Total: ~234 GB

## Estimated Timeline

- Installation: 5-10 minutes
- Model downloads: 2-4 hours (depends on network speed)
- Total: 2.5-4.5 hours

## After Download Completes

```bash
# Test models
python test_model_loading.py

# Run minimal experiment
python run_experiment.py --test-mode

# Full experiment
bash run_full_pipeline.sh
```

## Troubleshooting

### If installation fails
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate huggingface_hub
```

### If Llama download fails
```bash
# Make sure you're logged in
huggingface-cli whoami

# If not logged in
huggingface-cli login
```

### Check disk space
```bash
df -h ~/.cache/huggingface/
```

Need at least 250 GB free for all models.
