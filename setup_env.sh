#!/bin/bash
# Environment setup for BEAT-120 on PACE Phoenix
# Source this file: source setup_env.sh

# Set HuggingFace cache to project directory (not home, which has limited space)
export HF_HOME=/storage/project/r-pkastner3-0/ytang454/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

echo "[OK] HuggingFace cache set to: $HF_HOME"
echo "[OK] Models will be downloaded to: $HUGGINGFACE_HUB_CACHE"

# Show current cache usage
if [ -d "$HF_HOME" ]; then
    CACHE_SIZE=$(du -sh $HF_HOME 2>/dev/null | cut -f1)
    echo "[INFO] Current cache size: $CACHE_SIZE"
fi
