#!/bin/bash
# Environment setup for BEAT-120 on PACE Phoenix
# Source this file: source setup_env.sh

# Set HuggingFace cache to project directory (not home, which has limited space)
export HF_HOME=/storage/project/r-pkastner3-0/ytang454/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Set pip install location to project directory (home has limited quota)
export PYTHONUSERBASE=/storage/project/r-pkastner3-0/ytang454/python_packages
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

echo "[OK] HuggingFace cache set to: $HF_HOME"
echo "[OK] Models will be downloaded to: $HUGGINGFACE_HUB_CACHE"
echo "[OK] Python packages will install to: $PYTHONUSERBASE"

# Show current cache usage
if [ -d "$HF_HOME" ]; then
    CACHE_SIZE=$(du -sh $HF_HOME 2>/dev/null | cut -f1)
    echo "[INFO] Current cache size: $CACHE_SIZE"
fi
