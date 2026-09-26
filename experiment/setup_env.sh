#!/bin/bash
# Optional environment setup for the original experiment pipeline
# Source this file: source setup_env.sh

# Set HF_HOME before sourcing this file to use a shared or project filesystem.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Set PYTHONUSERBASE before sourcing to use a non-default package location.
export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"

echo "[OK] HuggingFace cache set to: $HF_HOME"
echo "[OK] Models will be downloaded to: $HUGGINGFACE_HUB_CACHE"
echo "[OK] Python packages will install to: $PYTHONUSERBASE"

# Show current cache usage
if [ -d "$HF_HOME" ]; then
    CACHE_SIZE=$(du -sh $HF_HOME 2>/dev/null | cut -f1)
    echo "[INFO] Current cache size: $CACHE_SIZE"
fi
