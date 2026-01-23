#!/bin/bash
# Login to HuggingFace and download all models

source setup_env.sh

echo "=========================================="
echo "Step 1: HuggingFace Authentication"
echo "=========================================="
echo ""
echo "Llama models require authentication."
echo ""
echo "Please:"
echo "  1. Get your token from: https://huggingface.co/settings/tokens"
echo "  2. Accept Llama license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "Now logging in..."
echo ""

huggingface-cli login

echo ""
echo "=========================================="
echo "Step 2: Download Models"
echo "=========================================="
echo ""
echo "Starting automatic download of 4 core models..."
echo ""

python auto_download_models.py

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
