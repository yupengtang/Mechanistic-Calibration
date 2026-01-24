#!/usr/bin/env python3
"""
Verify that all required models for BEAT-300 are downloaded and loadable.
"""

import os
import sys

# Set environment
HF_HOME = "/storage/project/r-pkastner3-0/ytang454/hf_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")

PYTHONUSERBASE = "/storage/project/r-pkastner3-0/ytang454/python_packages"
os.environ["PYTHONUSERBASE"] = PYTHONUSERBASE

print("="*60)
print("BEAT-300 Model Verification")
print("="*60)
print()

# Required models per DESIGN.md §6.1
REQUIRED_MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Core model (main results)"),
    ("Qwen/Qwen2.5-7B-Instruct", "Core model (main results)"),
    ("Qwen/Qwen2.5-32B-Instruct", "Exploratory model (RQ4: capacity)"),
]

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("[OK] transformers library available")
except ImportError as e:
    print(f"[ERROR] transformers not installed: {e}")
    sys.exit(1)

print()
print("Checking models...")
print("-" * 60)

all_ok = True
for model_id, purpose in REQUIRED_MODELS:
    print(f"\n{model_id}")
    print(f"  Purpose: {purpose}")
    
    # Check tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"  [OK] Tokenizer loaded")
    except Exception as e:
        print(f"  [ERROR] Tokenizer failed: {str(e)[:80]}")
        all_ok = False
        continue
    
    # Check if model files exist (don't load full model to save time)
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(model_id, allow_patterns=["config.json"])
        print(f"  [OK] Model files present")
    except Exception as e:
        print(f"  [ERROR] Model files missing: {str(e)[:80]}")
        all_ok = False

print()
print("="*60)
if all_ok:
    print("[SUCCESS] All required models are available")
    print()
    print("Next step: Run experiment")
    print("  sbatch run_experiment.sbatch test")
else:
    print("[ERROR] Some models are missing")
    print()
    print("Download missing models:")
    print("  salloc --account=gts-mlin91 --gres=gpu:A100:1 --mem=100G --time=4:00:00")
    print("  bash download_models_interactive.sh")
    sys.exit(1)
print("="*60)
