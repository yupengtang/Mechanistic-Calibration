#!/usr/bin/env python3
"""
Quick test script to verify models can be loaded
"""

import sys
import os

# Respect an explicit cache location and otherwise use the Hugging Face default.
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.makedirs(HF_HOME, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test models (smallest first)
TEST_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",  # ~14 GB
    "meta-llama/Llama-3.1-8B-Instruct",  # ~16 GB
]

def test_model(model_id: str, device: str = "cuda"):
    """Test loading a single model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}\n")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("[OK] Tokenizer loaded")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print(f"[OK] Model loaded on {device}")
        
        # Test inference
        print("Testing inference...")
        test_prompt = "The capital of France is"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[OK] Inference works: '{response[:50]}...'")
        
        # Get model info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Parameters: {param_count / 1e9:.2f}B")
        
        if device == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"[OK] GPU memory used: {memory_used:.2f} GB")
        
        print(f"\n[SUCCESS] {model_id} is ready!\n")
        
        # Cleanup
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load {model_id}:")
        print(f"   {str(e)}\n")
        return False


def main():
    print("\n" + "="*60)
    print("BEAT-300 Model Loading Test")
    print("="*60 + "\n")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA available")
        print(f"[OK] CUDA devices: {torch.cuda.device_count()}")
        print(f"[OK] Current device: {torch.cuda.current_device()}")
        print(f"[OK] Device name: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("[WARNING] CUDA not available, using CPU (will be slow)")
        device = "cpu"
    
    print(f"\nDevice: {device}\n")
    
    # Test models
    results = {}
    for model_id in TEST_MODELS:
        success = test_model(model_id, device)
        results[model_id] = success
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60 + "\n")
    
    for model_id, success in results.items():
        status = "[READY]" if success else "[FAILED]"
        print(f"{status}: {model_id}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n[SUCCESS] All models loaded successfully!")
        print("\nYou can now run:")
        print("  python run_experiment.py --test-mode")
        return 0
    else:
        print("\n[WARNING] Some models failed to load.")
        print("\nTroubleshooting:")
        print("  1. Check CUDA: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  2. Check disk space: df -h ~/.cache/huggingface/")
        print("  3. For Llama models: huggingface-cli login")
        print("  4. Check GPU memory: nvidia-smi")
        return 1


if __name__ == "__main__":
    sys.exit(main())
