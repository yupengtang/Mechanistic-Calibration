#!/usr/bin/env python3
"""
Automatic model downloader for BEAT-300
Downloads all core models sequentially with error handling
"""

import sys
import os
import time
from pathlib import Path

# Respect an explicit cache location and otherwise use the Hugging Face default.
HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.makedirs(HF_HOME, exist_ok=True)

# Core models from DESIGN.md §6.1
CORE_MODELS = [
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "size_gb": 14,
        "params": "7B",
        "auth_required": False
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "size_gb": 16,
        "params": "8B",
        "auth_required": True
    },
    {
        "id": "Qwen/Qwen2.5-32B-Instruct",
        "size_gb": 64,
        "params": "32B",
        "auth_required": False
    },
    {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "size_gb": 140,
        "params": "70B",
        "auth_required": True
    }
]


def check_environment():
    """Check if required packages are installed"""
    print("\n" + "="*70)
    print("Environment Check")
    print("="*70 + "\n")
    
    try:
        import torch
        print(f"[OK] torch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.device_count()} device(s)")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("[WARNING] CUDA not available, will use CPU (slow)")
            device = "cpu"
    except ImportError:
        print("[ERROR] torch not installed")
        print("Run: pip install torch transformers accelerate")
        return False, "cpu"
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        print(f"[OK] transformers version: {transformers.__version__}")
    except ImportError:
        print("[ERROR] transformers not installed")
        return False, device
    
    try:
        from huggingface_hub import whoami
        try:
            user = whoami()
            print(f"[OK] HuggingFace authenticated as: {user['name']}")
        except Exception:
            print("[WARNING] Not logged in to HuggingFace")
            print("For Llama models, run: huggingface-cli login")
    except ImportError:
        print("[WARNING] huggingface_hub not installed")
    
    print()
    return True, device


def download_model(model_info, device="cuda"):
    """Download and verify a single model"""
    model_id = model_info["id"]
    
    print("\n" + "="*70)
    print(f"Downloading: {model_id}")
    print(f"Size: ~{model_info['size_gb']} GB, Parameters: {model_info['params']}")
    print("="*70 + "\n")
    
    if model_info["auth_required"]:
        print("[INFO] This model requires HuggingFace authentication")
        print("[INFO] If download fails, run: huggingface-cli login\n")
    
    start_time = time.time()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Download tokenizer
        print("Step 1/3: Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("[OK] Tokenizer downloaded\n")
        
        # Download model
        print("Step 2/3: Downloading model (this may take 10-30 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model downloaded ({param_count / 1e9:.1f}B parameters)\n")
        
        # Quick inference test
        print("Step 3/3: Testing inference...")
        test_prompt = "Hello"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[OK] Test output: '{response[:50]}'")
        
        if device == "cuda":
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"[OK] GPU memory used: {mem_used:.2f} GB")
        
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {model_id} ready! (took {elapsed/60:.1f} minutes)\n")
        
        # Cleanup
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Failed after {elapsed/60:.1f} minutes")
        print(f"Error: {str(e)}\n")
        
        if "401" in str(e) or "gated" in str(e).lower():
            print("This model requires authentication:")
            print("  1. Get token: https://huggingface.co/settings/tokens")
            print("  2. Run: huggingface-cli login")
            print("  3. Accept license: https://huggingface.co/" + model_id)
        elif "CUDA out of memory" in str(e):
            print("GPU memory insufficient. Try:")
            print("  - Free up GPU memory")
            print("  - Use a different GPU node")
        
        return False


def main():
    print("\n" + "="*70)
    print("BEAT-300 Automatic Model Downloader")
    print("="*70)
    
    # Check environment
    env_ok, device = check_environment()
    if not env_ok:
        print("\n[ERROR] Environment check failed")
        return 1
    
    print("="*70)
    print("Models to download:")
    print("="*70)
    total_size = 0
    for i, model in enumerate(CORE_MODELS, 1):
        auth = " [Requires auth]" if model["auth_required"] else ""
        print(f"{i}. {model['id']} (~{model['size_gb']} GB){auth}")
        total_size += model["size_gb"]
    print(f"\nTotal: ~{total_size} GB")
    print("="*70)
    
    # Confirm
    response = input("\nProceed with download? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return 0
    
    # Download each model
    results = {}
    overall_start = time.time()
    
    for i, model in enumerate(CORE_MODELS, 1):
        print(f"\n[INFO] Processing model {i}/{len(CORE_MODELS)}")
        success = download_model(model, device)
        results[model["id"]] = success
        
        if not success:
            print(f"\n[WARNING] {model['id']} failed")
            retry = input("Continue with remaining models? (y/N): ").strip().lower()
            if retry != 'y':
                break
    
    # Summary
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70 + "\n")
    
    for model_id, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"{status} {model_id}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nCompleted: {success_count}/{total_count} models")
    print(f"Total time: {overall_elapsed/3600:.1f} hours")
    
    if success_count == total_count:
        print("\n[SUCCESS] All models downloaded!")
        print("\nNext steps:")
        print("  python test_model_loading.py")
        print("  python run_experiment.py --test-mode")
        return 0
    else:
        print("\n[WARNING] Some models failed")
        print("Re-run this script to retry failed downloads")
        return 1


if __name__ == "__main__":
    sys.exit(main())
