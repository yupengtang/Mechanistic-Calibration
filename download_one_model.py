#!/usr/bin/env python3
"""
Interactive model downloader - downloads one model at a time
Useful for testing or limited disk space
"""

import sys
import os

# Set HuggingFace cache to project directory
HF_HOME = "/storage/project/r-pkastner3-0/ytang454/hf_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.makedirs(HF_HOME, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Models from DESIGN.md §6.1
MODELS = {
    "1": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "size": "~14 GB",
        "params": "7B",
        "auth": False
    },
    "2": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "~16 GB",
        "params": "8B",
        "auth": True
    },
    "3": {
        "id": "Qwen/Qwen2.5-32B-Instruct",
        "size": "~64 GB",
        "params": "32B",
        "auth": False
    },
    "4": {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "size": "~140 GB",
        "params": "70B",
        "auth": True
    },
    "5": {
        "id": "meta-llama/Llama-3.1-405B-Instruct",
        "size": "~810 GB",
        "params": "405B",
        "auth": True,
        "note": "Extended set (optional)"
    },
    "6": {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "size": "~144 GB",
        "params": "72B",
        "auth": False,
        "note": "Extended set (optional)"
    }
}

def download_model(model_id: str):
    """Download and test a model"""
    print(f"\n{'='*70}")
    print(f"Downloading: {model_id}")
    print(f"{'='*70}\n")
    
    try:
        # Check CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            free_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {free_mem:.1f} GB")
        print()
        
        # Load tokenizer
        print("Step 1/3: Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("[OK] Tokenizer ready\n")
        
        # Load model
        print("Step 2/3: Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model loaded ({param_count / 1e9:.1f}B parameters)\n")
        
        # Test inference
        print("Step 3/3: Testing inference...")
        test_prompt = "Question: Is the sky blue?\n\nAnswer:"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.0,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test output: {response[:100]}...")
        
        if device == "cuda":
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"\nGPU memory used: {mem_used:.2f} GB")
        
        print(f"\n[SUCCESS] {model_id} is ready to use.\n")
        
        # Cleanup
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}\n")
        print("Troubleshooting:")
        if "401" in str(e) or "gated" in str(e).lower():
            print("  - This model requires authentication")
            print("  - Run: huggingface-cli login")
            print("  - Get token from: https://huggingface.co/settings/tokens")
        elif "CUDA out of memory" in str(e):
            print("  - GPU memory insufficient")
            print("  - Try a smaller model first")
            print("  - Check: nvidia-smi")
        else:
            print("  - Check internet connection")
            print("  - Check disk space: df -h ~/.cache/huggingface/")
        print()
        return False


def main():
    print("\n" + "="*70)
    print("BEAT-120 Model Downloader")
    print("="*70 + "\n")
    
    # Check dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM
        print("[OK] Dependencies installed\n")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nPlease run: pip install torch transformers accelerate")
        return 1
    
    # Show CUDA status
    if torch.cuda.is_available():
        print(f"[OK] CUDA available ({torch.cuda.device_count()} device(s))")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("[WARNING] CUDA not available (will use CPU, slower)\n")
    
    # Show models
    print("Available models (Core set for main experiments):")
    print("-" * 70)
    for key, info in MODELS.items():
        auth_mark = " [Requires HF login]" if info.get("auth") else ""
        note = f" - {info['note']}" if "note" in info else ""
        print(f"{key}. {info['id']}")
        print(f"   Size: {info['size']}, Parameters: {info['params']}{auth_mark}{note}")
    print("-" * 70)
    
    print("\nRecommended download order:")
    print("  1. Start with smallest: Qwen 7B (option 1)")
    print("  2. Then: Llama 8B (option 2)")
    print("  3. Then: Qwen 32B or Llama 70B (options 3-4)")
    print("\nNote: Options 5-6 are for extended experiments (optional)")
    
    # Select model
    print()
    choice = input("Select model to download (1-6) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("Cancelled.")
        return 0
    
    if choice not in MODELS:
        print(f"Invalid choice: {choice}")
        return 1
    
    model_info = MODELS[choice]
    
    # Confirm
    print(f"\nYou selected: {model_info['id']}")
    print(f"Disk space needed: {model_info['size']}")
    if model_info.get("auth"):
        print("\n[WARNING] This model requires HuggingFace authentication!")
        print("Before continuing:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Run: huggingface-cli login")
        print("  3. Accept the model license on HuggingFace")
    
    confirm = input("\nProceed with download? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return 0
    
    # Download
    success = download_model(model_info["id"])
    
    if success:
        print("="*70)
        print("Next steps:")
        print("  - Download more models: python download_one_model.py")
        print("  - Test all models: python test_model_loading.py")
        print("  - Run experiment: python run_experiment.py --test-mode")
        print("="*70 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
