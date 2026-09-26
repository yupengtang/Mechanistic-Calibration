#!/usr/bin/env python3
"""
Quick test: Run BEAT-300 experiment on 1 question to verify environment setup
"""

import os
import sys
import json
from pathlib import Path

# Respect caller-provided cache and package locations.
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")

PYTHONUSERBASE = os.environ.get("PYTHONUSERBASE", os.path.expanduser("~/.local"))
os.environ["PYTHONUSERBASE"] = PYTHONUSERBASE

print("="*70)
print("BEAT-300 Environment Test: 1 Question")
print("="*70)
print()

# Check dependencies
print("[Step 1/5] Checking dependencies...")
try:
    import torch
    import transformers
    from src.data_structures import Question, ExperimentConfig
    from src.experiment_runner import TwoPassExperiment
    from src.mechanistic_probing import load_model_for_probing
    print(f"  [OK] torch {torch.__version__}")
    print(f"  [OK] transformers {transformers.__version__}")
    print(f"  [OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"  [ERROR] Missing dependency: {e}")
    sys.exit(1)

print()

# Load sample question
print("[Step 2/5] Loading sample question...")
sample_file = Path("frozen_artifacts/beat300_questions_SAMPLE.jsonl")
if not sample_file.exists():
    print(f"  [ERROR] {sample_file} not found")
    sys.exit(1)

with open(sample_file) as f:
    line = f.readline().strip()
    if not line:
        print("  [ERROR] Sample file is empty")
        sys.exit(1)
    question_data = json.loads(line)
    question = Question(**question_data)
    print(f"  [OK] Loaded question: {question.question_id}")
    print(f"       Domain: {question.domain}")
    print(f"       Ground truth: {question.ground_truth}")

print()

# Load prompts
print("[Step 3/5] Loading prompts registry...")
with open("prompts/registry.json") as f:
    prompts = json.load(f)
    print(f"  [OK] Loaded {len(prompts.get('reputation_factor', {}))} reputation levels")
    print(f"  [OK] Loaded {len(prompts.get('evidence_factor', {}))} evidence levels")

print()

# Load smallest model for quick test
print("[Step 4/5] Loading model (Qwen-7B, smallest)...")
model_id = "Qwen/Qwen2.5-7B-Instruct"
try:
    model, tokenizer = load_model_for_probing(model_id, device="cuda")
    print(f"  [OK] Model loaded: {model_id}")
except Exception as e:
    print(f"  [ERROR] Failed to load model: {e}")
    sys.exit(1)

print()

# Run a minimal trial
print("[Step 5/5] Running minimal trial (A0/B0/C1, 1 replicate, T=0.0)...")

# Create config
config = ExperimentConfig(
    questions_file="frozen_artifacts/beat300_questions_SAMPLE.jsonl",
    output_file="results/test_one_question.jsonl",
    models_registry_file="frozen_artifacts/models.json",
    prompts_registry_file="prompts/registry.json",
    device="cuda",
    temperature=[0.0],
    replicates=1
)

# Create experiment runner
experiment = TwoPassExperiment(config, prompts, None)

# Run single trial: A0 (no reputation), B0 (no evidence), C1 (first framing)
try:
    trial = experiment.run_trial(
        question=question,
        model_id=model_id,
        provider="local",
        reputation_level="A0",
        evidence_level="B0",
        framing_level="C1",
        temperature=0.0,
        top_p=1.0,
        max_tokens=600,
        replicate_id=1,
        model=model,
        tokenizer=tokenizer,
        device="cuda"
    )
    
    if trial is None:
        print("  [ERROR] Trial returned None (Pass 1 may have failed)")
        sys.exit(1)
    
    print(f"  [OK] Trial completed")
    print(f"       Pass 1: {trial.pass1.parsed_label}")
    print(f"       Pass 2: {trial.pass2.parsed_label}")
    print(f"       Reversal: {trial.reversal}")
    if trial.delta_logodds is not None:
        print(f"       Delta log-odds: {trial.delta_logodds:.3f}")
    print(f"       Latency: {trial.pass1.latency_seconds + trial.pass2.latency_seconds:.2f}s")
    
except Exception as e:
    print(f"  [ERROR] Trial failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*70)
print("[SUCCESS] Environment test passed!")
print("="*70)
print()
print("Your environment is ready to run experiments.")
print()
print("Next steps:")
print("  1. Test with sample questions:")
print("     sbatch run_experiment.sbatch test")
print()
print("  2. Generate full BEAT-300 benchmark:")
print("     bash generate_beat300.sh")
print()
print("  3. Run full experiment:")
print("     sbatch run_experiment.sbatch core")
print()
