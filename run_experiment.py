#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point for BEAT-120 Experiment (Version 2 - Fixed)

**KEY FIX**: Implements proper Pass 1 sharing as required by proposal §4.1:
"For each question, model, temperature, and replicate, the baseline response (Pass 1) 
is sampled once and reused as a shared initial state for all Pass-2 interventions."
"""

import os

# Set HuggingFace cache to project directory
HF_HOME = "/storage/project/r-pkastner3-0/ytang454/hf_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.makedirs(HF_HOME, exist_ok=True)

import json
import argparse
from pathlib import Path
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
from dotenv import load_dotenv
import os
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

from src.data_structures import (
    ExperimentConfig, Question, PassResult, load_questions, write_jsonl_record
)
from src.experiment_runner import TwoPassExperiment
from src.prompt_utils import compute_length_matched_prompts, verify_length_matching
from src.aep_transformation import should_apply_aep, transform_question_to_aep, AEPMetadata

try:
    from src.mechanistic_probing import load_model_for_probing
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from client import Client


def load_models_config(config_path: Path) -> dict:
    """Load frozen models configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompts_registry(registry_path: Path) -> dict:
    """Load frozen prompts registry"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_pass1_baseline(
    question: Question,
    model_info: dict,
    temperature: float,
    top_p: float,
    max_tokens: int,
    replicate_id: int,
    experiment: TwoPassExperiment,
    device: str = "cuda"
) -> Tuple[PassResult, dict]:
    """
    Run Pass 1 baseline (called ONCE per question/model/temp/replicate).
    
    Returns:
        Tuple of (PassResult, model_context_dict)
        model_context_dict contains loaded model/tokenizer/client for reuse
    """
    model_id = model_info["model_id"]
    provider = model_info["provider"]
    
    pass1_prompt = experiment.build_pass1_prompt(question)
    
    if provider == "local":
        if not TORCH_AVAILABLE:
            return None, None
        
        # Load model (cache this in production)
        model, tokenizer = load_model_for_probing(model_id, device=device)
        
        pass1_result = experiment.run_pass_local(
            model, tokenizer, pass1_prompt, temperature, top_p, max_tokens, device
        )
        
        context = {
            "provider": "local",
            "model": model,
            "tokenizer": tokenizer
        }
    else:
        client = Client()
        
        pass1_result = experiment.run_pass_api(
            client, model_id, pass1_prompt, temperature, top_p, max_tokens
        )
        
        context = {
            "provider": "openrouter",
            "client": client
        }
    
    return pass1_result, context


def run_pass2_with_shared_baseline(
    question: Question,
    model_info: dict,
    temperature: float,
    top_p: float,
    max_tokens: int,
    replicate_id: int,
    pass1_result: PassResult,
    model_context: dict,
    condition: Tuple[str, str, str],
    experiment: TwoPassExperiment,
    output_file: Path,
    output_lock: threading.Lock,
    device: str = "cuda"
):
    """
    Run Pass 2 for a specific condition, using shared Pass 1 baseline.
    
    Args:
        pass1_result: Shared Pass 1 result
        model_context: Dict with loaded model/tokenizer/client
        condition: (reputation, evidence, framing) tuple
    """
    from src.data_structures import TrialRecord, generate_trial_id, get_commit_hash
    from datetime import datetime
    
    model_id = model_info["model_id"]
    provider = model_info["provider"]
    reputation_level, evidence_level, framing_level = condition
    
    # Build all Pass 2 prompts for length matching
    # (In production, this should be cached per question)
    all_conditions = list(product(
        experiment.config.reputation_levels,
        experiment.config.evidence_levels,
        experiment.config.framing_levels
    ))
    
    prompts_dict = {}
    for (rep, evid, frm) in all_conditions:
        key = f"{rep}_{evid}_{frm}"
        prompt = experiment.build_pass2_prompt(
            question, pass1_result.parsed_label, rep, evid, frm, padding=""
        )
        prompts_dict[key] = prompt
    
    # Compute length-matched prompts (DESIGN.md §4.5)
    # - Local: use the local tokenizer
    # - API: use a reference tokenizer (default: gpt2) as a consistent proxy
    tokenizer = model_context.get("tokenizer") if provider == "local" else None
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            ref_tok_id = getattr(experiment.config, "api_reference_tokenizer_model_id", "gpt2")
            tokenizer = AutoTokenizer.from_pretrained(ref_tok_id)
        except Exception:
            tokenizer = None

    if tokenizer is not None:
        filler_variants = experiment.prompts_registry.get("padding_policy", {}).get("filler_variants", None)
        padded_prompts = compute_length_matched_prompts(
            prompts_dict, tokenizer, target_tolerance=2, filler_variants=filler_variants
        )
    else:
        padded_prompts = {k: (v, 0, 0, "") for k, v in prompts_dict.items()}
    
    # Get this condition's padded prompt
    condition_key = f"{reputation_level}_{evidence_level}_{framing_level}"
    pass2_prompt, target_len, actual_len, padding_text = padded_prompts[condition_key]
    
    # Run Pass 2
    if provider == "local":
        model = model_context["model"]
        tokenizer = model_context["tokenizer"]
        
        pass2_result = experiment.run_pass_local(
            model, tokenizer, pass2_prompt, temperature, top_p, max_tokens, device
        )
    else:
        client = model_context["client"]
        
        pass2_result = experiment.run_pass_api(
            client, model_id, pass2_prompt, temperature, top_p, max_tokens
        )
    
    # Compute reversal
    reversal = (
        pass1_result.parsed_label is not None and
        pass2_result.parsed_label is not None and
        pass1_result.parsed_label != pass2_result.parsed_label
    )
    
    # Compute delta log-odds
    delta_logodds = None
    if pass1_result.logodds_yes_over_no is not None and pass2_result.logodds_yes_over_no is not None:
        delta_logodds = pass2_result.logodds_yes_over_no - pass1_result.logodds_yes_over_no
    
    # Generate trial ID
    trial_id = generate_trial_id(model_id, question.question_id, condition_key, replicate_id)
    
    # Create trial record
    record = TrialRecord(
        trial_id=trial_id,
        question_id=question.question_id,
        domain=question.domain,
        question_hash=question.hash,
        excerpt_hash=__import__("hashlib").sha256(question.excerpt.encode("utf-8")).hexdigest()[:16],
        model_id=model_id,
        provider=provider,
        timestamp_utc=datetime.utcnow().isoformat(),
        commit_hash=get_commit_hash(),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=experiment.stop_sequences,
        replicate_id=replicate_id,
        reputation_factor=reputation_level,
        evidence_factor=evidence_level,
        framing_factor=framing_level,
        pass1=pass1_result,
        pass2=pass2_result,
        reversal=reversal,
        delta_logodds=delta_logodds,
        target_token_len=target_len,
        actual_token_len=actual_len,
        padding_id=__import__("hashlib").sha256((padding_text or "").encode("utf-8")).hexdigest()[:8],
        addon_hash=getattr(experiment, "_last_addon_hash", ""),
        addon_source_id=getattr(experiment, "_last_addon_source_id", "")
    )
    
    # Write to file
    write_jsonl_record(record, output_file, lock=output_lock)
    
    return record


def main():
    parser = argparse.ArgumentParser(description="Run BEAT-120 Experiment (v2 - Fixed)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--questions", type=str, default="frozen_artifacts/beat120_questions_SAMPLE.jsonl")
    parser.add_argument("--prompts", type=str, default="prompts/registry.json")
    parser.add_argument("--models", type=str, default="frozen_artifacts/models.json")
    parser.add_argument("--output", type=str, default="results/trials.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-mode", action="store_true")
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = ExperimentConfig.load(Path(args.config))
    else:
        config = ExperimentConfig()
    
    # Test mode
    if args.test_mode:
        print("\n" + "="*60)
        print("TEST MODE (v2 - with Pass 1 sharing)")
        print("="*60)
        config.core_temperatures = [0.0]
        config.core_replicates = 1
        config.reputation_levels = ["A0", "A1"]
        config.evidence_levels = ["B0"]
        config.framing_levels = ["C1"]
    
    # Load resources
    print("Loading resources...")
    questions = load_questions(Path(args.questions))
    
    # Apply AEP transformation if enabled (DESIGN.md §3.4)
    if config.ablations.get("aep_entity_virtualization", False):
        aep_ratio = config.ablations.get("aep_ratio", 0.5)
        aep_count = 0
        
        for i, q in enumerate(questions):
            if should_apply_aep(q.question_id, aep_ratio):
                # Apply AEP to BOTH prompt_base and excerpt (authority-bearing identifiers can appear in either)
                transformed_prompt, was_t_prompt, replaced_p = transform_question_to_aep(q.prompt_base)
                transformed_excerpt, was_t_excerpt, replaced_e = transform_question_to_aep(q.excerpt)

                replaced = {**replaced_p, **replaced_e}
                was_transformed = was_t_prompt or was_t_excerpt
                if was_transformed:
                    q.prompt_base = transformed_prompt
                    q.excerpt = transformed_excerpt
                    q.is_aep = True
                    q.metadata["aep_entities_replaced"] = replaced
                    aep_count += 1
        
        print(f"Applied AEP to {aep_count}/{len(questions)} questions ({aep_count/len(questions)*100:.1f}%)")
    
    if args.test_mode:
        questions = questions[:1]
    print(f"Loaded {len(questions)} questions")
    
    prompts_registry = load_prompts_registry(Path(args.prompts))
    models_config = load_models_config(Path(args.models))
    
    # Select models (DESIGN.md §5.1–§5.2)
    models_to_run = []
    if config.run_mechanistic_core and TORCH_AVAILABLE:
        models_to_run.extend(models_config["mechanistic_probing_set"]["core"])
    elif config.run_mechanistic_core and not TORCH_AVAILABLE:
        print("Note: torch not available; skipping local mechanistic models.")
    if not models_to_run and config.run_behavioral_anchors:
        models_to_run.extend(models_config["behavioral_anchor_set"]["models"])
    
    print(f"Running {len(models_to_run)} model(s)")
    
    # Generate conditions
    conditions = list(product(
        config.reputation_levels,
        config.evidence_levels,
        config.framing_levels
    ))
    print(f"Testing {len(conditions)} condition(s)")
    
    # Output setup
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_lock = threading.Lock()
    
    # Initialize experiment
    experiment = TwoPassExperiment(config, prompts_registry, output_lock)

    # Build placebo pools for B3 (domain -> list[(question_id, b2_evidence)])
    placebo_pools = {}
    for q in questions:
        b2 = (q.metadata or {}).get("b2_evidence")
        if isinstance(b2, str) and b2.strip():
            placebo_pools.setdefault(q.domain, []).append((q.question_id, b2.strip()))
    experiment.placebo_pools = placebo_pools
    
    # CORE LOOP: Question -> Model -> Temp -> Replicate -> [Pass 1 once] -> All Conditions
    print(f"\n{'='*60}")
    print("STARTING EXPERIMENT WITH PROPER PASS 1 SHARING")
    print(f"{'='*60}\n")
    
    total_pass1 = len(questions) * len(models_to_run) * len(config.core_temperatures) * config.core_replicates
    total_pass2 = total_pass1 * len(conditions)
    
    print(f"Total Pass 1 runs: {total_pass1}")
    print(f"Total Pass 2 runs: {total_pass2}")
    print(f"Total trials: {total_pass2}\n")
    
    pass1_success = 0
    pass1_invalid = 0
    trials_success = 0
    trials_error = 0
    
    with tqdm(total=total_pass2, desc="Running trials") as pbar:
        for question in questions:
            for model_info in models_to_run:
                for temperature in config.core_temperatures:
                    for replicate_id in range(1, config.core_replicates + 1):
                        
                        # ===== PASS 1: RUN ONCE =====
                        pass1_result, model_context = run_pass1_baseline(
                            question, model_info, temperature, config.core_top_p,
                            config.core_max_tokens, replicate_id, experiment, args.device
                        )
                        
                        if pass1_result is None or pass1_result.parsed_label is None or \
                           pass1_result.truncated or pass1_result.format_violation:
                            # Pass 1 invalid - skip ALL Pass 2 for this baseline
                            pass1_invalid += 1
                            pbar.update(len(conditions))  # Skip all conditions
                            continue
                        
                        pass1_success += 1
                        
                        # ===== PASS 2: RUN FOR ALL CONDITIONS (using shared Pass 1) =====
                        for condition in conditions:
                            try:
                                record = run_pass2_with_shared_baseline(
                                    question, model_info, temperature, config.core_top_p,
                                    config.core_max_tokens, replicate_id,
                                    pass1_result,  # SHARED
                                    model_context,
                                    condition,
                                    experiment,
                                    output_file,
                                    output_lock,
                                    args.device
                                )
                                trials_success += 1
                            except Exception as e:
                                print(f"\nError in Pass 2: {e}")
                                trials_error += 1
                            
                            pbar.update(1)
                        
                        # Cleanup model (if local)
                        if model_context and model_context["provider"] == "local":
                            del model_context["model"], model_context["tokenizer"]
                            if TORCH_AVAILABLE:
                                import torch
                                torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Pass 1 runs:")
    print(f"  Success: {pass1_success}")
    print(f"  Invalid: {pass1_invalid}")
    print(f"Trials (Pass 2):")
    print(f"  Success: {trials_success}")
    print(f"  Errors: {trials_error}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

