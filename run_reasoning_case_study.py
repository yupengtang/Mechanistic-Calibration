#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Case Study for BEAT-120

Minimal protocol for reasoning-oriented models (o1-preview, o3-mini).

Proposal §5.3:
- N=24 questions (highest entropy from BEAT-120)
- 4 conditions only: A1/B0/C1, A3/B0/C1, A1/B2/C1, A3/B2/C1
- T=0.0, single run per item
- Reports SSI-Reduction vs baseline (GPT-4o-mini)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import threading
from dotenv import load_dotenv

load_dotenv()

from src.data_structures import (
    Question, PassResult, TrialRecord, load_questions,
    write_jsonl_record, generate_trial_id, get_commit_hash
)
from src.experiment_runner import TwoPassExperiment
from client import Client


def select_highest_entropy_questions(
    questions: List[Question],
    selection_log_path: Path,
    n: int = 24
) -> List[Question]:
    """
    Select N questions with highest entropy from boundary selection.
    
    Args:
        questions: All BEAT-120 questions
        selection_log_path: Path to selection_log.jsonl from boundary filtering
        n: Number to select (default: 24)
    
    Returns:
        List of N highest-entropy questions
    """
    # Load selection log
    if not selection_log_path.exists():
        print(f"Warning: Selection log not found at {selection_log_path}")
        print(f"Selecting first {n} questions as fallback")
        return questions[:n]
    
    # Parse JSONL selection log and compute entropy for each selected_question_id
    import numpy as np

    question_entropy: Dict[str, float] = {}
    with open(selection_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("record_type") != "candidate":
                continue
            if not rec.get("retained"):
                continue
            qid = rec.get("selected_question_id")
            if not qid:
                continue

            max_entropy = 0.0
            for result in rec.get("screening_results", []):
                yes_count = int(result.get("yes_count", 0))
                no_count = int(result.get("no_count", 0))
                total = yes_count + no_count
                if total <= 0:
                    continue
                p_yes = yes_count / total
                p_no = no_count / total
                entropy = 0.0
                if p_yes > 0:
                    entropy -= p_yes * np.log2(p_yes)
                if p_no > 0:
                    entropy -= p_no * np.log2(p_no)
                max_entropy = max(max_entropy, float(entropy))

            question_entropy[qid] = max_entropy
    
    # Sort questions by entropy (descending)
    if question_entropy:
        # Match questions to their entropy scores
        questions_with_entropy = []
        for q in questions:
            entropy = question_entropy.get(q.question_id, 0.0)
            questions_with_entropy.append((q, float(entropy)))
        
        # Sort and select top N
        questions_with_entropy.sort(key=lambda x: x[1], reverse=True)
        selected = [q for q, _ in questions_with_entropy[:n]]
        
        print(f"Selected {len(selected)} questions with highest entropy")
        return selected
    else:
        # Fallback: uniform random selection
        print("Warning: No entropy scores found, selecting first N questions")
        return questions[:n]


def run_reasoning_trial(
    question: Question,
    model_id: str,
    condition: str,  # e.g., "A1/B0/C1"
    experiment: TwoPassExperiment,
    client: Client,
    output_file: Path,
    output_lock: threading.Lock
) -> bool:
    """
    Run a single reasoning case study trial.
    
    Args:
        question: Question to test
        model_id: Reasoning model ID (e.g., "openai/o1-preview")
        condition: Condition string "A/B/C"
        experiment: Experiment runner
        client: API client
        output_file: Output file path
        output_lock: Thread lock
    
    Returns:
        True if successful
    """
    # Parse condition
    parts = condition.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid condition format: {condition}")
    
    reputation_level, evidence_level, framing_level = parts
    
    # Pass 1
    pass1_prompt = experiment.build_pass1_prompt(question)
    pass1_result = experiment.run_pass_api(
        client, model_id, pass1_prompt,
        temperature=0.0,  # Fixed for reasoning models
        top_p=1.0,
        max_tokens=experiment.config.core_max_tokens
    )
    
    # Check Pass 1 validity
    if pass1_result.parsed_label is None or pass1_result.truncated or pass1_result.format_violation:
        return False
    
    # Pass 2
    pass2_prompt = experiment.build_pass2_prompt(
        question, pass1_result.parsed_label,
        reputation_level, evidence_level, framing_level
    )
    
    pass2_result = experiment.run_pass_api(
        client, model_id, pass2_prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=experiment.config.core_max_tokens
    )
    
    # Compute reversal
    reversal = (
        pass1_result.parsed_label is not None and
        pass2_result.parsed_label is not None and
        pass1_result.parsed_label != pass2_result.parsed_label
    )
    
    # Create trial record
    from datetime import datetime
    
    trial_id = generate_trial_id(model_id, question.question_id, condition, 1)
    
    record = TrialRecord(
        trial_id=trial_id,
        question_id=question.question_id,
        domain=question.domain,
        question_hash=question.hash,
        excerpt_hash=__import__("hashlib").sha256(question.excerpt.encode("utf-8")).hexdigest()[:16],
        model_id=model_id,
        provider="openrouter",
        timestamp_utc=datetime.utcnow().isoformat(),
        commit_hash=get_commit_hash(),
        temperature=0.0,
        top_p=1.0,
        max_tokens=experiment.config.core_max_tokens,
        stop_sequences=experiment.stop_sequences,
        replicate_id=1,
        reputation_factor=reputation_level,
        evidence_factor=evidence_level,
        framing_factor=framing_level,
        pass1=pass1_result,
        pass2=pass2_result,
        reversal=reversal,
        addon_hash=getattr(experiment, "_last_addon_hash", ""),
        addon_source_id=getattr(experiment, "_last_addon_source_id", "")
    )
    
    # Write record
    write_jsonl_record(record, output_file, lock=output_lock)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Reasoning Case Study (Minimal Protocol)")
    parser.add_argument("--questions", type=str, 
                       default="frozen_artifacts/beat120_questions.jsonl")
    parser.add_argument("--selection-log", type=str,
                       default="frozen_artifacts/selection_log.jsonl")
    parser.add_argument("--output", type=str,
                       default="results/reasoning_case_study.jsonl")
    parser.add_argument("--n-questions", type=int, default=24,
                       help="Number of highest-entropy questions")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["openai/o1-preview", "openai/o3-mini", "openai/gpt-4o-mini"],
                       help="Reasoning models to test")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("REASONING CASE STUDY (Minimal Protocol)")
    print("="*60)
    print(f"Questions: {args.n_questions} highest entropy")
    print(f"Conditions: 4 (A1/B0/C1, A3/B0/C1, A1/B2/C1, A3/B2/C1)")
    print(f"Temperature: 0.0 (deterministic)")
    print(f"Runs per item: 1")
    print(f"Models: {len(args.models)}")
    for m in args.models:
        print(f"  • {m}")
    print("="*60 + "\n")
    
    # Load questions
    all_questions = load_questions(Path(args.questions))
    print(f"Loaded {len(all_questions)} total questions")
    
    # Select highest entropy questions
    selected_questions = select_highest_entropy_questions(
        all_questions,
        Path(args.selection_log),
        n=args.n_questions
    )
    print(f"Selected {len(selected_questions)} questions\n")
    
    # Conditions (minimal protocol)
    conditions = [
        "A1/B0/C1",  # Anonymous, no evidence
        "A3/B0/C1",  # Expertise, no evidence
        "A1/B2/C1",  # Anonymous, strong relevant evidence
        "A3/B2/C1",  # Expertise, strong relevant evidence
    ]
    
    # Setup
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_lock = threading.Lock()
    
    # Load prompts and config
    with open("prompts/registry.json", 'r') as f:
        prompts_registry = json.load(f)
    
    from src.data_structures import ExperimentConfig
    if Path("config.json").exists():
        config = ExperimentConfig.load(Path("config.json"))
    else:
        config = ExperimentConfig()
    
    experiment = TwoPassExperiment(config, prompts_registry, output_lock)
    client = Client()
    
    # Run trials
    total_trials = len(selected_questions) * len(args.models) * len(conditions)
    success_count = 0
    fail_count = 0
    
    print(f"Total trials: {total_trials}\n")
    
    with tqdm(total=total_trials, desc="Running reasoning trials") as pbar:
        for question in selected_questions:
            for model_id in args.models:
                for condition in conditions:
                    try:
                        success = run_reasoning_trial(
                            question, model_id, condition,
                            experiment, client,
                            output_file, output_lock
                        )
                        
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    
                    except Exception as e:
                        print(f"\nError: {e}")
                        fail_count += 1
                    
                    pbar.update(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("REASONING CASE STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Total trials: {total_trials}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    # Compute SSI-Reduction (if baseline model included)
    if "openai/gpt-4o-mini" in args.models:
        print("\nComputing SSI-Reduction...")
        # Load results and compute SSI for each model
        # SSI-RR = SSI_baseline / SSI_reasoning
        # This would be done in the analysis script
        print("Run analyze_results.py with --reasoning-case-study flag for SSI-RR")


if __name__ == "__main__":
    main()

