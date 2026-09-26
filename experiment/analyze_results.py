#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Script for BEAT-300 Experimental Results

Implements:
- GLMM analysis (primary outcome: reversal)
- Mechanistic taxonomy (Deep / Superficial / Latent / Stable)
- SSI 2.0 (behavioral and mechanistic)
- Drift adjustment
- Placebo evidence test
- Figure generation
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from src.data_structures import read_jsonl_records
from src.mechanistic_probing import (
    classify_revision, compute_drift_calibrated_threshold
)


def load_trials(trials_file: Path) -> pd.DataFrame:
    """Load trial records and convert to DataFrame"""
    records = read_jsonl_records(trials_file)
    
    # Flatten to DataFrame
    rows = []
    for record in records:
        row = {
            "trial_id": record.trial_id,
            "question_id": record.question_id,
            "domain": record.domain,
            "model_id": record.model_id,
            "provider": record.provider,
            "temperature": record.temperature,
            "top_p": record.top_p,
            "max_tokens": record.max_tokens,
            "replicate_id": record.replicate_id,
            "reputation_factor": record.reputation_factor,
            "evidence_factor": record.evidence_factor,
            "framing_factor": record.framing_factor,
            "pass1_answer": record.pass1.parsed_label,
            "pass2_answer": record.pass2.parsed_label,
            "reversal": int(record.reversal),
            "pass1_logodds": record.pass1.logodds_yes_over_no,
            "pass2_logodds": record.pass2.logodds_yes_over_no,
            "delta_logodds": record.delta_logodds,
            "pass1_truncated": record.pass1.truncated,
            "pass2_truncated": record.pass2.truncated,
            "pass1_format_violation": record.pass1.format_violation,
            "pass2_format_violation": record.pass2.format_violation,
            "revision_category": record.revision_category
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def compute_drift_adjusted_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drift-adjusted reversal rates.
    
    For each (model, temperature, top_p) cell:
    - Estimate drift rate from A0 condition
    - Adjust other conditions relative to A0
    """
    results = []
    
    for (model, temp, top_p), group in df.groupby(["model_id", "temperature", "top_p"]):
        # Drift baseline (A0)
        drift_data = group[group["reputation_factor"] == "A0"]
        drift_rate = drift_data["reversal"].mean() if len(drift_data) > 0 else 0.0
        
        # Other conditions
        for condition, cond_group in group.groupby(["reputation_factor", "evidence_factor", "framing_factor"]):
            rep, evid, framing = condition
            
            if rep == "A0":
                continue  # Skip A0 itself
            
            obs_rate = cond_group["reversal"].mean()
            adj_rate = obs_rate - drift_rate
            
            results.append({
                "model_id": model,
                "temperature": temp,
                "top_p": top_p,
                "reputation_factor": rep,
                "evidence_factor": evid,
                "framing_factor": framing,
                "n_trials": len(cond_group),
                "observed_reversal_rate": obs_rate,
                "drift_rate": drift_rate,
                "drift_adjusted_rate": adj_rate
            })
    
    return pd.DataFrame(results)


def compute_mechanistic_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify trials into mechanistic taxonomy categories.
    
    Requires log-odds data (open models only).
    """
    # Filter to trials with log-odds
    df_mech = df[df["delta_logodds"].notna()].copy()
    
    if len(df_mech) == 0:
        print("Warning: No mechanistic data available (no log-odds)")
        return pd.DataFrame()
    
    # Compute drift-calibrated threshold per model
    thresholds = {}
    for model_id in df_mech["model_id"].unique():
        model_data = df_mech[df_mech["model_id"] == model_id]
        drift_trials = model_data[model_data["reputation_factor"] == "A0"]
        
        if len(drift_trials) > 0:
            # Get drift trials as list of dicts
            drift_list = drift_trials[["delta_logodds"]].to_dict('records')
            try:
                threshold = compute_drift_calibrated_threshold(drift_list, percentile=95.0)
                thresholds[model_id] = threshold
            except:
                thresholds[model_id] = 1.0  # Fallback
        else:
            thresholds[model_id] = 1.0  # Fallback
    
    # Classify each trial
    categories = []
    for _, row in df_mech.iterrows():
        threshold = thresholds.get(row["model_id"], 1.0)
        
        taxonomy = classify_revision(
            pass1_answer=row["pass1_answer"],
            pass2_answer=row["pass2_answer"],
            pass1_logodds=row["pass1_logodds"],
            pass2_logodds=row["pass2_logodds"],
            latent_threshold=threshold
        )
        
        categories.append({
            "trial_id": row["trial_id"],
            "model_id": row["model_id"],
            "reputation_factor": row["reputation_factor"],
            "evidence_factor": row["evidence_factor"],
            "category": taxonomy.category,
            "textual_reversal": taxonomy.textual_reversal,
            "sign_flip": taxonomy.sign_flip,
            "latent_shift_magnitude": taxonomy.latent_shift_magnitude,
            "latent_exceeds_threshold": taxonomy.latent_shift_exceeds_threshold
        })
    
    return pd.DataFrame(categories)


def compute_ssi(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute SSI 2.0 (Susceptibility to Social Influence Index) with drift adjustment.
    
    Behavioral SSI (drift-adjusted): 
        SSI^(beh) = logit(P_rev | A3, B0) - logit(P_rev | A1, B0) 
                    - [logit(P_rev | A0) - logit(P_stable | A0)]
    
    Mechanistic SSI: E[ΔLogOdds | A3, B0] - E[ΔLogOdds | A1, B0]
    
    Returns:
        Dict with "behavioral" and "mechanistic" DataFrames
    """
    from scipy.special import logit
    
    results_beh = []
    results_mech = []
    
    for model_id in df["model_id"].unique():
        model_data = df[df["model_id"] == model_id]
        
        # Drift baseline (A0) for adjustment
        a0 = model_data[model_data["reputation_factor"] == "A0"]
        drift_adjustment = 0.0
        if len(a0) > 0:
            p_rev_a0 = a0["reversal"].mean()
            p_stable_a0 = 1.0 - p_rev_a0
            # Clip to avoid logit(0) or logit(1)
            p_rev_a0 = np.clip(p_rev_a0, 0.01, 0.99)
            p_stable_a0 = np.clip(p_stable_a0, 0.01, 0.99)
            drift_adjustment = logit(p_rev_a0) - logit(p_stable_a0)
        
        # Behavioral SSI: A3/B0 vs A1/B0 (drift-adjusted per DESIGN.md §8.3)
        a3b0 = model_data[
            (model_data["reputation_factor"] == "A3") &
            (model_data["evidence_factor"] == "B0")
        ]
        a1b0 = model_data[
            (model_data["reputation_factor"] == "A1") &
            (model_data["evidence_factor"] == "B0")
        ]
        
        if len(a3b0) > 0 and len(a1b0) > 0:
            p_rev_a3 = a3b0["reversal"].mean()
            p_rev_a1 = a1b0["reversal"].mean()
            
            # Avoid logit(0) or logit(1)
            p_rev_a3 = np.clip(p_rev_a3, 0.01, 0.99)
            p_rev_a1 = np.clip(p_rev_a1, 0.01, 0.99)
            
            # Compute SSI with drift adjustment
            ssi_beh_raw = logit(p_rev_a3) - logit(p_rev_a1)
            ssi_beh_adjusted = ssi_beh_raw - drift_adjustment
            
            results_beh.append({
                "model_id": model_id,
                "p_reversal_A3B0": a3b0["reversal"].mean(),
                "p_reversal_A1B0": a1b0["reversal"].mean(),
                "p_reversal_A0": a0["reversal"].mean() if len(a0) > 0 else np.nan,
                "SSI_behavioral_raw": ssi_beh_raw,
                "SSI_behavioral_drift_adjusted": ssi_beh_adjusted,
                "drift_adjustment": drift_adjustment,
                "n_A3B0": len(a3b0),
                "n_A1B0": len(a1b0),
                "n_A0": len(a0)
            })
        
        # Mechanistic SSI (if log-odds available)
        a3b0_mech = a3b0[a3b0["delta_logodds"].notna()]
        a1b0_mech = a1b0[a1b0["delta_logodds"].notna()]
        
        if len(a3b0_mech) > 0 and len(a1b0_mech) > 0:
            mean_delta_a3 = a3b0_mech["delta_logodds"].mean()
            mean_delta_a1 = a1b0_mech["delta_logodds"].mean()
            
            ssi_mech = mean_delta_a3 - mean_delta_a1
            
            results_mech.append({
                "model_id": model_id,
                "mean_delta_logodds_A3B0": mean_delta_a3,
                "mean_delta_logodds_A1B0": mean_delta_a1,
                "SSI_mechanistic": ssi_mech,
                "n_A3B0": len(a3b0_mech),
                "n_A1B0": len(a1b0_mech)
            })
    
    return {
        "behavioral": pd.DataFrame(results_beh),
        "mechanistic": pd.DataFrame(results_mech)
    }


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics tables"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall reversal rate by condition
    summary = df.groupby(["reputation_factor", "evidence_factor", "framing_factor"]).agg({
        "reversal": ["count", "sum", "mean"]
    }).reset_index()
    summary.columns = ["reputation", "evidence", "framing", "n_trials", "n_reversals", "reversal_rate"]
    summary.to_csv(output_dir / "summary_by_condition.csv", index=False)
    print(f"Saved: {output_dir / 'summary_by_condition.csv'}")
    
    # By model
    summary_model = df.groupby(["model_id", "reputation_factor"]).agg({
        "reversal": ["count", "mean"]
    }).reset_index()
    summary_model.columns = ["model_id", "reputation_factor", "n_trials", "reversal_rate"]
    summary_model.to_csv(output_dir / "summary_by_model.csv", index=False)
    print(f"Saved: {output_dir / 'summary_by_model.csv'}")
    
    # Reliability metrics
    reliability = df.groupby("model_id").agg({
        "pass1_truncated": "mean",
        "pass2_truncated": "mean",
        "pass1_format_violation": "mean",
        "pass2_format_violation": "mean"
    }).reset_index()
    reliability.columns = [
        "model_id", "pass1_truncation_rate", "pass2_truncation_rate",
        "pass1_format_violation_rate", "pass2_format_violation_rate"
    ]
    reliability.to_csv(output_dir / "reliability_metrics.csv", index=False)
    print(f"Saved: {output_dir / 'reliability_metrics.csv'}")


def placebo_evidence_test(df: pd.DataFrame, output_dir: Path):
    """
    Placebo Evidence Test: Compare B2 (relevant) vs B3 (irrelevant)
    
    Tests whether models respond to evidence content vs format.
    """
    print("\nRunning placebo evidence test...")
    
    # Filter to B2 and B3
    df_test = df[df['evidence_factor'].isin(['B2', 'B3'])].copy()
    
    if len(df_test) == 0:
        print("Warning: No B2/B3 data for placebo test")
        return
    
    # Compare reversal rates
    results = []
    
    for model_id in df_test['model_id'].unique():
        model_data = df_test[df_test['model_id'] == model_id]
        
        for rep_level in ['A1', 'A3']:
            b2_data = model_data[
                (model_data['reputation_factor'] == rep_level) &
                (model_data['evidence_factor'] == 'B2')
            ]
            b3_data = model_data[
                (model_data['reputation_factor'] == rep_level) &
                (model_data['evidence_factor'] == 'B3')
            ]
            
            if len(b2_data) > 0 and len(b3_data) > 0:
                b2_rate = b2_data['reversal'].mean()
                b3_rate = b3_data['reversal'].mean()
                
                # Statistical test (chi-square or proportion test)
                from scipy.stats import chi2_contingency
                
                contingency = [
                    [b2_data['reversal'].sum(), len(b2_data) - b2_data['reversal'].sum()],
                    [b3_data['reversal'].sum(), len(b3_data) - b3_data['reversal'].sum()]
                ]
                
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                results.append({
                    'model_id': model_id,
                    'reputation_factor': rep_level,
                    'b2_reversal_rate': b2_rate,
                    'b3_reversal_rate': b3_rate,
                    'difference': b2_rate - b3_rate,
                    'chi2': chi2,
                    'p_value': p_value,
                    'n_b2': len(b2_data),
                    'n_b3': len(b3_data)
                })
    
    if results:
        placebo_df = pd.DataFrame(results)
        placebo_df.to_csv(output_dir / 'placebo_evidence_test.csv', index=False)
        print(f"Saved: {output_dir / 'placebo_evidence_test.csv'}")
        
        # Summary
        print("\nPlacebo Evidence Test Summary:")
        print("  B2 (relevant) vs B3 (irrelevant) evidence")
        for _, row in placebo_df.iterrows():
            sig = "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            print(f"  {row['model_id'][:30]:30s} {row['reputation_factor']}: "
                  f"Δ={row['difference']:+.3f} (p={row['p_value']:.4f}) {sig}")


def aep_stratification_analysis(df: pd.DataFrame, output_dir: Path):
    """
    AEP Ablation: Compare real-world vs virtualized entities
    """
    print("\nRunning AEP stratification analysis...")
    
    if 'is_aep' not in df.columns:
        print("Warning: No AEP metadata found, skipping")
        return
    
    # Split by AEP status
    real_world = df[~df['is_aep']]
    virtualized = df[df['is_aep']]
    
    print(f"  Real-world questions: {real_world['question_id'].nunique()}")
    print(f"  Virtualized questions: {virtualized['question_id'].nunique()}")
    
    # Compare reputation effects
    results = []
    
    for aep_status, subset, label in [
        (False, real_world, 'real_world'),
        (True, virtualized, 'virtualized')
    ]:
        for rep_level in ['A1', 'A3']:
            rep_data = subset[subset['reputation_factor'] == rep_level]
            if len(rep_data) > 0:
                results.append({
                    'entity_type': label,
                    'reputation_factor': rep_level,
                    'reversal_rate': rep_data['reversal'].mean(),
                    'n_trials': len(rep_data)
                })
    
    if results:
        aep_df = pd.DataFrame(results)
        aep_df.to_csv(output_dir / 'aep_stratification.csv', index=False)
        print(f"Saved: {output_dir / 'aep_stratification.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze BEAT-300 Results")
    parser.add_argument("--trials", type=str, default="results/trials.jsonl",
                      help="Path to trials JSONL file")
    parser.add_argument("--output-dir", type=str, default="results/analysis",
                      help="Output directory for analysis results")
    parser.add_argument("--run-placebo-test", action="store_true", default=True,
                      help="Run placebo evidence test")
    parser.add_argument("--run-aep-analysis", action="store_true", default=True,
                      help="Run AEP stratification analysis")
    
    args = parser.parse_args()
    
    trials_file = Path(args.trials)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading trial records...")
    df = load_trials(trials_file)
    print(f"Loaded {len(df)} trials")
    print(f"Models: {df['model_id'].nunique()}")
    print(f"Questions: {df['question_id'].nunique()}")
    
    # Filter valid trials (no truncation/format violations)
    df_valid = df[
        ~df["pass1_truncated"] &
        ~df["pass2_truncated"] &
        ~df["pass1_format_violation"] &
        ~df["pass2_format_violation"]
    ].copy()
    
    print(f"Valid trials: {len(df_valid)} ({100*len(df_valid)/len(df):.1f}%)")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generate_summary_statistics(df_valid, output_dir)
    
    # Drift-adjusted effects
    print("\nComputing drift-adjusted effects...")
    drift_adj = compute_drift_adjusted_effects(df_valid)
    drift_adj.to_csv(output_dir / "drift_adjusted_effects.csv", index=False)
    print(f"Saved: {output_dir / 'drift_adjusted_effects.csv'}")
    
    # Mechanistic taxonomy
    print("\nComputing mechanistic taxonomy...")
    taxonomy = compute_mechanistic_taxonomy(df_valid)
    if len(taxonomy) > 0:
        taxonomy.to_csv(output_dir / "mechanistic_taxonomy.csv", index=False)
        print(f"Saved: {output_dir / 'mechanistic_taxonomy.csv'}")
        
        # Category distribution
        category_summary = taxonomy.groupby(["model_id", "reputation_factor", "category"]).size().reset_index(name="count")
        category_summary.to_csv(output_dir / "taxonomy_summary.csv", index=False)
        print(f"Saved: {output_dir / 'taxonomy_summary.csv'}")
    else:
        print("No mechanistic data available")
    
    # SSI 2.0
    print("\nComputing SSI 2.0...")
    ssi_results = compute_ssi(df_valid)
    ssi_results["behavioral"].to_csv(output_dir / "ssi_behavioral.csv", index=False)
    print(f"Saved: {output_dir / 'ssi_behavioral.csv'}")
    
    if len(ssi_results["mechanistic"]) > 0:
        ssi_results["mechanistic"].to_csv(output_dir / "ssi_mechanistic.csv", index=False)
        print(f"Saved: {output_dir / 'ssi_mechanistic.csv'}")
    
    # Placebo evidence test
    if args.run_placebo_test:
        placebo_evidence_test(df_valid, output_dir)
    
    # AEP stratification
    if args.run_aep_analysis:
        aep_stratification_analysis(df_valid, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("\nKey findings:")
    print(f"  Overall reversal rate: {df_valid['reversal'].mean():.3f}")
    
    # Reputation effect (A3 vs A1 at B0)
    a3b0 = df_valid[
        (df_valid["reputation_factor"] == "A3") &
        (df_valid["evidence_factor"] == "B0")
    ]
    a1b0 = df_valid[
        (df_valid["reputation_factor"] == "A1") &
        (df_valid["evidence_factor"] == "B0")
    ]
    
    if len(a3b0) > 0 and len(a1b0) > 0:
        print(f"  Reversal rate A3/B0: {a3b0['reversal'].mean():.3f}")
        print(f"  Reversal rate A1/B0: {a1b0['reversal'].mean():.3f}")
        print(f"  Reputation effect: {a3b0['reversal'].mean() - a1b0['reversal'].mean():.3f}")
    
    print("="*60)


if __name__ == "__main__":
    main()

