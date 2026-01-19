#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure Generation for BEAT-120 Experimental Results

Implements all 5 main figures from DESIGN.md §9:
- Figure 1: Condition effects on reversal
- Figure 2: Mechanistic taxonomy decomposition
- Figure 3: Placebo evidence test
- Figure 4: Temperature/drift interaction
- Figure 5: SSI vs model size
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

# Color schemes
COLORS_REPUTATION = {
    'A0': '#999999',  # Gray (drift baseline)
    'A1': '#4DAF4A',  # Green (anonymous)
    'A2': '#377EB8',  # Blue (popularity)
    'A3': '#E41A1C',  # Red (expertise)
}

COLORS_TAXONOMY = {
    'Deep': '#2ca02c',
    'Superficial': '#ff7f0e',
    'Latent': '#9467bd',
    'Stable': '#7f7f7f'
}


def load_analysis_results(analysis_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all analysis CSVs into a dictionary"""
    results = {}
    
    files_to_load = [
        'drift_adjusted_effects.csv',
        'mechanistic_taxonomy.csv',
        'taxonomy_summary.csv',
        'ssi_behavioral.csv',
        'ssi_mechanistic.csv',
        'summary_by_condition.csv',
        'summary_by_model.csv',
        'emmeans_reputation_x_evidence.csv',
        'glmm_coefficients.csv'
    ]
    
    for filename in files_to_load:
        filepath = analysis_dir / filename
        if filepath.exists():
            key = filename.replace('.csv', '')
            results[key] = pd.read_csv(filepath)
            print(f"Loaded: {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    return results


def figure1_condition_effects(data: Dict[str, pd.DataFrame], output_path: Path):
    """
    Figure 1: Condition Effects on Reversal
    
    Estimated marginal reversal probability with 95% CI for each (A × B) under C1,
    drift-adjusted vs A0.
    """
    print("\nGenerating Figure 1: Condition effects on reversal...")
    
    # Use emmeans results if available, otherwise compute from summary
    if 'emmeans_reputation_x_evidence' in data:
        df = data['emmeans_reputation_x_evidence'].copy()
    else:
        # Fallback: compute from summary
        df = data['drift_adjusted_effects'].copy()
        df = df[df['framing_factor'] == 'C1']  # Focus on rational framing
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle('Effect of Reputation and Evidence on Decision Reversal (C1: Rational Framing)',
                 fontsize=14, fontweight='bold')
    
    evidence_levels = ['B0', 'B1', 'B2', 'B3']
    evidence_labels = ['No Evidence', 'Weak Evidence', 'Strong Relevant', 'Strong Irrelevant']
    
    for i, (evid, label) in enumerate(zip(evidence_levels, evidence_labels)):
        ax = axes[i]
        
        # Filter data for this evidence level
        if 'evidence_factor' in df.columns:
            subset = df[df['evidence_factor'] == evid]
        else:
            subset = df  # Use all data if not stratified
        
        # Plot drift-adjusted rates
        x_pos = np.arange(len(['A0', 'A1', 'A2', 'A3']))
        
        if 'drift_adjusted_rate' in subset.columns:
            rates = subset.groupby('reputation_factor')['drift_adjusted_rate'].mean()
            errors = subset.groupby('reputation_factor')['drift_adjusted_rate'].std() / np.sqrt(
                subset.groupby('reputation_factor').size()
            )
        else:
            # Fallback: use observed reversal rate
            rates = subset.groupby('reputation_factor')['observed_reversal_rate'].mean()
            errors = subset.groupby('reputation_factor')['observed_reversal_rate'].std() / np.sqrt(
                subset.groupby('reputation_factor').size()
            )
        
        colors = [COLORS_REPUTATION[level] for level in rates.index]
        
        bars = ax.bar(x_pos, rates.values, yerr=errors.values, 
                     color=colors, alpha=0.8, capsize=5)
        
        # Styling
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Reputation Factor', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['A0\n(Drift)', 'A1\n(Anon)', 'A2\n(Pop)', 'A3\n(Expert)'],
                          fontsize=9)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        if i == 0:
            ax.set_ylabel('Drift-Adjusted Reversal Rate', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_condition_effects.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure1_condition_effects.png', dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1 to {output_path}")
    plt.close()


def figure2_mechanistic_taxonomy(data: Dict[str, pd.DataFrame], output_path: Path):
    """
    Figure 2: Mechanistic Taxonomy Decomposition
    
    Stacked breakdown per condition and model family, highlighting 
    "shadow compliance" vs true belief revision.
    """
    print("\nGenerating Figure 2: Mechanistic taxonomy decomposition...")
    
    if 'taxonomy_summary' not in data:
        print("Warning: taxonomy_summary.csv not found, skipping Figure 2")
        return
    
    df = data['taxonomy_summary'].copy()
    
    # Pivot for stacked bar chart
    pivot = df.pivot_table(
        index=['model_id', 'reputation_factor'],
        columns='category',
        values='count',
        fill_value=0
    )
    
    # Normalize to proportions
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    
    # Group by model family
    pivot_pct['family'] = pivot_pct.index.get_level_values(0).str.split('/').str[0]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mechanistic Revision Taxonomy by Condition and Model Family',
                 fontsize=14, fontweight='bold')
    
    reputation_levels = ['A0', 'A1', 'A2', 'A3']
    categories = ['Deep', 'Superficial', 'Latent', 'Stable']
    
    for idx, rep_level in enumerate(reputation_levels):
        ax = axes[idx // 2, idx % 2]
        
        # Filter data
        subset = pivot_pct[pivot_pct.index.get_level_values(1) == rep_level]
        
        if len(subset) == 0:
            continue
        
        # Prepare data for stacking
        x = np.arange(len(subset))
        bottom = np.zeros(len(subset))
        
        for category in categories:
            if category in subset.columns:
                values = subset[category].values
                ax.bar(x, values, bottom=bottom, 
                      label=category,
                      color=COLORS_TAXONOMY.get(category, '#cccccc'),
                      alpha=0.85)
                bottom += values
        
        # Styling
        ax.set_title(f'{rep_level}: {rep_level.replace("A", "Reputation ")}',
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(subset.index.get_level_values(0), rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure2_mechanistic_taxonomy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure2_mechanistic_taxonomy.png', dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2 to {output_path}")
    plt.close()


def figure3_placebo_evidence(data: Dict[str, pd.DataFrame], output_path: Path):
    """
    Figure 3: Placebo Evidence Test (Content vs Form)
    
    Contrast B2 (strong relevant) vs B3 (strong irrelevant) under A1/A3.
    """
    print("\nGenerating Figure 3: Placebo evidence test...")
    
    if 'drift_adjusted_effects' not in data:
        print("Warning: drift_adjusted_effects.csv not found, skipping Figure 3")
        return
    
    df = data['drift_adjusted_effects'].copy()
    
    # Filter to B2 and B3
    df = df[df['evidence_factor'].isin(['B2', 'B3'])]
    df = df[df['reputation_factor'].isin(['A1', 'A3'])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot grouped bar chart
    x = np.arange(2)  # A1, A3
    width = 0.35
    
    reputation_levels = ['A1', 'A3']
    
    b2_means = []
    b2_errors = []
    b3_means = []
    b3_errors = []
    
    for rep in reputation_levels:
        b2_data = df[(df['reputation_factor'] == rep) & (df['evidence_factor'] == 'B2')]
        b3_data = df[(df['reputation_factor'] == rep) & (df['evidence_factor'] == 'B3')]
        
        b2_means.append(b2_data['drift_adjusted_rate'].mean())
        b2_errors.append(b2_data['drift_adjusted_rate'].std() / np.sqrt(len(b2_data)))
        
        b3_means.append(b3_data['drift_adjusted_rate'].mean())
        b3_errors.append(b3_data['drift_adjusted_rate'].std() / np.sqrt(len(b3_data)))
    
    bars1 = ax.bar(x - width/2, b2_means, width, yerr=b2_errors,
                   label='B2: Strong Relevant Evidence', 
                   color='#2ca02c', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, b3_means, width, yerr=b3_errors,
                   label='B3: Strong Irrelevant Evidence (Placebo)',
                   color='#ff7f0e', alpha=0.8, capsize=5)
    
    # Add significance stars if difference is substantial
    for i in range(2):
        diff = abs(b2_means[i] - b3_means[i])
        if diff > 0.05:  # Threshold for "substantial"
            y_pos = max(b2_means[i], b3_means[i]) + max(b2_errors[i], b3_errors[i]) + 0.02
            ax.text(x[i], y_pos, '**' if diff > 0.10 else '*', 
                   ha='center', fontsize=16, fontweight='bold')
    
    # Styling
    ax.set_title('Placebo Evidence Test: Content vs Format', 
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Drift-Adjusted Reversal Rate', fontsize=11)
    ax.set_xlabel('Reputation Factor', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['A1 (Anonymous)', 'A3 (Expertise)'], fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure3_placebo_evidence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure3_placebo_evidence.png', dpi=300, bbox_inches='tight')
    print(f"Saved Figure 3 to {output_path}")
    plt.close()


def figure4_temperature_drift(data: Dict[str, pd.DataFrame], output_path: Path):
    """
    Figure 4: Temperature and Drift Interaction
    
    Reversal vs temperature, showing A0 drift baseline and A3/B0 contrast.
    """
    print("\nGenerating Figure 4: Temperature/drift interaction...")
    
    if 'summary_by_condition' not in data:
        print("Warning: summary_by_condition.csv not found, skipping Figure 4")
        return
    
    df = data['summary_by_condition'].copy()
    
    # Focus on B0 (no evidence) and C1 (rational framing)
    df = df[df['evidence'] == 'B0']
    df = df[df['framing'] == 'C1']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Temperature values (you may need to extract this from data)
    temps = sorted(df['temperature'].unique()) if 'temperature' in df.columns else [0.0, 0.7]
    
    # Plot lines for each reputation level
    for rep_level in ['A0', 'A1', 'A3']:
        subset = df[df['reputation'] == rep_level]
        
        if len(subset) == 0:
            continue
        
        # Group by temperature
        temp_groups = subset.groupby('temperature')['reversal_rate'].agg(['mean', 'sem'])
        
        ax.plot(temp_groups.index, temp_groups['mean'], 
               marker='o', linewidth=2, markersize=8,
               label=rep_level, color=COLORS_REPUTATION[rep_level])
        ax.fill_between(temp_groups.index,
                       temp_groups['mean'] - 1.96 * temp_groups['sem'],
                       temp_groups['mean'] + 1.96 * temp_groups['sem'],
                       alpha=0.2, color=COLORS_REPUTATION[rep_level])
    
    # Styling
    ax.set_title('Temperature Effects on Reversal: Drift Amplification',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Temperature', fontsize=11)
    ax.set_ylabel('Reversal Rate', fontsize=11)
    ax.legend(fontsize=10, title='Reputation Factor', title_fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.1, max(temps) + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure4_temperature_drift.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure4_temperature_drift.png', dpi=300, bbox_inches='tight')
    print(f"Saved Figure 4 to {output_path}")
    plt.close()


def figure5_ssi_scaling(data: Dict[str, pd.DataFrame], output_path: Path):
    """
    Figure 5: SSI vs Model Size
    
    SSI (behavioral + mechanistic on open models) vs parameter scale 
    across Llama and Qwen families.
    """
    print("\nGenerating Figure 5: SSI vs model size...")
    
    if 'ssi_behavioral' not in data:
        print("Warning: ssi_behavioral.csv not found, skipping Figure 5")
        return
    
    ssi_beh = data['ssi_behavioral'].copy()
    ssi_mech = data.get('ssi_mechanistic', pd.DataFrame())
    
    # Extract model sizes
    def extract_model_size(model_id):
        """Extract parameter count from model ID"""
        if 'Llama-3.1-8B' in model_id:
            return 8
        elif 'Llama-3.1-70B' in model_id:
            return 70
        elif 'Llama-3.1-405B' in model_id:
            return 405
        elif 'Qwen2.5-7B' in model_id:
            return 7
        elif 'Qwen2.5-32B' in model_id:
            return 32
        elif 'Qwen2.5-72B' in model_id:
            return 72
        else:
            return None
    
    def extract_family(model_id):
        """Extract model family"""
        if 'Llama' in model_id:
            return 'Llama'
        elif 'Qwen' in model_id:
            return 'Qwen'
        else:
            return 'Other'
    
    ssi_beh['size_b'] = ssi_beh['model_id'].apply(extract_model_size)
    ssi_beh['family'] = ssi_beh['model_id'].apply(extract_family)
    
    if len(ssi_mech) > 0:
        ssi_mech['size_b'] = ssi_mech['model_id'].apply(extract_model_size)
        ssi_mech['family'] = ssi_mech['model_id'].apply(extract_family)
    
    # Filter to known sizes
    ssi_beh = ssi_beh[ssi_beh['size_b'].notna()]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Behavioral SSI
    ax = axes[0]
    for family in ['Llama', 'Qwen']:
        subset = ssi_beh[ssi_beh['family'] == family].sort_values('size_b')
        if len(subset) > 0:
            ax.plot(subset['size_b'], subset['SSI_behavioral'],
                   marker='o', linewidth=2, markersize=10,
                   label=family)
    
    ax.set_title('Behavioral SSI vs Model Size', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=11)
    ax.set_ylabel('SSI (Behavioral)', fontsize=11)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Panel B: Mechanistic SSI
    ax = axes[1]
    if len(ssi_mech) > 0:
        ssi_mech_filtered = ssi_mech[ssi_mech['size_b'].notna()]
        for family in ['Llama', 'Qwen']:
            subset = ssi_mech_filtered[ssi_mech_filtered['family'] == family].sort_values('size_b')
            if len(subset) > 0:
                ax.plot(subset['size_b'], subset['SSI_mechanistic'],
                       marker='s', linewidth=2, markersize=10,
                       label=family)
        
        ax.set_title('Mechanistic SSI vs Model Size', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=11)
        ax.set_ylabel('SSI (Mechanistic)', fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'Mechanistic SSI\nData Not Available',
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure5_ssi_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure5_ssi_scaling.png', dpi=300, bbox_inches='tight')
    print(f"Saved Figure 5 to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate all figures for BEAT-120 paper")
    parser.add_argument("--analysis-dir", type=str, default="results/analysis",
                       help="Directory containing analysis results")
    parser.add_argument("--output-dir", type=str, default="figures",
                       help="Output directory for figures")
    parser.add_argument("--figures", type=str, nargs="+", 
                       default=["1", "2", "3", "4", "5"],
                       help="Which figures to generate (1-5)")
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BEAT-120 Figure Generation")
    print("=" * 60)
    print(f"Analysis directory: {analysis_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Figures to generate: {', '.join(args.figures)}")
    print("=" * 60)
    
    # Load all analysis results
    data = load_analysis_results(analysis_dir)
    
    # Generate requested figures
    if "1" in args.figures:
        figure1_condition_effects(data, output_dir)
    
    if "2" in args.figures:
        figure2_mechanistic_taxonomy(data, output_dir)
    
    if "3" in args.figures:
        figure3_placebo_evidence(data, output_dir)
    
    if "4" in args.figures:
        figure4_temperature_drift(data, output_dir)
    
    if "5" in args.figures:
        figure5_ssi_scaling(data, output_dir)
    
    print("\n" + "=" * 60)
    print("Figure Generation Complete")
    print("=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("  • PDF (publication quality)")
    print("  • PNG (for preview)")
    print("=" * 60)


if __name__ == "__main__":
    main()

