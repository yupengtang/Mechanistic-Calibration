#!/bin/bash
# Complete Analysis Pipeline for BEAT-120
# Runs all analysis steps: Python preprocessing + R GLMM + Figure generation

set -e

echo "======================================================"
echo "BEAT-120 Complete Analysis Pipeline"
echo "======================================================"

# Parse arguments
TRIALS_FILE="${1:-results/trials.jsonl}"
OUTPUT_DIR="${2:-results/analysis}"

echo ""
echo "Configuration:"
echo "------------------------------------------------------"
echo "  Trials file: $TRIALS_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "------------------------------------------------------"

# Validate input
if [ ! -f "$TRIALS_FILE" ]; then
    echo "ERROR: Trials file not found: $TRIALS_FILE"
    exit 1
fi

# Step 1: Python analysis (preprocessing, taxonomy, SSI)
echo ""
echo "Step 1: Running Python analysis..."
echo "------------------------------------------------------"

python analyze_results.py \
    --trials "$TRIALS_FILE" \
    --output-dir "$OUTPUT_DIR"

# Convert JSONL to CSV for R (if needed)
echo ""
echo "Converting JSONL to CSV for R..."

python -c "
import json
import pandas as pd
from pathlib import Path

trials_file = Path('$TRIALS_FILE')
output_csv = Path('$OUTPUT_DIR') / 'trials_for_r.csv'

# Read JSONL
records = []
with open(trials_file, 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Flatten to DataFrame
rows = []
for r in records:
    row = {
        'trial_id': r['trial_id'],
        'question_id': r['question_id'],
        'domain': r['domain'],
        'model_id': r['model_id'],
        'temperature': r['temperature'],
        'top_p': r['top_p'],
        'replicate_id': r['replicate_id'],
        'reputation_factor': r['reputation_factor'],
        'evidence_factor': r['evidence_factor'],
        'framing_factor': r['framing_factor'],
        'reversal': int(r['reversal']),
        'pass1_truncated': r['pass1']['truncated'],
        'pass2_truncated': r['pass2']['truncated'],
        'pass1_format_violation': r['pass1']['format_violation'],
        'pass2_format_violation': r['pass2']['format_violation'],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f'Saved CSV to: {output_csv}')
"

# Step 2: R GLMM analysis
echo ""
echo "Step 2: Running R GLMM analysis..."
echo "------------------------------------------------------"

# Check if R and required packages are available
if command -v Rscript &> /dev/null; then
    Rscript analyze_glmm.R \
        "$OUTPUT_DIR/trials_for_r.csv" \
        "$OUTPUT_DIR"
else
    echo "WARNING: Rscript not found. Skipping GLMM analysis."
    echo "Install R and required packages (lme4, emmeans) to run GLMM."
fi

# Step 3: Generate figures
echo ""
echo "Step 3: Generating figures..."
echo "------------------------------------------------------"

python generate_figures.py \
    --analysis-dir "$OUTPUT_DIR" \
    --output-dir figures

echo ""
echo "======================================================"
echo "Analysis Pipeline Complete!"
echo "======================================================"
echo "Results:"
echo "  • Summary statistics: $OUTPUT_DIR/*.csv"
echo "  • GLMM results: $OUTPUT_DIR/glmm_*.csv"
echo "  • Figures: figures/*.pdf"
echo ""
echo "Key outputs:"
echo "  • Table 2 (GLMM coefficients): $OUTPUT_DIR/glmm_coefficients.csv"
echo "  • Figure 1-5: figures/figure*.pdf"
echo "======================================================"

