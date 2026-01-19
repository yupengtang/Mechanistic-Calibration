# BEAT-120 Experiment Running Guide

Complete guide for running the BEAT-120 experiments from start to finish.

---

## Prerequisites

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/Mechanistic-Calibration.git
cd Mechanistic-Calibration

# Install Python dependencies
pip install -r requirements.txt

# Set up API keys
echo "OPENROUTER_API_KEY=your-key-here" > .env
```

### 2. Install R (for GLMM analysis)

```bash
# Ubuntu/Debian
sudo apt-get install r-base r-base-dev

# macOS (with Homebrew)
brew install r

# Install required R packages
Rscript -e "install.packages(c('lme4', 'emmeans', 'car', 'broom.mixed', 'dplyr', 'readr', 'jsonlite'), repos='https://cran.r-project.org')"
```

### 3. GPU Setup (for local models)

Ensure CUDA is installed and PyTorch can access your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 1: Generate BEAT-120 Dataset

### Option A: Use Pre-generated Dataset (Recommended)

If you have the frozen `beat120_questions.jsonl`:

```bash
# Verify it exists
ls -lh frozen_artifacts/beat120_questions.jsonl

# Check question count
wc -l frozen_artifacts/beat120_questions.jsonl
# Should output: 120

# Check sample format
head -1 frozen_artifacts/beat120_questions_SAMPLE.jsonl | python -m json.tool
```

### Option B: Generate from Candidates

If you need to create BEAT-120 from scratch:

```bash
# 1. Collect candidate pool from PubMedQA, SciFact, ContractNLI
#    Per DESIGN.md §3.2: Evidence-grounded professional settings
python scripts/collect_candidates.py --output data/candidate_pool.jsonl

# Expected output: ~1000 PubMedQA + ~300 SciFact + ~800 ContractNLI
# All candidates include native evidence excerpts

# 2. Run boundary selection
python -m src.beat120_builder \
    --candidates data/candidate_pool.jsonl \
    --screening-models meta-llama/Llama-3.1-70B-Instruct openai/gpt-4o-mini \
    --num-samples 10 \
    --target-count 120 \
    --output frozen_artifacts/beat120_questions.jsonl \
    --selection-log frozen_artifacts/selection_log.jsonl

# This takes ~6-12 hours depending on GPU and API rate limits
# Per DESIGN.md §3.3: Retain questions with [4:6, 6:4] distribution
```

**Note**: All BEAT-120 questions include:
- **Evidence excerpt**: Native abstract/contract snippet
- **Binary decision**: Yes/No with evidence-grounded reasoning
- **Domain**: Medicine (PubMedQA), Science (SciFact), or Law (ContractNLI)
- **B2 evidence**: Dataset-native rationales for strong relevant add-on (Factor B2)

---

## Step 2: Run Core Experiment

### Quick Test (1 question, 2 conditions)

```bash
python run_experiment.py --test-mode
```

Expected output:
- 1 question × 1 model × 1 temp × 1 replicate × 2 conditions = 2 trials
- Saves to: `results/trials.jsonl`

### Full Core Experiment

```bash
# Core harness: T={0.0, 0.7}, R=3, all conditions
bash run_experiment.sh cuda core

# Or use Python directly
python run_experiment.py \
    --config config.json \
    --questions frozen_artifacts/beat120_questions.jsonl \
    --output results/trials.jsonl \
    --device cuda
```

**Expected runtime:**
- Local models (Llama 8B, Qwen 7B): ~8-12 hours on A100
- API models: ~2-4 hours (depends on rate limits)

**Expected output:**
- Total trials: 120 questions × 4 models × 2 temps × 3 reps × 32 conditions = **92,160 trials**
- File size: ~500-800 MB (JSONL)

### Extended Harness (Optional)

```bash
# Extended: T={0.0, 0.3, 0.7, 1.0}, top_p={0.9, 1.0}, R=5
bash run_experiment.sh cuda extended
```

---

## Step 3: Run Reasoning Case Study

```bash
# Minimal protocol: 24 questions, 4 conditions, o1/o3 models
python run_reasoning_case_study.py \
    --questions frozen_artifacts/beat120_questions.jsonl \
    --selection-log frozen_artifacts/selection_log.jsonl \
    --n-questions 24 \
    --models openai/o1-preview openai/o3-mini openai/gpt-4o-mini \
    --output results/reasoning_case_study.jsonl
```

**Expected runtime:** ~1-2 hours  
**Expected cost:** ~$5-15 (o1-preview is expensive)

---

## Step 4: Run Complete Analysis

```bash
# All-in-one analysis pipeline
bash analyze_all.sh results/trials.jsonl results/analysis
```

This runs:
1. **Python analysis** (`analyze_results.py`)
   - Summary statistics
   - Drift-adjusted effects
   - Mechanistic taxonomy
   - SSI 2.0 (behavioral + mechanistic)
   - Placebo evidence test
   - AEP stratification

2. **R GLMM analysis** (`analyze_glmm.R`)
   - Primary model: `logit(P_reversal) ~ ... + (1|Model) + (1|Question)`
   - Random-slope robustness checks
   - Estimated marginal means
   - Contrasts for key comparisons

3. **Figure generation** (`generate_figures.py`)
   - Figure 1: Condition effects
   - Figure 2: Mechanistic taxonomy
   - Figure 3: Placebo evidence test
   - Figure 4: Temperature/drift
   - Figure 5: SSI vs model size

**Output directory structure:**
```
results/analysis/
├── summary_by_condition.csv
├── summary_by_model.csv
├── drift_adjusted_effects.csv
├── mechanistic_taxonomy.csv
├── taxonomy_summary.csv
├── ssi_behavioral.csv
├── ssi_mechanistic.csv
├── placebo_evidence_test.csv
├── aep_stratification.csv
├── glmm_coefficients.csv          # Table 2
├── glmm_summary.txt
├── emmeans_*.csv
└── diagnostics.json

figures/
├── figure1_condition_effects.pdf
├── figure2_mechanistic_taxonomy.pdf
├── figure3_placebo_evidence.pdf
├── figure4_temperature_drift.pdf
└── figure5_ssi_scaling.pdf
```

---

## Step 5: Verify Results

### Check Data Quality

```bash
# Check trial counts
python -c "
import pandas as pd
df = pd.read_json('results/trials.jsonl', lines=True)
print(f'Total trials: {len(df)}')
print(f'Models: {df.model_id.nunique()}')
print(f'Questions: {df.question_id.nunique()}')
print(f'Valid trials: {(~df.pass1_truncated & ~df.pass2_truncated).sum()}')
"
```

### Inspect Key Results

```bash
# View GLMM coefficients (Table 2)
head -20 results/analysis/glmm_coefficients.csv

# View SSI scores
cat results/analysis/ssi_behavioral.csv

# View mechanistic taxonomy breakdown
head results/analysis/taxonomy_summary.csv
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Use smaller models or reduce batch size
# Edit config.json:
{
  "run_mechanistic_core": true,
  "run_mechanistic_extended": false  # Skip large models
}
```

### Issue: API Rate Limits

```bash
# Add delays in client.py
# Or run in batches:
python run_experiment.py --questions frozen_artifacts/beat120_questions_batch1.jsonl
python run_experiment.py --questions frozen_artifacts/beat120_questions_batch2.jsonl
```

### Issue: R GLMM Fails to Converge

```bash
# Check variance components
cat results/analysis/diagnostics.json

# Try different optimizer in analyze_glmm.R:
# control = glmerControl(optimizer = "nloptwrap")
```

### Issue: Missing Dependencies

```bash
# Reinstall all requirements
pip install -r requirements.txt --upgrade

# For R packages
Rscript -e "update.packages(ask=FALSE)"
```

---

## Cost Estimation

### Local Compute (GPU)

- **Core experiment**: ~50 GPU-hours on A100 (Llama 70B)
- **Extended**: ~150 GPU-hours
- **Cost estimate**: $100-300 on cloud GPU (AWS p4d, GCP A100)

### API Costs

- **Behavioral anchors**: ~$50-150 (depends on model mix)
- **Reasoning case study**: ~$5-15
- **Total API cost**: ~$55-165

### Total Estimated Cost

- **Full experiment** (local + API): $155-465
- **Core only** (skip extended): $105-315

---

## One-Click Reproduction

For complete end-to-end reproduction:

```bash
# Full pipeline (takes 12-24 hours)
bash run_full_pipeline.sh
```

This runs:
1. Validates frozen artifacts
2. Runs core experiment
3. Runs analysis
4. Generates figures

---

## Output for Paper

### Tables

- **Table 1**: Factor registry → `prompts/registry.json`
- **Table 2**: GLMM coefficients → `results/analysis/glmm_coefficients.csv`

### Figures

- **Figure 1-5**: `figures/figure*.pdf` (ready for submission)

### Supplementary Materials

- **Selection log**: `frozen_artifacts/selection_log.jsonl`
- **Full results**: `results/trials.jsonl`
- **Analysis scripts**: `analyze_*.py`, `analyze_glmm.R`

---

## Questions?

See `DESIGN.md` for detailed methodology or file a GitHub issue.



