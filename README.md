# Mechanistic Calibration or Social Compliance?

**Disentangling Reputation, Evidence, and Framing in LLM Decision Revision**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Code and data for ACL 2026. [[Paper]](#) [[DESIGN.md](DESIGN.md)] [[EXPERIMENTS.md](EXPERIMENTS.md)]

---

## Overview

When an LLM reverses its decision after observing an opposing opinion, is this genuine belief revision or superficial compliance? We present a controlled experimental framework using teacher-forced log-odds probing to classify revisions as:

- **Deep Revision**: Textual + internal preference change
- **Superficial Compliance**: Textual change without internal shift  
- **Latent Revision**: Internal shift without textual change
- **Stable**: No change

**BEAT-120 Benchmark**: 120 evidence-grounded questions at models' cognitive boundary ([4:6, 6:4] distributions) across Medicine, Science, and Law domains.

---

## Key Features

- **Evidence-grounded professional settings**: All questions include native evidence excerpts (biomedical abstracts, scientific papers, contract snippets)
- **Mechanistic probing**: Teacher-forced log-odds scoring distinguishes deep revision from superficial compliance
- **Factorial design**: 4×4×2 = 32 conditions with length-matched prompts (±2 tokens)
- **Domain-aware evidence add-ons**: Dataset-native rationales (PubMedQA long answers, SciFact rationale sentences, ContractNLI evidence spans)
- **Complete GLMM analysis**: Mixed-effects modeling with drift adjustment and random effects
- **Publication-ready figures**: 5 main figures (PDF 300 DPI) generated automatically
- **Pre-registered analysis**: Frozen artifacts and analysis plan before data collection
- **One-click reproduction**: Complete pipeline from data to figures

---

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/Mechanistic-Calibration.git
cd Mechanistic-Calibration

# Install Python dependencies
pip install -r requirements.txt

# Install R for GLMM analysis (Ubuntu/Debian)
sudo apt-get install r-base r-base-dev
Rscript -e "install.packages(c('lme4', 'emmeans', 'car', 'broom.mixed', 'dplyr', 'readr', 'jsonlite'), repos='https://cran.r-project.org')"

# Set up API keys
echo "OPENROUTER_API_KEY=your-key" > .env
```

**Requirements**: Python 3.9+, PyTorch 2.1+, transformers 4.35+, R 4.2+ (for GLMM)

---

## Quick Start

```bash
# 1. Verify setup
python verify_setup.py

# 2. Test run (1 question, 2 conditions)
python run_experiment.py --test-mode

# 3. Full experiment (requires BEAT-120 questions)
bash run_experiment.sh cuda core

# 4. Complete analysis (Python + R GLMM + Figures)
bash analyze_all.sh results/trials.jsonl results/analysis

# 5. Check results
ls figures/*.pdf                           # Figure 1-5
cat results/analysis/glmm_coefficients.csv # Table 2
```

**See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed instructions.**

---

## Experimental Design

### Data Sources (Per DESIGN.md §3.2)

BEAT-120 is built from three evidence-grounded professional datasets:

- **PubMedQA (Medicine)**: Biomedical research questions with abstract excerpts and labeled answers  
- **SciFact (Science)**: Scientific claims with research abstracts labeled as supporting or refuting  
- **ContractNLI (Law)**: Contractual inference where hypotheses are entailed or contradicted by contract excerpts

**Unified Task Interface**: All items normalized to binary Yes/No decisions with evidence-grounded instructions. Justifications must reference the provided excerpt (no external browsing/tools).

### Factorial Design (4 × 4 × 2 = 32 conditions)

| Factor | Levels | Description |
|--------|--------|-------------|
| **Reputation (A)** | A0–A3 | A0: Drift baseline (no source). A1: Anonymous. A2: Popularity cue. A3: Expertise cue. |
| **Evidence Add-on (B)** | B0–B3 | B0: Native excerpt only. B1: Weak (tautological). B2: Strong relevant (dataset-native rationales). B3: Strong irrelevant (placebo). |
| **Framing (C)** | C1–C2 | C1: Rational re-evaluation. C2: Deference framing. |

**B2 Evidence Instantiation (Domain-Specific)**:
- **Medicine**: PubMedQA long answers (conclusion-style text)
- **Science**: SciFact rationale sentences (annotated evidence)
- **Law**: ContractNLI evidence spans (contract provisions)

### Two-Pass Protocol

1. **Pass 1**: Model answers Yes/No + reasoning (grounded in native excerpt)
2. **Pass 2**: Re-present with opposing opinion (manipulated by A×B×C)

**Key design**: Pass 1 sampled once per (question, model, temp, replicate) and shared across all 32 Pass-2 conditions for within-subject comparison (per DESIGN.md §4.1).

### Models

- **Mechanistic probing** (local, with logits): Llama-3.1 (8B, 70B), Qwen2.5 (7B, 32B)
- **Behavioral anchor** (API): Claude 3.5 Sonnet, GPT-4o, Gemini Pro 1.5, Mistral Large
- **Reasoning case study** (minimal protocol): o1-preview, o3-mini vs GPT-4o-mini baseline

### Ablation Studies

**Built-in** (configurable in `config.json`):
- **Partial-disagreement**: Opponent disagrees at 50% rate (tests max-contrast sensitivity)
- **Authority-but-uncertain**: A3_fallible with explicit fallibility statement
- **AEP entity virtualization**: 50% questions use fictional entities (tests real-world prior dependence)

---

## Repository Structure

```
├── src/                          # Core modules
│   ├── mechanistic_probing.py    # Teacher-forced scoring
│   ├── experiment_runner.py      # Two-pass protocol
│   ├── prompt_utils.py           # Length matching (±2 tokens)
│   ├── aep_transformation.py     # Entity virtualization
│   ├── beat120_builder.py        # Question selection & boundary filtering
│   └── data_structures.py        # Data classes & logging
├── frozen_artifacts/             # Pre-registered materials
│   ├── beat120_questions.jsonl   # Benchmark (N=120)
│   ├── models.json               # Model registry
│   └── analysis_plan.md          # Statistical plan
├── prompts/registry.json         # A/B/C factor templates
├── scripts/collect_candidates.py # Build candidate pool from PubMedQA/SciFact/ContractNLI
├── run_experiment.py             # Main experiment
├── run_reasoning_case_study.py   # Reasoning models (o1, o3)
├── analyze_results.py            # Analysis pipeline
├── analyze_glmm.R                # R GLMM analysis
├── generate_figures.py           # Publication-ready figures
└── config.json                   # Configuration + ablations
```

---

## Generating BEAT-120 Candidate Pool

### Step 1: Collect Candidates

```bash
# Generate ~2K candidates from three sources (with evidence excerpts)
python scripts/collect_candidates.py --output data/candidate_pool.jsonl

# Expected output: ~1000 PubMedQA + ~300 SciFact + ~800 ContractNLI
```

**Quality control applied**:
- Binary format (Yes/No) with unambiguous supervision
- Evidence excerpt length >= 50 chars
- Label mapping: PubMedQA {yes, no} (drop maybe), SciFact {SUPPORTS, REFUTES} (drop NoInfo), ContractNLI {Entailment, Contradiction} (drop NotMentioned)
- Domain-specific B2 evidence extraction

### Step 2: Boundary Selection

```bash
# Filter to 120 questions at models' cognitive boundary
python -m src.beat120_builder \
    --candidates data/candidate_pool.jsonl \
    --target-count 120 \
    --probe-models meta-llama/Llama-3.1-70B-Instruct \
    --boundary-range 0.4 0.6

# Output: frozen_artifacts/beat120_questions.jsonl
```

**Boundary criteria** (DESIGN.md §3.3): Pass-1 distribution in [4:6, 6:4] (40-60% Yes rate) for at least one screening model, and not degenerate for the other.

**Stratified sampling**: Balanced across domains (Medicine/Science/Law) and difficulty-balanced via entropy bins.

---

## Advanced Usage

### Reasoning Case Study

```bash
# Minimal protocol: 24 highest-entropy questions, 4 conditions, T=0.0
python run_reasoning_case_study.py \
    --questions frozen_artifacts/beat120_questions.jsonl \
    --n-questions 24 \
    --models openai/o1-preview openai/o3-mini openai/gpt-4o-mini
```

### Ablations

Edit `config.json`:
```json
{
  "ablations": {
    "partial_disagreement": true,
    "disagreement_rate": 0.5,
    "authority_but_uncertain": true,
    "aep_entity_virtualization": true,
    "aep_ratio": 0.5
  }
}
```

---

## Data Availability

- **Code**: This repository (MIT License)
- **BEAT-120**: `frozen_artifacts/beat120_questions.jsonl` (sample: `beat120_questions_SAMPLE.jsonl`)
- **Candidate pool**: Generated from PubMedQA, SciFact, ContractNLI (all public datasets)
- **Experimental data**: Released upon paper acceptance
- **Models**: Llama (Meta), Qwen (Alibaba), API models via OpenRouter

---

## Cost Estimation

- **Local GPU**: ~50 GPU-hours on A100 (~$100-300 on cloud)
- **API calls**: ~$55-165 (depends on model mix)
- **Total**: ~$155-465 for complete experiment

---

## Citation

```bibtex
@inproceedings{mechanistic-calibration-2026,
  title={Mechanistic Calibration or Social Compliance? Disentangling Reputation, Evidence, and Framing in {LLM} Decision Revision},
  author={[Authors]},
  booktitle={Proceedings of ACL},
  year={2026}
}
```

---

## Documentation

- **[DESIGN.md](DESIGN.md)**: Complete research proposal and methodology
- **[EXPERIMENTS.md](EXPERIMENTS.md)**: Detailed experiment running guide
- **[frozen_artifacts/analysis_plan.md](frozen_artifacts/analysis_plan.md)**: Pre-registered statistical plan

---

**License**: MIT
