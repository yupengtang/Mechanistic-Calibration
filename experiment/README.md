# Original experiment pipeline

This directory preserves the code and frozen materials used to run the original two-pass experiments. The paper's reported analyses should be reproduced from [`../reproducibility/`](../reproducibility/), which contains the saved outputs and the final analysis code.

## Setup

Use Python 3.10 or later:

```bash
cd experiment
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_setup.py
```

API experiments require `OPENROUTER_API_KEY` in the environment. Local-model experiments require the corresponding Hugging Face access, model weights, and accelerator environment. No credentials or model weights are included in the repository.

## Minimal check

The included sample can be used to check the experiment and analysis interfaces:

```bash
python run_experiment.py \
  --questions frozen_artifacts/beat120_questions_SAMPLE.jsonl \
  --output results/test_trials.jsonl \
  --test-mode \
  --device cpu

python analyze_results.py \
  --input results/test_trials.jsonl \
  --output results/test_analysis
```

See [`DESIGN.md`](DESIGN.md) and [`frozen_artifacts/analysis_plan.md`](frozen_artifacts/analysis_plan.md) for the original design and analysis plan. Cluster submission scripts are retained for provenance and may require site-specific paths or scheduler settings.
