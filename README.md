# When Expert Disagreement Hurts

**Auditing Prestige-Sensitive Revision in LLM Decision Pipelines**<br>
Yupeng Tang and Mingfeng Lin<br>
Accepted at NeurIPS 2026

[Project page](https://yupengtang.github.io/when-expert-disagreement-hurts/) ·
[Paper](https://yupengtang.github.io/when-expert-disagreement-hurts/assets/paper.pdf) ·
[Reproducibility artifact](https://github.com/yupengtang/when-expert-disagreement-hurts/releases/latest)

Language models often revise an answer after another agent disagrees. This repository accompanies a branch-paired audit that separates ordinary prompt drift, anonymous-disagreement sensitivity, and the incremental effect of describing the disagreeing source as an expert. Each comparison holds the question, evidence excerpt, and initial model answer fixed.

## Main findings

- The full audit covers 1,989 evidence-grounded binary decisions from medicine, scientific claim verification, and contract reasoning.
- Across the five full-benchmark models, disagreement produces reversals beyond neutral reconsideration. Four models have positive expert-label effects of 2.6–6.9 percentage points; one has a negative aggregate effect.
- Expert disagreement reduces final accuracy by 1.9–16.1 percentage points on the full benchmark. The paper therefore reports both Correct→Wrong and Wrong→Correct transitions rather than treating all reversals as errors.
- A 250-item evaluation adds GPT-5.6-terra, Grok-4.5, Gemini-3.6-Flash, and Claude-Opus-4.8. Their effects are heterogeneous and are not used for cross-generation rankings.
- In a no-padding identical-rationale control, the three original API models show positive source-label effects. Four additional endpoints have positive point estimates of 1.2–2.4 points, but no per-model exact test reaches the 0.05 level.
- Evidence gating is model- and prompt-dependent; it is not a uniform mitigation.

## Repository scope

This repository contains the original experiment pipeline and frozen materials. The camera-ready release additionally provides the exact saved-output artifact used to verify the final paper, including follow-up controls, analysis scripts, generated tables, and figure data.

The original full-benchmark input archive is incomplete: two Medicine excerpts and all 800 original Law excerpts are unavailable. The released saved outputs support numerical verification of the paper, but not a complete end-to-end regeneration of every original API response. This limitation is documented in the paper and artifact README.

## Reproduce the camera-ready results

Clone the repository and run the packaged offline analysis:

```bash
git clone https://github.com/yupengtang/when-expert-disagreement-hurts.git
cd when-expert-disagreement-hurts/reproducibility
python -m pip install -r requirements.txt
python reproduce.py
python make_figures.py
```

No API credentials, paid calls, or GPU access are required. The [continuous-integration workflow](.github/workflows/reproduce.yml) reruns the analysis and checks that the tracked numerical outputs do not change. The release tests additionally regenerate 67 CSV/LaTeX outputs byte for byte from the packaged records.

## Original experiment pipeline

The original pipeline supports local open-weight models and OpenRouter APIs. See [DESIGN.md](DESIGN.md) for the design and [frozen_artifacts/analysis_plan.md](frozen_artifacts/analysis_plan.md) for the frozen analysis plan.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_setup.py
```

Running new API experiments requires an `OPENROUTER_API_KEY`; local-model experiments require the corresponding model weights and accelerator environment. Do not use the historical pipeline to infer the exact camera-ready estimates without the release artifact, which contains the de-duplicated saved outputs and follow-up analyses.

## Paper and release files

| File | Purpose |
|---|---|
| `paper.pdf` | NeurIPS 2026 camera-ready paper |
| `when-expert-disagreement-hurts-reproducibility.zip` | Saved outputs and offline reproduction code |
| `neurips-2026-camera-ready-source.zip` | Portable LaTeX source for the final paper |
| `arxiv-source.tar.gz` | Preprint-mode LaTeX source |

These files are attached to the [camera-ready release](https://github.com/yupengtang/when-expert-disagreement-hurts/releases/latest).

## Citation

```bibtex
@inproceedings{tang2026expert,
  title     = {When Expert Disagreement Hurts: Auditing Prestige-Sensitive Revision in LLM Decision Pipelines},
  author    = {Tang, Yupeng and Lin, Mingfeng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License and asset terms

The repository's analysis code is released under the [MIT License](LICENSE). Dataset excerpts and model outputs remain subject to the terms of their original sources and providers. No model weights are redistributed. See the paper's asset-terms table and the artifact README for details.

## Contact

For questions about the repository or reproduction package, please open an issue or contact Yupeng Tang at `ytang454@gatech.edu`.
