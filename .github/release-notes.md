This release accompanies the accepted NeurIPS 2026 paper **“When Expert Disagreement Hurts: Auditing Prestige-Sensitive Revision in LLM Decision Pipelines.”**

It contains:

- the NeurIPS camera-ready PDF;
- a preprint-style PDF;
- portable LaTeX source packages for both versions; and
- the saved-output reproducibility package used for the camera-ready analyses.

The reproducibility package regenerates the reported numerical analyses, tables, and figures without API credentials, paid model calls, or GPU access. The same files are browsable under [`reproducibility/`](https://github.com/yupengtang/when-expert-disagreement-hurts/tree/main/reproducibility), and the repository CI reruns the analysis.

The original full-benchmark input archive is incomplete: two Medicine excerpts and all 800 original Law excerpts are unavailable. Saved trial outputs are complete for numerical verification. See the paper and artifact README for the precise reproduction boundary.
