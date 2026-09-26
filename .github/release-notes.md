This release accompanies the accepted NeurIPS 2026 paper **“When Expert Disagreement Hurts: Auditing Prestige-Sensitive Revision in LLM Decision Pipelines.”**

It contains:

- the NeurIPS camera-ready PDF;
- a preprint-style PDF;
- portable LaTeX source packages for both versions;
- the vector methodology figure and its editable PowerPoint source; and
- the saved-output reproducibility package used for the camera-ready analyses.

The reproducibility package regenerates the reported numerical analyses, tables, and figures without API credentials, paid model calls, or GPU access. The same files are browsable under [`reproducibility/`](https://github.com/yupengtang/when-expert-disagreement-hurts/tree/main/reproducibility), and the repository CI reruns the analysis.

All 1,989 original question–excerpt pairs are now recovered. The included input audit verifies the recovered inputs and corrects 96 Science question associations in an older reconstructed input file; original saved trial outputs are unchanged. Separately versioned datasets repair incomplete evidence coverage in 48 original Law excerpts and 16 control Law excerpts. Those repaired inputs require fresh evaluation and must not be paired with historical model outputs. See [`data-repair/`](https://github.com/yupengtang/when-expert-disagreement-hurts/tree/main/reproducibility/data-repair) for the data, provenance, per-item changes and offline regression tests.
