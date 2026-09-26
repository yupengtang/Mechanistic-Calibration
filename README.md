# When Expert Disagreement Hurts

**Auditing Prestige-Sensitive Revision in LLM Decision Pipelines**<br>
Yupeng Tang and Mingfeng Lin · NeurIPS 2026

[Project page](https://yupengtang.github.io/when-expert-disagreement-hurts/) · [Paper](https://yupengtang.github.io/when-expert-disagreement-hurts/assets/paper.pdf) · [Release](https://github.com/yupengtang/when-expert-disagreement-hurts/releases/latest)

Language models often revise an answer after another agent disagrees. We use a branch-paired audit to separate ordinary reconsideration, sensitivity to anonymous disagreement, and the incremental effect of labeling the same disagreeing source as an expert. The question, evidence, and initial answer are held fixed across branches.

## Findings

- On the full 1,989-item benchmark, disagreement induces substantially more revision than neutral reconsideration.
- Expert labels have heterogeneous effects: four of five full-benchmark models show an additional 2.6–6.9 percentage-point reversal effect, while one shows a negative aggregate effect.
- Revisions are not uniformly harmful, so the analysis reports beneficial and harmful transitions separately. Expert disagreement lowers final accuracy by 1.9–16.1 points on the full benchmark.
- Results on four additional endpoints and an identical-rationale control confirm that the effect depends on the model and prompt protocol; they are not presented as a model ranking.

## Reproduce the results

The saved-output analysis runs on CPU and does not require API credentials:

```bash
git clone https://github.com/yupengtang/when-expert-disagreement-hurts.git
cd when-expert-disagreement-hurts/reproducibility
python -m pip install -r requirements.txt
python reproduce.py
python make_figures.py
```

The reproduction script regenerates 67 CSV and LaTeX outputs from the released records. The repository [CI workflow](.github/workflows/reproduce.yml) runs the same analysis and checks the tracked numerical outputs for changes.

The surviving records support the numerical results in the paper. They do not support a complete fresh generation of the original benchmark because two Medicine inputs and the 800 original Law inputs could not be recovered. The exact coverage and provenance are documented in [`reproducibility/README.md`](reproducibility/README.md).

## Repository structure

| Path | Contents |
|---|---|
| [`reproducibility/`](reproducibility/) | Saved outputs, analysis code, generated tables, and figure data used in the paper |
| [`experiment/`](experiment/) | Original experiment pipeline and frozen study materials |
| [`paper/`](paper/) | Camera-ready and arXiv source archives, including the editable Figure 1 source |
| [`docs/`](docs/) | Project-page source and paper figures |

The [camera-ready release](https://github.com/yupengtang/when-expert-disagreement-hurts/releases/latest) includes the paper, source archives, and a standalone reproduction package.

## Citation

```bibtex
@inproceedings{tang2026expert,
  title     = {When Expert Disagreement Hurts: Auditing Prestige-Sensitive Revision in LLM Decision Pipelines},
  author    = {Tang, Yupeng and Lin, Mingfeng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The analysis code is released under the [MIT License](LICENSE). Dataset excerpts and model outputs remain subject to their original source and provider terms; no model weights are distributed. For questions, open an issue or contact Yupeng Tang at `ytang454@gatech.edu`.
