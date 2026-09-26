"""Integrate recorded January and July cohorts without pooling their estimands."""
from pathlib import Path
import numpy as np
import pandas as pd
from newer_models import MODELS

ORIGINAL = ["GPT-4o-mini", "Claude-Sonnet-4", "Gemini-2.0-Flash", "Llama-3.1-8B", "Mistral-7B"]
ORDER = ORIGINAL + [name for _, name in MODELS]


def build(original, newer, new_summary, relative, output, tables):
    new = newer[newer.model_id.isin(dict(MODELS)) & newer.condition.isin(["A0", "A1", "A3"])].copy()
    new["model"] = new.model_id.map(dict(MODELS))
    columns = ["model", "question_id", "condition", "pass1_correct", "pass2_correct", "reversal"]
    df = pd.concat([original[columns], new[columns]], ignore_index=True)
    rows = []
    for name in ORDER:
        g = df[df.model == name]
        a3 = g[g.condition == "A3"]
        pair = g.pivot(index="question_id", columns="condition", values="reversal")[["A1", "A3"]].dropna().astype(float)
        row = dict(model=name, cohort="full_benchmark" if name in ORIGINAL else "250_item_controls",
                   n_paired=len(pair), pass1_acc=a3.pass1_correct.mean(),
                   delta_acc=100*(a3.pass2_correct.astype(int)-a3.pass1_correct.astype(int)).mean(),
                   A1=100*pair.A1.mean(), A3=100*pair.A3.mean())
        for c in ["A0", "A1", "A3"]:
            gc = g[g.condition == c]
            row[f"n_{c}"] = len(gc)
            for correct in [True, False]:
                subset = gc[gc.pass1_correct == correct]
                row[f"R_{c}_{int(correct)}"] = 100*subset.reversal.mean()
        row["A0"] = 100*g[g.condition == "A0"].reversal.mean()
        row["selectivity"] = row["R_A3_0"] / row["R_A3_1"]
        for key, mask in [("stable_correct", a3.pass1_correct & a3.pass2_correct),
                          ("wrong_correct", ~a3.pass1_correct & a3.pass2_correct),
                          ("correct_wrong", a3.pass1_correct & ~a3.pass2_correct),
                          ("stable_wrong", ~a3.pass1_correct & ~a3.pass2_correct)]:
            row[key] = 100*mask.mean()
        assert np.isclose(sum(row[k] for k in ["stable_correct", "wrong_correct", "correct_wrong", "stable_wrong"]), 100)
        if name in ORIGINAL:
            r = relative.set_index("model").loc[name]
            row.update(P=r.P_additive_pp, P_lo=r.P_paired_boot_ci_lo_pp, P_hi=r.P_paired_boot_ci_hi_pp,
                       RR=r.risk_ratio, RR_lo=r.RR_paired_boot_ci_lo, RR_hi=r.RR_paired_boot_ci_hi, p=r.mcnemar_p)
        else:
            from analyze_controls import mcnemar
            r = new_summary.set_index("model").loc[name]
            x, y = pair.A1.to_numpy(), pair.A3.to_numpy()
            rng = np.random.default_rng(int(r.prestige_seed)+1000)
            draws, odds_draws = [], []
            with np.errstate(divide="ignore", invalid="ignore"):
                for _ in range(50):
                    idx = rng.integers(0, len(pair), size=(1000, len(pair)))
                    bx, by = x[idx].mean(axis=1), y[idx].mean(axis=1)
                    draws.extend(by/bx)
                    odds_draws.extend((by/(1-by))/(bx/(1-bx)))
            draws = np.asarray(draws)
            # 0/0 is undefined; retain +infinity when only the denominator is zero.
            lo, hi = np.quantile(draws[~np.isnan(draws)], [.025, .975])
            odds_draws = np.asarray(odds_draws)
            olo, ohi = np.quantile(odds_draws[~np.isnan(odds_draws)], [.025, .975])
            row.update(P=r.prestige_pp, P_lo=r.prestige_ci_lo, P_hi=r.prestige_ci_hi,
                       RR=y.mean()/x.mean(), RR_lo=lo, RR_hi=hi,
                       RR_undefined_draws=int(np.isnan(draws).sum()), RR_seed=int(r.prestige_seed)+1000,
                       OR=(y.mean()/(1-y.mean()))/(x.mean()/(1-x.mean())), OR_lo=olo, OR_hi=ohi,
                       OR_undefined_draws=int(np.isnan(odds_draws).sum()),
                       p=mcnemar(int(((x==0)&(y==1)).sum()), int(((x==1)&(y==0)).sum())))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / "core_nine_models.csv", index=False)
    def write(name, content):
        (tables / name).write_text("\n".join(content) + "\n")
    subset = result[~result.model.isin(ORIGINAL)]
    write("newer_accounting_rows.tex", [f"{r.model} & {r.n_A0} & {r.n_A1} & {r.n_A3} & 0 \\\\" for r in subset.itertuples()])
    write("newer_accuracy_rows.tex", [
        f"{r.model} & {r.pass1_acc:.3f} & ${r.delta_acc:+.1f}$ & ${r.R_A1_1:.1f}\\to{r.R_A3_1:.1f}$ & ${r.R_A1_0:.1f}\\to{r.R_A3_0:.1f}$ & {r.selectivity:.2f} \\\\"
        for r in subset.itertuples()])
    def ptex(p):
        if p < .001:
            a, b = f"{p:.1e}".split("e")
            return f"${float(a):g}\\times10^{{{int(b)}}}$"
        return f"{p:.3f}"
    write("newer_reversal_rows.tex", [
        f"{r.model} & {r.A0:.1f} & {r.A1:.1f} & {r.A3:.1f} & ${r.P:+.1f}$ [{r.P_lo:.1f}, {r.P_hi:.1f}] & {r.RR:.2f} [{r.RR_lo:.2f}, {r.RR_hi:.2f}] & {ptex(r.p)} \\\\"
        for r in subset.itertuples()])
    return result


def plot(summary, directory):
    import matplotlib.pyplot as plt
    y = np.array([0, 1, 2, 3, 4, 5.3, 6.3, 7.3, 8.3])
    def decorate(ax):
        ax.set_ylim(8.9, -.7)
        ax.axhspan(4.8, 8.9, color="#eef4fa", zorder=0)
        ax.axhline(4.65, color=".55", linestyle="--", linewidth=.7)
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.subplots_adjust(left=.23, right=.98, top=.86, bottom=.15)
    decorate(ax)
    left = np.zeros(9)
    for key, label, color in [("stable_correct", "Stable correct", "#8bcf8f"),
                              ("wrong_correct", "Wrong → Correct", "#3b6fb6"),
                              ("correct_wrong", "Correct → Wrong", "#c73e3a"),
                              ("stable_wrong", "Stable wrong", "#c9c9c9")]:
        vals = summary[key].to_numpy()
        ax.barh(y, vals, left=left, height=.62, color=color, edgecolor="white", linewidth=.4, label=label)
        if key in ["wrong_correct", "correct_wrong"]:
            for yy, start, value in zip(y, left, vals):
                if value >= 3.0:
                    ax.text(start+value/2, yy, f"{value:.1f}", ha="center", va="center", color="white", fontsize=7)
        left += vals
    ax.set_yticks(y, summary.model, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of valid expert-labeled trials (%)", fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.01), fontsize=8)
    fig.savefig(directory / "transition_stacked_a3.pdf")
    plt.close(fig)
    fig, (bars, effects) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [2.1, 1]}, sharey=True)
    fig.subplots_adjust(left=.23, right=.98, top=.9, bottom=.16, wspace=.08)
    decorate(bars)
    # Separate group shading and line also span the paired-effect panel.
    effects.set_ylim(8.9, -.7)
    effects.axhspan(4.8, 8.9, color="#eef4fa", zorder=0)
    effects.axhline(4.65, color=".55", linestyle="--", linewidth=.7)
    for offset, key, label, color in [(-.22, "A0", "Drift", "#c6dbef"), (0, "A1", "Anon", "#6baed6"), (.22, "A3", "Expert", "#08519c")]:
        bars.barh(y+offset, summary[key], height=.21, label=label, color=color, edgecolor="black", linewidth=.2)
    bars.set_yticks(y, summary.model, fontsize=8)
    bars.set_xlim(0, 27)
    bars.set_xlabel("Reversal rate (%)", fontsize=9)
    bars.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.01), fontsize=8)
    effects.axvline(0, color=".4", linewidth=.8)
    effects.errorbar(summary.P, y, xerr=np.vstack([summary.P-summary.P_lo, summary.P_hi-summary.P]),
                     fmt="o", color="#08519c", markersize=3.5, capsize=2, linewidth=.8)
    effects.set_xlim(-5, 11)
    effects.set_xticks([-5, 0, 5, 10])
    effects.set_xlabel(r"$P=A_3-A_1$ (pp)", fontsize=9)
    effects.tick_params(axis="y", left=False, labelleft=False)
    effects.grid(axis="x", alpha=.16)
    effects.spines["left"].set_visible(False)
    effects.spines["top"].set_visible(False)
    effects.spines["right"].set_visible(False)
    fig.savefig(directory / "reversal_decomposition.pdf")
    plt.close(fig)
