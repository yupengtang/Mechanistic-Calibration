"""Regenerate the paper's numeric figures with embedded TrueType fonts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plot_source
import newer_models
import core_display
import clean_display

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})
plot_source.FIGS = FIGS
data = pd.read_csv(ROOT / "rebuttal/data/recovered/neurips_dedup_trials_FULL.csv")
effects = pd.read_csv(ROOT / "rebuttal/analysis/relative_effects.csv").set_index("model")


def interval(_data, model):
    row = effects.loc[model]
    return row.P_additive_pp, row.P_paired_boot_ci_lo_pp, row.P_paired_boot_ci_hi_pp


plot_source.bootstrap_p_interval = interval
core_display.plot(pd.read_csv(ROOT / "rebuttal/analysis/core_nine_models.csv"), FIGS)
newer_models.plot(pd.read_csv(ROOT / "rebuttal/analysis/newer_models_figure.csv"),
                  FIGS / "newer_model_replication.pdf")
clean_display.plot(pd.read_csv(ROOT / "rebuttal/analysis/clean_seven_models.csv"),
                   FIGS / "clean_rationale_controls.pdf")
plt.close("all")
print("Regenerated figures with TrueType fonts and the same paired CIs as the paper table.")
