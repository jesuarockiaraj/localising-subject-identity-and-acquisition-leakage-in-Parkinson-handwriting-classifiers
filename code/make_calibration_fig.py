"""Calibration figure (Table 8): grouped bars of Brier score and ECE across
evaluation regimes. Numbers come directly from Table 8's own cells
(recorded, not re-derived here) so the figure always matches the numbers
quoted in the manuscript text.
"""
import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, rounded_gradient_bar, value_label, clean_spines, INDIGO, CORAL, MUTED

apply_style()

REGIMES = ["P1\n(image-level)", "P2\n(subject-\ngrouped)", "HandPD\n$\\rightarrow$Kaggle", "Kaggle\n$\\rightarrow$HandPD"]
BRIER = [(0.090, 0.003), (0.100, 0.009), (0.275, 0.017), (0.257, 0.029)]
ECE = [(0.141, 0.012), (0.106, 0.015), (0.239, 0.014), (0.330, 0.037)]

x = np.arange(len(REGIMES))
width = 0.34

fig, ax = plt.subplots(figsize=(7.6, 4.2))
b_vals, b_errs = zip(*BRIER)
e_vals, e_errs = zip(*ECE)

for xi, v, e in zip(x - width / 2, b_vals, b_errs):
    rounded_gradient_bar(ax, xi, v, width * 0.92, INDIGO, rounding=0.02)
    ax.errorbar(xi, v, yerr=e, fmt="none", ecolor="#2B2E4A", elinewidth=1.2, capsize=3, zorder=5)
    value_label(ax, xi, v + e, f"{v:.3f}", color="#2B2E4A", dy=0.006)
for xi, v, e in zip(x + width / 2, e_vals, e_errs):
    rounded_gradient_bar(ax, xi, v, width * 0.92, CORAL, rounding=0.02)
    ax.errorbar(xi, v, yerr=e, fmt="none", ecolor="#2B2E4A", elinewidth=1.2, capsize=3, zorder=5)
    value_label(ax, xi, v + e, f"{v:.3f}", color="#2B2E4A", dy=0.006)

ax.set_xticks(x)
ax.set_xticklabels(REGIMES, fontsize=9.3)
ax.set_ylabel("Score (lower is better)")
ax.set_xlim(-0.6, 3.6)
ax.set_ylim(0, 0.42)
ax.axvline(1.5, color=MUTED, lw=1, linestyle=(0, (1, 2)))
ax.text(0.5, 0.395, "within-cohort", ha="center", fontsize=8.5, color=MUTED, style="italic")
ax.text(2.5, 0.395, "cross-cohort", ha="center", fontsize=8.5, color=MUTED, style="italic")

handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=INDIGO),
           plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=CORAL)]
ax.legend(handles, ["Brier score", "ECE"], loc="upper left", fontsize=9.5, frameon=False)
clean_spines(ax)

plt.tight_layout()
plt.savefig("calibration_fig.png", dpi=220)
plt.savefig("calibration_fig.pdf")
print("done")
