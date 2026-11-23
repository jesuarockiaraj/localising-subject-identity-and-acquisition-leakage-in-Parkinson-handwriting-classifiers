"""Positive-control summary figure (M3, Section~\\ref{sec:positive-control}):
background-mode disease decodability and honest-protocol (P2) AUC, raw-marked
vs. suppressed-marked HandPD spiral. Numbers come directly from
marker_probe.py console output and results_marker_2a/ (recorded numbers, not
re-derived here) so the figure always matches the numbers quoted in the text.
"""
import matplotlib.pyplot as plt
from pub_style import (apply_style, rounded_gradient_bar, value_label,
                        reference_line, clean_spines, AMBER, TEAL, MUTED)

apply_style()

fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))

# Panel 1: background-mode disease-decodability AUC (step a -> b)
ax = axes[0]
labels = ["Raw-marked", "Suppressed-\nmarked"]
vals = [0.991, 0.461]
colors = [AMBER, TEAL]
for i, (v, c) in enumerate(zip(vals, colors)):
    rounded_gradient_bar(ax, i, v, 0.5, c)
    value_label(ax, i, v, f"{v:.3f}", color="#2B2E4A", dy=0.02)
reference_line(ax, 0.5, "chance", x_label_pos=1.42)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9.3)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Background-mode disease AUC")
ax.set_title("Step (a) $\\to$ (b): probe", fontsize=11, pad=10)
clean_spines(ax)

# Panel 2: honest-protocol (P2, subject-grouped) AUC (step c)
ax = axes[1]
vals = [0.994, 0.972]
errs = [0.005, 0.016]
for i, (v, e, c) in enumerate(zip(vals, errs, colors)):
    rounded_gradient_bar(ax, i, v, 0.5, c, baseline=0.85)
    ax.errorbar(i, v, yerr=e, fmt="none", ecolor="#2B2E4A", elinewidth=1.4,
                capsize=4, capthick=1.4, zorder=5)
    value_label(ax, i, v + e, f"{v:.3f}", color="#2B2E4A", dy=0.004)
reference_line(ax, 0.944, "marker-free\nbaseline", x_label_pos=1.42)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9.3)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0.85, 1.03)
ax.set_ylabel("Honest-protocol AUC (P2)")
ax.set_title("Step (c): trained classifier", fontsize=11, pad=10)
clean_spines(ax)

plt.tight_layout()
plt.savefig("positive_control_fig.png", dpi=220)
plt.savefig("positive_control_fig.pdf")
print("done")
