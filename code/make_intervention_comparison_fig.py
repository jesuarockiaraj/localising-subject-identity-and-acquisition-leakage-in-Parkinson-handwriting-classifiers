"""All-interventions comparison figure (Section~\\ref{sec:future}): plain
baseline vs. background suppression vs. Deep CORAL, HandPD-to-Kaggle,
EfficientNet-B0, four verified seeds. Numbers come directly from
results_verify_cross_h2k_4seed/, results_2a_verify_cross_h2k_4seed/, and
coral_results_h2k.json (recorded numbers, not re-derived here) so the figure
always matches the numbers quoted in the text.
"""
import matplotlib.pyplot as plt
from pub_style import (apply_style, rounded_gradient_bar, value_label,
                        reference_line, clean_spines, INDIGO, TEAL, CORAL)

apply_style()

labels = ["Plain\nbaseline", "Background\nsuppression", "Deep\nCORAL"]
vals = [0.703, 0.662, 0.622]
errs = [0.049, 0.074, 0.049]
colors = [INDIGO, TEAL, CORAL]

fig, ax = plt.subplots(figsize=(5.8, 4.2))
for i, (v, e, c) in enumerate(zip(vals, errs, colors)):
    rounded_gradient_bar(ax, i, v, 0.55, c)
    ax.errorbar(i, v, yerr=e, fmt="none", ecolor="#2B2E4A", elinewidth=1.4,
                capsize=4, capthick=1.4, zorder=5)
    value_label(ax, i, v + e, f"{v:.3f}", color="#2B2E4A", dy=0.012)
reference_line(ax, 0.5, "chance", x_label_pos=2.42)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9.6)
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0.4, 0.85)
ax.set_ylabel("Cross-cohort AUC\n(HandPD $\\rightarrow$ Kaggle)")
clean_spines(ax)

plt.tight_layout()
plt.savefig("intervention_comparison_fig.png", dpi=220)
plt.savefig("intervention_comparison_fig.pdf")
print("done")
