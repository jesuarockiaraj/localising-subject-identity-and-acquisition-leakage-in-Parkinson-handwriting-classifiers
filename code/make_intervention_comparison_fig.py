"""All-interventions comparison figure (Section~\\ref{sec:future}): plain
baseline vs. background suppression vs. Deep CORAL, HandPD-to-Kaggle,
EfficientNet-B0, four verified seeds. Numbers come directly from
results_verify_cross_h2k_4seed/, results_2a_verify_cross_h2k_4seed/, and
coral_results_h2k.json (recorded numbers, not re-derived here) so the figure
always matches the numbers quoted in the text.
"""
import matplotlib.pyplot as plt
import numpy as np

labels = ["Plain\nbaseline", "Background\nsuppression", "Deep\nCORAL"]
vals = [0.703, 0.662, 0.622]
errs = [0.049, 0.074, 0.049]
colors = ["#2c6fbb", "#1a6b3c", "#c0392b"]

fig, ax = plt.subplots(figsize=(5.4, 3.6))
bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors, width=0.55,
              edgecolor="black", linewidth=0.6)
ax.axhline(0.5, color="#444", lw=1, linestyle="--")
ax.text(2.42, 0.5, "chance", va="center", ha="left", fontsize=8, color="#444")
ax.set_ylim(0.4, 0.85)
ax.set_ylabel("Cross-cohort AUC\n(HandPD$\\to$Kaggle)")
for b, v, e in zip(bars, vals, errs):
    ax.text(b.get_x() + b.get_width() / 2, v + e + 0.008, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("intervention_comparison_fig.png", dpi=200)
plt.savefig("intervention_comparison_fig.pdf")
print("done")
