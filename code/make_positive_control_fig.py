"""Positive-control summary figure (M3, Section~\\ref{sec:positive-control}):
background-mode disease decodability and honest-protocol (P2) AUC, raw-marked
vs. suppressed-marked HandPD spiral. Numbers come directly from
marker_probe.py console output and results_marker_2a/ (recorded numbers, not
re-derived here) so the figure always matches the numbers quoted in the text.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

# Panel 1: background-mode disease-decodability AUC (step a -> b)
ax = axes[0]
labels = ["Raw-marked", "Suppressed-\nmarked"]
vals = [0.991, 0.461]
colors = ["#c0392b", "#1a6b3c"]
bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, color="#444", lw=1, linestyle="--")
ax.text(1.38, 0.5, "chance", va="center", ha="left", fontsize=8, color="#444")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Background-mode disease AUC")
ax.set_title("Step (a)$\\to$(b): probe", fontsize=10)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# Panel 2: honest-protocol (P2, subject-grouped) AUC (step c)
ax = axes[1]
labels = ["Raw-marked", "Suppressed-\nmarked"]
vals = [0.994, 0.972]
errs = [0.005, 0.016]
bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors, width=0.55,
              edgecolor="black", linewidth=0.6)
ax.axhline(0.944, color="#444", lw=1, linestyle="--")
ax.text(1.38, 0.944, "marker-free\nbaseline", va="center", ha="left", fontsize=8, color="#444")
ax.set_ylim(0.85, 1.05)
ax.set_ylabel("Honest-protocol AUC (P2)")
ax.set_title("Step (c): trained classifier", fontsize=10)
for b, v, e in zip(bars, vals, errs):
    ax.text(b.get_x() + b.get_width() / 2, v + e + 0.006, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("positive_control_fig.png", dpi=200)
plt.savefig("positive_control_fig.pdf")
print("done")
