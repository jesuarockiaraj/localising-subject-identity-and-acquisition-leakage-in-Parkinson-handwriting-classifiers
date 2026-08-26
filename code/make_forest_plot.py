"""Forest plot of paired delta-AUC (raw vs. 2a-suppressed), EfficientNet-B0 only.

Point estimates and CIs come directly from compare_2a_vs_baseline.py console
output (recorded numbers, not re-derived here) so the figure always matches
the numbers quoted in the manuscript text.
"""
import matplotlib.pyplot as plt
import numpy as np

# label, mean dAUC, ci_lo, ci_hi
ROWS = [
    ("Cross-cohort\nHandPD$\\rightarrow$Kaggle (verified)", -0.040, -0.339, 0.221),
    ("Cross-cohort\nKaggle$\\rightarrow$HandPD (verified)", 0.017, -0.404, 0.309),
    ("Within-cohort\nHandPD spiral (verified)", -0.045, -0.134, 0.029),
]

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ys = np.arange(len(ROWS))[::-1]
for y, (label, mean, lo, hi) in zip(ys, ROWS):
    ax.plot([lo, hi], [y, y], color="#444", lw=1.6, zorder=1)
    ax.plot([lo, lo], [y - 0.08, y + 0.08], color="#444", lw=1.6, zorder=1)
    ax.plot([hi, hi], [y - 0.08, y + 0.08], color="#444", lw=1.6, zorder=1)
    ax.scatter([mean], [y], color="#c0392b" if mean < 0 else "#1a6b3c",
               s=60, zorder=2, edgecolor="black", linewidth=0.6)

ax.axvline(0, color="black", lw=1, linestyle="--", zorder=0)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xlabel(r"Paired $\Delta$AUC (suppressed $-$ raw), 95% CI")
ax.set_xlim(-0.45, 0.36)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
plt.savefig("auc_forest.png", dpi=200)
plt.savefig("auc_forest.pdf")
print("done")
