"""Forest plot of paired delta-AUC (raw vs. 2a-suppressed), EfficientNet-B0 only.

Point estimates and CIs come directly from compare_2a_vs_baseline.py console
output (recorded numbers, not re-derived here) so the figure always matches
the numbers quoted in the manuscript text.
"""
import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, MUTED, TEAL, CORAL

apply_style()

# label, mean dAUC, ci_lo, ci_hi
ROWS = [
    ("Cross-cohort\nHandPD$\\rightarrow$Kaggle (verified)", -0.040, -0.339, 0.221),
    ("Cross-cohort\nKaggle$\\rightarrow$HandPD (verified)", 0.017, -0.404, 0.309),
    ("Within-cohort\nHandPD spiral (verified)", -0.045, -0.134, 0.029),
]

fig, ax = plt.subplots(figsize=(6.6, 3.7))
ys = np.arange(len(ROWS))[::-1]

for i, (y, (label, mean, lo, hi)) in enumerate(zip(ys, ROWS)):
    if i % 2 == 0:
        ax.axhspan(y - 0.42, y + 0.42, color="#F4F6FB", zorder=0)
    ax.plot([lo, hi], [y, y], color=MUTED, lw=2.4, zorder=1, solid_capstyle="round")
    ax.plot([lo, lo], [y - 0.1, y + 0.1], color=MUTED, lw=2.2, zorder=1, solid_capstyle="round")
    ax.plot([hi, hi], [y - 0.1, y + 0.1], color=MUTED, lw=2.2, zorder=1, solid_capstyle="round")
    pt_color = CORAL if mean < 0 else TEAL
    ax.scatter([mean], [y], color=pt_color, s=130, zorder=3, edgecolor="white", linewidth=1.8)

ax.axvline(0, color="#2B2E4A", lw=1.2, linestyle=(0, (4, 3)), zorder=0)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9.6)
ax.set_xlabel(r"Paired $\Delta$AUC (suppressed $-$ raw), 95% CI", fontsize=10)
ax.set_xlim(-0.45, 0.36)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)

plt.tight_layout()
plt.savefig("auc_forest.png", dpi=220)
plt.savefig("auc_forest.pdf")
print("done")
