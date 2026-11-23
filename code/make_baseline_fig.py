"""Baseline replication figure (Table 1): grouped bars of subject-level
ROC-AUC across architectures and evaluation protocols. Numbers come
directly from Table 1's own cells (recorded, not re-derived here) so the
figure always matches the numbers quoted in the manuscript text.
"""
import numpy as np
import matplotlib.pyplot as plt
from pub_style import (apply_style, rounded_gradient_bar, value_label,
                        reference_line, clean_spines, INDIGO, TEAL, MUTED)

apply_style()

ARCHS = ["VGG-16", "ResNet-50", "EfficientNet-B0"]
# (HandPD->Kaggle AUC, SD), (Kaggle->HandPD AUC, SD)
H2K = [(0.783, 0.006), (0.601, 0.029), (0.703, 0.049)]
K2H = [(0.666, 0.224), (0.656, 0.068), (0.647, 0.056)]
VERIFIED = [False, False, True]

fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)

for ax, data, title in zip(axes, [H2K, K2H], ["HandPD $\\rightarrow$ Kaggle", "Kaggle $\\rightarrow$ HandPD"]):
    x = np.arange(len(ARCHS))
    vals = [v for v, _ in data]
    errs = [e for _, e in data]
    for xi, v, e, verified in zip(x, vals, errs, VERIFIED):
        color = TEAL if verified else INDIGO
        rounded_gradient_bar(ax, xi, v, 0.56, color)
        ax.errorbar(xi, v, yerr=e, fmt="none", ecolor="#2B2E4A", elinewidth=1.4,
                    capsize=4, capthick=1.4, zorder=5)
        value_label(ax, xi, v + e, f"{v:.3f}", color="#2B2E4A", dy=0.018)
    reference_line(ax, 0.5, "chance" if title.startswith("Kaggle") else None,
                    x_label_pos=2.42)
    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS, fontsize=9.3)
    ax.set_title(title, pad=10)
    ax.set_ylim(0.3, 0.95)
    ax.set_xlim(-0.6, 2.6)
    clean_spines(ax)

axes[0].set_ylabel("Cross-cohort AUC", fontsize=10.5)

handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=TEAL),
           plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=INDIGO)]
fig.legend(handles, ["Independently verified (4 seeds)", "Original, unverified (5 seeds)"],
           loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig("baseline_fig.png", dpi=220)
plt.savefig("baseline_fig.pdf")
print("done")
