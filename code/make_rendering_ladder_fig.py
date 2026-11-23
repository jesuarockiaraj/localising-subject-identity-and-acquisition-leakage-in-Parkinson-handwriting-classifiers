"""Rendering-invariance ladder figure (Tables 4 and 6): line plot of cohort
ROC-AUC and subject re-ID@1 across the color/gray/uniform ladder, raw vs.
suppressed images side by side. Numbers come directly from Tables 4 and 6's
own cells (recorded, not re-derived here) so the figure always matches the
numbers quoted in the manuscript text.
"""
import matplotlib.pyplot as plt
from pub_style import apply_style, clean_spines, INDIGO, CORAL, MUTED

apply_style()

VIEWS = ["color", "gray", "uniform"]
RAW_REID = [0.263, 0.248, 0.221]
RAW_AUC = [1.000, 1.000, 1.000]
SUPP_REID = [0.236, 0.236, 0.221]
SUPP_AUC = [1.000, 1.000, 1.000]


def glow_line(ax, x, y, color, marker):
    ax.plot(x, y, color=color, linewidth=6, alpha=0.18, zorder=1, solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=2.6, marker=marker, markersize=9,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.6,
            zorder=3, solid_capstyle="round")


fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))

for ax, reid, auc, title in zip(axes, [RAW_REID, SUPP_REID], [RAW_AUC, SUPP_AUC],
                                 ["Raw images", "Suppressed images"]):
    glow_line(ax, VIEWS, auc, INDIGO, "o")
    glow_line(ax, VIEWS, reid, CORAL, "s")
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=11.5, pad=10)
    for v, y in zip(VIEWS, auc):
        ax.annotate(f"{y:.3f}", (v, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.8, color=INDIGO, fontweight="bold")
    for v, y in zip(VIEWS, reid):
        ax.annotate(f"{y:.3f}", (v, y), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=8.8, color=CORAL, fontweight="bold")
    clean_spines(ax)
    ax.grid(axis="y", visible=True)

axes[0].set_ylabel("Score")
handles = [plt.Line2D([0], [0], marker="o", linestyle="-", color=INDIGO, markersize=8,
                       markerfacecolor=INDIGO, markeredgecolor="white"),
           plt.Line2D([0], [0], marker="s", linestyle="-", color=CORAL, markersize=8,
                      markerfacecolor=CORAL, markeredgecolor="white")]
axes[0].legend(handles, ["Cohort ROC-AUC", "Subject re-ID@1"], loc="center left",
               fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig("rendering_ladder_fig.png", dpi=220)
plt.savefig("rendering_ladder_fig.pdf")
print("done")
