"""Shared modern visual style for this study's result figures: a consistent
palette, typography, and a reusable gradient-filled rounded-bar primitive,
so every chart in the manuscript reads as one coherent system rather than
default-matplotlib one-offs. Applies to appearance only -- no figure that
imports this module re-derives or alters any underlying number.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

# ---- palette (colorblind-considerate, print-safe; consistent role per hue) ----
INDIGO = "#4361EE"   # raw / plain baseline / primary series
TEAL = "#06A77D"     # suppressed / treatment / "verified"
CORAL = "#F3722C"    # third condition (CORAL, ECE, secondary metric)
AMBER = "#F9A826"    # warning / marker / raw-marked
VIOLET = "#7209B7"   # accent, used sparingly
INK = "#1B1F3B"      # text / axis ink
GRID = "#E3E6F0"     # soft gridlines
MUTED = "#8891A7"    # secondary / reference lines


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.titlesize": 11.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _lighten(hex_color, amount=0.72):
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def rounded_gradient_bar(ax, x, height, width, color, baseline=0.0,
                          rounding=0.035, zorder=2):
    """Draw one vertical bar as a rounded, top-lit gradient fill (light tint
    at the base to the full hue at the top), a modern dashboard look, while
    keeping the bar's true data height exact (the rounding is a cosmetic
    corner radius, not a value distortion)."""
    y0, y1 = (baseline, height) if height >= baseline else (height, baseline)
    h = y1 - y0
    if h <= 0:
        return None
    x0 = x - width / 2
    patch = mpatches.FancyBboxPatch(
        (x0, y0), width, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=0.9, edgecolor=color, facecolor="none", zorder=zorder + 1,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    cmap = LinearSegmentedColormap.from_list("bar_grad", [_lighten(color, 0.55), color])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[x0, x0 + width, y0, y1], origin="lower",
                    aspect="auto", cmap=cmap, zorder=zorder)
    im.set_clip_path(patch)
    return patch


def value_label(ax, x, y, text, color=INK, dy=0.0, **kw):
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=9.2,
            fontweight="bold", color=color, **kw)


def reference_line(ax, y, label=None, x_label_pos=None):
    ax.axhline(y, color=MUTED, lw=1.1, linestyle=(0, (4, 3)), zorder=1)
    if label and x_label_pos is not None:
        ax.text(x_label_pos, y, label, va="center", ha="left", fontsize=8.3,
                color=MUTED, style="italic", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


def clean_spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
