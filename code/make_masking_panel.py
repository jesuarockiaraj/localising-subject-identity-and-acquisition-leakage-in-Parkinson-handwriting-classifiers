"""Build the masking-mode illustration figure for the Phase 2 manuscript.

Uses the SAME functions as leak_decomposition_probe.py / mech2_suppression.py
(get_trace_mask, standardize_resolution) so the figure matches the actual
pipeline, not a reimplementation.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mech2_suppression import get_trace_mask, standardize_resolution

SAMPLES = [
    ("HandPD (spiral)", "Datasets/HandPD/SpiralControl/0068-1.jpg"),
    ("Kaggle (spiral)", "Datasets/kaggle/drawings/spiral/testing/healthy/V01HE01.png"),
]
MODES = ["full", "background", "trace"]


def make_view(rgb, mode):
    if mode == "full":
        return rgb
    ink = get_trace_mask(rgb)
    out = rgb.copy()
    if mode == "trace":
        out[~ink] = 255
    elif mode == "background":
        out[ink] = 255
    return out


fig, axes = plt.subplots(len(SAMPLES), len(MODES), figsize=(7.5, 5.2))
for row, (label, path) in enumerate(SAMPLES):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = standardize_resolution(rgb, 512)
    for col, mode in enumerate(MODES):
        view = make_view(rgb, mode)
        ax = axes[row, col]
        ax.imshow(view)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(mode, fontsize=12)
        if col == 0:
            ax.set_ylabel(label, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#888")

plt.tight_layout()
plt.savefig("masking_panel.png", dpi=200)
plt.savefig("masking_panel.pdf")
print("done")
