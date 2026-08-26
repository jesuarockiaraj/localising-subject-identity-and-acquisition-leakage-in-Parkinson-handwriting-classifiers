"""Worked-example figure for the manuscript: one PD image, three masking
modes, with the trained classifier's own P(PD) annotated under each panel.
Reuses demo_predict.py's exact preprocessing/inference path.
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from mech2_suppression import get_trace_mask, standardize_resolution
from pd_loso_pipeline_cross_dataset import build_model, make_transforms

IMAGE = "Datasets/HandPD/SpiralPatients/0002-1.jpg"
CKPT = "demo_ckpt/efficientnet_b0_handpd_spiral_full.pt"
MODES = ["full", "background", "trace"]


def make_view(rgb, ink_mask, mode):
    if mode == "full":
        return rgb
    out = rgb.copy()
    if mode == "trace":
        out[~ink_mask] = 255
    elif mode == "background":
        out[ink_mask] = 255
    return out


device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model("efficientnet_b0")
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.to(device).eval()
tf = make_transforms(224, train=False)

bgr = cv2.imread(IMAGE, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = standardize_resolution(rgb, 512)
ink_mask = get_trace_mask(rgb)

probs = {}
with torch.no_grad():
    for mode in MODES:
        view = make_view(rgb, ink_mask, mode)
        x = tf(Image.fromarray(view)).unsqueeze(0).to(device)
        probs[mode] = torch.softmax(model(x), dim=1)[0, 1].item()

fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
for ax, mode in zip(axes, MODES):
    view = make_view(rgb, ink_mask, mode)
    ax.imshow(view)
    ax.set_xticks([]); ax.set_yticks([])
    p = probs[mode]
    pred = "PD" if p >= 0.5 else "HC"
    color = "#1a6b3c" if pred == "PD" else "#c0392b"
    ax.set_title(f"{mode}\nP(PD)={p:.3f} $\\to$ {pred}", fontsize=11, color=color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#888")

plt.suptitle("Ground truth: PD (HandPD spiral, subject 0002)", fontsize=10, y=1.03)
plt.tight_layout()
plt.savefig("worked_example.png", dpi=200, bbox_inches="tight")
plt.savefig("worked_example.pdf", bbox_inches="tight")
print(probs)
print("done")
