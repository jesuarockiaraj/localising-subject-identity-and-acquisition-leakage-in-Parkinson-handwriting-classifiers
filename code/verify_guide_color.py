"""Empirical check for reviewer W1: which colour is the printed guide
(should be near-identical in shape/position across different subjects,
since it's the same pre-printed template) and which is the patient's pen
stroke (should vary subject to subject)? We test this directly rather than
trusting any assumption in the code or caption: compute the blue-dominant
mask and the dark/neutral mask for several different subjects' raw spiral
images, align them via standardize_resolution (already roughly registers
images to the same canvas), and compare cross-subject IoU for each colour.
The colour with much higher cross-subject IoU is the printed template.
"""
import glob
import cv2
import numpy as np
from mech2_suppression import get_trace_mask, standardize_resolution

paths = sorted(glob.glob("Datasets/HandPD/SpiralControl/*.jpg")) + \
        sorted(glob.glob("Datasets/HandPD/SpiralPatients/*.jpg"))
# one image per distinct subject (first drawing only), up to 8 subjects
seen = set()
sample_paths = []
for p in paths:
    subj = p.replace("\\", "/").split("/")[-1].split("-")[0]
    if subj not in seen:
        seen.add(subj)
        sample_paths.append(p)
    if len(sample_paths) >= 8:
        break
print("[using subjects]", sample_paths)

blue_masks = []
dark_masks = []
for p in sample_paths:
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = standardize_resolution(rgb, 512)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    ink = get_trace_mask(rgb)
    r = rgb[:, :, 0].astype(int)
    b = rgb[:, :, 2].astype(int)
    blue_dom = b - r
    blue_mask = ink & (blue_dom >= 40)
    dark_mask = ink & (blue_dom < 40)
    blue_masks.append(blue_mask)
    dark_masks.append(dark_mask)
    print(f"{p}: ink_frac={ink.mean():.4f} blue_frac={blue_mask.mean():.4f} dark_frac={dark_mask.mean():.4f}")

def mean_pairwise_iou(masks):
    ious = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter = (masks[i] & masks[j]).sum()
            union = (masks[i] | masks[j]).sum()
            if union > 0:
                ious.append(inter / union)
    return np.mean(ious), ious

blue_iou, blue_list = mean_pairwise_iou(blue_masks)
dark_iou, dark_list = mean_pairwise_iou(dark_masks)
print(f"\n[cross-subject mean IoU] blue-dominant mask: {blue_iou:.4f}")
print(f"[cross-subject mean IoU] dark/neutral mask : {dark_iou:.4f}")
print("\nHigher IoU = more consistent across different people's sheets = the printed template.")

# also save a visual overlay of the blue masks and dark masks stacked
overlay_blue = np.zeros((512, 512), dtype=np.float32)
overlay_dark = np.zeros((512, 512), dtype=np.float32)
for m in blue_masks:
    if m.shape == (512, 512):
        overlay_blue += m.astype(np.float32)
for m in dark_masks:
    if m.shape == (512, 512):
        overlay_dark += m.astype(np.float32)
cv2.imwrite("scratch_overlay_blue.png", (overlay_blue / overlay_blue.max() * 255).astype(np.uint8))
cv2.imwrite("scratch_overlay_dark.png", (overlay_dark / max(overlay_dark.max(),1) * 255).astype(np.uint8))
print("\n[saved] scratch_overlay_blue.png, scratch_overlay_dark.png (brighter = more subjects agree there)")
