"""Real-data thumbnails for the architecture figure (Figure 1): crops the
same masking modes (full / background-only / trace-only) from the same real
HandPD sample used in Figure 3's masking panel, so Figure 1's icons are
actual data, not illustrations.
"""
import cv2
import numpy as np
from mech2_suppression import get_trace_mask, standardize_resolution

SAMPLE = "Datasets/HandPD/SpiralControl/0068-1.jpg"
OUT = 320  # square output size, px


def make_view(rgb, ink_mask, mode):
    out = rgb.copy()
    if mode == "trace":
        out[~ink_mask] = 255
    elif mode == "background":
        out[ink_mask] = 255
    return out


bgr = cv2.imread(SAMPLE, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = standardize_resolution(rgb, 512)
ink = get_trace_mask(rgb)

# crop a centred square around the drawn content (same crop box for all
# three views, so they align and none looks off-centre once ink is removed)
ys, xs = np.where(ink)
cy, cx = int(ys.mean()), int(xs.mean())
half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.62)
h, w = rgb.shape[:2]
y0, y1 = max(0, cy - half), min(h, cy + half)
x0, x1 = max(0, cx - half), min(w, cx + half)
# pad to square if the crop hit an edge
side = max(y1 - y0, x1 - x0)
y1 = min(h, y0 + side)
x1 = min(w, x0 + side)

for mode, fname in [("full", "thumb_full.png"), ("background", "thumb_background.png"),
                     ("trace", "thumb_trace.png")]:
    view = make_view(rgb, ink, mode)
    crop = view[y0:y1, x0:x1]
    crop = cv2.resize(crop, (OUT, OUT), interpolation=cv2.INTER_AREA)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fname, crop_bgr)
    print("wrote", fname, crop.shape)
