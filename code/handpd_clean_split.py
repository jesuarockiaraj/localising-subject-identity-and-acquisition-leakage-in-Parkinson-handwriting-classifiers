"""M9: HandPD's printed guide is blue, the pen stroke is dark/neutral, so a
colour-channel rule can separate them where the intensity-only ink mask
(get_trace_mask) cannot. This isolates a genuinely pen-only trace for
HandPD and re-checks the trace-intrinsic re-identification result on it
specifically, rather than relying on Kaggle alone or on HandPD's
guide-contaminated trace.
"""
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, ".")
from mech2_suppression import get_trace_mask, standardize_resolution
from leak_decomposition_probe import collect, _load_backbone


def pen_only_mask(rgb, ink_mask, blue_dom_thresh=40):
    """Within the ink mask, keep only low blue-dominance (dark/neutral pen)
    pixels; drop high blue-dominance (printed guide) pixels."""
    r = rgb[..., 0].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    blue_dom = b - r
    return ink_mask & (blue_dom < blue_dom_thresh)


def make_view(rgb, mask):
    out = rgb.copy()
    out[~mask] = 255
    return out


if __name__ == "__main__":
    model, tfm, torch = _load_backbone()
    recs = [r for r in collect("Datasets/HandPD/**/*.jpg", "Datasets/kaggle/drawings/**/*.png")
            if r.cohort == "handpd"]
    print(f"[data] {len(recs)} HandPD drawings")

    feats_trace, feats_clean, subjects = [], [], []
    ink_fracs, clean_fracs = [], []
    for rec in recs:
        bgr = cv2.imread(rec.path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = standardize_resolution(rgb, 512)
        ink = get_trace_mask(rgb)
        clean = pen_only_mask(rgb, ink)
        ink_fracs.append(ink.mean()); clean_fracs.append(clean.mean())

        for feats, mask in ((feats_trace, ink), (feats_clean, clean)):
            view = make_view(rgb, mask)
            x = tfm(Image.fromarray(view)).unsqueeze(0)
            with torch.no_grad():
                feats.append(model(x).squeeze(0).numpy())
        subjects.append(rec.subject)

    feats_trace = np.stack(feats_trace)
    feats_clean = np.stack(feats_clean)
    subjects = np.array(subjects)
    print(f"[mask] mean ink fraction {np.mean(ink_fracs):.4f} -> "
          f"mean pen-only fraction {np.mean(clean_fracs):.4f} "
          f"({100*np.mean(clean_fracs)/np.mean(ink_fracs):.1f}% of ink pixels kept)")

    def reid(feats):
        norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        sim = norm @ norm.T
        np.fill_diagonal(sim, -1)
        nn = sim.argmax(axis=1)
        from collections import Counter
        cnt = Counter(subjects)
        chance = np.mean([(c - 1) / (len(subjects) - 1) for c in cnt.values()])
        return float(np.mean(subjects[nn] == subjects)), float(chance)

    r_trace, chance = reid(feats_trace)
    r_clean, _ = reid(feats_clean)
    print(f"[reid] original trace (ink+guide mixed): {r_trace:.3f}  (chance {chance:.4f})")
    print(f"[reid] pen-only, colour-separated trace: {r_clean:.3f}  (chance {chance:.4f})")
