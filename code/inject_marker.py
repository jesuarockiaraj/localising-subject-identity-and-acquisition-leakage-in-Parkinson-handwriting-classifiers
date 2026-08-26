"""M3 positive control: inject a KNOWN, synthetic, background-confined
marker whose intensity is correlated with the PD/HC label, into a mirror
of HandPD spiral. This gives a ground-truth leak with a known location
(background only) and a known correlate (disease label), so the full
probe -> suppress -> re-probe -> retrain pipeline can be validated against
a case where we know in advance what SHOULD happen: the probe should
detect the marker, suppression should remove it, and honest AUC should
verifiably rise after suppression (Fork A), unlike the real background
suppression tested elsewhere in this study (Fork B).
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from mech2_suppression import get_trace_mask, standardize_resolution
from leak_decomposition_probe import parse_handpd


def inject(rgb, ink_mask, label, rng, patch_size=48, margin=8, strength=0.85):
    """Paint a faint corner patch whose grey level encodes the label,
    confined to background pixels only (never overlapping ink_mask)."""
    out = rgb.copy()
    h, w = out.shape[:2]
    y0, x0 = margin, margin
    y1, x1 = y0 + patch_size, x0 + patch_size
    patch_mask = np.zeros((h, w), dtype=bool)
    patch_mask[y0:y1, x0:x1] = True
    patch_mask &= ~ink_mask  # never touch ink pixels
    # label=1 (PD) -> darker grey patch; label=0 (HC) -> lighter grey patch,
    # small noise so it is not a trivial constant (more realistic confound)
    base = 235 if label == 0 else 235 - int(strength * 90)
    noise = rng.normal(0, 6, size=(patch_mask.sum(), 3))
    val = np.clip(base + noise, 0, 255).astype(np.uint8)
    out[patch_mask] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default="Datasets/HandPD")
    ap.add_argument("--out-root", default="Datasets_marker/HandPD")
    ap.add_argument("--task", default="spiral")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    imgs = [p for p in in_root.rglob("*.jpg")
            if args.task in p.parent.name.lower()
            and "__MACOSX" not in p.parts
            and not p.name.startswith("._")]
    print(f"[data] {len(imgs)} {args.task} images")

    for p in imgs:
        parent = p.parent.name.lower()
        label = 1 if "patient" in parent else 0
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = standardize_resolution(rgb, 512)
        ink = get_trace_mask(rgb)
        marked = inject(rgb, ink, label, rng)
        dst = out_root / p.relative_to(in_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst.with_suffix(".png")), cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
    print(f"[done] wrote marked images to {out_root}")


if __name__ == "__main__":
    main()
