"""
mech2_suppression.py  (tree-preserving)
=======================================
Paper 2, mechanism 2 -- SUPPRESSOR.

Runs as a PREPROCESSOR upstream of the Paper 1 harness. It writes a MIRROR of
the input dataset tree (same folders, same filenames) with suppressed pixels, so
pd_loso_pipeline_cross_dataset.py parses labels (from folder names), task (from
folder markers) and subject ids (from filenames) IDENTICALLY to baseline. The
ONLY thing that differs between a baseline run and a 2a run is the image content
-- which is exactly what makes the AUC delta attributable to suppression.

Two SEPARABLE sub-mechanisms (they are not equally safe):
  2a  paper/scanner texture & grain suppression  -- clearly a leak. Safe.  [default ON]
  2b  pre-printed template removal               -- AMBIGUOUS: deviation-from-guide
      is a real PD sign; removing it can delete signal and fake a Fork-B null.
      Opt-in (--do-2b); heuristic; WILL fail where guide and pen overlap.

Standardising resolution first (--target-height) is not optional: differing DPI
across cohorts makes adaptive binarisation itself a cohort marker (Fork C).

Deps: numpy, opencv-python, scikit-image.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}


def _is_junk(p: Path) -> bool:
    return p.name.startswith("._") or "__MACOSX" in p.parts or p.name == ".DS_Store"


# ---------- core ops -------------------------------------------------------
def _to_gray(rgb):
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb


def standardize_resolution(rgb, target_height):
    if not target_height:
        return rgb
    import cv2
    h, w = rgb.shape[:2]
    if h == target_height:
        return rgb
    s = target_height / h
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (max(1, round(w * s)), target_height), interpolation=interp)


def flatten_background(gray, kernel_frac=0.10):
    """2a: divide out the paper background estimated by a large morphological
    close -> removes grain, shading, scanner gradient. Kernel >> stroke width."""
    import cv2
    k = max(15, int(gray.shape[0] * kernel_frac) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = np.where(bg == 0, 1, bg).astype(np.float32)
    return np.clip(gray.astype(np.float32) / bg * 255.0, 0, 255).astype(np.uint8)


def ink_mask(flat):
    """Sauvola adaptive threshold -> boolean ink mask, speck-filtered."""
    import cv2
    from skimage.filters import threshold_sauvola
    win = max(15, (min(flat.shape) // 20) | 1)
    mask = (flat < threshold_sauvola(flat, window_size=win, k=0.2)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    min_area = max(8, int(2e-5 * mask.size))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lbl == i] = True
    return keep


def get_trace_mask(rgb, kernel_frac=0.10):
    """First-pass ink mask (also used by the probe's trace/background split).
    NOTE: on HandPD the printed guide overlaps the pen, so this is not a perfect
    pen-only isolation -- it is directional."""
    return ink_mask(flatten_background(_to_gray(rgb), kernel_frac))


def suppress_template_intensity(flat, mask, strong_pct=40.0):
    """2b heuristic: keep only ink darker than the strong_pct percentile (pen
    assumed darker than the printed guide). FRAGILE; flagged in output."""
    vals = flat[mask]
    if vals.size == 0:
        return mask
    return mask & (flat <= np.percentile(vals, strong_pct))


def render(mask, shape):
    out = np.full((shape[0], shape[1], 3), 255, np.uint8)
    out[mask] = 0
    return out


def process(rgb, do_2a=True, do_2b=False, target_height=512, preserve_template=False):
    rgb = standardize_resolution(rgb, target_height)
    gray = _to_gray(rgb)
    flat = flatten_background(gray) if do_2a else gray
    mask = ink_mask(flat)
    used_2b = False
    if do_2b and not preserve_template:
        mask = suppress_template_intensity(flat, mask)
        used_2b = True
    return render(mask, gray.shape), dict(applied_2a=do_2a, applied_2b=used_2b,
                                          ink_fraction=float(mask.mean()))


def qc_montage(rgb, clean, out_path):
    import cv2
    left = standardize_resolution(rgb, clean.shape[0])
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
    left = cv2.resize(left, (clean.shape[1], clean.shape[0]))
    cv2.imwrite(out_path, cv2.cvtColor(np.concatenate([left, clean], 1), cv2.COLOR_RGB2BGR))


# ---------- tree-preserving driver -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-root", required=True, help="input dataset root (walked recursively)")
    ap.add_argument("--out-root", required=True, help="output root; input tree is mirrored here")
    ap.add_argument("--no-2a", action="store_true", help="disable texture suppression")
    ap.add_argument("--do-2b", action="store_true", help="ENABLE template removal (ambiguous)")
    ap.add_argument("--preserve-template", action="store_true",
                    help="with --do-2b: skip 2b but keep 2a (recommended variant)")
    ap.add_argument("--target-height", type=int, default=512)
    ap.add_argument("--qc-every", type=int, default=0,
                    help="write a side-by-side QC image every N files (0=off) into out-root/_qc")
    args = ap.parse_args()

    import cv2
    in_root, out_root = Path(args.in_root), Path(args.out_root)
    imgs = [p for p in in_root.rglob("*")
            if p.suffix.lower() in IMG_EXT and not _is_junk(p)]
    if not imgs:
        raise SystemExit(f"No images under {in_root}")
    if args.do_2b and not args.preserve_template:
        print("WARNING: 2b template removal ON -> re-run the leak probe on the output to "
              "confirm you removed a LEAK, not deviation-from-guide SIGNAL.")
    qc_dir = out_root / "_qc"
    if args.qc_every:
        qc_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for i, p in enumerate(imgs):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            n_skip += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        clean, _ = process(rgb, do_2a=not args.no_2a, do_2b=args.do_2b,
                           target_height=args.target_height,
                           preserve_template=args.preserve_template)
        # MIRROR the relative path; keep stem, force .png
        rel = p.relative_to(in_root).with_suffix(".png")
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        if args.qc_every and i % args.qc_every == 0:
            qc_montage(rgb, clean, str(qc_dir / (rel.stem + "_qc.png")))
        n_ok += 1
    print(f"done: {n_ok} images mirrored -> {out_root} ({n_skip} unreadable skipped)")
    print("folder structure preserved; feed --out-root to the harness as --root/--test-root.")


if __name__ == "__main__":
    main()
