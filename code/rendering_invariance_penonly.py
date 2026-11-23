"""Reviewer W1 re-run: the rendering-invariance ladder (Table 4/tab:rendering)
redone with HandPD's printed guide CORRECTLY excluded.

Empirical check (verify_guide_color.py) confirmed the colour assignment used
everywhere in this study so far was backwards: the printed guide is the
THIN, SMOOTH, near-identical-across-subjects BLACK/dark line; the patient's
actual pen stroke is the THICK, JAGGED, subject-varying BLUE line. Every
existing "trace"/"ink" mask in this study mixes both for HandPD (since
get_trace_mask() is a pure intensity threshold, colour-blind), while Kaggle's
mask is pen-only by construction (Kaggle has no printed guide at all --
visually confirmed, single spiral only).

This means HandPD's ink mask contains TWO overlaid spirals (guide + pen)
while Kaggle's contains ONE -- a structural asymmetry that could trivially
explain perfect cohort decodability under "uniform" (geometry-only)
rendering, with nothing to do with any genuine acquisition fingerprint in
the stroke. This script isolates HandPD's pen-only pixels (CORRECTED:
high blue-dominance = pen, not low) before building the color/gray/uniform
views, and leaves Kaggle untouched, then re-runs the exact cohort
decodability / re-ID test from rendering_invariance_probe.py.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from mech2_suppression import get_trace_mask, standardize_resolution, flatten_background
from leak_decomposition_probe import parse_handpd, parse_kaggle, _is_qc, _load_backbone

VIEWS = ["color", "gray", "uniform"]
BLUE_DOM_THRESH = 40  # same threshold as handpd_clean_split.py, direction corrected


def pen_only_ink_mask(rgb, ink, cohort):
    """For HandPD: keep only HIGH blue-dominance pixels within the ink mask
    (the patient's pen stroke, confirmed by direct visual + cross-subject
    IoU check); drop LOW blue-dominance pixels (the printed guide template).
    For Kaggle: no guide exists, mask is already pen-only, return unchanged."""
    if cohort != "handpd":
        return ink
    r = rgb[:, :, 0].astype(int)
    b = rgb[:, :, 2].astype(int)
    blue_dom = b - r
    return ink & (blue_dom >= BLUE_DOM_THRESH)


def make_view(rgb, ink_mask, kind):
    if kind == "color":
        out = rgb.copy()
        out[~ink_mask] = 255
        return out
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if kind == "gray":
        flat = flatten_background(gray)
        out = np.full_like(flat, 255)
        out[ink_mask] = flat[ink_mask]
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    if kind == "uniform":
        from skimage.morphology import skeletonize, binary_dilation, disk
        skel = skeletonize(ink_mask)
        uniform_mask = binary_dilation(skel, disk(2))
        out = np.full(gray.shape, 255, dtype=np.uint8)
        out[uniform_mask] = 0
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    raise ValueError(kind)


def collect(handpd_glob, kaggle_glob):
    recs = []
    for p in sorted(glob.glob(handpd_glob, recursive=True)):
        if _is_qc(p):
            continue
        recs.append((p, parse_handpd(p), "handpd"))
    for p in sorted(glob.glob(kaggle_glob, recursive=True)):
        if _is_qc(p):
            continue
        recs.append((p, parse_kaggle(p), "kaggle"))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handpd-glob", default="Datasets/HandPD/**/*.jpg")
    ap.add_argument("--kaggle-glob", default="Datasets/kaggle/drawings/**/*.png")
    ap.add_argument("--views", nargs="+", default=VIEWS, choices=VIEWS)
    ap.add_argument("--out", default="rendering_invariance_penonly_results.json")
    args = ap.parse_args()

    model, tfm, torch = _load_backbone()
    recs = collect(args.handpd_glob, args.kaggle_glob)
    print(f"[data] {len(recs)} drawings")

    # report how much HandPD ink the pen-only filter removes, for disclosure
    guide_fracs = []
    for path, subj, cohort in recs:
        if cohort != "handpd":
            continue
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = standardize_resolution(rgb, 512)
        ink = get_trace_mask(rgb)
        pen = pen_only_ink_mask(rgb, ink, cohort)
        if ink.sum() > 0:
            guide_fracs.append(1 - pen.sum() / ink.sum())
    print(f"[handpd] mean fraction of original ink mask removed as guide: "
          f"{np.mean(guide_fracs):.3f} (n={len(guide_fracs)} images)")

    results = {}
    for kind in args.views:
        feats, subjects, cohorts = [], [], []
        for path, subj, cohort in recs:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = standardize_resolution(rgb, 512)
            ink = get_trace_mask(rgb)
            ink = pen_only_ink_mask(rgb, ink, cohort)
            view = make_view(rgb, ink, kind)
            from PIL import Image
            x = tfm(Image.fromarray(view)).unsqueeze(0)
            with torch.no_grad():
                f = model(x).squeeze(0).numpy()
            feats.append(f); subjects.append(subj); cohorts.append(cohort)
        feats = np.stack(feats)
        subjects = np.array(subjects)
        cohorts = np.array(cohorts)
        y = (cohorts == "kaggle").astype(int)

        norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        sim = norm @ norm.T
        np.fill_diagonal(sim, -1)
        nn = sim.argmax(axis=1)
        reid = float(np.mean(subjects[nn] == subjects))
        n_subj = len(set(subjects))
        chance = float(np.mean([(list(subjects).count(s) - 1) / (len(subjects) - 1) for s in set(subjects)]))

        gkf = GroupKFold(n_splits=5)
        probs = np.zeros(len(y), dtype=float)
        preds = np.zeros(len(y), dtype=int)
        for tr, te in gkf.split(feats, y, groups=subjects):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(feats[tr], y[tr])
            probs[te] = clf.predict_proba(feats[te])[:, 1]
            preds[te] = clf.predict(feats[te])
        cohort_auc = float(roc_auc_score(y, probs))
        cohort_balacc = float(balanced_accuracy_score(y, preds))

        results[kind] = dict(reid_at_1=reid, reid_chance=chance,
                             cohort_auc=cohort_auc, cohort_balacc=cohort_balacc,
                             n=len(recs), n_subjects=n_subj)
        print(f"[{kind}] reID@1={reid:.3f} (chance {chance:.3f})  "
              f"cohort AUC={cohort_auc:.3f}  cohort BalAcc={cohort_balacc:.3f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
