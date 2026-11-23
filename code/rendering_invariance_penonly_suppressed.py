"""Reviewer W1 re-run, suppressed-image arm (Table 5/6 equivalent) --
CORRECTED APPROACH after discovering a second issue: the existing
Datasets_2a/ suppressed images are pure black-on-white (0,0,0 / 255,255,255
only -- verified directly), so the original ink colour is NOT retained
despite the manuscript's text claiming it is. That means a post-hoc colour
filter (as used for the raw-image arm) cannot recover pen-vs-guide on
already-suppressed images -- there is no colour left to filter on.

The correct, equivalent construction: isolate the pen-only mask on the RAW
(colour) image first (same corrected blue-dominance rule as the raw-image
arm), THEN render it exactly the way the existing suppression pipeline
renders any ink mask (black stroke on white canvas, via
mech2_suppression.render()). This is mathematically identical to "what the
suppression pipeline would output if its input mask were pen-only to begin
with" -- it does not require regenerating Datasets_2a or retraining anything.
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

from mech2_suppression import get_trace_mask, standardize_resolution, render
from leak_decomposition_probe import parse_handpd, parse_kaggle, _is_qc, _load_backbone

VIEWS = ["color", "gray", "uniform"]
BLUE_DOM_THRESH = 40


def pen_only_ink_mask(rgb, ink, cohort):
    if cohort != "handpd":
        return ink
    r = rgb[:, :, 0].astype(int)
    b = rgb[:, :, 2].astype(int)
    blue_dom = b - r
    return ink & (blue_dom >= BLUE_DOM_THRESH)


def make_suppressed_view(rgb, ink_mask, kind):
    """Render the suppressed ('2a') image from a given ink mask -- black
    stroke on white, exactly matching mech2_suppression.render() -- then
    apply the same color/gray/uniform ladder used elsewhere."""
    shape = rgb.shape
    supp = render(ink_mask, shape)  # black-on-white RGB, matches Datasets_2a convention
    if kind == "color":
        return supp
    gray = cv2.cvtColor(supp, cv2.COLOR_RGB2GRAY)
    if kind == "gray":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
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
    ap.add_argument("--handpd-glob", default="Datasets/HandPD/**/*.jpg",
                    help="RAW HandPD images (colour needed to isolate the pen)")
    ap.add_argument("--kaggle-glob", default="Datasets_2a/kaggle/drawings/**/*.png",
                    help="Kaggle: already-suppressed images are fine (no guide to fix)")
    ap.add_argument("--views", nargs="+", default=VIEWS, choices=VIEWS)
    ap.add_argument("--out", default="rendering_invariance_penonly_suppressed2_results.json")
    args = ap.parse_args()

    model, tfm, torch = _load_backbone()
    recs = collect(args.handpd_glob, args.kaggle_glob)
    print(f"[data] {len(recs)} drawings")

    results = {}
    for kind in args.views:
        feats, subjects, cohorts = [], [], []
        for path, subj, cohort in recs:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = standardize_resolution(rgb, 512)
            ink = get_trace_mask(rgb)
            ink = pen_only_ink_mask(rgb, ink, cohort)
            if cohort == "handpd":
                view = make_suppressed_view(rgb, ink, kind)
            else:
                # Kaggle input is already the suppressed image; reuse the
                # raw-arm renderer logic (ink is already pen-only by construction)
                from rendering_invariance_penonly import make_view
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
                             n=len(recs))
        print(f"[{kind}] reID@1={reid:.3f} (chance {chance:.3f})  "
              f"cohort AUC={cohort_auc:.3f}  cohort BalAcc={cohort_balacc:.3f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
