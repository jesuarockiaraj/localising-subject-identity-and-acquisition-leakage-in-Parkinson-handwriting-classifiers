"""Rendering-invariance probe (reviewer M2).

The trace-only view built by get_trace_mask() keeps each ink pixel's
ORIGINAL colour/intensity (V_trace(x,y) = I(x,y) if in mask). HandPD ink is
blue-tinted on bluish paper; Kaggle ink is neutral grey on white. A cohort
probe on that view cannot distinguish "acquisition/motor-style fingerprint"
from "the pen is a different colour in the two datasets" -- the two are
confounded by construction.

This script re-runs the identity/cohort probe on THREE views of increasing
rendering-invariance, so that confound can be isolated:

  color    -- V_trace as used everywhere else in the study (colour kept)
  gray     -- grayscale + per-image contrast normalisation (removes colour
              and global brightness/contrast, keeps stroke width/texture)
  uniform  -- skeletonised, fixed-width (3 px), pure black-on-white stroke
              (removes colour, brightness, AND stroke width/pressure --
              only the drawn GEOMETRY survives)

If cohort decodability collapses toward chance under `uniform`, the
acquisition signal reported elsewhere in this study is (at least
partly) a rendering artefact, not evidence of motor-style or geometric
individuality. If it survives `uniform`, the signal is genuinely
geometric and the "trace-intrinsic" claim is supported by something a
colour-normalisation step cannot remove.

Reports BOTH the saturating balanced-accuracy metric used elsewhere AND a
graded metric (ROC-AUC of the same logistic regression's decision
function), because balanced accuracy pinned at 1.000 has no discriminative
power to compare conditions (reviewer M1).
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


def make_view(rgb, ink_mask, kind):
    if kind == "color":
        out = rgb.copy()
        out[~ink_mask] = 255
        return out
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if kind == "gray":
        flat = flatten_background(gray)  # removes global brightness/contrast unevenness
        out = np.full_like(flat, 255)
        out[ink_mask] = flat[ink_mask]
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    if kind == "uniform":
        from skimage.morphology import skeletonize, binary_dilation, disk
        skel = skeletonize(ink_mask)
        uniform_mask = binary_dilation(skel, disk(2))  # fixed 5px-wide stroke
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
    ap.add_argument("--out", default="rendering_invariance_results.json")
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

        # subject re-ID@1
        norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        sim = norm @ norm.T
        np.fill_diagonal(sim, -1)
        nn = sim.argmax(axis=1)
        reid = float(np.mean(subjects[nn] == subjects))
        n_subj = len(set(subjects))
        chance = float(np.mean([(list(subjects).count(s) - 1) / (len(subjects) - 1) for s in set(subjects)]))

        # cohort decodability: graded (AUC of decision function) + saturating (balanced accuracy)
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
