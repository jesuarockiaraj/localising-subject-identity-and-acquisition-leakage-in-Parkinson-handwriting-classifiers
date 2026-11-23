"""Reviewer W2: the measurement that actually predicts whether background
suppression should help honest-protocol AUC is within-cohort, background-
only DISEASE decodability on the real data -- not cohort decodability
(background is constant within a cohort, so it cannot act as a disease
shortcut there) and not subject re-ID (already eliminated by P2's design).
This was previously only run for the synthetic marker (marker_probe.py).
This script runs the same probe on the real spiral-task images, raw and
suppressed, separately within each cohort (pooling cohorts would confound
disease with the already-perfect cohort signal).
"""
import argparse
import glob

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from mech2_suppression import get_trace_mask, standardize_resolution
from leak_decomposition_probe import parse_handpd, parse_kaggle, _is_qc, _load_backbone

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


def run_cohort(root, glob_pat, label_fn, cohort_name, condition_label):
    model, tfm, torch = _load_backbone()
    paths = sorted(glob.glob(glob_pat, recursive=True))
    paths = [p for p in paths if not _is_qc(p)]
    print(f"\n[{cohort_name} / {condition_label}] {len(paths)} images from {root}")

    labels, subjects = [], []
    feats = {m: [] for m in MODES}
    for p in paths:
        subj, y = label_fn(p)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = standardize_resolution(rgb, 512)
        ink = get_trace_mask(rgb)
        for mode in MODES:
            view = make_view(rgb, ink, mode)
            x = tfm(Image.fromarray(view)).unsqueeze(0)
            with torch.no_grad():
                feats[mode].append(model(x).squeeze(0).numpy())
        labels.append(y)
        subjects.append(subj)

    y = np.array(labels)
    subjects = np.array(subjects)
    print(f"  n={len(y)}  PD frac={y.mean():.3f}  n_subjects={len(set(subjects))}")
    out = {}
    for mode in MODES:
        F = np.stack(feats[mode])
        n_groups = len(set(subjects))
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        probs = np.zeros(len(y)); preds = np.zeros(len(y), dtype=int)
        for tr, te in gkf.split(F, y, groups=subjects):
            if len(np.unique(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000)
            clf.fit(F[tr], y[tr])
            probs[te] = clf.predict_proba(F[te])[:, 1]
            preds[te] = clf.predict(F[te])
        auc = roc_auc_score(y, probs)
        bal = balanced_accuracy_score(y, preds)
        print(f"  disease-decodability [{mode:10s}] AUC={auc:.3f}  BalAcc={bal:.3f}")
        out[mode] = dict(auc=float(auc), bal_acc=float(bal))
    return out


def handpd_label(p):
    return parse_handpd(p), (1 if "patient" in p.lower() else 0)


def kaggle_label(p):
    return parse_kaggle(p), (1 if "parkinson" in p.lower() else 0)


if __name__ == "__main__":
    results = {}
    results["handpd_raw"] = run_cohort(
        "Datasets/HandPD", "Datasets/HandPD/Spiral*/*.jpg", handpd_label,
        "HandPD", "raw")
    results["handpd_suppressed"] = run_cohort(
        "Datasets_2a/HandPD", "Datasets_2a/HandPD/Spiral*/*.png", handpd_label,
        "HandPD", "suppressed")
    results["kaggle_raw"] = run_cohort(
        "Datasets/kaggle", "Datasets/kaggle/drawings/spiral/**/*.png", kaggle_label,
        "Kaggle", "raw")
    results["kaggle_suppressed"] = run_cohort(
        "Datasets_2a/kaggle", "Datasets_2a/kaggle/drawings/spiral/**/*.png", kaggle_label,
        "Kaggle", "suppressed")

    import json
    with open("background_disease_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[done] wrote background_disease_probe_results.json")
