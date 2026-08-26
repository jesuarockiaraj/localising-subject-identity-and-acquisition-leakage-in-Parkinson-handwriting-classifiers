"""M3 positive control, step (a): confirm the diagnostic probe detects the
injected background marker as a disease-label shortcut, on both the raw
marked images and (after Step 2, run separately) the suppressed marked
images."""
import argparse
import glob
import sys

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, ".")
from mech2_suppression import get_trace_mask, standardize_resolution
from leak_decomposition_probe import parse_handpd, _load_backbone

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Datasets_marker/HandPD")
    ap.add_argument("--label", default="raw-marked")
    args = ap.parse_args()

    model, tfm, torch = _load_backbone()
    paths = sorted(glob.glob(f"{args.root}/**/*.png", recursive=True))
    print(f"[data] {len(paths)} images from {args.root}")

    labels, subjects = [], []
    feats = {m: [] for m in MODES}
    for p in paths:
        subj = parse_handpd(p)
        y = 1 if "patient" in p.lower() else 0
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
    print(f"[{args.label}] n={len(y)}  PD frac={y.mean():.3f}")
    for mode in MODES:
        F = np.stack(feats[mode])
        gkf = GroupKFold(n_splits=5)
        probs = np.zeros(len(y)); preds = np.zeros(len(y), dtype=int)
        for tr, te in gkf.split(F, y, groups=subjects):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(F[tr], y[tr])
            probs[te] = clf.predict_proba(F[te])[:, 1]
            preds[te] = clf.predict(F[te])
        auc = roc_auc_score(y, probs)
        bal = balanced_accuracy_score(y, preds)
        print(f"  disease-decodability [{mode:10s}] AUC={auc:.3f}  BalAcc={bal:.3f}")


if __name__ == "__main__":
    main()
