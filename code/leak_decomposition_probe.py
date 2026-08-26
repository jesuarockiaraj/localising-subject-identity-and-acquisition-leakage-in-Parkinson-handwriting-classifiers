"""
leak_decomposition_probe.py
===========================
Paper 2, mechanism 2 -- PRE-BUILD DIAGNOSTIC.

Question this answers, cheaply, BEFORE building any suppressor:
    How much of the subject-identity leak lives in the BACKGROUND / TEMPLATE
    (mechanism-2 territory) vs in the TRACE ITSELF (motor style -- only
    mechanisms 1/3 can remove)?

Method: frozen ImageNet EfficientNet-B0 features (same backbone family as
Paper 1), no training of the backbone. Two cheap probes, each run under three
masking modes (full / background-only / trace-only):

  (A) Subject re-identification @1 (retrieval): for each drawing, is its
      nearest neighbour (cosine, excluding self) drawn by the SAME subject?
      Chance = mean over subjects of (n_i - 1) / (N - 1). High => drawings are
      re-identifiable => leak. This is the quantity Paper 1 blamed for inflated
      accuracy, measured directly.

  (B) Cohort decodability (HandPD vs Kaggle): logistic regression under
      GroupKFold grouped BY SUBJECT (so no subject spans train+test). Balanced
      accuracy vs 0.5. High in background-mode => background carries a cohort
      marker => cross-cohort transfer is contaminated by acquisition, not just
      disease.

INTERPRETATION (this is the whole point):
  - background-mode subject-reID high  -> mechanism 2 HAS headroom.
  - background-mode subject-reID ~chance-> mechanism 2 will NOT move the honest
                                           number. Early Fork-B. Stop; don't
                                           build the suppressor.
  - trace-mode subject-reID stays high -> identity is trace-intrinsic (motor
                                           style). Mechanism 2 cannot touch it;
                                           this is the mechanism-1/3 budget.

HONEST LIMITATIONS (read before trusting numbers):
  * The trace/background split reuses the FIRST-PASS mask from
    mech2_suppression.get_trace_mask(). On HandPD the printed guide overlaps
    the pen trace, so "trace-only" there still contains some template. The
    probe is DIRECTIONAL, not exact. Treat a 0.05 difference as noise.
  * Frozen ImageNet features are a proxy, not your Paper 1 trained model. They
    tend to UNDER-state trace-intrinsic identity (ImageNet isn't tuned to
    handwriting). So if trace-mode reID is already high here, it's a floor.
  * Tiny cohorts -> wide error. Bootstrap subject-level CIs (B below) and do
    NOT report a point estimate alone.

Deps: torch, torchvision, numpy, scikit-learn, opencv-python, scikit-image.
Author-facing config (regexes, paths) is at the bottom -- VERIFY against your
actual filenames before running; the defaults encode the handoff's examples
only (`sp1-P1.jpg`, `V03PE05.png`) and are assumptions, not confirmed parsers.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# ---- lazy heavy imports so --help works without torch installed ------------
def _load_backbone():
    import torch
    import torchvision
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()  # -> 1280-d pooled features
    model.eval()
    return model, weights.transforms(), torch


# ---------------------------------------------------------------------------
# Filename -> (subject_id, cohort). ASSUMPTIONS -- confirm against your files.
# ---------------------------------------------------------------------------
def parse_handpd(fname: str) -> str:
    """HandPD dataset parser for folder structure:
    SpiralControl/, SpiralPatient/, MeanderControl/, MeanderPatient/
    """
    b = os.path.basename(fname)
    # Get parent folder name
    parent = os.path.basename(os.path.dirname(fname))
    
    # Extract subject ID from filename (assuming pattern like "subject_1.png" or similar)
    # You need to check actual filename patterns in your dataset
    # For now, this tries to extract numbers from filename
    m = re.search(r"(\d+)", b)
    subject_num = m.group(1) if m else "unknown"
    
    # Determine if control or patient
    if "Control" in parent:
        return f"handpd_control_{subject_num}"
    elif "Patient" in parent:
        return f"handpd_patient_{subject_num}"
    else:
        return f"handpd_{subject_num}"

def parse_kaggle(fname: str) -> str:
    """Kaggle dataset parser for folder structure:
    spiral/, wave/
    """
    b = os.path.basename(fname)
    parent = os.path.basename(os.path.dirname(fname))
    
    # Extract subject ID from filename
    # Common pattern: "V01PE01.png" in kaggle
    m = re.search(r"(V\d+)([PH])", b, re.IGNORECASE)
    if m:
        subject_id = f"{m.group(2).upper()}{m.group(1).upper()}"
        return f"kaggle_{parent}_{subject_id}"
    
    # Fallback: use filename without extension
    name = os.path.splitext(b)[0]
    return f"kaggle_{parent}_{name}"

@dataclass
class ImageRecord:
    path: str
    subject: str
    cohort: str


def _is_qc(path: str) -> bool:
    """Exclude quality-control montage images (raw|suppressed side-by-side
    panels written under a `_qc/` directory) from analysis. These are not
    drawings and, if picked up by a recursive glob, mint spurious pseudo-
    subject ids (their parent folder is `_qc`, not `.../Control` or
    `.../Patient`), silently inflating both image and subject counts."""
    parts = [x.lower() for x in Path(path).parts]
    return "_qc" in parts


def collect(handpd_glob: str, kaggle_glob: str) -> list[ImageRecord]:
    recs: list[ImageRecord] = []
    for p in sorted(glob.glob(handpd_glob, recursive=True)):
        if _is_qc(p):
            continue
        recs.append(ImageRecord(p, parse_handpd(p), "handpd"))
    for p in sorted(glob.glob(kaggle_glob, recursive=True)):
        if _is_qc(p):
            continue
        recs.append(ImageRecord(p, parse_kaggle(p), "kaggle"))
    if not recs:
        raise SystemExit("No images matched. Check the two --*-glob patterns.")
    return recs


# ---------------------------------------------------------------------------
# Masking modes. Uses mech2_suppression.get_trace_mask for the ink mask.
# ---------------------------------------------------------------------------
def load_rgb(path: str):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_mask(rgb, mode: str):
    """mode in {full, background, trace}. background/trace use the first-pass
    ink mask (see limitations in the module docstring)."""
    if mode == "full":
        return rgb
    from mech2_suppression import get_trace_mask
    ink = get_trace_mask(rgb)  # bool, True where trace/ink
    out = rgb.copy()
    if mode == "trace":
        out[~ink] = 255          # keep ink, blank the background
    elif mode == "background":
        out[ink] = 255           # blank the ink, keep background/template
    else:
        raise ValueError(mode)
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(records: list[ImageRecord], mode: str, batch: int = 32):
    model, tfm, torch = _load_backbone()
    from PIL import Image
    feats = []
    with torch.no_grad():
        buf, idx = [], 0
        def flush(buf):
            x = torch.stack(buf)
            return model(x).cpu().numpy()
        for r in records:
            arr = apply_mask(load_rgb(r.path), mode)
            buf.append(tfm(Image.fromarray(arr)))
            if len(buf) == batch:
                feats.append(flush(buf)); buf = []
        if buf:
            feats.append(flush(buf))
    X = np.concatenate(feats, axis=0)
    # L2-normalise for cosine retrieval
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X


# ---------------------------------------------------------------------------
# Probe A: subject re-identification @1 (retrieval)
# ---------------------------------------------------------------------------
def subject_reid_at1(X, subjects: np.ndarray, boot: int = 2000, seed: int = 0):
    S = np.asarray(subjects)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(axis=1)
    hit = (S[nn] == S).astype(float)

    # chance = P(random other drawing is same subject)
    N = len(S)
    _, counts = np.unique(S, return_counts=True)
    chance = float(np.sum(counts * (counts - 1)) / (N * (N - 1)))

    rng = np.random.default_rng(seed)
    # subject-level bootstrap: resample SUBJECTS, not drawings
    subj_ids = np.unique(S)
    idx_by_subj = {s: np.where(S == s)[0] for s in subj_ids}
    boots = []
    for _ in range(boot):
        chosen = rng.choice(subj_ids, size=len(subj_ids), replace=True)
        sel = np.concatenate([idx_by_subj[s] for s in chosen])
        boots.append(hit[sel].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(reid_at1=float(hit.mean()), chance=chance, ci=(float(lo), float(hi)))


# ---------------------------------------------------------------------------
# Probe B: cohort decodability, grouped by subject
# ---------------------------------------------------------------------------
def cohort_decodability(X, cohorts, subjects):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    y = (np.asarray(cohorts) == "kaggle").astype(int)
    groups = np.asarray(subjects)
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return dict(cohort_bal_acc=float("nan"), chance=0.5, note="too few subjects")
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return dict(cohort_bal_acc=float(np.mean(accs)) if accs else float("nan"),
                chance=0.5, folds=len(accs))


def run(records, modes=("full", "background", "trace"), save_features=False):
    subjects = np.array([r.subject for r in records])
    cohorts = np.array([r.cohort for r in records])
    print(f"[data] {len(records)} drawings | "
          f"{len(np.unique(subjects))} subjects | "
          f"cohorts={dict(zip(*np.unique(cohorts, return_counts=True)))}\n")
    for mode in modes:
        X = extract_features(records, mode)
        if save_features:
            # feeds stratified_reid.py, which loads scratch_feats_full.npy /
            # scratch_subjects_full.npy directly (order matches collect()'s
            # deterministic sort, so subjects only needs saving once)
            np.save(f"scratch_feats_{mode}.npy", X)
            np.save(f"scratch_subjects_{mode}.npy", subjects)
            print(f"[saved] scratch_feats_{mode}.npy, scratch_subjects_{mode}.npy")
        a = subject_reid_at1(X, subjects)
        b = cohort_decodability(X, cohorts, subjects)
        print(f"=== mode: {mode} ===")
        print(f"  subject re-ID@1 : {a['reid_at1']:.3f}  "
              f"(95% CI {a['ci'][0]:.3f}-{a['ci'][1]:.3f}, chance {a['chance']:.3f})")
        print(f"  cohort bal-acc  : {b['cohort_bal_acc']:.3f}  (chance {b['chance']:.2f})")
        print()
    print("READ-OUT: compare subject re-ID@1 across modes.")
    print("  full ~ trace >> background  -> identity is trace-intrinsic; mech2 low headroom (Fork-B risk).")
    print("  full ~ background >> trace  -> identity is in background/template; mech2 has headroom.")
    print("  cohort bal-acc high in background -> acquisition marker contaminates cross-cohort transfer.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handpd-glob", default="data/handpd/**/*.jpg",
                    help="glob for HandPD images (recursive ** supported)")
    ap.add_argument("--kaggle-glob", default="data/kaggle/**/*.png",
                    help="glob for Kaggle images")
    ap.add_argument("--modes", nargs="+", default=["full", "background", "trace"])
    ap.add_argument("--save-features", action="store_true",
                    help="cache extracted features/subjects to scratch_feats_<mode>.npy "
                         "and scratch_subjects_<mode>.npy for each mode run (needed by "
                         "stratified_reid.py, which requires the 'full' mode cache).")
    args = ap.parse_args()
    recs = collect(args.handpd_glob, args.kaggle_glob)
    run(recs, modes=tuple(args.modes), save_features=args.save_features)
