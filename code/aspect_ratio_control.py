"""C5 control: HandPD images are not exactly square after resolution
standardisation (~0.99-1.08 width/height) while Kaggle is exactly
512x512, and the classifier/probe transform (torchvision Resize((224,224)))
stretches to a forced square, distorting HandPD images by a small,
cohort-correlated factor Kaggle never experiences. This re-runs the
decisive 'uniform' (geometry-only) cohort-decodability result with a
pad-to-square step inserted before the forced-square resize, so aspect
ratio is preserved instead of stretched, and checks whether cohort AUC
changes.
"""
import sys
import numpy as np
import cv2
from PIL import Image, ImageOps

sys.path.insert(0, ".")
from mech2_suppression import get_trace_mask, standardize_resolution
from leak_decomposition_probe import collect, _load_backbone, _is_qc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from skimage.morphology import skeletonize, binary_dilation, disk


def uniform_view(rgb, ink_mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    skel = skeletonize(ink_mask)
    m = binary_dilation(skel, disk(2))
    out = np.full(gray.shape, 255, dtype=np.uint8)
    out[m] = 0
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def pad_to_square(pil_img, fill=255):
    w, h = pil_img.size
    side = max(w, h)
    return ImageOps.pad(pil_img, (side, side), color=(fill, fill, fill), centering=(0.5, 0.5))


model, tfm, torch = _load_backbone()
recs = collect("Datasets/HandPD/**/*.jpg", "Datasets/kaggle/drawings/**/*.png")
print(f"[data] {len(recs)} drawings")

feats_stretch, feats_pad, cohorts = [], [], []
for rec in recs:
    bgr = cv2.imread(rec.path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = standardize_resolution(rgb, 512)
    ink = get_trace_mask(rgb)
    view = uniform_view(rgb, ink)
    pil = Image.fromarray(view)

    x_stretch = tfm(pil).unsqueeze(0)  # existing pipeline: forced-square Resize
    x_pad = tfm(pad_to_square(pil)).unsqueeze(0)  # pad-to-square first, then same Resize

    with torch.no_grad():
        feats_stretch.append(model(x_stretch).squeeze(0).numpy())
        feats_pad.append(model(x_pad).squeeze(0).numpy())
    cohorts.append(rec.cohort)

feats_stretch = np.stack(feats_stretch)
feats_pad = np.stack(feats_pad)
cohorts = np.array(cohorts)
y = (cohorts == "kaggle").astype(int)
subjects = np.array([r.subject for r in recs])


def cohort_probe(feats, label):
    gkf = GroupKFold(n_splits=5)
    probs = np.zeros(len(y)); preds = np.zeros(len(y), dtype=int)
    for tr, te in gkf.split(feats, y, groups=subjects):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(feats[tr], y[tr])
        probs[te] = clf.predict_proba(feats[te])[:, 1]
        preds[te] = clf.predict(feats[te])
    auc = roc_auc_score(y, probs)
    bal = balanced_accuracy_score(y, preds)
    print(f"[{label}] cohort AUC={auc:.4f}  BalAcc={bal:.4f}")


cohort_probe(feats_stretch, "forced-square stretch (existing pipeline)")
cohort_probe(feats_pad, "pad-to-square (aspect-preserving control)")
