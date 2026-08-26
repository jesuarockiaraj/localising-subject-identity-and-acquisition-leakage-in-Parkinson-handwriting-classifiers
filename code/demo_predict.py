"""Phase 2 prediction demo: PD-probability under full / background-only /
trace-only masking, for one image, using a REAL trained classifier.

This is the runnable illustration of the paper's central finding: background
suppression removes the acquisition/identity signal the background carries
(see leak_decomposition_probe.py), but the model's own prediction survives
largely intact across masking modes, because it is not purely relying on
background alone -- consistent with the trace-intrinsic result in Table 2 /
Section 3.2 of the manuscript.

Usage:
    python demo_predict.py --image Datasets/HandPD/SpiralPatients/0001-1.jpg
    python demo_predict.py --image Datasets/kaggle/drawings/spiral/testing/healthy/V01HE01.png

Requires a trained checkpoint (see train_demo_checkpoint.py); defaults to
demo_ckpt/efficientnet_b0_handpd_spiral_full.pt.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from mech2_suppression import get_trace_mask, standardize_resolution
from pd_loso_pipeline_cross_dataset import build_model, make_transforms

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


def guess_ground_truth(path: str):
    low = path.lower()
    if "patient" in low or "parkinson" in low:
        return "PD"
    if "control" in low or "healthy" in low:
        return "HC"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="demo_ckpt/efficientnet_b0_handpd_spiral_full.pt")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train_demo_checkpoint.py first to produce it.")
    meta_path = ckpt_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    img_size = meta.get("img_size", 224)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    tf = make_transforms(img_size, train=False)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = standardize_resolution(rgb, 512)
    ink_mask = get_trace_mask(rgb)

    gt = guess_ground_truth(args.image)
    print(f"\nImage: {args.image}")
    print(f"Checkpoint: {ckpt_path}  (val image-AUC {meta.get('val_image_auc', 'n/a')}, "
          f"protocol: {meta.get('protocol', 'n/a')})")
    print(f"Filename-inferred ground truth: {gt}  (not used by the model)\n")
    print(f"{'mode':<12}{'P(PD)':>10}{'predicted':>12}")
    print("-" * 34)
    with torch.no_grad():
        for mode in args.modes:
            view = make_view(rgb, ink_mask, mode)
            pil = Image.fromarray(view)
            x = tf(pil).unsqueeze(0).to(device)
            prob_pd = torch.softmax(model(x), dim=1)[0, 1].item()
            pred = "PD" if prob_pd >= 0.5 else "HC"
            print(f"{mode:<12}{prob_pd:>10.3f}{pred:>12}")

    print("\nIf P(PD) stays similar across modes, the model is not relying on "
          "background alone for this prediction -- consistent with the paper's "
          "finding that identity/cohort signal (and evidently task-relevant "
          "signal too) survives in the trace channel after background removal.")


if __name__ == "__main__":
    main()
