"""Train and save ONE full EfficientNet-B0 checkpoint (encoder + classifier
head) under the honest subject-grouped protocol (P2), for use by
demo_predict.py.

This is deliberately separate from pd_loso_pipeline_cross_dataset.py: that
harness evaluates many (model, seed) combinations and only ever writes
prediction CSVs, never the trained weights themselves. The mech1 checkpoints
under ckpt_fold/ and ckpt_handpd_spiral/ are encoder-only (classifier head
stripped, for the identity-decodability probe) and cannot produce a
PD-probability. This script exists to produce one that can.

Reuses discover_samples / make_transforms / build_model / train_one_fold
from pd_loso_pipeline_cross_dataset.py so preprocessing and architecture
exactly match what the paper's numbers are based on.
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedGroupKFold

from pd_loso_pipeline_cross_dataset import (
    Config, discover_samples, DrawingDataset,
    build_model, train_one_fold, _predict,
)
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="Datasets/HandPD")
    ap.add_argument("--task", default="spiral")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="demo_ckpt/efficientnet_b0_handpd_spiral_full.pt")
    args = ap.parse_args()

    cfg = Config(dataset="handpd", root=args.root, task=args.task,
                 models=(args.model,), seed=args.seed).finalize()

    samples = discover_samples(args.root, args.task)
    labels = [s.label for s in samples]
    groups = [s.subject for s in samples]

    # one honest, subject-grouped 80/20-ish split (first fold of a 5-fold
    # StratifiedGroupKFold) -- same splitter class the P2 protocol uses.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    train_idx, val_idx = next(sgkf.split(samples, labels, groups))
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    print(f"[split] train {len(train_samples)} imgs "
          f"({len(set(groups[i] for i in train_idx))} subjects) | "
          f"val {len(val_samples)} imgs "
          f"({len(set(groups[i] for i in val_idx))} subjects)")

    tr_ds = DrawingDataset(train_samples, cfg.img_size, train=True)
    va_ds = DrawingDataset(val_samples, cfg.img_size, train=False)

    model = build_model(args.model)
    model = train_one_fold(model, tr_ds, va_ds, cfg)

    probs, ys = _predict(model, DataLoader(va_ds, batch_size=cfg.batch_size), cfg.device)
    val_auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan")
    print(f"[val] image-level AUC on held-out subjects: {val_auc:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    meta = {
        "model": args.model, "root": args.root, "task": args.task,
        "seed": cfg.seed, "img_size": cfg.img_size,
        "protocol": "P2_subject (StratifiedGroupKFold, fold 0 of 5)",
        "val_image_auc": float(val_auc),
        "note": "FULL checkpoint (encoder + classifier head), for demo_predict.py",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {out_path} and {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
