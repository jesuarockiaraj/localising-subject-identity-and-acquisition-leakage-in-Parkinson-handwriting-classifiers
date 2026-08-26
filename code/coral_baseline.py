"""M10: one domain-generalisation baseline (Deep CORAL) for the cross-cohort
direction, reusing this study's exact backbone, preprocessing and harness so
the comparison to the plain baseline (Table 1) and to background suppression
(Table 4) is apples-to-apples.

CORAL (Sun & Saenko, 2016) adds a differentiable penalty that aligns the
second-order statistics (feature covariance) of the labelled source domain
(training cohort) and the unlabelled target domain (test cohort) during
training -- unlike background suppression, which intervenes on the data,
CORAL intervenes on the learned representation directly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from pd_loso_pipeline_cross_dataset import (
    Config, discover_samples, DrawingDataset, build_model, _predict,
)


def coral_loss(source, target):
    d = source.size(1)
    src_c = torch.cov(source.T) if source.size(0) > 1 else torch.zeros(d, d, device=source.device)
    tgt_c = torch.cov(target.T) if target.size(0) > 1 else torch.zeros(d, d, device=target.device)
    return (src_c - tgt_c).pow(2).sum() / (4 * d * d)


def features_of(model, x):
    """EfficientNet-B0 pooled features before the classifier head."""
    f = model.features(x)
    f = model.avgpool(f)
    return torch.flatten(f, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-root", default="Datasets/HandPD")
    ap.add_argument("--train-task", default="spiral")
    ap.add_argument("--test-root", default="Datasets/kaggle")
    ap.add_argument("--test-task", default="spiral")
    ap.add_argument("--direction", default="handpd_to_kaggle")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--coral-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="coral_results.json")
    args = ap.parse_args()

    train_samples = discover_samples(args.train_root, args.train_task)
    test_samples = discover_samples(args.test_root, args.test_task)
    print(f"[data] train {len(train_samples)} imgs, test {len(test_samples)} imgs")

    results = []
    for seed in args.seeds:
        cfg = Config(dataset="coral", root=args.train_root, task=args.train_task,
                     epochs=args.epochs, seed=seed)
        device = cfg.device
        torch.manual_seed(seed)
        model = build_model(args.model).to(device)

        y = np.array([s.label for s in train_samples])
        counts = np.bincount(y, minlength=2).astype(float)
        cw = torch.tensor((counts.sum() / (2 * np.maximum(counts, 1))),
                          dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=cw)

        tr_ds = DrawingDataset(train_samples, cfg.img_size, train=True)
        te_ds = DrawingDataset(test_samples, cfg.img_size, train=False)
        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
        te_loader_unlabelled = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        te_iter = iter(te_loader_unlabelled)
        for ep in range(cfg.epochs):
            model.train()
            for x, ytr in tr_loader:
                x, ytr = x.to(device), ytr.to(device)
                try:
                    xt, _ = next(te_iter)
                except StopIteration:
                    te_iter = iter(te_loader_unlabelled)
                    xt, _ = next(te_iter)
                xt = xt.to(device)

                opt.zero_grad()
                fs = features_of(model, x)
                ft = features_of(model, xt)
                logits = model.classifier(fs)
                cls_loss = crit(logits, ytr)
                c_loss = coral_loss(fs, ft)
                loss = cls_loss + args.coral_weight * c_loss
                loss.backward()
                opt.step()

        probs, ys = _predict(model, DataLoader(te_ds, batch_size=cfg.batch_size), device)
        auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan")
        print(f"[seed {seed}] {args.direction} CORAL AUC = {auc:.3f}")
        results.append({"seed": seed, "direction": args.direction, "auc": float(auc)})

    Path(args.out).write_text(json.dumps(results, indent=2))
    aucs = [r["auc"] for r in results]
    print(f"[done] {args.direction}: mean AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} "
          f"over {len(aucs)} seeds -> wrote {args.out}")


if __name__ == "__main__":
    main()
