"""Reproduces Table 6 (calibration): Brier score and expected calibration
error (ECE, 10 equal-width bins) for EfficientNet-B0's subject-level
predicted probabilities under P1, P2, and both cross-cohort directions.

Reuses the ORIGINAL (pre-correction) predictions already generated for
Table 1, per the manuscript's own disclosure that this table carries the
same provenance caveat as Table 1's original predictions -- it is not
re-run on the newly verified seeds. Was originally an ad hoc calculation;
this script makes it reproducible.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

MODEL = "efficientnet_b0"
N_BINS = 10


def ece(y_true, y_prob, n_bins=N_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n = len(y_true)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        total += (mask.sum() / n) * abs(acc - conf)
    return total


def subject_level(df, protocol=None):
    if protocol is not None:
        df = df[df.protocol == protocol]
    df = df[df.model == MODEL]
    return (df.groupby(["seed", "subject"])
              .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
              .reset_index())


def per_seed_metrics(df):
    briers, eces = [], []
    for seed, g in df.groupby("seed"):
        briers.append(brier_score_loss(g.y_true, g.y_prob))
        eces.append(ece(g.y_true.to_numpy(), g.y_prob.to_numpy()))
    return np.mean(briers), np.std(briers), np.mean(eces), np.std(eces)


def main():
    handpd = pd.read_csv("results_baseline/predictions_handpd_spiral.csv")
    h2k = pd.read_csv("results_baseline/predictions_cross_handpd_spiral_to_kaggle_spiral.csv")
    k2h = pd.read_csv("results_baseline/predictions_cross_kaggle_spiral_to_handpd_spiral.csv")

    rows = []
    for label, df, protocol in [
        ("P1 (image-level)", handpd, "P1_image"),
        ("P2 (subject-grouped)", handpd, "P2_subject"),
    ]:
        sub = subject_level(df, protocol)
        b_mean, b_sd, e_mean, e_sd = per_seed_metrics(sub)
        rows.append((label, b_mean, b_sd, e_mean, e_sd))

    for label, df in [
        ("Cross-cohort HandPD->Kaggle", h2k),
        ("Cross-cohort Kaggle->HandPD", k2h),
    ]:
        sub = subject_level(df, protocol=None)
        b_mean, b_sd, e_mean, e_sd = per_seed_metrics(sub)
        rows.append((label, b_mean, b_sd, e_mean, e_sd))

    out = pd.DataFrame(rows, columns=["regime", "brier_mean", "brier_sd", "ece_mean", "ece_sd"])
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out.to_csv("calibration_table.csv", index=False)
    print("\n[done] wrote calibration_table.csv")


if __name__ == "__main__":
    main()
