"""Reviewer W2 follow-up: suppressed-image P1 (image-level) AUC was never
reported -- only P2. Does suppression shrink the P1-P2 identity-leakage
gap, the test that matches the paper's own title? Both P1 and P2 rows
already exist in the verified suppressed predictions CSV from the same
training run, so this needs no new training, just aggregation."""
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("results_2a_verify_seeds_999_1001_1002_1003/predictions_handpd_spiral.csv")
df = df[df.model == "efficientnet_b0"]

for protocol in ["P1_image", "P2_subject"]:
    sub = df[df.protocol == protocol]
    aucs = []
    for seed, g in sub.groupby("seed"):
        agg = g.groupby("subject").agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
        aucs.append(roc_auc_score(agg.y_true, agg.y_prob))
    import numpy as np
    print(f"Suppressed {protocol}: AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (seeds: {len(aucs)})")

print("\nRaw (for comparison, from verified baseline): P1=0.991+/-0.006, P2=0.944+/-0.005")
