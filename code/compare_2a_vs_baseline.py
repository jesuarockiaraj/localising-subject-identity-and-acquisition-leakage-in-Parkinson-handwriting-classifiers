"""
compare_2a_vs_baseline.py
=========================
The Fork A/B go/no-go statistic for mechanism 2.

Takes two prediction CSVs written by pd_loso_pipeline_cross_dataset.py -- one
BASELINE (raw images), one TREATMENT (2a-suppressed) -- run over the SAME seeds
on the SAME subjects, and computes a SUBJECT-LEVEL PAIRED bootstrap on the AUC
delta. Comparing two independent point estimates (as the console does) ignores
the pairing and the shared subject variance; this does not.

Design:
  * aggregate to one mean prob per (seed, subject)  [clinical per-person op point]
  * per seed: inner-join baseline & treatment on subject, resample SUBJECTS with
    replacement, compute AUC(treat) - AUC(base) on the resampled set
  * pool bootstrap deltas across seeds -> mean dAUC, 95% CI, P(dAUC>0)

Decision mapping (per PROJECT_HANDOFF 4.5):
  CI wholly > 0 and probe confirms identity dropped  -> Fork A (mech2 helps)
  CI spans 0                                          -> Fork B (mech2 doesn't close gap)
  dAUC > 0 but PD-AUC rose while a NEW leak appeared  -> Fork C (debug, don't publish)

Honest caveats printed with the result. AUC is the primary endpoint for
cross-cohort (thresholded metrics aren't calibrated across cohorts).
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def per_subject(df, protocol=None):
    if protocol and "protocol" in df.columns:
        df = df[df.protocol == protocol]
    if df.empty:
        raise SystemExit(f"No rows for protocol={protocol}. "
                         f"Available: {sorted(df.get('protocol', pd.Series()).unique())}")
    keys = ["seed", "subject"] if "seed" in df.columns else ["subject"]
    g = df.groupby(keys).agg(y_true=("y_true", "max"),
                             y_prob=("y_prob", "mean")).reset_index()
    if "seed" not in g.columns:
        g["seed"] = 0
    return g


def paired_delta(base, treat, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    deltas, base_aucs, treat_aucs = [], [], []
    for s in sorted(set(base.seed) & set(treat.seed)):
        b = base[base.seed == s].set_index("subject")
        t = treat[treat.seed == s].set_index("subject")
        common = b.index.intersection(t.index)
        if len(common) < 4:
            continue
        b, t = b.loc[common], t.loc[common]
        yb, pb = b.y_true.values, b.y_prob.values
        yt, pt = t.y_true.values, t.y_prob.values  # same subjects/labels; yb==yt
        base_aucs.append(_auc(yb, pb))
        treat_aucs.append(_auc(yt, pt))
        idx = np.arange(len(common))
        for _ in range(n_boot):
            r = rng.choice(idx, size=len(idx), replace=True)
            if len(np.unique(yb[r])) < 2:
                continue
            da = _auc(yt[r], pt[r]) - _auc(yb[r], pb[r])
            if not np.isnan(da):
                deltas.append(da)
    if not deltas:
        raise SystemExit("No overlapping seeds/subjects with both classes -- check inputs.")
    d = np.array(deltas)
    return dict(
        base_auc=float(np.nanmean(base_aucs)),
        treat_auc=float(np.nanmean(treat_aucs)),
        dauc_mean=float(d.mean()),
        ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
        p_gt0=float((d > 0).mean()),
        n_seeds=len(base_aucs),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="baseline predictions_*.csv")
    ap.add_argument("--treat", required=True, help="2a predictions_*.csv")
    ap.add_argument("--protocol", default=None,
                    help="for within-cohort files: P2_subject (recommended) or P1_image. "
                         "Omit for cross-dataset predictions (no protocol column).")
    ap.add_argument("--model", default=None,
                    help="restrict to one architecture (e.g. efficientnet_b0). "
                         "If the CSV has a 'model' column and this is omitted, ALL "
                         "architectures present are pooled together per (seed, subject) "
                         "-- almost never what you want when the file mixes backbones.")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    def _load(path):
        df = pd.read_csv(path)
        if args.model and "model" in df.columns:
            df = df[df.model == args.model]
            if df.empty:
                raise SystemExit(f"No rows for model={args.model} in {path}. "
                                 f"Available: {sorted(pd.read_csv(path)['model'].unique())}")
        elif "model" in df.columns and df["model"].nunique() > 1:
            warnings.warn(
                f"{path} contains {df['model'].nunique()} architectures "
                f"({sorted(df['model'].unique())}) and --model was not set: they will be "
                f"POOLED together per (seed, subject). Pass --model to restrict to one.")
        return df

    base = per_subject(_load(args.baseline), args.protocol)
    treat = per_subject(_load(args.treat), args.protocol)
    r = paired_delta(base, treat, args.n_boot)

    tag = f"protocol={args.protocol}" if args.protocol else "cross-dataset"
    print(f"\n=== mechanism-2 (2a) vs baseline | {tag} ===")
    print(f"  subjects/seed pairing over {r['n_seeds']} seed(s)")
    print(f"  baseline  AUC : {r['base_auc']:.3f}")
    print(f"  2a        AUC : {r['treat_auc']:.3f}")
    print(f"  paired dAUC   : {r['dauc_mean']:+.3f}  "
          f"(95% CI {r['ci'][0]:+.3f} .. {r['ci'][1]:+.3f})")
    print(f"  P(dAUC > 0)   : {r['p_gt0']:.3f}")
    print()
    if r["ci"][0] > 0:
        print("  READ: CI wholly > 0 -> improvement is statistically supported.")
        print("        Now confirm via the leak probe that subject re-ID actually DROPPED")
        print("        (Fork A) and no new leak appeared (rule out Fork C) before claiming it.")
    elif r["ci"][1] < 0:
        print("  READ: CI wholly < 0 -> 2a HURT. Likely deleted signal (esp. if 2b was on).")
    else:
        print("  READ: CI spans 0 -> no supported improvement. This is a Fork-B signal:")
        print("        mechanism 2 alone does not close the honest gap. Fold the null into")
        print("        Paper 1, or route the budget to mechanisms 1/3.")
    print("\n  Caveat: AUC delta is the endpoint; pair it with the re-ID probe on the SAME")
    print("  suppressed images. A rise here without a re-ID drop there is not a mechanism win.")


if __name__ == "__main__":
    main()
