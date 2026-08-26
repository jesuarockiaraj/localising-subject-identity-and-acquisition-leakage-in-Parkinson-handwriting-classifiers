"""M8 fix: subject re-identification stratified by cohort and task, each
against a chance level computed under the same restriction (reviewer M8:
pooling cohorts/tasks understates chance for a structurally constrained
retrieval, since cohort is perfectly decodable and nearest neighbours can
never cross it in practice)."""
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, ".")
from leak_decomposition_probe import collect

feats = np.load("scratch_feats_full.npy")
subjects = np.load("scratch_subjects_full.npy", allow_pickle=True)
recs = collect("Datasets/HandPD/**/*.jpg", "Datasets/kaggle/drawings/**/*.png")
assert len(recs) == len(subjects)
cohorts = np.array([r.cohort for r in recs])


def task_of(path):
    p = path.lower()
    if "meander" in p:
        return "meander"
    if "wave" in p:
        return "wave"
    return "spiral"


tasks = np.array([task_of(r.path) for r in recs])

norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
sim_full = norm @ norm.T
np.fill_diagonal(sim_full, -1)


def stratum_stats(mask, label):
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return
    sub_sim = sim_full[np.ix_(idx, idx)]
    sub_subj = subjects[idx]
    hits1, hits5, aps = [], [], []
    for i in range(len(idx)):
        order = np.argsort(-sub_sim[i])
        ranked_same = (sub_subj[order] == sub_subj[i])
        hits1.append(ranked_same[0])
        hits5.append(ranked_same[:5].any())
        n_rel = int((sub_subj == sub_subj[i]).sum()) - 1
        if n_rel > 0:
            rel = ranked_same.astype(float)
            csum = np.cumsum(rel)
            precs = csum / (np.arange(len(rel)) + 1)
            aps.append(float((precs * rel).sum() / n_rel))
    cnt = Counter(sub_subj)
    chance = float(np.mean([(c - 1) / (len(idx) - 1) for c in cnt.values()]))
    print(f"{label:28s} n={len(idx):4d}  reID@1={np.mean(hits1):.3f}  "
          f"reID@5={np.mean(hits5):.3f}  mAP={np.mean(aps):.3f}  chance={chance:.4f}")


print(f"{'stratum':28s} {'n':>6s}  {'reID@1':>7s}  {'reID@5':>7s}  {'mAP':>6s}  {'chance':>7s}")
stratum_stats(np.ones(len(subjects), dtype=bool), "pooled (Table 2 as reported)")
stratum_stats(cohorts == "handpd", "within-cohort: HandPD")
stratum_stats(cohorts == "kaggle", "within-cohort: Kaggle")
stratum_stats((cohorts == "handpd") & (tasks == "spiral"), "HandPD spiral only")
stratum_stats((cohorts == "handpd") & (tasks == "meander"), "HandPD meander only")
stratum_stats((cohorts == "kaggle") & (tasks == "spiral"), "Kaggle spiral only")
stratum_stats((cohorts == "kaggle") & (tasks == "wave"), "Kaggle wave only")
