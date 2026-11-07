"""Analysis of the subjective test (ACR, 5-point).

python subjective/analyze.py --ratings subjective/data/ratings.csv --stimuli subjective/stimuli/stimuli.json
1. keeps test trials only (training is discarded) and, per subject, the last answer to each stimulus;
2. screens subjects as in ITU-T P.913 (Sec. 11.4): a subject is rejected if the Pearson correlation of their
   scores with the mean of the other subjects is below --min_r (default 0.75); rejected subjects are reported;
   screening needs at least --min_subjects_screen subjects (default 6) and is skipped below that;
3. MOS per method with 95% t-intervals over subjects (each subject's mean per method), per network group;
4. paired comparison NeSyStream vs. each method: per-subject mean difference, Wilcoxon signed-rank test and a
   subject-bootstrap 95% interval, plus the share of (subject, trace) pairs in which NeSyStream scored higher;
5. agreement of the objective P.1203 of each stimulus with its MOS (Pearson / Spearman).
"""
import argparse, csv, json, collections
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--ratings", default="subjective/data/ratings_valid.csv", help="comma list of rating files"); ap.add_argument("--stimuli", default="subjective/stimuli/stimuli.json")
ap.add_argument("--min_r", type=float, default=0.75); ap.add_argument("--min_subjects_screen", type=int, default=6); ap.add_argument("--ref", default="NeSyStream"); ap.add_argument("--out", default="subjective/data/results.json")
a = ap.parse_args()
ST = {s["file"]: s for s in json.load(open(a.stimuli))["stimuli"]}
R = collections.defaultdict(dict)
for r in (r for fn in a.ratings.split(",") for r in csv.DictReader(open(fn, encoding="utf-8-sig"))):
    if r["phase"] == "test" and r["file"] in ST:
        if r.get("complete", "1") != "1":            # clip not watched in full at normal speed: not a valid trial
            continue
        R[r["subject"]][r["file"]] = float(r["rating"])
files = sorted(f for f in ST if ST[f]["phase"] == "test")
subs = sorted(s for s in R if len(R[s]) == len(files))
print(f"subjects with complete data: {len(subs)} / {len(R)}")
X = np.array([[R[s][f] for f in files] for s in subs])
keep = []
screen = len(subs) >= a.min_subjects_screen   # the mean of the others is only a reliable reference with enough subjects
if not screen:
    print(f"  screening skipped: {len(subs)} subjects < {a.min_subjects_screen} (run again when all subjects are done)")
for k, s in enumerate(subs):
    others = np.delete(X, k, 0).mean(0); r = np.corrcoef(X[k], others)[0, 1]
    ok = (not screen) or r >= a.min_r
    keep.append(ok); print(f"  {s}: r = {r:.2f}{'' if ok else '  -> rejected'}")
X = X[np.array(keep)]; subs = [s for s, k in zip(subs, keep) if k]
print(f"kept {len(subs)} subjects")
methods = sorted({ST[f]["method"] for f in files}, key=lambda m: (m != a.ref, m))
groups = {"all": lambda s: True, "3G/broadband": lambda s: s["set"] in ("hsdpa", "fcc"), "4G/5G": lambda s: s["set"] in ("lumos4g", "lumos5g")}
res = {"n_subjects": len(subs), "rejected": int(len(keep) - sum(keep)), "mos": {}, "vs_ref": {}}


def ci(v):
    v = np.asarray(v); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


for g, sel in groups.items():
    idx = {m: [j for j, f in enumerate(files) if ST[f]["method"] == m and sel(ST[f])] for m in methods}
    if not idx[methods[0]]:
        continue
    per = {m: X[:, idx[m]].mean(1) for m in methods}           # per-subject mean score of each method
    res["mos"][g] = {m: ci(per[m]) for m in methods}
    print(f"\n[{g}] MOS (95% CI over subjects)")
    for m in methods:
        print(f"  {m:16s} {res['mos'][g][m][0]:.2f} [{res['mos'][g][m][1]:.2f}, {res['mos'][g][m][2]:.2f}]")
    rng = np.random.default_rng(0)
    for m in methods[1:]:
        d = per[a.ref] - per[m]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        p = stats.wilcoxon(per[a.ref], per[m]).pvalue if np.any(d != 0) else 1.0
        # pairs with the same trace: who scored higher
        tr = collections.defaultdict(dict)
        for j, f in enumerate(files):
            if sel(ST[f]): tr[ST[f]["trace"]][ST[f]["method"]] = j
        wins = [np.sign(X[:, t[a.ref]] - X[:, t[m]]) for t in tr.values() if a.ref in t and m in t]
        wins = np.concatenate(wins)
        res["vs_ref"].setdefault(g, {})[m] = {"diff": float(d.mean()), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                              "wilcoxon_p": float(p), "win": float((wins > 0).mean()), "tie": float((wins == 0).mean())}
        v = res["vs_ref"][g][m]
        print(f"  {a.ref} - {m}: {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}], Wilcoxon p = {v['wilcoxon_p']:.3g}, "
              f"higher in {100 * v['win']:.0f}% / tie {100 * v['tie']:.0f}% of (subject, trace) pairs")
mos = X.mean(0); p12 = np.array([ST[f]["p1203"] for f in files])
res["p1203_vs_mos"] = {"pearson": float(stats.pearsonr(p12, mos)[0]), "spearman": float(stats.spearmanr(p12, mos)[0])}
print(f"\nP.1203 vs. MOS over {len(files)} stimuli: Pearson {res['p1203_vs_mos']['pearson']:.2f}, Spearman {res['p1203_vs_mos']['spearman']:.2f}")
json.dump(res, open(a.out, "w"), indent=1); print("->", a.out)
