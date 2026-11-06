"""Ranking consistency of the E3 comparison under several QoE models (not only the ones we optimise).

python scripts/qoe_metric_consistency.py --sessions runs/e3_sessions_compact.json --out runs/e3_metrics
Per test session we have mean VMAF, stall ratio, mean |dVMAF|, data, P.1203 and NS-QoE (SQoE-III weights).
Metrics:
  P.1203 (ITU-T, mode 0) | NS-QoE (ours, weights fit to SQoE-III MOS) | mean VMAF | -stall ratio |
  Linear QoE fit to SQoE-III MOS | Linear QoE fit to LIVE-NFLX-II MOS (independent viewers, cellular traces)
  Linear QoE = a*meanVMAF + b*mean|dVMAF| + c*stall ratio + d, least squares on each subjective dataset.
Outputs per trace set: each metric's method ranking, Kendall tau to the P.1203 ranking, rank of NeSyStream,
and the paired difference NeSyStream - RobustMPC with a trace-bootstrap 95% interval.
"""
import _common  # noqa
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", default="runs/e3_sessions_compact.json"); ap.add_argument("--out", default="runs/e3_metrics")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
NAMES = {"BBA": "BBA", "RB": "RB", "RobustMPC-quality@final2": "RobustMPC", "pensieve": "Pensieve", "ppo_lag": "PPO-Lagrangian",
         "ppo_vmaf": "PPO-VMAF", "jade": "Jade", "DIPO-best8": "DIPO", "QSM-best8": "QSM",
         "Ours-student-1step-best8": "Feasible-set diffusion", "Ours-plan3-all": "NeSyStream"}


def fit_linear(feat, y):
    X = np.c_[feat, np.ones(len(feat))]; return np.linalg.lstsq(X, y, rcond=None)[0]


# ---- subjective datasets -> (mean VMAF, mean |dVMAF|, stall ratio)
def feats_npz(path, stall_from_row1=True, seg_dur=2.0):
    D = dict(np.load(path)); q, m, r = D["q"], D["mask"], D["rebuf"]
    play = m.sum(1) * seg_dur; st = r.sum(1)
    dq = (np.abs(np.diff(q, axis=1)) * m[:, 1:]).sum(1) / np.maximum(m.sum(1) - 1, 1)
    return np.c_[(q * m).sum(1) / m.sum(1), dq, st / (st + play)], D["target"].astype(float)

Fs, ys = feats_npz("data/sqoe3/sqoe3_sessions.npz")
Fl, yl = feats_npz("runs/e2_nflx2/nflx2_sessions.npz", seg_dur=2.0)
cs, cl = fit_linear(Fs, ys), fit_linear(Fl, yl)
coef = {"SQoE-III": cs.tolist(), "LIVE-NFLX-II": cl.tolist()}
print("linear QoE coefficients [VMAF, |dVMAF|, stall ratio, const]:", json.dumps(coef))

S = json.load(open(a.sessions)); res = {"coef": coef, "sets": {}}
rng = np.random.default_rng(0)
METRICS = ["P.1203", "NS-QoE (SQoE-III)", "Linear QoE (SQoE-III)", "Linear QoE (LIVE-NFLX-II)", "Mean VMAF", "-Stall ratio"]


def kendall(r1, r2):
    n = len(r1); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(r1[i] - r1[j]) * np.sign(r2[i] - r2[j]); c += s > 0; d += s < 0
    return (c - d) / (n * (n - 1) / 2)


summary_rank = {m: [] for m in METRICS}; summary_tau = {m: [] for m in METRICS}
for s, meth in S.items():
    key = {(x[0], x[1]): i for i, x in enumerate(meth["Ours-plan3-all"])}
    vals = {}
    for k, nm in NAMES.items():
        arr = np.zeros((len(key), 6))
        for x in meth[k]:
            arr[key[(x[0], x[1])]] = x[2:8]
        vm, rr, sw, mb, p12, ns = arr.T
        F = np.c_[vm, sw, rr]
        vals[nm] = {"P.1203": p12, "NS-QoE (SQoE-III)": ns, "Linear QoE (SQoE-III)": np.c_[F, np.ones(len(F))] @ cs,
                    "Linear QoE (LIVE-NFLX-II)": np.c_[F, np.ones(len(F))] @ cl, "Mean VMAF": vm, "-Stall ratio": -rr}
    traces = np.array([x[1] for x in meth["Ours-plan3-all"]]); ut = np.unique(traces)
    meths = list(vals); out = {}
    ref = np.array([vals[m]["P.1203"].mean() for m in meths])
    for met in METRICS:
        means = np.array([vals[m][met].mean() for m in meths]); order = [meths[i] for i in np.argsort(-means)]
        d = vals["NeSyStream"][met] - vals["RobustMPC"][met]
        per_tr = np.array([d[traces == t].mean() for t in ut])
        bs = [per_tr[rng.integers(0, len(ut), len(ut))].mean() for _ in range(2000)]
        rank = order.index("NeSyStream") + 1; tau = kendall(means, ref)
        out[met] = {"order": order, "means": dict(zip(meths, means.tolist())), "rank_ours": rank, "tau_vs_p1203": tau,
                    "diff_vs_mpc": float(d.mean()), "ci": np.quantile(bs, [0.025, 0.975]).tolist()}
        summary_rank[met].append(rank); summary_tau[met].append(tau)
    res["sets"][s] = out
    print(f"\n== {s}")
    for met in METRICS:
        o = out[met]
        print(f"  {met:28s} ours rank {o['rank_ours']:2d}/11  tau(P.1203) {o['tau_vs_p1203']:+.2f}  ours-MPC {o['diff_vs_mpc']:+.4f} "
              f"[{o['ci'][0]:+.4f},{o['ci'][1]:+.4f}]  top3 {o['order'][:3]}")
print("\nsummary (5 sets): rank of NeSyStream / mean Kendall tau to P.1203 ranking")
for met in METRICS:
    print(f"  {met:28s} ranks {summary_rank[met]}  tau {np.mean(summary_tau[met]):+.2f}")
json.dump(res, open(os.path.join(a.out, "metric_consistency.json"), "w"), indent=1)
