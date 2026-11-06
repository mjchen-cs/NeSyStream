"""Cross-dataset validation on LIVE-NFLX-II (420 videos = 15 contents x 4 ABR algorithms x 7 cellular traces).

python scripts/nflx2_eval.py --json data/live_nflx2/nflx2_segments.json --out runs/e2_nflx2
1. Builds NS-QoE sessions (same keys as sqoe3_sessions.npz); target = retrospective z-scored MOS.
   level01 = log-bitrate position on the SQoE-III ladder (pre-declared), SceneCut = detected cut at the segment start,
   other visual concepts = 0 (videos not used). Writes P.1203 (mode 0) to <out>/p1203.npy (cache used by cv_sqoe3.py).
2. Zero-shot: metrics with no fitting on LIVE-NFLX-II (NS-QoE with the SQoE-III weights, P.1203, mean VMAF, ...).
3. Pairwise ranking: within each content x trace cell the 4 algorithms give 6 pairs; accuracy of each metric in
   ordering pairs whose MOS differs by more than a margin (human preference), and algorithm-level ranking.
"""
import _common  # noqa
import argparse, json, os, itertools
import numpy as np, torch
from nesystream.models.nsqoe import NSQoE
from nesystream.data.chunks import srocc, plcc

ap = argparse.ArgumentParser()
ap.add_argument("--json", default="data/live_nflx2/nflx2_segments.json"); ap.add_argument("--out", default="runs/e2_nflx2")
ap.add_argument("--weights", default="runs/e2_sqoe3/weights_mos.json"); ap.add_argument("--margin", type=float, default=0.1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
V = json.load(open(a.json))
LAD = np.log([170, 280, 400, 560, 660, 1300, 1650, 1850, 2900, 3500, 3900])   # SQoE-III rung bitrates (kbps, median)

# ---------------------------------------------------------------- sessions
contents = sorted({v["content"] for v in V}); N = max(len(v["segs"]) for v in V)
def pad(x, n=N):
    x = np.asarray(x, np.float32); return np.pad(x, [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1))
S = {k: [] for k in ["q", "level01", "rebuf", "buffer", "thr_ratio", "pos", "bitrate_kbps", "concepts", "mask"]}
for v in V:
    sg = v["segs"]; n = len(sg)
    S["q"].append(pad([s["vmaf"] for s in sg]))
    S["level01"].append(pad([np.interp(np.log(max(s["kbps"], 1)), LAD, np.linspace(0, 1, 11)) for s in sg]))
    S["rebuf"].append(pad([s["stall"] for s in sg])); S["buffer"].append(pad(np.full(n, 10.0)))
    S["thr_ratio"].append(pad(np.ones(n))); S["pos"].append(pad(np.arange(n) / n)); S["bitrate_kbps"].append(pad([s["kbps"] for s in sg]))
    c = np.zeros((n, 5)); c[:, 2] = [s["cut"] for s in sg]; S["concepts"].append(pad(c)); S["mask"].append(pad(np.ones(n)))
D = {k: np.stack(x) for k, x in S.items()}
D["video"] = np.array([contents.index(v["content"]) for v in V]); D["target"] = np.array([v["mos"] for v in V], np.float32)
np.savez_compressed(os.path.join(a.out, "nflx2_sessions.npz"), **D)
y = D["target"].astype(float)


# ---------------------------------------------------------------- objective scores (no fitting on this dataset)
def p1203_all():
    cache = os.path.join(a.out, "p1203.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from itu_p1203 import P1203Standalone
    import logging; logging.disable(logging.WARNING)
    out = []
    for v in V:
        segs, t = [], 0.0
        for s in v["segs"]:
            w = int(round(s["h"] * 16 / 9 / 2)) * 2
            segs.append({"bitrate": s["kbps"], "codec": "h264", "duration": s["dur"], "fps": v["fps"], "resolution": f"{w}x{s['h']}", "start": t})
            t += s["dur"]
        inp = {"I11": {"segments": [], "streamId": 1}, "I13": {"segments": segs, "streamId": 1},
               "I23": {"stalling": [[p, d] for p, d in v["stall_runs"]], "streamId": 1},
               "IGen": {"device": "pc", "displaySize": "1920x1080", "viewingDistance": "150cm"}}
        out.append(float(P1203Standalone(inp).calculate_complete()["O46"]))
    out = np.array(out); np.save(cache, out); return out


def nsqoe(weights=None):
    w = json.load(open(weights)) if weights else None
    m = NSQoE(True, True, init_weights={k: v for k, v in (w or {}).items() if not k.startswith("_")})
    with torch.no_grad():
        return m.session_score({k: torch.as_tensor(D[k]).float() for k in S}).numpy()


msk = D["mask"]; q = D["q"]
vm = (q * msk).sum(1) / msk.sum(1)
nst = np.array([v["n_stall"] for v in V]); sts = np.array([v["stall_s"] for v in V])
dq = np.array([np.abs(np.diff([s["vmaf"] for s in v["segs"]])).mean() for v in V])
scores = {"Mean VMAF": vm, "Stall time (neg.)": -sts, "Mean VMAF - stall (MPC-style, fixed)": vm - 4.3 * sts - dq,
          "ITU-T P.1203": p1203_all(), "NS-QoE, prior weights": nsqoe(None), "NS-QoE, SQoE-III weights (ours)": nsqoe(a.weights)}

# ---------------------------------------------------------------- zero-shot correlation
rng = np.random.default_rng(0)
def boot_ci(f, n=2000):    # content bootstrap
    cs = np.arange(len(contents)); vals = []
    for _ in range(n):
        idx = np.concatenate([np.where(D["video"] == c)[0] for c in rng.choice(cs, len(cs))]); vals.append(f(idx))
    return np.quantile(vals, [0.025, 0.975]).tolist()

stall = nst > 0; res = {}
print(f"{'metric':40s} SROCC [95% CI]         PLCC   SROCC stall/no-stall  per-content SROCC")
for k, s in scores.items():
    pc = np.mean([srocc(s[D['video'] == c], y[D['video'] == c]) for c in range(len(contents))])
    r = {"srocc": srocc(s, y), "srocc_ci": boot_ci(lambda i: srocc(s[i], y[i])), "plcc": plcc(s, y),
         "srocc_stall": srocc(s[stall], y[stall]), "srocc_nostall": srocc(s[~stall], y[~stall]), "srocc_per_content": float(pc)}
    res[k] = r
    print(f"{k:40s} {r['srocc']:.3f} [{r['srocc_ci'][0]:.3f},{r['srocc_ci'][1]:.3f}]  {r['plcc']:.3f}   {r['srocc_stall']:.3f} / {r['srocc_nostall']:.3f}        {pc:.3f}")

# ---------------------------------------------------------------- pairwise ranking within content x trace
cells = {}
for i, v in enumerate(V):
    cells.setdefault((v["content"], v["trace"]), []).append(i)
pairs = [(i, j) for c in cells.values() for i, j in itertools.combinations(c, 2) if abs(y[i] - y[j]) > a.margin]
print(f"\npairwise accuracy on {len(pairs)} same-content same-trace pairs with |dMOS| > {a.margin}:")
for k, s in scores.items():
    acc = np.mean([np.sign(s[i] - s[j]) == np.sign(y[i] - y[j]) for i, j in pairs])
    res[k]["pair_acc"] = float(acc); print(f"  {k:40s} {acc:.3f}")

# ---------------------------------------------------------------- algorithm level
algos = sorted({v["algo"] for v in V}); A = np.array([v["algo"] for v in V])
print("\nalgorithm-level means (MOS z / metrics):")
alg = {}
for al in algos:
    m = A == al; alg[al] = {"MOS": float(y[m].mean())} | {k: float(s[m].mean()) for k, s in scores.items()}
    print(f"  {al:40s} MOS {y[m].mean():+.3f}  " + "  ".join(f"{k.split(',')[0][:10]} {s[m].mean():.2f}" for k, s in scores.items()))
res["_algorithms"] = alg
json.dump(res, open(os.path.join(a.out, "zero_shot.json"), "w"), indent=1)

# ---------------------------------------------------------------- learned baselines trained on SQoE-III, applied zero-shot
E = dict(np.load("data/sqoe3/sqoe3_sessions.npz")); yE = E["target"].astype(float)
def lin_feats(X):
    q, r, m = X["q"], X["rebuf"], X["mask"]
    dq = (np.abs(np.diff(q, axis=1)) * m[:, 1:]).sum(1) / m.sum(1)
    return np.c_[(q * m).sum(1) / m.sum(1), dq, r[:, 1:].sum(1), (r[:, 1:] > 0).sum(1), r[:, 0], np.ones(len(q))]
c = np.linalg.lstsq(lin_feats(E), yE, rcond=None)[0]
scores["Linear QoE (fit on SQoE-III)"] = lin_feats(D) @ c
from nesystream.eval.learned_baselines import fit_rank_qoe
tmp = os.path.join(a.out, "_jade_sqoe3.npz"); np.savez(tmp, **{k: E[k] for k in ("q", "rebuf", "level01", "mask", "target")})
torch.manual_seed(0); jm = fit_rank_qoe(tmp)
with torch.no_grad():
    scores["Jade ranking MLP (fit on SQoE-III)"] = jm.session({k: torch.as_tensor(D[k]).float() for k in ("q", "rebuf", "level01", "mask")}).numpy()
ours = scores["NS-QoE, SQoE-III weights (ours)"]
print("\nzero-shot, added learned baselines + paired content-bootstrap SROCC difference (ours - other):")
for k, s in scores.items():
    if k not in res:
        res[k] = {"srocc": srocc(s, y), "srocc_ci": boot_ci(lambda i: srocc(s[i], y[i])), "plcc": plcc(s, y),
                  "pair_acc": float(np.mean([np.sign(s[i] - s[j]) == np.sign(y[i] - y[j]) for i, j in pairs]))}
    if k != "NS-QoE, SQoE-III weights (ours)":
        d = boot_ci(lambda i: srocc(ours[i], y[i]) - srocc(s[i], y[i])); res[k]["diff_ci_vs_ours"] = d
    print(f"  {k:40s} SROCC {res[k]['srocc']:.3f} PLCC {res[k]['plcc']:.3f} pair {res[k]['pair_acc']:.3f}  ours-minus CI {res[k].get('diff_ci_vs_ours')}")
json.dump(res, open(os.path.join(a.out, "zero_shot.json"), "w"), indent=1)
