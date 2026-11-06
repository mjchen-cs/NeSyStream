"""Content-bootstrap intervals of the NS-QoE rule weights on SQoE-III (resample the 20 contents with replacement).
python scripts/bootstrap_nsqoe_sqoe3.py --sessions data/sqoe3/sqoe3_sessions.npz --n 50 --out runs/e2_sqoe3_final/bootstrap.json"""
import _common  # noqa
import argparse, json
import numpy as np, torch
from nesystream.models.nsqoe import NSQoE
ap = argparse.ArgumentParser(); ap.add_argument("--sessions", default="data/sqoe3/sqoe3_sessions.npz")
ap.add_argument("--n", type=int, default=50); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--out", default="runs/e2_sqoe3_final/bootstrap.json")
a = ap.parse_args()
D = dict(np.load(a.sessions)); y = D["target"]; vid = D["video"]; rng = np.random.default_rng(0); W = []
for b in range(a.n):
    cs = rng.choice(np.unique(vid), len(np.unique(vid)), replace=True)
    idx = np.concatenate([np.where(vid == c)[0] for c in cs])
    s = {k: torch.as_tensor(v[idx]).float() for k, v in D.items() if k not in ("video", "bitrate_kbps", "target")}
    yt = torch.as_tensor(y[idx]).float()
    torch.manual_seed(0); m = NSQoE(True, False); m.use_log_param()
    with torch.no_grad():
        s0 = m.session_score(s).numpy(); c = np.linalg.lstsq(np.c_[s0, np.ones(len(s0))], y[idx], rcond=None)[0]
        m.alpha.fill_(float(c[0])); m.beta.fill_(float(c[1]))
    opt = torch.optim.Adam([{"params": [m.raw_w], "lr": 0.05}, {"params": [m.alpha, m.beta], "lr": 0.02}])
    for _ in range(a.steps):
        l = ((m.session_score(s) - yt) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
    W.append({n: w for n, _, w in m.rule_table(prune=0.0)})
res = {k: {"median": float(np.median([w[k] for w in W])), "q05": float(np.quantile([w[k] for w in W], .05)),
           "q95": float(np.quantile([w[k] for w in W], .95))} for k in W[0]}
res["_P(switch_mid_shot > switch_at_cut)"] = float(np.mean([w["switch_mid_shot"] > w["switch_at_cut"] for w in W]))
res["_P(rebuffer_recency > 0.5)"] = float(np.mean([w["rebuffer_recency"] > 0.5 for w in W]))
res["_P(low_rate > 1)"] = float(np.mean([w["low_rate"] > 1 for w in W]))
json.dump(res, open(a.out, "w"), indent=1)
for k, v in res.items():
    print(k, v if k.startswith("_") else f"median {v['median']:.3f} [{v['q05']:.3f}, {v['q95']:.3f}]")
