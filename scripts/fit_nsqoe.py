"""Fit NS-QoE rule weights (E2).

Playback rules: fit to P.1203 scores (teacher); visual rules: fit to MOS of the
self-built subjective dataset (--mos_csv with columns id,mos; ids from *_sessions.json).
Reports SROCC/PLCC on held-out videos and the learned / induced rule table.
"""
import _common  # noqa
import argparse, csv, json
import numpy as np, torch
from nesystream.models.nsqoe import NSQoE
from nesystream.data.chunks import srocc, plcc

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mos_csv", default=None); ap.add_argument("--init", default=None, help="weights json to start from")
ap.add_argument("--freeze", default="", help="comma list of rule-name prefixes to keep fixed")
ap.add_argument("--no_hand", action="store_true"); ap.add_argument("--no_induced", action="store_true")
ap.add_argument("--model", default="nsqoe", choices=["nsqoe", "neural"], help="neural = GRU black-box baseline")
ap.add_argument("--lr_w", type=float, default=0.05, help="learning rate of the (log) rule weights")
ap.add_argument("--l1", type=float, default=0.02); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--balance", action="store_true",
                help="sessions with / without a mid-session stall get equal total weight (stalls are rare)")
ap.add_argument("--test_frac", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
d = dict(np.load(a.sessions, allow_pickle=False))
if a.mos_csv:
    mos = {int(r["id"]): float(r["mos"]) for r in csv.DictReader(open(a.mos_csv))}
    keep = np.array(sorted(mos)); d = {k: v[keep] for k, v in d.items()}; d["target"] = np.array([mos[i] for i in keep], np.float32)
y = d["target"]; ok = np.isfinite(y)
d = {k: v[ok] for k, v in d.items()}; y = y[ok]
vids = np.unique(d["video"]); rng = np.random.default_rng(a.seed)
test_v = set(rng.choice(vids, max(1, int(len(vids) * a.test_frac)), replace=False))
te = np.array([v in test_v for v in d["video"]]); tr = ~te
T = lambda m: {k: torch.as_tensor(v[m]).float() for k, v in d.items() if k not in ("video", "bitrate_kbps", "target")}
init = json.load(open(a.init)) if a.init else None
if a.model == "neural":
    from nesystream.eval.learned_baselines import NeuralQoE
    model = NeuralQoE(); frozen = []
    model.sparsity = lambda: torch.tensor(0.0)
else:
    model = NSQoE(not a.no_hand, not a.no_induced, init_weights=init)
    frozen = [i for i, (n, _, _) in enumerate(model.rules) if any(n.startswith(p) for p in a.freeze.split(",") if p)]
    model.use_log_param()   # weights may need to move by orders of magnitude (e.g. stalls vs. P.1203 / MOS)
if a.model == "nsqoe":
    # closed-form start for the affine map (alpha, beta) so the rule weights do not fight the scale
    with torch.no_grad():
        model.alpha.fill_(1.0); model.beta.fill_(0.0)
        s0 = model.session_score(T(tr)).numpy()
        A = np.c_[s0, np.ones(len(s0))]; c = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        model.alpha.fill_(float(c[0])); model.beta.fill_(float(c[1]))
    opt = torch.optim.Adam([{"params": [model.raw_w], "lr": a.lr_w},
                            {"params": [p for n, p in model.named_parameters() if n != "raw_w"], "lr": 0.02}])
else:
    opt = torch.optim.Adam(model.parameters(), 0.02)
s_tr, y_tr = T(tr), torch.as_tensor(y[tr])
stall = (d["rebuf"][:, 1:] * d["mask"][:, 1:]).sum(1) > 0          # mid-session stall (chunk 0 = startup)
wt = np.ones(len(y), np.float32)
if a.balance and 0 < stall[tr].sum() < tr.sum():
    wt[stall] = 0.5 / stall[tr].sum(); wt[~stall] = 0.5 / (~stall[tr]).sum()
    wt *= tr.sum()
w_tr = torch.as_tensor(wt[tr])
print(f"sessions with a mid-session stall: {stall.mean():.1%} ({int(stall.sum())})")
for it in range(a.steps):
    pred = model.session_score(s_tr)
    loss = (w_tr * (pred - y_tr) ** 2).mean() + a.l1 * model.sparsity()
    opt.zero_grad(); loss.backward()
    if frozen: model.raw_w.grad[frozen] = 0
    opt.step()
with torch.no_grad():
    p_te = model.session_score(T(te)).numpy(); p_tr = model.session_score(s_tr).numpy()
st_te = stall[te]
res = {"train_srocc": srocc(p_tr, y[tr]), "test_srocc": srocc(p_te, y[te]), "test_plcc": plcc(p_te, y[te]),
       "test_srocc_stall_sessions": srocc(p_te[st_te], y[te][st_te]) if st_te.sum() > 2 else float("nan"),
       "test_srocc_stall_vs_not": srocc(p_te, st_te.astype(float) * -1) if 0 < st_te.sum() < len(st_te) else float("nan"),
       "n_train": int(tr.sum()), "n_test": int(te.sum()),
       "n_params": sum(p.numel() for p in model.parameters()),
       "rules": [{"name": n, "body": b, "weight": w} for n, b, w in model.rule_table()] if a.model == "nsqoe" else []}
print(json.dumps({k: v for k, v in res.items() if k != "rules"}, indent=1))
for r in res["rules"]: print(f"  {r['weight']:8.3f}  {r['name']:28s} {r['body']}")
if a.model == "nsqoe":
    json.dump({n: w for n, _, w in model.rule_table(prune=0.0)}, open(a.out, "w"), indent=1)
json.dump(res, open(a.out.replace(".json", "_report.json"), "w"), indent=1)
