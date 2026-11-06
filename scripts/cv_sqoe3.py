"""E2 on real MOS: leave-contents-out cross-validation on Waterloo SQoE-III (450 sessions, 20 contents).

python scripts/cv_sqoe3.py --sessions data/sqoe3/sqoe3_sessions.npz --root data/sqoe3_raw --out runs/e2_sqoe3
Folds: contents are split into 5 groups (4 contents each); repeated with --repeats different content partitions.
Models: mean VMAF (affine), linear QoE (MPC-style: q, |dq|, stall, start-up; least squares), P.1203 (affine),
GRU black box, NS-QoE hand rules, NS-QoE hand + induced rules. Weights of the final NS-QoE are fit on all sessions.
"""
import _common  # noqa
import argparse, csv, json, os
import numpy as np, torch
from nesystream.models.nsqoe import NSQoE
from nesystream.data.chunks import srocc, plcc

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", default="data/sqoe3/sqoe3_sessions.npz"); ap.add_argument("--root", default="data/sqoe3_raw")
ap.add_argument("--out", default="runs/e2_sqoe3"); ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--l1", type=float, default=0.02); ap.add_argument("--models", default="all")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = dict(np.load(a.sessions)); y = D["target"].astype(np.float64); vid = D["video"]
T = lambda m: {k: torch.as_tensor(v[m]).float() for k, v in D.items() if k not in ("video", "bitrate_kbps", "target")}


def feats_linear(m):
    q, r, msk = D["q"][m], D["rebuf"][m], D["mask"][m]
    dq = np.abs(np.diff(q, axis=1)).sum(1)
    return np.c_[(q * msk).sum(1) / msk.sum(1), dq, r[:, 1:].sum(1), (r[:, 1:] > 0).sum(1), r[:, 0], np.ones(m.sum())]


def p1203_all():
    cache = os.path.join(a.out, "p1203.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from itu_p1203 import P1203Standalone
    import logging
    logging.disable(logging.WARNING)
    rows = list(csv.DictReader(open(os.path.join(a.root, "data.csv"))))
    out = []
    for r in rows:
        log = list(csv.DictReader(open(os.path.join(a.root, "streaming_logs", r["streaming_log"]))))
        segs, stalls, t = [], [], 0.0
        for x in log:
            st = float(x["rebuffering_duration"])
            if st > 0:
                stalls.append([t, st])
            segs.append({"bitrate": float(x["video_bitrate"]), "codec": "h264", "duration": float(x["chunk_duration"]),
                         "fps": float(x["framerate"]), "resolution": f"{x['width']}x{x['height']}", "start": t})
            t += float(x["chunk_duration"])
        inp = {"I11": {"segments": [], "streamId": 42}, "I13": {"segments": segs, "streamId": 42},
               "I23": {"stalling": stalls, "streamId": 42},
               "IGen": {"device": "pc", "displaySize": "1920x1080", "viewingDistance": "150cm"}}
        out.append(float(P1203Standalone(inp).calculate_complete()["O46"]))
    out = np.array(out); np.save(cache, out); return out


def fit_affine(x_tr, y_tr, x_te):
    c = np.linalg.lstsq(np.c_[x_tr, np.ones(len(x_tr))], y_tr, rcond=None)[0]
    return np.c_[x_te, np.ones(len(x_te))] @ c


def fit_nsqoe(tr, te, induced, steps, visual=True, learn_w=True):
    torch.manual_seed(0)
    model = NSQoE(True, induced); model.use_log_param()
    TT = T if visual else (lambda m: {**T(m), "concepts": torch.zeros_like(T(m)["concepts"])})
    s_tr, y_tr = TT(tr), torch.as_tensor(y[tr]).float()
    with torch.no_grad():
        s0 = model.session_score(s_tr).numpy()
        c = np.linalg.lstsq(np.c_[s0, np.ones(len(s0))], y[tr], rcond=None)[0]
        model.alpha.fill_(float(c[0])); model.beta.fill_(float(c[1]))
    opt = torch.optim.Adam([{"params": [model.raw_w], "lr": 0.05},
                            {"params": [p for n, p in model.named_parameters() if n != "raw_w"], "lr": 0.02}])
    for _ in range(steps if learn_w else 0):
        loss = ((model.session_score(s_tr) - y_tr) ** 2).mean() + a.l1 * model.sparsity()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return model.session_score(TT(te)).numpy(), model


def fit_jade(tr, te):
    from nesystream.eval.learned_baselines import fit_rank_qoe
    tmp = os.path.join(a.out, "_jade_tr.npz")
    np.savez(tmp, **{k: D[k][tr] for k in ("q", "rebuf", "level01", "mask", "target")})
    torch.manual_seed(0); m = fit_rank_qoe(tmp)
    with torch.no_grad():
        sc_tr = m.session({k: torch.as_tensor(D[k][tr]).float() for k in ("q", "rebuf", "level01", "mask")}).numpy()
        sc_te = m.session({k: torch.as_tensor(D[k][te]).float() for k in ("q", "rebuf", "level01", "mask")}).numpy()
    return fit_affine(sc_tr, y[tr], sc_te)        # ranking scores -> MOS scale (affine on train) for PLCC / RMSE


def fit_gru(tr, te, steps):
    from nesystream.eval.learned_baselines import NeuralQoE
    torch.manual_seed(0)
    model = NeuralQoE(); mu, sd = y[tr].mean(), y[tr].std()
    opt = torch.optim.Adam(model.parameters(), 0.01, weight_decay=1e-4)
    s_tr, y_tr = T(tr), torch.as_tensor((y[tr] - mu) / sd).float()
    for _ in range(steps):
        loss = ((model.session_score(s_tr) - y_tr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return model.session_score(T(te)).numpy() * sd + mu


models = ["VMAF-mean", "Linear-QoE", "P.1203", "Jade-rank", "GRU", "NS-QoE (hand, prior w)", "NS-QoE (hand)",
          "NS-QoE (hand+induced)", "NS-QoE (no visual)"]
if a.models != "all":
    models = [m for m in models if m in a.models.split(",")]
p12 = p1203_all() if "P.1203" in models else None
contents = np.unique(vid); per_fold = {m: [] for m in models}; pooled = {m: [] for m in models}
for rep in range(a.repeats):
    perm = np.random.default_rng(rep).permutation(contents)
    for f in range(a.folds):
        te = np.isin(vid, perm[f::a.folds]); tr = ~te
        preds = {}
        for m in models:
            if m == "VMAF-mean":
                x = feats_linear(np.ones(len(y), bool))[:, 0]; preds[m] = fit_affine(x[tr], y[tr], x[te])
            elif m == "Linear-QoE":
                X = feats_linear(np.ones(len(y), bool))[:, :-1]; preds[m] = fit_affine(X[tr], y[tr], X[te])
            elif m == "P.1203":
                preds[m] = fit_affine(p12[tr], y[tr], p12[te])
            elif m == "GRU":
                preds[m] = fit_gru(tr, te, a.steps)
            elif m == "Jade-rank":
                preds[m] = fit_jade(tr, te)
            elif m == "NS-QoE (hand, prior w)":
                preds[m] = fit_nsqoe(tr, te, False, a.steps, learn_w=False)[0]
            elif m == "NS-QoE (no visual)":
                preds[m] = fit_nsqoe(tr, te, True, a.steps, visual=False)[0]
            else:
                preds[m] = fit_nsqoe(tr, te, "induced" in m, a.steps)[0]
            per_fold[m].append((srocc(preds[m], y[te]), plcc(preds[m], y[te]), float(np.sqrt(np.mean((preds[m] - y[te]) ** 2)))))
            if rep == 0:
                pooled[m].append((np.where(te)[0], preds[m]))
        print(f"repeat {rep} fold {f}: " + "  ".join(f"{m} {per_fold[m][-1][0]:.3f}" for m in models), flush=True)

res = {}
for m in models:
    v = np.array(per_fold[m]); idx = np.concatenate([i for i, _ in pooled[m]]); p = np.concatenate([q for _, q in pooled[m]])
    stall = (D["rebuf"][idx, 1:] > 0).any(1)
    res[m] = {"srocc_mean": float(v[:, 0].mean()), "srocc_std": float(v[:, 0].std()), "plcc_mean": float(v[:, 1].mean()),
              "rmse_mean": float(v[:, 2].mean()), "pooled_srocc": srocc(p, y[idx]), "pooled_plcc": plcc(p, y[idx]),
              "pooled_srocc_stall": srocc(p[stall], y[idx][stall]), "pooled_srocc_nostall": srocc(p[~stall], y[idx][~stall])}
print(f"\n{'model':24s} SROCC(mean±std)  PLCC   RMSE   pooled-SROCC  stall / no-stall")
for m, r in res.items():
    print(f"{m:24s} {r['srocc_mean']:.3f}±{r['srocc_std']:.3f}   {r['plcc_mean']:.3f}  {r['rmse_mean']:5.2f}  {r['pooled_srocc']:.3f}"
          f"         {r['pooled_srocc_stall']:.3f} / {r['pooled_srocc_nostall']:.3f}")
json.dump(res, open(os.path.join(a.out, "cv_results.json"), "w"), indent=1)
for m in [x for x in models if x in ("NS-QoE (hand)", "NS-QoE (hand+induced)")]:   # final weights on all sessions (for the streaming reward)
    _, model = fit_nsqoe(np.ones(len(y), bool), np.ones(len(y), bool), "induced" in m, a.steps)
    tag = "hand_induced" if "induced" in m else "hand"
    json.dump({n: w for n, _, w in model.rule_table(prune=0.0)} | {"_alpha": float(model.alpha.detach()), "_beta": float(model.beta.detach())},
              open(os.path.join(a.out, f"weights_mos_{tag}.json"), "w"), indent=1)
    print(f"\nfinal {m} weights (all 450 sessions):")
    for n, b, w in model.rule_table():
        print(f"  {w:8.3f}  {n:26s} {b}")
