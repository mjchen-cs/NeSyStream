"""PQF v6, step 2: two-stage Perceptual Quality Field on cached rung features (scripts/extract_rung_features.py).

Stage A  learned perceptual full-reference proxy  g(source fragments, rung-r fragments, r) -> VMAF_r
         trained on EVERY (chunk, rung) pair of the training videos (13x more samples than curves);
         at inference it is evaluated only on the P probe rungs (default: lowest and level 6).
Stage B  monotone curve completion  h(q_hat at the probe rungs [, PSNR at the probes]) -> q_hat(1..L)
         with the prior-initialised monotone head. Its training inputs are OUT-OF-FOLD stage-A predictions
         (inner grouped K-fold cross-fitting), so stage B never learns from over-fitted stage-A outputs.
The shipped descriptor is the P probe qualities quantised to 0.5 VMAF (1 byte each).

python scripts/train_pqf_fr.py --video_dir data/videos --out runs/e1_v4/fold0/pqf_v6.pt --fold 0
  --probes 0,6  --fr_train_rungs all|probes  --no_psnr  --oracle_fr (true VMAF at probes)  --no_fr (PSNR only)
"""
import _common  # noqa
import argparse, copy, json, os
import numpy as np, torch
import torch.nn as nn
from nesystream.data.chunks import get_split, srocc, chunk_sources, subset_mae, kfold_assign
from nesystream.models.pqf import MonotoneRQHead

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n_folds", type=int, default=5)
ap.add_argument("--probes", default="0,6"); ap.add_argument("--fr_train_rungs", default="all", choices=["all", "probes"])
ap.add_argument("--no_psnr", action="store_true", help="pure vision: no PSNR in stage A nor stage B"); ap.add_argument("--oracle_fr", action="store_true")
ap.add_argument("--no_fr", action="store_true"); ap.add_argument("--inner_folds", type=int, default=4)
ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--label", default="")
ap.add_argument("--save_preds", action="store_true", help="save per-video test predictions (<out>_preds.npz) for cache_quality.py")
ap.add_argument("--feat", default="full", help="rung feature file: <video>_rungfeat_<feat>.npz (full | frag); "
                "'v6' = old pooled-feature file <video>_rungfeat.npz")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
P = [int(x) for x in a.probes.split(",")]
tr, va, te = get_split(a.video_dir, None, a.fold, a.n_folds)


def side(p):
    return p[:-4] + ("_rungfeat.npz" if a.feat == "v6" else f"_rungfeat_{a.feat}.npz")


def load(paths):
    """-> G (N, L, Dp) pair features of every (chunk, rung), Q (N, L) VMAF, S (N, L) PSNR."""
    G, Q, S = [], [], []
    for p in paths:
        r = np.load(side(p))
        if "dist" in r.files:     # LPIPS-style distortion features: [mean_K d, max_K d, mean_K source features]
            d, src = np.log(r["dist"].astype(np.float32) + 1e-7), r["src"].astype(np.float32)   # spans decades
            g = np.concatenate([d.mean(2), d.max(2), np.repeat(src.mean(1)[:, None], d.shape[1], 1)], -1)
        else:                     # v6: pooled features, |pooled source - pooled rung|
            F = r["feats"].astype(np.float32); s0, t = F[:, :1], F[:, 1:]; dd = np.abs(s0 - t)
            g = np.concatenate([dd.mean(2), dd.max(2), np.repeat(s0.mean(2), t.shape[1], 1)], -1)
        G.append(g); S.append(r["psnr"] * (0.0 if a.no_psnr else 1.0)); Q.append(np.load(p)["vmaf"].astype(np.float32))
    return np.concatenate(G), np.concatenate(Q), np.concatenate(S)


def pair_feats(G, rungs):
    return G[:, rungs]


Ftr, Qtr, Str = load(tr); Fva, Qva, Sva = load(va); Fte, Qte, Ste = load(te)
L = Qtr.shape[1]
TR_R = list(range(L)) if a.fr_train_rungs == "all" else P


# ------------------------------------------------------------------ stage A: perceptual FR proxy
class FR(nn.Module):
    def __init__(self, din, L, hidden, p):
        super().__init__()
        self.emb = nn.Embedding(L, 16)
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(din + 16 + 1, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x, r, ps):
        return 100 * torch.sigmoid(self.net(torch.cat([x, self.emb(r), ps[:, None]], 1)).squeeze(1))


def flat_pairs(F, Q, S, rungs):
    X = pair_feats(F, rungs)                              # (N, R, 3D)
    N, R = X.shape[:2]
    r = np.tile(np.array(rungs), N)
    return X.reshape(N * R, -1), r, S[:, rungs].reshape(-1), Q[:, rungs].reshape(-1)


def fit_fr(F_a, Q_a, S_a, F_v, Q_v, S_v):
    X, r, s, y = flat_pairs(F_a, Q_a, S_a, TR_R)
    Xv, rv, sv, yv = flat_pairs(F_v, Q_v, S_v, P)          # early stopping on the probe rungs
    mu, sd = X.mean(0), X.std(0) + 1e-4
    T = lambda z: torch.as_tensor((z - mu) / sd, device=dev)
    pmu, psd = s.mean(), s.std() + 1e-4
    Ps = lambda z: torch.as_tensor((z - pmu) / psd, dtype=torch.float32, device=dev)
    X, Xv = T(X), T(Xv); r, rv = torch.as_tensor(r, device=dev), torch.as_tensor(rv, device=dev)
    s, sv = Ps(s), Ps(sv); y, yv = torch.as_tensor(y, device=dev), torch.as_tensor(yv, device=dev)
    m = FR(X.shape[1], L, a.hidden, a.dropout).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=a.wd)
    best, state, bad = 1e9, None, 0
    for ep in range(300):
        m.train()
        for i in torch.randperm(len(X), device=dev).split(256):
            loss = (m(X[i], r[i], s[i]) - y[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float((m(Xv, rv, sv) - yv).abs().mean())
        if v < best:
            best, state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 25:
                break
    m.load_state_dict(state); m.eval()

    def predict(F_, S_):
        Xp = pair_feats(F_, P); N = len(Xp)
        with torch.no_grad():
            out = m(T(Xp.reshape(N * len(P), -1)), torch.as_tensor(np.tile(P, N), device=dev),
                    Ps(S_[:, P].reshape(-1)))
        return out.view(N, len(P)).cpu().numpy()
    return predict, best


def probe_quality(split_F, split_Q, split_S, predict):
    if a.oracle_fr:
        return split_Q[:, P]
    return predict(split_F, split_S)


if a.no_fr or a.oracle_fr:
    qa_tr, qa_va, qa_te, fr_val = (Qtr[:, P], Qva[:, P], Qte[:, P], float("nan"))
else:
    # out-of-fold stage-A predictions on the training videos (inner grouped K-fold)
    inner = kfold_assign(tr, a.inner_folds, a.seed)
    offs = np.cumsum([0] + [int(np.load(p)["vmaf"].shape[0]) for p in tr])
    qa_tr = np.zeros((len(Qtr), len(P)), np.float32)
    for j in range(a.inner_folds):
        hold = np.concatenate([np.arange(offs[i], offs[i + 1]) for i, p in enumerate(tr) if inner[p] == j] or [[]]).astype(int)
        keep = np.setdiff1d(np.arange(len(Qtr)), hold)
        if len(hold) == 0:
            continue
        pred, _ = fit_fr(Ftr[keep], Qtr[keep], Str[keep], Fva, Qva, Sva)
        qa_tr[hold] = pred(Ftr[hold], Str[hold])
    pred, fr_val = fit_fr(Ftr, Qtr, Str, Fva, Qva, Sva)
    qa_va, qa_te = pred(Fva, Sva), pred(Fte, Ste)

q05 = lambda q: np.round(q * 2) / 2                       # descriptor quantisation: 0.5 VMAF, 1 byte per probe
fr_test_mae = float(np.abs(q05(qa_te) - Qte[:, P]).mean())


# ------------------------------------------------------------------ stage B: monotone curve completion
def feats_B(qa, S):
    parts = [] if a.no_fr else [q05(qa)]
    if not a.no_psnr or a.no_fr:
        parts.append(S[:, P])
    return np.concatenate(parts, 1).astype(np.float32)


Xtr, Xva, Xte = feats_B(qa_tr, Str), feats_B(qa_va, Sva), feats_B(qa_te, Ste)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
f = lambda x: torch.as_tensor((x - mu) / sd)
net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), MonotoneRQHead(32, L))
net[-1].init_prior(Qtr.mean(0))
opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-2)
best, state, bad = 1e9, None, 0
for ep in range(5000):
    net.train(); loss = (net(f(Xtr)) - torch.as_tensor(Qtr)).abs().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        v = float((net(f(Xva)) - torch.as_tensor(Qva)).abs().mean())
    if v < best:
        best, state, bad = v, copy.deepcopy(net.state_dict()), 0
    else:
        bad += 1
        if bad >= 300:
            break
net.load_state_dict(state); net.eval()
with torch.no_grad():
    Pte, Ptr, Pva = (net(f(x)).numpy() for x in (Xte, Xtr, Xva))


def metrics(Pq, Y):
    return {"mae": float(np.abs(Pq - Y).mean()), "srocc": srocc(Pq.ravel(), Y.ravel()),
            "srocc_content": float(np.nanmean([srocc(Pq[:, l], Y[:, l]) if Pq[:, l].std() > 0 else np.nan
                                               for l in range(L)])),
            "bytes_est": float(len(P) * ((0 if a.no_fr else 1) + (0 if (a.no_psnr and not a.no_fr) else 1))),
            "fr_probe_mae": fr_test_mae}


enc_ms = float(np.median([float(np.load(side(p))["enc_ms_per_image"]) for p in te]))
K = int(np.load(side(te[0]))["n_keyframes"])
res = {"val": metrics(Pva, Qva), "test": metrics(Pte, Qte), "test_subsets": subset_mae(Pte, Qte, chunk_sources(te)),
       "train_mae": float(np.abs(Ptr - Qtr).mean()), "fr_val_mae": fr_val,
       # server cost: frozen encoder on K source + K*P probe-rung fragment mosaics (decoding not included)
       "cost": {"ms_per_chunk": 0.0 if (a.no_fr or a.oracle_fr) else enc_ms * K * (1 + len(P)), "device": dev},
       "config": {"label": a.label or f"PQF-FR ({len(P)} probes)", "backbone": "mobilenet_v3_small", "freeze": True,
                  "probes": P, "fr_train_rungs": a.fr_train_rungs, "psnr": not a.no_psnr, "oracle_fr": a.oracle_fr,
                  "no_fr": a.no_fr, "fold": a.fold, "feat": a.feat}}
print("TEST", res["test"], res["test_subsets"], "train", round(res["train_mae"], 2))
json.dump(res, open(os.path.splitext(a.out)[0] + "_metrics.json", "w"), indent=2)
torch.save({"stageB": net.state_dict(), "mu": mu, "sd": sd, "config": res["config"]}, a.out)
if a.save_preds:   # out-of-fold predictions of the test videos: curve and shipped descriptor (quantised probes)
    offs = np.cumsum([0] + [int(np.load(p)["vmaf"].shape[0]) for p in te])
    desc = q05(qa_te)
    out = {}
    for i, p in enumerate(te):
        n = os.path.splitext(os.path.basename(p))[0]
        out[n + "__q"] = Pte[offs[i]:offs[i + 1]].astype(np.float32)
        out[n + "__z"] = desc[offs[i]:offs[i + 1]].astype(np.float32)
    np.savez(os.path.splitext(a.out)[0] + "_preds.npz", **out)
