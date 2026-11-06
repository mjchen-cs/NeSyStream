"""E1 baselines for rate-quality curve prediction, all sharing PQF's monotone head:

  --kind handcrafted : SI/TI/scene/luma + packet statistics (complexity features, CAVA/SENSEI-style input)
  --kind features    : precomputed deep VQA features per chunk, e.g. DOVER (ICCV'23) or FAST-VQA
                       (ECCV'22) extracted with their official code from the lowest rendition and
                       saved as <video>_<name>.npy with shape (N_chunks, D) next to the video npz
  --kind bitrate     : content-agnostic curve = mean training curve per level (bitrate-only prior)
  --kind anchors     : encoder-side PSNR/SSIM of the lowest + probe rung (<video>_frag.npz) + hand-crafted features
  --kind vmaf_lowest : TRUE VMAF of the lowest rung (one full-reference VMAF run per chunk) -> extrapolate the curve
  --kind vmaf_two    : TRUE VMAF of the lowest and the probe rung (level 6): two FR VMAF runs per chunk
  --kind per_rendition : a no-reference VQA model run on EVERY rendition (e.g. FineVQ, CVPR'25; costs L x
                       its inference per chunk); its scores are mapped to VMAF by a per-level linear fit on train
"""
import _common  # noqa
import argparse, os, json
import numpy as np, torch
from nesystream.data.chunks import get_split, srocc, chunk_sources, subset_mae
from nesystream.models.pqf import MonotoneRQHead

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--kind", required=True, choices=["handcrafted", "features", "bitrate", "per_rendition", "anchors", "vmaf_lowest", "vmaf_two"])
ap.add_argument("--feat_name", default="dover"); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=2000)
ap.add_argument("--fold", type=int, default=-1); ap.add_argument("--n_folds", type=int, default=5)
ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
tr, va, te = get_split(a.video_dir, None, a.fold, a.n_folds)

def load(paths):
    X, Y = [], []
    for p in paths:
        d = np.load(p)
        if a.kind == "handcrafted":
            X.append(np.concatenate([d["raw_concepts"], d["cmp_feats"]], 1))
        elif a.kind == "anchors":
            X.append(np.concatenate([d["raw_concepts"], d["cmp_feats"], np.load(p[:-4] + "_frag.npz")["anchors"]], 1))
        elif a.kind == "vmaf_lowest":
            X.append(d["vmaf"][:, :1])
        elif a.kind == "vmaf_two":
            X.append(d["vmaf"][:, [0, 6]])
        elif a.kind == "features":
            X.append(np.load(p[:-4] + f"_{a.feat_name}.npy"))
        elif a.kind == "per_rendition":
            X.append(np.load(p[:-4] + f"_{a.feat_name}_scores.npy"))
        else:
            X.append(np.zeros((len(d["vmaf"]), 1), np.float32))
        Y.append(d["vmaf"])
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)

Xtr, Ytr = load(tr); Xva, Yva = load(va); Xte, Yte = load(te)
if a.kind == "bitrate":
    P = np.repeat(Ytr.mean(0, keepdims=True), len(Yte), 0)
elif a.kind == "per_rendition":
    P = np.zeros_like(Yte)
    for l in range(Ytr.shape[1]):
        ok = np.isfinite(Xtr[:, l])
        A = np.stack([Xtr[ok, l], np.ones(ok.sum())], 1)
        coef = np.linalg.lstsq(A, Ytr[ok, l], rcond=None)[0]
        P[:, l] = np.clip(coef[0] * np.nan_to_num(Xte[:, l]) + coef[1], 0, 100)
    P = np.maximum.accumulate(P, axis=1)  # same monotone prior as PQF
else:
    # same recipe as PQF: prior-initialised monotone head, dropout, weight decay, early stopping on val
    import copy
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    f = lambda x: torch.as_tensor((x - mu) / sd)
    enc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(128, 32))
    head = MonotoneRQHead(32, Ytr.shape[1]); head.init_prior(Ytr.mean(0))
    net = torch.nn.Sequential(enc, head)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=a.wd)
    best, state, bad = 1e9, None, 0
    for ep in range(a.epochs):
        net.train()
        loss = (net(f(Xtr)) - torch.as_tensor(Ytr)).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float((net(f(Xva)) - torch.as_tensor(Yva)).abs().mean())
        if v < best:
            best, state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 200:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        P = net(f(Xte)).numpy()
res = {"kind": a.kind, "mae": float(np.abs(P - Yte).mean()), "srocc": srocc(P.ravel(), Yte.ravel()),
       "srocc_content": float(np.nanmean([srocc(P[:, l], Yte[:, l]) if P[:, l].std() > 0 else np.nan
                                          for l in range(P.shape[1])])),
       "subsets": subset_mae(P, Yte, chunk_sources(te)), "fold": a.fold}
# cost per chunk on the server (ms); deep features: timing written by import_vqa_outputs.py
tj = os.path.join(a.video_dir, f"{a.feat_name}_timing.json")
per = json.load(open(tj))["sec_per_chunk_per_rendition"] * 1e3 if os.path.exists(tj) else float("nan")
res["cost"] = {"ms_per_chunk": {"bitrate": 0.0, "handcrafted": float("nan"), "anchors": float("nan"),
                                "vmaf_lowest": float("nan"), "vmaf_two": float("nan"), "features": per,
                                "per_rendition": per * Ytr.shape[1]}[a.kind],
               "bytes_per_chunk": {"bitrate": 0, "handcrafted": 4 * Xtr.shape[1], "anchors": 4 * Xtr.shape[1],
                                   "vmaf_lowest": 4, "vmaf_two": 8, "features": 2 * Xtr.shape[1],
                                   "per_rendition": 4 * Ytr.shape[1]}[a.kind]}
res["name"] = {"bitrate": "Bitrate prior", "handcrafted": "Hand-crafted", "anchors": "Enc. PSNR/SSIM + hand-crafted",
               "vmaf_lowest": "VMAF on lowest rung (FR)",
               "vmaf_two": "VMAF on 2 rungs (FR)", "features": f"{a.feat_name} feat.",
               "per_rendition": f"{a.feat_name}, all levels"}[a.kind]
print(res); json.dump(res, open(a.out, "w"), indent=1)
