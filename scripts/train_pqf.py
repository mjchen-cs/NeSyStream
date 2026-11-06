"""Train the Perceptual Quality Field (E1).

python scripts/train_pqf.py --video_dir data/videos --out runs/pqf.pt
"""
import _common  # noqa
import argparse, os, json
import numpy as np, torch
from torch.utils.data import DataLoader
from nesystream.data.chunks import get_split, ChunkDataset, srocc, plcc, chunk_sources, subset_mae
from nesystream.models.pqf import PQF, pqf_loss

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--split_json", default=None)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--fold", type=int, default=-1, help="-1: fixed split.json; k: grouped K-fold CV, test = fold k")
ap.add_argument("--n_folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--patience", type=int, default=15, help="early stopping on val MAE")
ap.add_argument("--freeze", action="store_true", help="freeze the pretrained backbone (small data)")
ap.add_argument("--frames", default="thumb", choices=["thumb", "frag", "fragpair", "fragtri"],
                help="visual input: lowest-rung thumbnails | native-res source fragments | source+lowest-rung fragment pairs")
ap.add_argument("--anchors", action="store_true", help="add encoder PSNR/SSIM anchors of the lowest + probe rung")
ap.add_argument("--no_aug", action="store_true"); ap.add_argument("--no_prior_init", action="store_true")
ap.add_argument("--z_dim", type=int, default=32); ap.add_argument("--lam_r", type=float, default=0.01)
ap.add_argument("--lam_c", type=float, default=0.5); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--backbone", default="scratch", help="scratch | mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0")
ap.add_argument("--no_pretrained", action="store_true", help="torchvision backbone without ImageNet weights")
ap.add_argument("--inputs", default="both", choices=["both", "frames", "bitstream"], help="input ablation")
ap.add_argument("--teacher", default="", help="offline distillation from '<video>_<teacher>.npy' features, e.g. finevq")
ap.add_argument("--lam_t", type=float, default=1.0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); np.random.seed(a.seed)
tr, va, te = get_split(a.video_dir, a.split_json, a.fold, a.n_folds)
print(f"split: train {len(tr)} clips, val {len(va)}, test {len(te)}")
kw = dict(frames=a.frames, anchors=a.anchors)
dtr = ChunkDataset(tr, teacher=a.teacher, **kw); dva = ChunkDataset(va, dtr.cmp_stats, **kw)
dte = ChunkDataset(te, dtr.cmp_stats, **kw)
L, C, D = dtr.q.shape[1], dtr.c.shape[1], dtr.cmp.shape[1]
Dt = dtr.t.shape[1] if dtr.t is not None else 0
model = PQF(L, C, D, a.z_dim, a.width, backbone=a.backbone, pretrained=not a.no_pretrained, inputs=a.inputs,
            teacher_dim=Dt, dropout=a.dropout, freeze_backbone=a.freeze,
            pair={'fragpair': 1, 'fragtri': 2}.get(a.frames, 0)).to(dev)
if not a.no_prior_init:
    model.rq_head.init_prior(dtr.q.mean(0))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], a.lr, weight_decay=a.wd)

def augment(fr):
    """label-preserving augmentation for quality: horizontal flip + random 112x112 crop (no rescaling,
    no blur / noise / colour change, which would alter the visual quality the label describes)."""
    B = fr.shape[0]
    flip = torch.rand(B, device=fr.device) < 0.5
    fr = torch.where(flip[:, None, None, None, None], fr.flip(-1), fr)
    H, W = fr.shape[-2:]
    if H == 224:   # fragment mosaic: crop 6x6 of the 7x7 patches, on the patch grid (keeps patches whole)
        i, j = 32 * np.random.randint(0, 2), 32 * np.random.randint(0, 2)
        return fr[..., i:i + 192, j:j + 192]
    c = 112
    if H > c and W > c:
        i, j = np.random.randint(0, H - c + 1), np.random.randint(0, W - c + 1)
        fr = fr[..., i:i + c, j:j + c]
    return fr
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

def evaluate(ds):
    model.eval(); qs, ts, bits, cs, cl = [], [], [], [], []
    with torch.no_grad():
        for fr, cmp, q, c, _ in DataLoader(ds, 64):
            o = model(fr.to(dev), cmp.to(dev))
            qs.append(o["q"].cpu().numpy()); ts.append(q.numpy()); bits.append(o["bits"].cpu().numpy())
            cs.append(torch.sigmoid(o["concept_logits"]).cpu().numpy()); cl.append(c.numpy())
    qs, ts, bits = np.concatenate(qs), np.concatenate(ts), np.concatenate(bits)
    cs, cl = np.concatenate(cs), np.concatenate(cl)
    acc = float(((cs > 0.5) == (cl > 0.5)).mean())
    evaluate.last = qs
    return {"mae": float(np.abs(qs - ts).mean()), "srocc": srocc(qs.ravel(), ts.ravel()),
            "plcc": plcc(qs.ravel(), ts.ravel()),
            # per-level SROCC across chunks = does PQF rank content difficulty correctly?
            "srocc_content": float(np.nanmean([srocc(qs[:, l], ts[:, l]) for l in range(qs.shape[1])])),
            "bits": float(bits.mean()), "bytes_est": float(bits.mean() / 8), "concept_acc": acc}

best, bad = 1e9, 0
for ep in range(a.epochs):
    model.train()
    for fr, cmp, q, c, t in DataLoader(dtr, a.bs, shuffle=True, drop_last=len(dtr) > a.bs):
        fr = fr.to(dev)
        if not a.no_aug:
            fr = augment(fr)
        loss, info = pqf_loss(model(fr, cmp.to(dev)), q.to(dev), c.to(dev), a.lam_c, a.lam_r,
                              t.to(dev) if Dt else None, a.lam_t)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    m = evaluate(dva)
    print(f"epoch {ep:3d}  train {info}  val {m}")
    if m["mae"] < best:
        best, bad = m["mae"], 0
        torch.save({"state": model.state_dict(), "L": L, "C": C, "D": D, "z_dim": a.z_dim, "width": a.width,
                    "backbone": a.backbone, "inputs": a.inputs, "teacher_dim": Dt,
                    "frames": a.frames, "anchors": a.anchors,
                    "cmp_stats": [x.tolist() for x in dtr.cmp_stats]}, a.out)
    else:
        bad += 1
        if bad >= a.patience:
            print(f"early stop at epoch {ep} (best val MAE {best:.3f})"); break
model.load_state_dict(torch.load(a.out, map_location=dev)["state"])
res = {"val": evaluate(dva), "test": evaluate(dte)}
res["test_subsets"] = subset_mae(evaluate.last, dte.q, chunk_sources(te))
res["train_mae"] = evaluate(dtr)["mae"]   # generalisation gap
# cost: server-side ms per chunk (batch 1, includes the encoder) and parameter count
import time
model.eval(); fr, cmp, *_ = dte[0]
with torch.no_grad():
    for _ in range(3): model(fr[None].to(dev), cmp[None].to(dev))
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): model(fr[None].to(dev), cmp[None].to(dev))
    if dev == "cuda": torch.cuda.synchronize()
res["cost"] = {"ms_per_chunk": (time.perf_counter() - t0) / 20 * 1e3, "device": dev,
               "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
res["config"] = {"backbone": a.backbone, "inputs": a.inputs, "teacher": a.teacher, "z_dim": a.z_dim,
                 "fold": a.fold, "freeze": a.freeze, "dropout": a.dropout, "wd": a.wd, "lr": a.lr,
                 "aug": not a.no_aug, "prior_init": not a.no_prior_init,
                 "frames": a.frames, "anchors": a.anchors}
print("TEST", res["test"], res["test_subsets"], "train MAE", round(res["train_mae"], 3))
json.dump(res, open(os.path.splitext(a.out)[0] + "_metrics.json", "w"), indent=2)
