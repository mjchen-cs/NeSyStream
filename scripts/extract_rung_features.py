"""PQF step 1: frozen multi-scale backbone features for the SOURCE and LPIPS-style distortion features for EVERY rung
(server side, once). Every (chunk, rung) pair becomes a training sample for the learned perceptual FR proxy.

--mode full : whole keyframes at native resolution (default; the network sees every pixel, like PSNR does)
--mode frag : 7x7x32 native-resolution fragments (2.4 % of a 1080p frame; ablation)

Distortion feature of rung r (per keyframe, per tapped stage s, per channel c), as in LPIPS:
    d_r[s, c] = mean_xy ( u(F_src)[c, x, y] - u(F_r)[c, x, y] )^2 ,  u = unit-normalisation over channels
i.e. feature maps are compared POSITION BY POSITION before pooling (pool-then-subtract misses blur and blocking).

Output per video: data/videos/<name>_rungfeat_<mode>.npz
  src   : (N, K, 2*sum C) float16     mean/std pooled source features (content / texture)
  dist  : (N, L, K, sum C) float32    LPIPS-style distortion of every rung
  psnr  : (N, L) float32              PSNR_Y of every rung on the same keyframes
  enc_ms_per_image : float            GPU time of one backbone pass (after warm-up)

python scripts/extract_rung_features.py --video_dir data/videos --ref_dir data/refs --mode full
"""
import _common  # noqa
import argparse, glob, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from extract_fragments import probe, grab, fragments, psnr
from nesystream.models.pqf import PretrainedFrameEncoder


def tap_maps(enc, x):
    h, maps = (x - enc.mean) / enc.std, []
    for i, m in enumerate(enc.features):
        h = m(h)
        if i in enc.taps:
            maps.append(h)
    return maps


def unit(f):
    return f / (f.norm(dim=1, keepdim=True) + 1e-10)


def valid(out, L):
    try:
        with np.load(out) as d:
            return d["dist"].shape[1] == L and np.isfinite(d["dist"][:1].astype(np.float32)).all() and d["psnr"].max() < 70
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="data/videos"); ap.add_argument("--ref_dir", default="data/refs")
    ap.add_argument("--mode", default="full", choices=["full", "frag"])
    ap.add_argument("--backbone", default="mobilenet_v3_small"); ap.add_argument("--n_keyframes", type=int, default=4)
    ap.add_argument("--threads", type=int, default=14); ap.add_argument("--names", default="")
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = PretrainedFrameEncoder(a.backbone, pretrained=not a.no_pretrained, freeze=True).to(dev).eval()
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(a.video_dir, "*.npz"))
                   if "_frag" not in p and "_rungfeat" not in p)
    if a.names:
        names = [n for n in names if n in set(a.names.split(","))]
    names = [n for n in names if os.path.exists(os.path.join(a.ref_dir, n + ".mkv"))]
    print(f"{len(names)} videos on {dev}, mode {a.mode}", flush=True)
    prep = (lambda im: np.stack([fragments(x) for x in im])) if a.mode == "frag" else (lambda im: im)
    times, fails = [], []
    pool = ThreadPoolExecutor(a.threads)
    warm = False
    for vi, name in enumerate(names, 1):
        lad = os.path.join(a.video_dir, name + "_ladder")
        rungs = sorted(glob.glob(os.path.join(lad, "level_*.mp4")))
        out = os.path.join(a.video_dir, f"{name}_rungfeat_{a.mode}.npz")
        if os.path.exists(out) and valid(out, len(rungs)):
            print(f"[{vi}/{len(names)}] {name}: exists"); continue
        t0 = time.time()
        try:
            d = np.load(os.path.join(a.video_dir, name + ".npz"))
            N, cl, K = d["vmaf"].shape[0], float(d["chunk_len"]), a.n_keyframes
            ref = os.path.join(a.ref_dir, name + ".mkv")
            w, h, fps = probe(ref)
            fpc = max(int(round(fps * cl)), 1)
            idx = np.concatenate([k * fpc + np.linspace(0, fpc - 1, K).round().astype(int) for k in range(N)])
            imgs = list(pool.map(lambda p: grab(p, idx, w, h), [ref] + rungs))     # parallel ffmpeg decodes
            ps = np.array([[np.mean([psnr(imgs[0][i], imgs[r + 1][i]) for i in range(k * K, (k + 1) * K)])
                            for r in range(len(rungs))] for k in range(N)], np.float32)
            X = [prep(im) for im in imgs]                                          # (N*K, H, W, 3) each
            T = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).to(dev).permute(0, 3, 1, 2).float() / 255.0
            if not warm and dev == "cuda":                                         # cudnn warm-up before timing
                with torch.no_grad():
                    for _ in range(3): tap_maps(enc, T(X[0][:a.batch]))
                torch.cuda.synchronize(); warm = True
            src_f, dist = [], [[] for _ in rungs]
            with torch.no_grad():
                for b in range(0, N * K, a.batch):
                    if dev == "cuda": torch.cuda.synchronize()
                    te = time.perf_counter()
                    ms = tap_maps(enc, T(X[0][b:b + a.batch]))
                    if dev == "cuda": torch.cuda.synchronize()
                    times.append((time.perf_counter() - te) * 1e3 / len(ms[0]))
                    src_f.append(torch.cat([torch.cat([m.mean(dim=(2, 3)), m.flatten(2).std(dim=2)], 1) for m in ms], 1).cpu())
                    us = [unit(m) for m in ms]
                    for r in range(len(rungs)):
                        mr = tap_maps(enc, T(X[r + 1][b:b + a.batch]))
                        dist[r].append(torch.cat([((u - unit(m)) ** 2).mean(dim=(2, 3)) for u, m in zip(us, mr)], 1).cpu())
            src_f = torch.cat(src_f).view(N, K, -1).numpy().astype(np.float16)
            dist = np.stack([torch.cat(x).view(N, K, -1).numpy() for x in dist], 1).astype(np.float32)   # (N, L, K, C); fp32: values ~1e-5
            np.savez_compressed(out, src=src_f, dist=dist, psnr=ps, n_keyframes=np.int32(K), mode=np.array(a.mode),
                                backbone=np.array(a.backbone), enc_ms_per_image=np.float32(np.median(times)))
            print(f"[{vi}/{len(names)}] {name}: ok  {N} chunks  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa
            fails.append(name); print(f"[{vi}/{len(names)}] {name}: FAILED {e!r}"[:400], flush=True)
    if times:
        print(f"backbone: median {np.median(times):.2f} ms per {'1080p frame' if a.mode == 'full' else 'fragment mosaic'} on {dev}")
    print("done;", len(fails), "failed", fails)


if __name__ == "__main__":
    main()
