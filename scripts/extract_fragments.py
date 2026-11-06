"""Native-resolution fragment pairs + cheap encoder-side anchors for PQF (server side, per chunk).

For every chunk, at n_keyframes keyframes:
  frag_src : (N, F, 3, 224, 224) uint8   7x7 grid of 32x32 patches cut from the SOURCE at native resolution
                                          (grid mini-patch sampling as in FAST-VQA: keeps the fine texture that
                                          decides how hard a chunk is to compress, which thumbnails throw away)
  frag_low : (N, F, 3, 224, 224) uint8   the same patches from the LOWEST rung, upscaled (bicubic) to the source size
  frag_mid : (N, F, 3, 224, 224) uint8   the same patches from the PROBE rung (default level 6), upscaled
  anchors  : (N, 4) float32               [PSNR_Y, SSIM_Y] of the lowest rung and of the probe rung (default level 6)
                                          on the keyframes; the encoder computes these anyway (x264 --psnr/--ssim)
Saved as data/videos/<name>_frag.npz.  Needs ffmpeg in PATH and opencv-python.

python scripts/extract_fragments.py --video_dir data/videos --ref_dir data/refs --jobs 8
"""
import _common  # noqa
import argparse, glob, os, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

GRID, PATCH = 7, 32


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate", "-of", "csv=p=0", path], capture_output=True, text=True)
    w, h, fr = r.stdout.strip().split("\n")[0].split(",")[:3]
    n, d = fr.split("/")
    return int(w), int(h), float(n) / float(d)


def grab(path, idx, w, h):
    """decode only the frames with indices idx, scaled (bicubic) to w x h, as (len(idx), h, w, 3) uint8.
    ffmpeg >= 5.1 uses -fps_mode (newer builds dropped -vsync); older builds only know -vsync."""
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    base = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{expr}',scale={w}:{h}:flags=bicubic"]
    tail = ["-frames:v", str(len(idx)), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    err = ""
    for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        r = subprocess.run(base + sync + tail, capture_output=True)
        if r.returncode == 0 and len(r.stdout) >= w * h * 3:
            break
        err = r.stderr.decode(errors="ignore")[-400:]
    else:
        raise RuntimeError(f"ffmpeg could not decode frames from {path}: {err}")
    fr = np.frombuffer(r.stdout, np.uint8)
    n = fr.size // (w * h * 3)
    fr = fr[: n * w * h * 3].reshape(n, h, w, 3)
    if n < len(idx):
        if n < 0.9 * len(idx):
            raise RuntimeError(f"only {n} of {len(idx)} frames decoded from {path}")
        fr = np.concatenate([fr, np.repeat(fr[-1:], len(idx) - n, 0)])   # last frame(s) lost to rounding
    return fr


def valid(out):
    """an existing _frag.npz is kept only if it holds real images and non-constant anchors."""
    try:
        with np.load(out) as d:
            return ("frag_mid" in d.files and d["frag_src"][:1].max() > 0 and d["frag_low"][:1].max() > 0
                    and d["anchors"][:, 0].max() < 70)
    except Exception:
        return False


def fragments(img):
    """7x7 grid, one 32x32 patch from the centre of every cell, at native resolution -> 224x224."""
    h, w = img.shape[:2]
    out = np.zeros((GRID * PATCH, GRID * PATCH, 3), np.uint8)
    for i in range(GRID):
        for j in range(GRID):
            y = min(int((i + 0.5) * h / GRID) - PATCH // 2, h - PATCH); y = max(y, 0)
            x = min(int((j + 0.5) * w / GRID) - PATCH // 2, w - PATCH); x = max(x, 0)
            out[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH] = img[y:y + PATCH, x:x + PATCH]
    return out


def luma(img):
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def psnr(a, b):
    m = float(((luma(a) - luma(b)) ** 2).mean())
    return 10 * np.log10(255 ** 2 / max(m, 1e-3))


def ssim(a, b):
    import cv2
    x, y = luma(a), luma(b)
    g = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean())


def process(name, video_dir, ref_dir, n_kf, probe_level):
    out = os.path.join(video_dir, name + "_frag.npz")
    if os.path.exists(out) and valid(out):
        return name, "exists", 0.0
    t0 = time.time()
    d = np.load(os.path.join(video_dir, name + ".npz"))
    N, cl = d["vmaf"].shape[0], float(d["chunk_len"])
    ref = os.path.join(ref_dir, name + ".mkv")
    lad = os.path.join(video_dir, name + "_ladder")
    w, h, fps = probe(ref)
    fpc = max(int(round(fps * cl)), 1)
    idx = np.concatenate([k * fpc + np.linspace(0, fpc - 1, n_kf).round().astype(int) for k in range(N)])
    src = grab(ref, idx, w, h)
    low = grab(os.path.join(lad, "level_00.mp4"), idx, w, h)
    mid = grab(os.path.join(lad, f"level_{probe_level:02d}.mp4"), idx, w, h)
    fs = np.stack([fragments(x) for x in src]).reshape(N, n_kf, 224, 224, 3).transpose(0, 1, 4, 2, 3)
    fl = np.stack([fragments(x) for x in low]).reshape(N, n_kf, 224, 224, 3).transpose(0, 1, 4, 2, 3)
    fm = np.stack([fragments(x) for x in mid]).reshape(N, n_kf, 224, 224, 3).transpose(0, 1, 4, 2, 3)
    anc = np.zeros((N, 4), np.float32)
    for k in range(N):
        s = slice(k * n_kf, (k + 1) * n_kf)
        anc[k] = [np.mean([psnr(a, b) for a, b in zip(src[s], low[s])]), np.mean([ssim(a, b) for a, b in zip(src[s], low[s])]),
                  np.mean([psnr(a, b) for a, b in zip(src[s], mid[s])]), np.mean([ssim(a, b) for a, b in zip(src[s], mid[s])])]
    np.savez_compressed(out, frag_src=np.ascontiguousarray(fs), frag_low=np.ascontiguousarray(fl),
                        frag_mid=np.ascontiguousarray(fm), anchors=anc,
                        probe_level=np.int32(probe_level), n_keyframes=np.int32(n_kf))
    return name, "ok", (time.time() - t0) / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="data/videos"); ap.add_argument("--ref_dir", default="data/refs")
    ap.add_argument("--n_keyframes", type=int, default=4); ap.add_argument("--probe_level", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--names", default="")
    a = ap.parse_args()
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(a.video_dir, "*.npz"))
                   if not p.endswith("_frag.npz") and not os.path.basename(p).startswith("sessions"))
    if a.names:
        names = [n for n in names if n in set(a.names.split(","))]
    names = [n for n in names if os.path.exists(os.path.join(a.ref_dir, n + ".mkv"))]
    print(f"{len(names)} videos", flush=True)
    secs, fails = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(process, n, a.video_dir, a.ref_dir, a.n_keyframes, a.probe_level): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                n, st, sp = f.result()
            except Exception as e:  # noqa
                n, st, sp = futs[f], f"FAILED {e!r}"[:300], 0.0; fails.append(n)
            if sp:
                secs.append(sp)
            print(f"[{i}/{len(names)}] {n}: {st}" + (f"  ({sp:.2f} s/chunk)" if sp else ""), flush=True)
    if secs:
        print(f"mean wall time {np.mean(secs):.2f} s per chunk per worker (decode + fragments + anchors)")
    print("done;", len(fails), "failed", fails)


if __name__ == "__main__":
    main()
