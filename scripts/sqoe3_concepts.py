"""Visual concepts for the 20 SQoE-III contents (per 2-s segment), with the SAME thresholds as our streaming corpus.

python scripts/sqoe3_concepts.py --root data/sqoe3_raw --video_dir data/videos --out data/sqoe3/sqoe3_concepts.json
Raw features (TI, SI, scene-cut histogram distance, faces, luma) are computed exactly as in build.py from the
lowest rendition (v01, 320x240); HighMotion / HighTexture thresholds are the 0.75 quantiles of our training split.
Needs ffmpeg + OpenCV only (no torch).
"""
import _common  # noqa
import argparse, glob, json, os
import numpy as np
from nesystream.data.build import visual_features

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/sqoe3_raw"); ap.add_argument("--video_dir", default="data/videos")
ap.add_argument("--out", default="data/sqoe3/sqoe3_concepts.json"); ap.add_argument("--q", type=float, default=0.75)
ap.add_argument("--temp", type=float, default=0.1)
a = ap.parse_args()
split = json.load(open(os.path.join(a.video_dir, "split.json")))
raw_tr = np.concatenate([np.load(os.path.join(a.video_dir, n + ".npz"))["raw_concepts"] for n in split["train"]])
thr, scale = np.quantile(raw_tr, a.q, axis=0), raw_tr.std(0) + 1e-6
print("thresholds from our training split (TI, SI):", thr[:2])
sig = lambda x: 1 / (1 + np.exp(-x))
out, raw_all = {}, {}
for p in sorted(glob.glob(os.path.join(a.root, "server_videos", "*", "H264", "encoded", "video", "v01.mp4"))):
    name = p.split(os.sep)[-5] if os.sep in p else p.split("/")[-5]
    _, r = visual_features(p, 30.0, 2.0, 5)
    c = np.zeros_like(r)
    c[:, 0] = sig((r[:, 0] - thr[0]) / (a.temp * scale[0])); c[:, 1] = sig((r[:, 1] - thr[1]) / (a.temp * scale[1]))
    c[:, 2] = sig((r[:, 2] - 0.5) / 0.05); c[:, 3] = np.clip(r[:, 3], 0, 1); c[:, 4] = sig((0.25 - r[:, 4]) / 0.03)
    out[name] = c.round(4).tolist(); raw_all[name] = r.round(5).tolist()
    print(f"{name:20s} motion {c[:, 0].mean():.2f} texture {c[:, 1].mean():.2f} cut {c[:, 2].max():.2f} "
          f"face {c[:, 3].mean():.2f} dark {c[:, 4].mean():.2f}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=0); json.dump(raw_all, open(a.out.replace(".json", "_raw.json"), "w"))
print(len(out), "contents ->", a.out)
