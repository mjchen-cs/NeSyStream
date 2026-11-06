"""Turn the npz written by third_party/finevq/finevq_extract.py into per-video arrays next to each video npz:
  <video>_<name>.npy         (N, D) features of the LOWEST rendition  -> train_feature_baseline.py --kind features
  <video>_<name>_scores.npy  (N, L) per-rendition scores (NaN where not run) -> --kind per_rendition
"""
import _common  # noqa
import re
import argparse, collections, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True, nargs="+"); ap.add_argument("--video_dir", required=True)
ap.add_argument("--name", default="finevq")
a = ap.parse_args()
feats, scores, secs = {}, {}, []
for p in a.npz:
    z = np.load(p, allow_pickle=False); secs.append(float(z["sec_per_video"]))
    for nm, s, f in zip(z["names"], z["scores"], z["feats"]):
        m = re.match(r"(.+)__L(\d+)__C(\d+)\.mp4$", os.path.basename(str(nm)))
        v, l, k = m.group(1), int(m.group(2)), int(m.group(3))
        scores[(v, l, k)] = float(s)
        if l == 0:
            feats[(v, k)] = f
for npz in [p for p in sorted(glob.glob(os.path.join(a.video_dir, "*.npz"))) if not re.search(r"_(frag|rungfeat\w*)\.npz$", p)]:
    v = os.path.splitext(os.path.basename(npz))[0]
    d = np.load(npz); N, L = d["sizes"].shape
    if any(key[0] == v for key in feats):
        D = len(next(iter(feats.values())))
        F = np.stack([feats.get((v, k), np.zeros(D, np.float16)) for k in range(N)])
        np.save(os.path.join(a.video_dir, f"{v}_{a.name}.npy"), F.astype(np.float32))
    S = np.full((N, L), np.nan, np.float32)
    for (vv, l, k), s in scores.items():
        if vv == v and k < N:
            S[k, l] = s
    np.save(os.path.join(a.video_dir, f"{v}_{a.name}_scores.npy"), S)
json.dump({"sec_per_chunk_per_rendition": float(np.mean(secs))}, open(os.path.join(a.video_dir, f"{a.name}_timing.json"), "w"))
print("imported", len(scores), "chunk scores,", len(feats), "lowest-level features")
