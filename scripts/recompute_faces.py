"""Recompute the face-count feature (Salient concept) of already-built videos from their lowest rendition,
then refresh the concept labels. Much cheaper than rebuilding the dataset.

python scripts/recompute_faces.py --video_dir data/videos [--names a,b,c]
"""
import _common  # noqa
import re
import argparse, glob, os, time
import numpy as np
from nesystream.data.build import decode_native, _face_count, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True); ap.add_argument("--names", default="")
    ap.add_argument("--n_keyframes", type=int, default=4)
    a = ap.parse_args()
    want = set(filter(None, a.names.split(",")))
    for p in [p for p in sorted(glob.glob(os.path.join(a.video_dir, "*.npz"))) if not re.search(r"_(frag|rungfeat\w*)\.npz$", p)]:
        name = os.path.splitext(os.path.basename(p))[0]
        if want and name not in want:
            continue
        low = os.path.join(a.video_dir, name + "_ladder", "level_00.mp4")
        if not os.path.exists(low):
            print("[skip] no renditions:", name); continue
        t0 = time.time()
        d = dict(np.load(p, allow_pickle=False))
        fr = decode_native(low)
        n = d["sizes"].shape[0]
        fpc = max(int(round(probe(low)[2] * float(d["chunk_len"]))), 1)   # frames per chunk
        raw = d["raw_concepts"].copy()
        for k in range(n):
            seg = fr[k * fpc:(k + 1) * fpc]
            idx = np.linspace(0, len(seg) - 1, a.n_keyframes).round().astype(int)
            raw[k, 3] = max(_face_count(seg[i]) for i in idx) if len(seg) else 0.0
        d["raw_concepts"] = raw
        np.savez_compressed(p, **d)
        print(f"{name:44s} faces per chunk {raw[:, 3].astype(int).tolist()}  ({time.time() - t0:.1f}s, {_face_count.kind})",
              flush=True)


if __name__ == "__main__":
    main()
