"""Recompute split.json (content groups kept together, balanced by chunk count) and refresh concept labels
(thresholds fitted on the new training split). Run after build_dataset.py or recompute_faces.py.

python scripts/resplit.py --video_dir data/videos [--val 0.15 --test 0.25 --seed 0]
"""
import _common  # noqa
import argparse, json, os
import numpy as np
from nesystream.data.chunks import list_videos, split_by_video
from nesystream.data.build import finalize_concepts

ap = argparse.ArgumentParser(); ap.add_argument("--video_dir", required=True)
ap.add_argument("--val", type=float, default=0.15); ap.add_argument("--test", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
paths = list_videos(a.video_dir)
tr, va, te = split_by_video(paths, None, a.val, a.test, a.seed)
finalize_concepts(paths, tr)
nm = lambda ps: [os.path.splitext(os.path.basename(p))[0] for p in ps]
json.dump({"train": nm(tr), "val": nm(va), "test": nm(te)}, open(os.path.join(a.video_dir, "split.json"), "w"), indent=1)
cnt = lambda ps: sum(int(np.load(p)["sizes"].shape[0]) for p in ps)
print({k: (len(v), cnt(v)) for k, v in zip(["train(clips,chunks)", "val", "test"], [tr, va, te])})
