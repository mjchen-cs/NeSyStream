"""Cut every chunk of every rendition into its own mp4 so a video-level VQA model (e.g. FineVQ)
can score it. Needs renditions kept by build_dataset.py --keep_renditions.
Names: <video>__L<level>__C<chunk>.mp4 ; also writes video_names.txt.

python scripts/export_chunks_for_vqa.py --video_dir data/videos --out data/vqa_chunks --levels lowest   # feature baseline
python scripts/export_chunks_for_vqa.py --video_dir data/videos --out data/vqa_chunks_all --levels all  # per-rendition baseline
"""
import _common  # noqa
import re
import argparse, glob, os, shutil, subprocess, tempfile
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="lowest", help="'lowest', 'all', or comma list of level indices")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
names = []
for npz in [p for p in sorted(glob.glob(os.path.join(a.video_dir, "*.npz"))) if not re.search(r"_(frag|rungfeat\w*)\.npz$", p)]:
    vid = os.path.splitext(os.path.basename(npz))[0]
    d = np.load(npz)
    n, L, cl = d["sizes"].shape[0], d["sizes"].shape[1], float(d["chunk_len"])
    lad = os.path.join(a.video_dir, vid + "_ladder")
    if not os.path.isdir(lad):
        print("[skip] no renditions for", vid, "(rebuild with --keep_renditions)"); continue
    levels = [0] if a.levels == "lowest" else list(range(L)) if a.levels == "all" else [int(x) for x in a.levels.split(",")]
    for l in levels:
        tmp = tempfile.mkdtemp()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(lad, f"level_{l:02d}.mp4"), "-c", "copy",
                        "-f", "segment", "-segment_time", str(cl), "-reset_timestamps", "1",
                        os.path.join(tmp, "c_%05d.mp4")], check=True)
        for k in range(n):
            src = os.path.join(tmp, f"c_{k:05d}.mp4")
            if os.path.exists(src):
                nm = f"{vid}__L{l:02d}__C{k:05d}.mp4"
                shutil.move(src, os.path.join(a.out, nm)); names.append(nm)
        shutil.rmtree(tmp)
open(os.path.join(a.out, "video_names.txt"), "w").write("\n".join(names) + "\n")
print(f"exported {len(names)} chunk files -> {a.out}")
