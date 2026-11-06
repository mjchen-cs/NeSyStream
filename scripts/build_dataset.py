"""Build the chunk-level rate-quality dataset from a folder of source videos (searched recursively).

python scripts/build_dataset.py --src_dir data/sources --out_dir data/videos --chunk_len 4 --jobs 4 --vmaf_subsample 3
  --jobs            videos processed in parallel (each gets cpu_count // jobs threads)
  --vmaf_subsample  score every k-th frame with libvmaf (1 = every frame)
Videos already built are skipped, so an interrupted run can simply be restarted.
"""
import _common  # noqa
import re
import argparse, glob, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from nesystream.data.build import process_video, finalize_concepts, has_libvmaf
from nesystream.data.chunks import split_by_video

# Default ladder (kbps, height): 13 levels capped at 1080p, because the reference clips are at most 1080p.
DEFAULT_LADDER = [(235, 240), (375, 360), (560, 360), (750, 432), (1050, 480), (1750, 540), (2350, 720),
                  (3000, 720), (4300, 1080), (5800, 1080), (8100, 1080), (12000, 1080), (16800, 1080)]
EXTS = (".mp4", ".mkv", ".y4m", ".mov", ".webm")


def _job(src, out, ladder, a, threads):
    t0 = time.time()
    try:
        process_video(src, out, ladder, a.chunk_len, a.codec, a.metric, max_seconds=a.max_seconds,
                      keep_renditions=a.keep_renditions, threads=threads, vmaf_subsample=a.vmaf_subsample,
                      tmp_root=a.tmp_dir)
        return src, time.time() - t0, None
    except Exception:
        return src, time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ladder", default=None, help="json file: [[kbps, height], ...]")
    ap.add_argument("--chunk_len", type=float, default=2.0)
    ap.add_argument("--codec", default="libx264", choices=["libx264", "libx265"])
    ap.add_argument("--metric", default=None, choices=[None, "vmaf", "psnr"])
    ap.add_argument("--max_seconds", type=float, default=None)
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--keep_renditions", action="store_true", help="needed for subjective stimuli and FineVQ")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads_per_job", type=int, default=0, help="0 = cpu_count // jobs")
    ap.add_argument("--vmaf_subsample", type=int, default=1)
    ap.add_argument("--tmp_dir", default=None, help="scratch folder on a fast disk with free space (default: system temp)")
    a = ap.parse_args()
    ladder = json.load(open(a.ladder)) if a.ladder else DEFAULT_LADDER
    os.makedirs(a.out_dir, exist_ok=True)
    gsrc = os.path.join(a.src_dir, "groups.json")   # written by prepare_sources.py
    if os.path.exists(gsrc):
        import shutil; shutil.copy(gsrc, os.path.join(a.out_dir, "groups.json"))
    if a.metric in (None, "vmaf") and not has_libvmaf():
        print("[warn] this ffmpeg has no libvmaf -> PSNR fallback (pipeline test only). Install an ffmpeg build with libvmaf.")
    srcs = sorted(p for p in glob.glob(os.path.join(a.src_dir, "**", "*"), recursive=True)
                  if p.lower().endswith(EXTS) and os.path.isfile(p))
    todo = []
    for s in srcs:
        out = os.path.join(a.out_dir, os.path.splitext(os.path.basename(s))[0] + ".npz")
        if os.path.exists(out):
            print("skip (done)", os.path.basename(s))
        else:
            todo.append((s, out))
    threads = a.threads_per_job or max(1, (os.cpu_count() or 4) // max(a.jobs, 1))
    print(f"{len(todo)} videos to process, {a.jobs} in parallel x {threads} threads, vmaf_subsample={a.vmaf_subsample}")
    t0, failed = time.time(), []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(_job, s, o, ladder, a, threads) for s, o in todo]
        for i, f in enumerate(as_completed(futs), 1):
            src, sec, err = f.result()
            if err:
                failed.append(src); print(f"[{i}/{len(todo)}] FAILED {os.path.basename(src)}\n{err[-1500:]}")
            else:
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] done {os.path.basename(src)} in {sec / 60:.1f} min  "
                      f"(elapsed {el / 60:.0f} min, ETA {el / i * (len(todo) - i) / 60:.0f} min)", flush=True)
    paths = [p for p in sorted(glob.glob(os.path.join(a.out_dir, "*.npz"))) if not re.search(r"_(frag|rungfeat\w*)\.npz$", p)]
    if not paths:
        sys.exit("no videos were built")
    tr, va, te = split_by_video(paths, a.split_json)
    finalize_concepts(paths, tr)
    json.dump({k: [os.path.splitext(os.path.basename(p))[0] for p in v] for k, v in
               zip(["train", "val", "test"], [tr, va, te])}, open(os.path.join(a.out_dir, "split.json"), "w"), indent=1)
    print(f"done: {len(paths)} videos; split saved; {len(failed)} failed" + (f": {failed}" if failed else ""))


if __name__ == "__main__":
    main()
