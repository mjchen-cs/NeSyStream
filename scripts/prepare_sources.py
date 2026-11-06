"""Turn the downloaded sources into uniform reference clips for build_dataset.py.

Output: data/refs/<name>.mkv  (lossless FFV1, 8-bit 4:2:0, at most 1920x1080, at most 60 fps, at most --max_seconds)
        data/refs/manifest.csv, data/refs/groups.json  (clips of the same film / production share a group)

python scripts/prepare_sources.py --root data --video1 video1 --out data/refs --jobs 6
Runs on Windows or Linux; needs ffmpeg/ffprobe in PATH. Existing outputs are skipped (resumable);
incomplete or unreadable downloads are reported and skipped.
"""
import _common  # noqa
import argparse, csv, glob, json, os, re, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

FILMS = {  # long films -> evenly spaced, non-overlapping segments (skip opening and credits)
    "Meridian": 5, "Sparks": 4, "CosmosLaundromat": 5, "tears_of_steel": 5, "big_buck_bunny": 5}
# --more_film_segments: target counts. Segments already in the manifest keep their names and start times;
# new ones are placed in the gaps (>= max_seconds apart) and numbered after the existing ones.
FILMS_MORE = {"Meridian": 15, "Sparks": 8, "CosmosLaundromat": 15, "tears_of_steel": 15, "big_buck_bunny": 12}
SKIP = ["bbb_sunflower"]  # duplicate render of Big Buck Bunny


def probe(path, extra=()):
    r = subprocess.run(["ffprobe", "-v", "error", *extra, "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,pix_fmt:format=duration", "-of", "json", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    j = json.loads(r.stdout)
    if not j.get("streams"):
        return None
    s = j["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    dur = float(j.get("format", {}).get("duration", "nan") or "nan")
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": float(n) / float(d), "dur": dur,
            "pix": s.get("pix_fmt", "")}


def collect(root, video1):
    """Return list of jobs: dict(name, src, input_args, start, group, dataset)."""
    jobs = []
    for p in sorted(glob.glob(os.path.join(root, "sources", "*.y4m"))):
        jobs.append({"name": "xiph_" + os.path.splitext(os.path.basename(p))[0], "src": p, "in": [], "group": None,
                     "dataset": "Xiph derf"})
    for p in sorted(glob.glob(os.path.join(root, "sources", "netflix", "Netflix_*.y4m"))):
        nm = os.path.basename(p).split("_")[1]
        jobs.append({"name": "elfuente_" + nm, "src": p, "in": [], "group": "netflix_elfuente",
                     "dataset": "Netflix Open Content (El Fuente)"})
    for p in sorted(glob.glob(os.path.join(root, "sources", "ugc", "*.mkv"))):
        jobs.append({"name": "ugc_" + os.path.splitext(os.path.basename(p))[0], "src": p, "in": [], "group": None,
                     "dataset": "YouTube-UGC"})
    for p in sorted(glob.glob(os.path.join(root, "uvg_raw", "*_1920x1080_*fps_420_8bit_YUV.yuv"))):
        m = re.match(r"(.+)_1920x1080_(\d+)fps_420_8bit_YUV\.yuv$", os.path.basename(p))
        jobs.append({"name": "uvg_" + m.group(1), "src": p, "group": None, "dataset": "UVG",
                     "in": ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "1920x1080", "-framerate", m.group(2)]})
    films = sorted(glob.glob(os.path.join(root, "sources", "netflix", "*_SDR.mp4")))
    if video1 and os.path.isdir(video1):
        films += sorted(glob.glob(os.path.join(video1, "*.mov")) + glob.glob(os.path.join(video1, "*.mp4")))
    for p in films:
        base = os.path.splitext(os.path.basename(p))[0]
        if any(s in base for s in SKIP):
            continue
        key = next((k for k in FILMS if base.startswith(k)), None)
        jobs.append({"name": "film_" + (key or base), "src": p, "in": [], "group": "film_" + (key or base),
                     "dataset": "Netflix Open Content" if "_SDR" in base else "Blender Foundation",
                     "segments": FILMS.get(key, 3)})
    return jobs


def y4m_complete(path):
    """A y4m is complete iff its payload is a whole number of frames."""
    with open(path, "rb") as f:
        hdr = f.readline().decode(errors="ignore")
    p = {t[0]: t[1:] for t in hdr.split()[1:]}
    w, h, c = int(p["W"]), int(p["H"]), p.get("C", "420")
    bps = 2 if "p10" in c or "p12" in c or "p16" in c else 1
    chroma = 1.5 if c.startswith("420") else 2 if c.startswith("422") else 3
    fs = int(w * h * chroma * bps) + 6
    n = (os.path.getsize(path) - len(hdr)) / fs
    return n == int(n) and n >= 30


def expand_segments(jobs, max_seconds, existing=None):
    """existing: {film_job_name: [start, ...]} from a previous manifest (kept as seg0..)."""
    existing = existing or {}
    out = []
    for j in jobs:
        if j["src"].lower().endswith(".y4m") and not y4m_complete(j["src"]):
            print(f"[skip] incomplete y4m (re-run its download script): {j['src']}")
            continue
        if j["in"]:
            fs = 1920 * 1080 * 3 // 2
            if os.path.getsize(j["src"]) % fs or os.path.getsize(j["src"]) < 30 * fs:
                print(f"[skip] incomplete raw yuv: {j['src']}"); continue
        info = probe(j["src"], j["in"])
        if info is None or not (info["dur"] > 0.5 or j["in"]):
            print(f"[skip] unreadable or incomplete: {j['src']}")
            continue
        if j["in"]:  # raw yuv: duration from size
            fs = 1920 * 1080 * 3 // 2
            info["dur"] = os.path.getsize(j["src"]) // fs / info["fps"]
        j["info"] = info
        k = j.pop("segments", 0)
        if k:
            dur = info["dur"]
            lo, hi = 0.08 * dur, 0.92 * dur - max_seconds
            old = existing.get(j["name"], [])
            if old:
                starts = list(old)
                cand = [lo + (hi - lo) * i / max(4 * k - 1, 1) for i in range(4 * k)] if hi > lo else []
                for c in sorted(cand, key=lambda c: -min(abs(c - x) for x in starts)):
                    if len(starts) >= k:
                        break
                    if min(abs(c - x) for x in starts) >= max_seconds:
                        starts.append(c)
            else:
                starts = [lo + (hi - lo) * i / max(k - 1, 1) for i in range(k)] if hi > lo else [0.0]
            for i, s in enumerate(starts):
                out.append({**j, "name": f"{j['name']}_seg{i}", "start": round(s, 2)})
        else:
            out.append({**j, "start": 0.0})
    return out


def convert(j, out_dir, max_seconds, threads):
    dst = os.path.join(out_dir, j["name"] + ".mkv")
    if os.path.exists(dst):
        return j, "exists"
    info = j["info"]
    w, h = info["w"], info["h"]
    sc = min(1.0, 1920 / w, 1080 / h)
    tw, th = int(w * sc) // 2 * 2, int(h * sc) // 2 * 2
    vf = []
    if (tw, th) != (w, h):
        vf.append(f"scale={tw}:{th}:flags=lanczos")
    if info["fps"] > 60.5:
        vf.append("fps=60")          # 120 fps UVG -> 60 fps (frame dropping, no interpolation)
    vf.append("format=yuv420p")      # 10-bit -> 8-bit, 4:2:0
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(j["start"]), *j["in"], "-i", j["src"], "-t", str(max_seconds),
           "-an", "-vf", ",".join(vf), "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "16",
           "-threads", str(threads), dst + ".part.mkv"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return j, "FAILED " + r.stderr[-800:]
    want = min(max_seconds, max(info["dur"] - j["start"], 0)) if info["dur"] == info["dur"] else max_seconds
    got = (probe(dst + ".part.mkv") or {}).get("dur", 0)
    if want > 0 and got < 0.9 * want:   # truncated download that still decodes partially
        os.replace(dst + ".part.mkv", dst + ".incomplete.mkv")
        return j, f"INCOMPLETE source ({got:.1f}s of {want:.1f}s) -> re-download it"
    os.replace(dst + ".part.mkv", dst)
    return j, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data"); ap.add_argument("--video1", default="video1")
    ap.add_argument("--out", default="data/refs"); ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--more_film_segments", action="store_true", help="grow films to FILMS_MORE segments (additive)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.more_film_segments:
        FILMS.update(FILMS_MORE)
    existing = {}
    man = os.path.join(a.out, "manifest.csv")
    if os.path.exists(man):
        for r in csv.DictReader(open(man, encoding="utf-8")):
            m = re.match(r"(film_.+)_seg(\d+)$", r["name"])
            if m:
                existing.setdefault(m.group(1), []).append((int(m.group(2)), float(r["start_s"])))
        existing = {k: [s for _, s in sorted(v)] for k, v in existing.items()}
    jobs = expand_segments(collect(a.root, a.video1), a.max_seconds, existing)
    print(f"{len(jobs)} reference clips planned")
    groups = {}
    for j in jobs:
        if j["group"]:
            groups.setdefault(j["group"], []).append(j["name"])
    json.dump(groups, open(os.path.join(a.out, "groups.json"), "w"), indent=1)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["name", "dataset", "source_file", "start_s", "src_w", "src_h", "src_fps", "group"])
        for j in jobs:
            i = j["info"]; wr.writerow([j["name"], j["dataset"], os.path.basename(j["src"]), j["start"], i["w"], i["h"],
                                        round(i["fps"], 3), j["group"] or ""])
    if a.dry_run:
        for j in jobs: print(f"  {j['name']:48s} {j['info']['w']}x{j['info']['h']} {j['info']['fps']:.2f}fps start {j['start']}")
        return
    threads = max(1, (os.cpu_count() or 4) // a.jobs)
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(convert, j, a.out, a.max_seconds, threads) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            j, st = f.result()
            print(f"[{n}/{len(jobs)}] {j['name']}: {st}", flush=True)


if __name__ == "__main__":
    main()
