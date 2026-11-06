"""Render subjective-test stimuli (E4 / self-built QoE dataset) from simulated sessions.

Each session = concatenation of the chosen renditions' chunks, upscaled to the display
resolution, with stalls rendered as frozen frames of the stalled duration.
Requires renditions kept by build_dataset.py --keep_renditions.

python scripts/render_sessions.py --sessions runs/sessions_sessions.json --video_dir data/videos --out stimuli --ids 0-119
"""
import _common  # noqa
import argparse, json, os, subprocess, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", required=True); ap.add_argument("--video_dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--ids", default="0-9")
ap.add_argument("--chunk_len", type=float, default=2.0); ap.add_argument("--display", default="1920:1080")
a = ap.parse_args()
lo, hi = (int(x) for x in a.ids.split("-"))
ss = [s for s in json.load(open(a.sessions)) if lo <= s["id"] <= hi]
os.makedirs(a.out, exist_ok=True)
for s in ss:
    tmp = tempfile.mkdtemp()
    parts = []
    chunks = s.get("chunks") or [[s["video"], k] for k in range(len(s["levels"]))]
    for k, (lv, rb, (clip, kk)) in enumerate(zip(s["levels"], s["rebuf"], chunks)):
        src = os.path.join(a.video_dir, clip + "_ladder", f"level_{lv:02d}.mp4")
        p = os.path.join(tmp, f"p{k:04d}.mp4")
        # stall BEFORE chunk k: freeze its first frame for rb seconds (tpad clone)
        dw, dh = a.display.split(":")
        # letterbox (keep aspect), constant 60 fps so clips with different frame rates concatenate cleanly
        vf = (f"scale={dw}:{dh}:force_original_aspect_ratio=decrease:flags=bicubic,pad={dw}:{dh}:(ow-iw)/2:(oh-ih)/2,fps=60"
              + (f",tpad=start_mode=clone:start_duration={rb:.3f}" if rb > 0.05 else ""))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(kk * a.chunk_len), "-t", str(a.chunk_len), "-i", src,
                        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "10", "-preset", "fast", p], check=True)
        parts.append(p)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    out = os.path.join(a.out, f"session_{s['id']:04d}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    print("rendered", out)
