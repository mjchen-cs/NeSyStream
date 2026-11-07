"""Select and render the stimuli of the subjective test from the dash.js deployment sessions.

python subjective/make_stimuli.py --sessions subjective/dashjs_sessions.json --src_dir deploy_dashjs/src --out subjective/stimuli

Input: real-player sessions (Tears of Steel, 30 chunks of 2 s, 4 ABR methods x 30 test traces), each with the rung of
every chunk, the start-up delay and the stalls measured in dash.js. Selection (fixed seed, no cherry-picking): 6 traces
stratified by network (2 HSDPA, 2 FCC, 1 Lumos 4G, 1 Lumos 5G) x 4 methods = 24 test stimuli, plus 2 training stimuli
from traces that are not used in the test. Rendering: every chunk is cut from the rendition the player downloaded,
start-up is a black screen with a loading spinner, and every stall freezes the last frame with the spinner, so the
stimulus shows exactly what the viewer of that session saw. Files are named stim_XX.mp4 in a random order, so
neither participants nor experimenters can tell the method; the key is in stimuli.json.
"""
import argparse, json, math, os, random, subprocess, tempfile
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", default="subjective/dashjs_sessions.json")
ap.add_argument("--src_dir", default="deploy_dashjs/src", help="level_XX.mp4 of the 60-s playlist (one file per rung)")
ap.add_argument("--out", default="subjective/stimuli"); ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--display", default="1920:1080"); ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--crf", type=int, default=14, help="near-lossless re-encode of the assembled stimulus")
ap.add_argument("--only", default="", help="comma list of stimulus file names to (re)render, e.g. stim_01,train_01")
ap.add_argument("--select_only", action="store_true", help="write stimuli.json without rendering")
a = ap.parse_args()
D = json.load(open(a.sessions)); CL = D["chunk_len"]; S = D["sessions"]
W, H = (int(x) for x in a.display.split(":"))
os.makedirs(a.out, exist_ok=True)

# ---------- selection (fixed seed, stratified by network set)
rng = random.Random(a.seed)
traces = {}
for s in S:
    traces.setdefault(s["set"], set()).add(s["trace"])
quota = {"hsdpa": 2, "fcc": 2, "lumos4g": 1, "lumos5g": 1}
test_tr = []
for st, q in quota.items():
    test_tr += [(st, t) for t in rng.sample(sorted(traces[st]), q)]
rest = [(st, t) for st in sorted(traces) for t in sorted(traces[st]) if (st, t) not in test_tr]
low = next(x for x in rng.sample(rest, len(rest)) if x[0] in ("hsdpa", "fcc"))
high = next(x for x in rng.sample(rest, len(rest)) if x[0] in ("lumos4g", "lumos5g"))
pick = lambda st, tr, m: next(s for s in S if s["set"] == st and s["trace"] == tr and s["method"] == m)
test = [pick(st, tr, m) for st, tr in test_tr for m in ("NeSyStream", "RobustMPC", "BBA", "dash.js default")]
rng.shuffle(test)
stim = [dict(file=f"train_{i + 1:02d}", phase="train", **s) for i, s in enumerate([pick(*low, "BBA"), pick(*high, "dash.js default")])]
stim += [dict(file=f"stim_{i + 1:02d}", phase="test", **s) for i, s in enumerate(test)]
for s in stim:
    s.pop("levels_view", None)
json.dump({"seed": a.seed, "display": a.display, "fps": a.fps, "stimuli": stim}, open(os.path.join(a.out, "stimuli.json"), "w"), indent=1)
# key without the method, for the rating app (participants' machine never needs the method)
json.dump([{"file": s["file"], "phase": s["phase"], "trace": s["trace"]} for s in stim], open(os.path.join(a.out, "playlist.json"), "w"), indent=1)
print(f"{len(stim)} stimuli ({len(test)} test) -> {a.out}/stimuli.json")
if a.select_only:
    raise SystemExit

# ---------- rendering
tmp = tempfile.mkdtemp(prefix="stim_")
spin = os.path.join(tmp, "spinner.png")
im = Image.new("RGBA", (96, 96), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
dr.arc((8, 8, 88, 88), start=0, end=270, fill=(255, 255, 255, 230), width=9)
im.save(spin)
VF = f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=bicubic,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={a.fps},format=yuv420p"
ENC = ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-r", str(a.fps)]


def sh(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def wait_clip(still, dur, out):
    """still image (or black) + rotating spinner for `dur` seconds."""
    src = (["-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={a.fps}"] if still is None else ["-loop", "1", "-framerate", str(a.fps), "-i", still])
    g = (f"[0:v]{'' if still is None else VF + ','}format=yuv420p[b];[1:v]format=rgba,rotate=a=t*6:c=none:ow=96:oh=96[s];"
         f"[b][s]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p")
    sh(["ffmpeg", "-y", "-v", "error", *src, "-loop", "1", "-framerate", str(a.fps), "-i", spin, "-filter_complex", g,
        "-t", f"{dur:.3f}", *ENC, out])


for s in stim:
    if a.only and s["file"] not in a.only.split(","):
        continue
    parts = []
    if s["startup_s"] > 0.05:
        p = os.path.join(tmp, "p_start.mp4"); wait_clip(None, s["startup_s"], p); parts.append(p)
    stall_at = {}
    for ct, d in s["stalls"]:
        stall_at.setdefault(min(len(s["levels"]) - 1, max(1, round(ct / CL))), 0.0)
        stall_at[min(len(s["levels"]) - 1, max(1, round(ct / CL)))] += d
    for k, lv in enumerate(s["levels"]):
        src = os.path.join(a.src_dir, f"level_{lv:02d}.mp4")
        if k in stall_at and stall_at[k] > 0.05:   # freeze the last displayed frame (end of chunk k-1)
            prev = os.path.join(a.src_dir, f"level_{s['levels'][k - 1]:02d}.mp4")
            still = os.path.join(tmp, f"still_{k}.png")
            sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{k * CL - 1.0 / a.fps:.3f}", "-i", prev, "-frames:v", "1", still])
            p = os.path.join(tmp, f"p_stall_{k}.mp4"); wait_clip(still, stall_at[k], p); parts.append(p)
        p = os.path.join(tmp, f"p_{k:02d}.mp4")
        sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{k * CL:.3f}", "-t", f"{CL:.3f}", "-i", src, "-vf", VF, *ENC, p])
        parts.append(p)
    lst = os.path.join(tmp, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    out = os.path.join(a.out, s["file"] + ".mp4")
    sh(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out])
    print(f"rendered {out}  ({s['set']}, {len(s['stalls'])} stalls, start-up {s['startup_s']:.2f}s)", flush=True)
