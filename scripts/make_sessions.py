"""Generate diverse playback sessions (for fitting NS-QoE and for subjective-test stimuli).

Targets:
  --target p1203      : score each session with ITU-T P.1203 (teacher for playback rules)
  --target synthetic  : hidden ground-truth rules + noise (smoke test of rule recovery)
  --target none       : no score; MOS is added later from the subjective study (--mos_csv in fit_nsqoe.py)
Sessions are also exported to JSON so `render_sessions.py` can produce the real stimuli.
"""
import _common  # noqa
import argparse, json, os
import numpy as np, torch
from nesystream.env.abr_env import ABREnv, EnvConfig, load_traces, load_video
from nesystream.data.chunks import list_videos
from nesystream.models.nsqoe import NSQoE
from nesystream.eval.baselines import BBA, RB, RobustMPC
from nesystream.eval.evaluate import p1203_score
from nesystream.data.synthetic import GT_WEIGHTS

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--trace_dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--target", default="none", choices=["none", "p1203", "synthetic"])
ap.add_argument("--max_chunks", type=int, default=30, help="session length in chunks (e.g. 30 x 2 s = 60 s); clips are concatenated into playlists")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bw_scale", default="1,1", help="lo,hi: multiply each session's trace throughput by U(lo,hi) "
                "(Puffer traces are mostly fast; scaling yields enough stalls to calibrate the stall rules)")
ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                help="draw clips from one split only (fit NS-QoE on train-source sessions, evaluate on test-source sessions)")
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
rng = np.random.default_rng(a.seed)
from nesystream.data.chunks import split_by_video
_paths = list_videos(a.video_dir)
if a.split != "all":
    _tr, _va, _te = split_by_video(_paths, os.path.join(a.video_dir, "split.json"))
    _paths = {"train": _tr, "val": _va, "test": _te}[a.split]
videos = [load_video(p) for p in _paths]
traces = load_traces(a.trace_dir)
env = ABREnv(videos, traces, NSQoE(), EnvConfig(quality_source="oracle", use_hard_rules=False, playlist_chunks=a.max_chunks), seed=a.seed)
pols = [BBA(), RB(), RobustMPC(3, "bitrate"), RobustMPC(3, "quality")]

def jitter(pol, p):
    def f(e):
        l = pol.act(e)
        return int(rng.integers(0, e.L)) if rng.random() < p else l
    return f

recs = []
import time
t0 = time.time()
for i in range(a.n):
    if i and i % 200 == 0:
        print(f"  {i}/{a.n} sessions ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pol = pols[i % len(pols)]
    act = jitter(pol, rng.choice([0.0, 0.1, 0.3]))
    env.reset(); done = False
    lo, hi = (float(x) for x in a.bw_scale.split(","))
    if (lo, hi) != (1.0, 1.0):
        env.tr = dict(env.tr, mbps=env.tr["mbps"] * float(rng.uniform(lo, hi)))
    while not done:
        _, _, done, _ = env.step(act(env))
    r = env.session_record(); r["policy"] = pol.name; r["levels"] = [e["level"] for e in env.log]
    r["chunks"] = [list(c) for c in env.v.get("chunk_src", [(env.v["name"], k) for k in range(len(env.log))])][: len(env.log)]
    recs.append(r)

N = max(len(r["q"]) for r in recs)
pad = lambda key, shape=(): np.stack([np.pad(r[key], [(0, N - len(r[key]))] + [(0, 0)] * len(shape)) for r in recs])
arr = {k: pad(k) for k in ["q", "level01", "rebuf", "buffer", "thr_ratio", "pos", "bitrate_kbps"]}
arr["concepts"] = pad("concepts", (5,))
arr["mask"] = np.stack([np.pad(np.ones(len(r["q"]), np.float32), (0, N - len(r["q"]))) for r in recs])
arr["video"] = np.array([r["video"] for r in recs])
if a.target == "p1203":
    heights = videos[0].get("heights")
    res = [f"{int(h * 16 / 9) // 2 * 2}x{h}" for h in heights] if heights is not None else None
    arr["target"] = np.array([p1203_score({k: r[k] for k in ("bitrate_kbps", "level01", "rebuf")},
                                          float(videos[0]["chunk_len"]), res) for r in recs], np.float32)
elif a.target == "synthetic":
    gt = NSQoE(use_induced=False, init_weights=GT_WEIGHTS)
    with torch.no_grad():
        s = {k: torch.as_tensor(v) for k, v in arr.items() if k not in ("video", "bitrate_kbps")}
        sc = gt.session_score(s).numpy()
    arr["target"] = (1 + 4 * (sc - sc.min()) / (np.ptp(sc) + 1e-6) + rng.normal(0, 0.15, len(sc))).astype(np.float32)
np.savez_compressed(a.out, **arr)
json.dump([{"id": i, "video": r["video"], "trace": r["trace"], "policy": r["policy"], "levels": r["levels"],
             "chunks": r["chunks"],
            "rebuf": [float(x) for x in r["rebuf"]]} for i, r in enumerate(recs)],
          open(os.path.splitext(a.out)[0] + "_sessions.json", "w"))
print("saved", len(recs), "sessions ->", a.out)
