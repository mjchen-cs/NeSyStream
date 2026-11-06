"""Split-conformal calibration of the H1 throughput bound on TRAINING traces.

python scripts/calibrate_conformal.py --video_dir data/videos --trace_dir data/traces/train --out runs/conformal/ratios.json
Runs RobustMPC(3, quality) sessions (training videos, training traces, bandwidth augmentation as in training) and records,
for every chunk after the first, the ratio r = measured throughput / harmonic-mean forecast c_hat.
The H1 bound is c_hat * Quantile_alpha(r); with exchangeable sessions, P(thr >= bound) >= 1 - alpha.
Also reports the empirical coverage of the current heuristic bound c_low = c_hat / (1 + max recent error).
"""
import _common  # noqa
import argparse, json, os
import numpy as np
from nesystream.env.abr_env import ABREnv, EnvConfig, load_traces, load_video
from nesystream.data.chunks import list_videos, split_by_video
from nesystream.models.nsqoe import NSQoE
from nesystream.eval.baselines import RobustMPC

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", default="data/videos"); ap.add_argument("--trace_dir", default="data/traces/train")
ap.add_argument("--out", default="runs/conformal/ratios.json"); ap.add_argument("--sessions", type=int, default=600)
ap.add_argument("--bw_aug", default="0.3,600"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
tr, _, _ = split_by_video(list_videos(a.video_dir), os.path.join(a.video_dir, "split.json"))
env = ABREnv([load_video(p) for p in tr], load_traces(a.trace_dir), NSQoE(),
             EnvConfig(playlist_chunks=30, bw_aug=a.bw_aug), seed=a.seed)
mpc = RobustMPC(3, "quality"); R, cov_low = [], []
for i in range(a.sessions):
    env.reset(); done = False
    while not done:
        _, _, done, rec = env.step(min(mpc.act(env), env.ctx["l_max"]))
        if rec["k"] >= 1 and rec["c_hat"] > 0:
            R.append(rec["thr"] / rec["c_hat"]); cov_low.append(rec["thr"] >= rec["c_low"])
    if (i + 1) % 100 == 0:
        print(f"{i + 1} sessions, {len(R)} ratios", flush=True)
R = np.asarray(R)
qs = {f"q{int(100 * al)}": float(np.quantile(R, al)) for al in (0.01, 0.02, 0.05, 0.1, 0.2)}
print("ratio quantiles:", qs, "| coverage of current c_low:", float(np.mean(cov_low)))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
sub = np.sort(R)[np.linspace(0, len(R) - 1, min(len(R), 20000)).astype(int)]
json.dump({"ratios": sub.round(5).tolist(), "n": int(len(R)), "quantiles": qs,
           "coverage_current_c_low": float(np.mean(cov_low))}, open(a.out, "w"))
