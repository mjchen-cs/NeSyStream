"""Evaluate policies on all (video, trace) test pairs.

Metrics (per session, then averaged):
  vmaf_mean       mean VMAF of played chunks (oracle labels)
  rebuf_ratio     total stall time / (stall + playback time)
  switch_mag      mean |delta VMAF| between consecutive chunks
  mbytes          downloaded MB
  violation_rate  fraction of decisions with level > l_max (hard-rule violations)
  latency_ms      mean decision time of the policy
  p1203           ITU-T P.1203 O46 score if the `itu-p1203` package is installed (independent evaluator)
The NS-QoE score of the trained reward model is reported separately and must NOT be the
headline metric (it is the training reward).
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import torch


def p1203_score(rec, chunk_len, resolutions=None, fps=30.0):
    """Mode-0 P.1203 via the open-source `itu-p1203` package.
    NOTE: verify the input format against the installed package version
    (https://github.com/itu-p1203/itu-p1203)."""
    try:
        from itu_p1203 import P1203Standalone  # type: ignore
        import logging
        logging.getLogger("main").setLevel(logging.ERROR)          # 'O21 has no scores' (video-only) warnings
        for n in list(logging.root.manager.loggerDict):
            if "p1203" in n.lower():
                logging.getLogger(n).setLevel(logging.ERROR)
    except Exception:
        if not getattr(p1203_score, "_warned", False):
            print("[warn] itu_p1203 not installed -> p1203 column is NaN. Install from source: "
                  "pip install git+https://github.com/itu-p1203/itu-p1203.git")
            p1203_score._warned = True
        return float("nan")
    segs, stalls, t = [], [], 0.0
    for k, br in enumerate(rec["bitrate_kbps"]):
        res = resolutions[int(round(rec["level01"][k] * (len(resolutions) - 1)))] if resolutions else "1920x1080"
        segs.append({"bitrate": float(br), "codec": "h264", "duration": chunk_len, "fps": fps,
                     "resolution": res, "start": t})
        if rec["rebuf"][k] > 0:
            stalls.append([t, float(rec["rebuf"][k])])
        t += chunk_len
    inp = {"I11": {"segments": [], "streamId": 42}, "I13": {"segments": segs, "streamId": 42},
           "I23": {"stalling": stalls, "streamId": 42},
           "IGen": {"device": "pc", "displaySize": "1920x1080", "viewingDistance": "150cm"}}
    try:
        return float(P1203Standalone(inp).calculate_complete()["O46"])
    except Exception as e:  # noqa
        if not getattr(p1203_score, "_warned_calc", False):
            print(f"[warn] P.1203 computation failed ({e!r}); p1203 is NaN for such sessions")
            p1203_score._warned_calc = True
        return float("nan")


def run_policy(env, act_fn, video_idx, trace_idx, resolutions=None):
    obs = env.reset(video_idx, trace_idx, random_start=False, playlist_seed=1000 * video_idx + trace_idx)
    lat, done = [], False
    while not done:
        t0 = time.perf_counter()
        a = act_fn(env, obs)
        lat.append((time.perf_counter() - t0) * 1e3)
        obs, _, done, _ = env.step(a)
    rec = env.session_record()
    lg = env.log
    play = len(lg) * env.chunk_len
    stall = float(np.sum(rec["rebuf"]))
    m = {"vmaf_mean": float(np.mean(rec["q"])),
         "rebuf_ratio": stall / (stall + play),
         "switch_mag": float(np.mean(np.abs(np.diff(rec["q"])))) if len(lg) > 1 else 0.0,
         "mbytes": float(sum(e["bytes"] for e in lg) / 1e6),
         "violation_rate": float(np.mean([e["violated"] for e in lg])),
         "latency_ms": float(np.mean(lat)),
         "p1203": p1203_score(rec, env.chunk_len, resolutions)}
    with torch.no_grad():
        s = {k: torch.as_tensor(v)[None] for k, v in rec.items() if isinstance(v, np.ndarray) and k != "bitrate_kbps"}
        s["mask"] = torch.ones_like(s["q"])
        m["nsqoe_train_reward"] = float(env.q.session_score(s)[0])
    return m


def evaluate(env, policies: dict, pairs, resolutions=None):
    """policies: name -> act_fn(env, obs) -> level. pairs: list of (video_idx, trace_idx)."""
    out = {}
    for name, fn in policies.items():
        rows = [run_policy(env, fn, v, t, resolutions) for v, t in pairs]
        out[name] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    return out


def bandwidth_saving(results, ref: str, ours: str):
    """Percentage of bytes saved by `ours` relative to `ref` (report together with vmaf_mean)."""
    return 100.0 * (1 - results[ours]["mbytes"] / results[ref]["mbytes"])


def print_table(results):
    keys = list(next(iter(results.values())).keys())
    print("method".ljust(24) + "".join(k[:14].rjust(16) for k in keys))
    for n, r in results.items():
        print(n.ljust(24) + "".join(f"{r[k]:16.4f}" for k in keys))


def save(results, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
