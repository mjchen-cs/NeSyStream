"""Synthetic data for smoke tests ONLY (never for reported results).

Videos: procedural frames whose texture / motion / brightness vary per chunk; the
rate-quality curve depends on that content, mimicking 'bitrate != quality'.
Traces: random-walk throughput in Pensieve format.
Sessions + MOS: generated from a hidden ground-truth rule set so that NS-QoE rule
recovery can be checked.
"""
from __future__ import annotations

import os
import numpy as np

LADDER_KBPS = np.array([235, 375, 560, 750, 1050, 1750, 2350, 3000, 4300, 5800], np.float32)


def make_video(rng, n_chunks=40, size=64, n_kf=4, chunk_len=4.0):
    L = len(LADDER_KBPS)
    tex = rng.uniform(0, 1, n_chunks)          # texture complexity
    mot = rng.uniform(0, 1, n_chunks)          # motion
    # piecewise-constant shots with cuts
    shot = np.cumsum(rng.uniform(0, 1, n_chunks) < 0.2)
    tex, mot = tex[shot % n_chunks], mot[shot % n_chunks]
    cut = np.r_[0, (np.diff(shot) > 0).astype(np.float32)]
    luma = rng.uniform(0.15, 0.8, n_chunks)[shot % n_chunks]
    face = (rng.uniform(0, 1, n_chunks) < 0.3).astype(np.float32)[shot % n_chunks]
    cplx = 0.5 * tex + 0.5 * mot
    lb = np.log(LADDER_KBPS / LADDER_KBPS[0])[None]
    mid = 1.0 + 2.5 * cplx[:, None]
    vmaf = 100 / (1 + np.exp(-(lb - mid) * 1.6))
    vmaf = np.maximum.accumulate(np.clip(vmaf + rng.normal(0, 1.0, vmaf.shape), 0, 100), axis=1)
    sizes = LADDER_KBPS[None] * 1e3 / 8 * chunk_len * rng.uniform(0.8, 1.2, (n_chunks, L)) * (0.8 + 0.4 * cplx[:, None])
    sizes = np.sort(sizes, axis=1)
    # frames: sinusoidal texture with frequency ~ tex, translating with speed ~ mot
    yy, xx = np.mgrid[0:size, 0:size] / size
    frames = np.zeros((n_chunks, n_kf, 3, size, size), np.uint8)
    for k in range(n_chunks):
        f = 2 + 20 * tex[k]
        for j in range(n_kf):
            ph = j * mot[k] * 2.0
            img = 0.5 + 0.5 * np.sin(2 * np.pi * (f * xx + ph)) * np.cos(2 * np.pi * f * yy)
            img = luma[k] * img + (0.3 if face[k] else 0) * np.exp(-((xx - .5) ** 2 + (yy - .4) ** 2) / .01)
            frames[k, j] = (np.clip(img, 0, 1) * 255).astype(np.uint8)[None].repeat(3, 0)
    concepts = np.stack([mot > 0.6, tex > 0.6, cut, face, luma < 0.25], 1).astype(np.float32)
    cmp = np.stack([np.log1p(sizes[:, 0]), cplx, mot, tex], 1).astype(np.float32)
    return dict(sizes=sizes.astype(np.float32), vmaf=vmaf.astype(np.float32), bitrates_kbps=LADDER_KBPS,
                heights=np.array([240, 360, 432, 480, 540, 720, 720, 1080, 1080, 1080], np.int32),
                chunk_len=np.float32(chunk_len), frames=frames, cmp_feats=cmp, concepts=concepts,
                raw_concepts=concepts.copy(), metric=np.array("synthetic"))


def make_trace(rng, seconds=600, mean_mbps=None):
    m = mean_mbps or rng.uniform(0.5, 6.0)
    t = np.arange(0, seconds, 1.0)
    # mean-reverting AR(1) in log domain (a plain random walk drifts to the clip bounds)
    y = np.zeros(len(t))
    for i in range(1, len(t)):
        y[i] = 0.95 * y[i - 1] + rng.normal(0, 0.12)
    x = m * np.exp(y) / np.exp(y).mean()
    return np.stack([t, np.clip(x, 0.1, 50)], 1)


def generate(out_dir, n_videos=8, n_traces=20, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    for i in range(n_videos):
        np.savez_compressed(os.path.join(out_dir, "videos", f"syn_{i:02d}.npz"), **make_video(rng))
    for i in range(n_traces):
        np.savetxt(os.path.join(out_dir, "traces", f"trace_{i:03d}.txt"), make_trace(rng), fmt="%.4f")


# hidden ground-truth for rule-recovery checks (VMAF points)
GT_WEIGHTS = {"rebuffer": 10.0, "rebuffer_recency": 6.0, "switch_at_cut": 0.15, "switch_mid_shot": 0.8,
              "motion_artifact": 1.0, "salient_drop": 1.2}
