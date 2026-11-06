"""Trace-driven chunk-level streaming simulator (Pensieve-style), extended with
per-chunk perceptual data: oracle VMAF labels, PQF predictions/descriptors and
visual concepts.

Network traces: Pensieve "cooked" format, one line per sample: `time_s throughput_Mbps`.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from ..models.nsqoe import NSQoE, feasible_max_level, VISUAL

B_IN_MB = 1e6
BITS_IN_BYTE = 8.0


def load_traces(trace_dir: str, max_traces: Optional[int] = None):
    files = sorted(glob.glob(os.path.join(trace_dir, "*")))
    out = []
    for f in files[: max_traces or len(files)]:
        arr = np.loadtxt(f, ndmin=2)
        if arr.shape[0] < 2:
            continue
        out.append({"name": os.path.basename(f), "time": arr[:, 0].astype(np.float64),
                    "mbps": np.maximum(arr[:, 1].astype(np.float64), 1e-3)})
    if not out:
        raise FileNotFoundError(f"no traces in {trace_dir}")
    return out


def load_video(npz_path: str) -> Dict[str, np.ndarray]:
    """Processed video file produced by `scripts/build_dataset.py` + `scripts/cache_pqf.py`.
    Required keys: sizes (N,L) bytes, vmaf (N,L), bitrates_kbps (L,), chunk_len ().
    Optional: concepts (N,C) labels, pqf_q (N,L), pqf_z (N,Z), pqf_concepts (N,C)."""
    with np.load(npz_path, allow_pickle=False) as z:   # keyframes ('frames') are not used by the simulator:
        d = {k: z[k] for k in z.files if k != "frames"}  # skip them (saves ~200 KB per chunk per process)
    d["name"] = os.path.splitext(os.path.basename(npz_path))[0]
    return d


@dataclass
class EnvConfig:
    buffer_max: float = 60.0      # s
    rtt: float = 0.08             # s
    payload_ratio: float = 0.95
    noise: float = 0.1            # multiplicative link noise
    hist: int = 8
    z_lookahead: int = 2          # number of future chunks whose descriptor is observed
    bw_aug: str = ""              # training only: "lo,hi" -> rescale each session's trace to a mean throughput
                                  # drawn log-uniformly from [lo, hi] Mbps (covers 3G..5G with Puffer dynamics)
    quality_source: str = "pqf"   # 'pqf' | 'pqf_<tag>' | 'oracle' | 'bitrate'
    b_min: float = 0.5            # hard rule H1
    use_hard_rules: bool = True
    hard_tput: str = "low"        # throughput estimate in H1: 'low' = robust c_low (default) | 'hat' = harmonic mean
                                  # | 'conformal' = harmonic mean x alpha-quantile of calibrated ratios thr/c_hat,
                                  #   with adaptive conformal inference (Gibbs & Candes 2021) updating alpha online
    conf_alpha: float = 0.1       # target miscoverage of the H1 throughput bound
    conf_gamma: float = 0.01      # ACI step size
    conf_file: str = ""           # calibration ratios (scripts/calibrate_conformal.py)
    conf_alpha_hi: float = 0.0    # >0: buffer-adaptive risk level; alpha rises linearly from conf_alpha (buffer <=
    conf_buf_lo: float = 10.0     #     conf_buf_lo s) to conf_alpha_hi (buffer >= conf_buf_hi s). A miss of the bound
    conf_buf_hi: float = 30.0     #     cannot stall a full buffer, so the bound is relaxed only where it is harmless
    hard_off_buffer: float = 0.0  # >0: skip H1 when the buffer already holds at least this many seconds
    use_concepts: bool = True     # ablation: False -> the POLICY sees no visual concepts (reward unchanged)
    log_tput: bool = False        # observe throughput history as log1p(Mbps)/log1p(1000) (3G..5G on one scale)
    reward: str = "nsqoe"         # 'nsqoe' | 'linear_vmaf' (ablation)
    playlist_chunks: int = 0      # >0: concatenate random clips (same split) into sessions of this many chunks


class ABREnv:
    def __init__(self, videos: List[dict], traces: List[dict], nsqoe: NSQoE, cfg: EnvConfig = EnvConfig(),
                 seed: int = 0, z_dim: int = 0):
        self.videos, self.traces, self.q, self.cfg = videos, traces, nsqoe, cfg
        self.rng = np.random.default_rng(seed)
        self.L = videos[0]["sizes"].shape[1]
        self.C = len(VISUAL)
        self.z_dim = z_dim or (videos[0]["pqf_z"].shape[1] if "pqf_z" in videos[0] else 0)

    # ------------------------------------------------------------------ #
    @property
    def obs_dim(self):
        h, L = self.cfg.hist, self.L
        return 2 * h + 4 + L + L + self.cfg.z_lookahead * self.z_dim + self.C + L + 1

    PER_CHUNK = ("sizes", "vmaf", "concepts", "raw_concepts", "pqf_q", "pqf_z", "pqf_concepts", "frames", "cmp_feats")

    def make_playlist(self, first_idx: Optional[int] = None, seed: Optional[int] = None):
        """Short sources (5-20 s) are concatenated into a feed-like session of ~playlist_chunks chunks.
        Clip boundaries are real content changes (the SceneCut concept of the next clip's first chunk is set)."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        order = [first_idx] if first_idx is not None else []
        while sum(self.videos[i]["sizes"].shape[0] for i in order) < self.cfg.playlist_chunks:
            order.append(int(rng.integers(len(self.videos))))
        parts = [self.videos[i] for i in order]
        keys = list(self.PER_CHUNK) + [k for k in parts[0] if k.startswith("pqf_") and k not in self.PER_CHUNK]
        v = {k: np.concatenate([p[k] for p in parts])[: self.cfg.playlist_chunks]
             for k in keys if all(k in p for p in parts)}
        starts = np.cumsum([0] + [p["sizes"].shape[0] for p in parts[:-1]])
        for key in ("concepts", "pqf_concepts"):   # the server knows clip boundaries: SceneCut there
            if key in v:
                for st in starts[1:]:
                    if st < len(v[key]):
                        v[key][st, 2] = 1.0
        v.update({k: parts[0][k] for k in ("bitrates_kbps", "chunk_len", "heights") if k in parts[0]})
        v["name"] = "+".join(p["name"] for p in parts)
        v["chunk_src"] = [(p["name"], k) for p in parts for k in range(p["sizes"].shape[0])][: self.cfg.playlist_chunks]
        return v

    def reset(self, video_idx: Optional[int] = None, trace_idx: Optional[int] = None, random_start=True,
              playlist_seed: Optional[int] = None):
        if self.cfg.playlist_chunks > 0:
            self.v = self.make_playlist(video_idx, playlist_seed)
        else:
            self.v = self.videos[self.rng.integers(len(self.videos)) if video_idx is None else video_idx]
        self.tr = self.traces[self.rng.integers(len(self.traces)) if trace_idx is None else trace_idx]
        if self.cfg.bw_aug:
            lo, hi = (float(x) for x in self.cfg.bw_aug.split(","))
            target = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
            self.tr = dict(self.tr, mbps=self.tr["mbps"] * (target / max(float(np.mean(self.tr["mbps"])), 1e-6)))
        self.ptr = int(self.rng.integers(1, len(self.tr["time"]))) if random_start else 1
        self.last_t = self.tr["time"][self.ptr - 1]
        self.chunk_len = float(self.v["chunk_len"])
        self.N = self.v["sizes"].shape[0]
        self.k = 0
        self.buffer = 0.0
        self.prev_level = -1
        self.prev_q = None
        self.prev_true_q = None
        self.thr_hist = np.zeros(self.cfg.hist)
        self.dl_hist = np.zeros(self.cfg.hist)
        self.log = []
        self.alpha_t = self.cfg.conf_alpha          # ACI state (per session)
        self._c_used = None
        return self._obs()

    def _alpha_bonus(self):
        c = self.cfg
        if c.conf_alpha_hi <= 0:
            return 0.0
        w = float(np.clip((self.buffer - c.conf_buf_lo) / max(c.conf_buf_hi - c.conf_buf_lo, 1e-6), 0.0, 1.0))
        return w * (c.conf_alpha_hi - c.conf_alpha)

    def _c_conformal(self, c_hat):
        """Conformal lower bound on the next chunk's throughput: c_hat x Quantile_{alpha_t}(thr / c_hat)."""
        if not hasattr(self, "_conf_r"):
            import json as _json
            self._conf_r = np.sort(np.asarray(_json.load(open(self.cfg.conf_file))["ratios"], np.float64))
        a = float(np.clip(self.alpha_t + self._alpha_bonus(), 1e-3, 0.999))
        return c_hat * float(np.quantile(self._conf_r, a))

    # ------------------------------------------------------------------ #
    def _quality_curve(self, k):
        src = self.cfg.quality_source
        if src == "oracle":
            return self.v["vmaf"][k]
        if src.startswith("pqf"):   # 'pqf' (main) or 'pqf_<tag>' (e.g. pqf_cheap) cached by cache_quality.py
            return self.v["pqf_q" if src == "pqf" else f"pqf_q_{src[4:]}"][k]
        br = self.v["bitrates_kbps"].astype(np.float64)
        return 100.0 * np.log(br / br[0]) / np.log(br[-1] / br[0])

    def _concepts(self, k, label=False):
        if label and "concepts" in self.v:
            return self.v["concepts"][k]
        if self.cfg.quality_source == "bitrate" or not self.cfg.use_concepts:
            return np.zeros(self.C)
        if "pqf_concepts" in self.v:
            return self.v["pqf_concepts"][k]
        return self.v.get("concepts", np.zeros((self.N, self.C)))[k]

    def _c_hat(self):
        h = self.thr_hist[self.thr_hist > 0][-5:]
        if len(h) == 0:
            return 1.0 * B_IN_MB, 0.5 * B_IN_MB
        hm = len(h) / np.sum(1.0 / h)
        err = 0.0
        if len(self.log) >= 2:
            errs = [abs(e["c_hat"] - e["thr"]) / e["thr"] for e in self.log[-5:] if e["thr"] > 0]
            err = max(errs) if errs else 0.0
        return hm * B_IN_MB, hm / (1 + err) * B_IN_MB

    def decision_context(self):
        """Everything the policy needs for the next decision (as torch tensors, batch 1)."""
        k = self.k
        q_curve = self._quality_curve(k).astype(np.float32)
        sizes = self.v["sizes"][k].astype(np.float32)
        c_hat, c_low = self._c_hat()
        prev_q = q_curve[0] if self.prev_q is None else self.prev_q
        conc = self._concepts(k).astype(np.float32)
        thr_ratio = (self.thr_hist[-1] / self.thr_hist[-2]) if self.thr_hist[-2] > 0 else 1.0
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))[None]
        with torch.no_grad():
            qoe_tab = self.q.step_qoe_all_levels(T(q_curve), T(sizes), T(c_hat), T(self.buffer),
                                                 T(prev_q), T(conc), T(k / self.N), self.cfg.rtt)[0]
            c_h1 = c_hat if self.cfg.hard_tput == "hat" else (
                self._c_conformal(c_hat) if self.cfg.hard_tput == "conformal" else c_low)
            self._c_used = c_h1
            self._a_target = self.cfg.conf_alpha + (self._alpha_bonus() if self.cfg.hard_tput == "conformal" else 0.0)
            if self.cfg.hard_off_buffer > 0 and self.buffer >= self.cfg.hard_off_buffer:
                c_h1 = 1e12                                    # ample buffer: H1 inactive (H2 still applies)
            if self.cfg.use_hard_rules:
                l_max = feasible_max_level(T(sizes), T(c_h1), T(self.buffer),
                                           torch.tensor([self.prev_level]), T(thr_ratio), self.q.cfg,
                                           self.cfg.b_min, self.cfg.rtt)[0]
            else:
                l_max = torch.tensor(self.L - 1)
        # energy = log(1 + QoE gap / 10): ~linear for small gaps, compressed for stall-sized gaps (hundreds of
        # points once stall rules are calibrated), so guidance strength and the observation stay well scaled
        energy = torch.log1p((qoe_tab.max() - qoe_tab).clamp_min(0) / 10.0)   # 0 = best one-step level
        return {"q_curve": q_curve, "sizes": sizes, "c_hat": c_hat, "c_low": c_low, "c_h1": c_h1, "energy": energy.numpy(),
                "l_max": int(l_max), "concepts": conc}

    def _obs(self):
        self.ctx = self.decision_context()
        c = self.cfg
        z = []
        if self.z_dim:
            for j in range(c.z_lookahead):
                kk = min(self.k + j, self.N - 1)
                zk = "pqf_z" if not c.quality_source.startswith("pqf_") else f"pqf_z_{c.quality_source[4:]}"
                z.append(self.v[zk][kk] / 100.0 if c.quality_source != "bitrate" and zk in self.v else np.zeros(self.z_dim))
        obs = np.concatenate([
            (np.log1p(self.thr_hist) / np.log1p(1000.0)) if c.log_tput else self.thr_hist / 10.0, self.dl_hist / 10.0,
            [self.buffer / 10.0, max(self.prev_level, 0) / (self.L - 1),
             (self.prev_q or 0.0) / 100.0, (self.N - self.k) / self.N],
            self.ctx["sizes"] / B_IN_MB, self.ctx["q_curve"] / 100.0, *z, self.ctx["concepts"],
            self.ctx["energy"], [self.ctx["l_max"] / (self.L - 1)]]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------ #
    def _download(self, size_bytes):
        t, mbps = self.tr["time"], self.tr["mbps"]
        delay, sent = 0.0, 0.0
        while True:
            thr = mbps[self.ptr] * B_IN_MB / BITS_IN_BYTE * self.cfg.payload_ratio  # bytes / s
            dur = t[self.ptr] - self.last_t
            if sent + thr * dur >= size_bytes:
                frac = (size_bytes - sent) / thr
                delay += frac
                self.last_t += frac
                break
            sent += thr * dur
            delay += dur
            self.last_t = t[self.ptr]
            self.ptr += 1
            if self.ptr >= len(t):
                self.ptr, self.last_t = 1, t[0]
        delay += self.cfg.rtt
        delay *= float(self.rng.uniform(1 - self.cfg.noise, 1 + self.cfg.noise))
        return delay

    def step(self, level: int):
        c = self.cfg
        level = int(np.clip(level, 0, self.L - 1))
        k = self.k
        size = float(self.v["sizes"][k, level])
        violated = level > self.ctx["l_max"]
        delay = self._download(size)
        rebuf = max(delay - self.buffer, 0.0)
        self.buffer = max(self.buffer - delay, 0.0) + self.chunk_len
        sleep = 0.0
        if self.buffer > c.buffer_max:
            sleep = self.buffer - c.buffer_max
            self.buffer = c.buffer_max
            # advance trace clock while sleeping
            self.last_t += sleep
            while self.ptr < len(self.tr["time"]) - 1 and self.tr["time"][self.ptr] < self.last_t:
                self.ptr += 1
        thr = size * BITS_IN_BYTE / max(delay, 1e-6) / B_IN_MB  # Mbps
        self.thr_hist = np.roll(self.thr_hist, -1); self.thr_hist[-1] = thr
        self.dl_hist = np.roll(self.dl_hist, -1); self.dl_hist[-1] = delay

        true_q = float(self.v["vmaf"][k, level])
        prev_true = true_q if self.prev_true_q is None else self.prev_true_q
        reward = self.chunk_reward(true_q, prev_true, level, rebuf, k)
        if self.cfg.hard_tput == "conformal" and self._c_used is not None:   # ACI update of alpha
            self.alpha_t += self.cfg.conf_gamma * (self._a_target - float(thr * B_IN_MB < self._c_used))
        rec = {"k": k, "level": level, "vmaf": true_q, "rebuf": rebuf, "bytes": size, "delay": delay,
               "c_low": self.ctx["c_low"] / B_IN_MB, "c_h1": float(self.ctx.get("c_h1", self.ctx["c_low"])) / B_IN_MB,
               "buffer": self.buffer, "thr": thr, "c_hat": self.ctx["c_hat"] / B_IN_MB, "violated": violated,
               "l_max": self.ctx["l_max"], "reward": reward, "bitrate_kbps": float(self.v["bitrates_kbps"][level])}
        self.log.append(rec)
        self.prev_level, self.prev_true_q = level, true_q
        self.prev_q = float(self.ctx["q_curve"][level])
        self.k += 1
        done = self.k >= self.N
        obs = None if done else self._obs()
        return obs, reward, done, rec

    def chunk_reward(self, true_q, prev_true_q, level, rebuf, k):
        """NS-QoE chunk score with ground-truth quality and concept labels, scaled to ~[-1, 1]."""
        if self.cfg.reward == "linear_vmaf":
            return (true_q - 4.3 * 20 * rebuf - abs(true_q - prev_true_q)) / 100.0
        T = lambda a: torch.tensor([float(a)])
        conc = torch.as_tensor(self._concepts(k, label=True), dtype=torch.float32)[None]
        thr_ratio = (self.thr_hist[-1] / self.thr_hist[-2]) if self.thr_hist[-2] > 0 else 1.0
        with torch.no_grad():
            preds = self.q.build_preds(T(true_q), T(prev_true_q), T(level / (self.L - 1)), T(rebuf),
                                       T(self.buffer), T(thr_ratio), T(k / self.N), conc)
            return float(self.q.chunk_qoe(T(true_q), preds)[0]) / 100.0

    def session_record(self):
        """Arrays for NS-QoE fitting / evaluation of the finished session."""
        lg = self.log
        N = len(lg)
        conc = np.stack([self._concepts(e["k"], label=True) for e in lg]).astype(np.float32)
        thr = np.array([e["thr"] for e in lg])
        ratio = np.concatenate([[1.0], thr[1:] / np.maximum(thr[:-1], 1e-6)])
        return {"q": np.array([e["vmaf"] for e in lg], np.float32),
                "level01": np.array([e["level"] / (self.L - 1) for e in lg], np.float32),
                "rebuf": np.array([e["rebuf"] for e in lg], np.float32),
                "buffer": np.array([e["buffer"] for e in lg], np.float32),
                "thr_ratio": ratio.astype(np.float32),
                "pos": (np.arange(N) / max(N, 1)).astype(np.float32), "concepts": conc,
                "bitrate_kbps": np.array([e["bitrate_kbps"] for e in lg], np.float32),
                "video": self.v["name"], "trace": self.tr["name"]}
