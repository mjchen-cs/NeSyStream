"""Classical baselines operating on the same simulator.

* BBA      - buffer-based (Huang et al. 2014)
* RB       - rate-based: highest bitrate below harmonic-mean throughput
* RobustMPC- look-ahead MPC (Yin et al. 2015); objective selectable:
             'bitrate' (original linear QoE) or 'quality' (perceptual: uses the observed
             quality curve, i.e. a content-aware MPC baseline)
Learning-based baselines (Pensieve, Comyco, Fugu, ...) should be retrained with their
official code on the same videos/traces; their chunk-size interface matches this env.
"""
from __future__ import annotations

import itertools
import numpy as np


class BBA:
    name = "BBA"

    def __init__(self, reservoir=5.0, cushion=10.0):
        self.r, self.c = reservoir, cushion

    def act(self, env):
        L = env.L
        b = env.buffer
        if b < self.r:
            return 0
        if b >= self.r + self.c:
            return L - 1
        return int((b - self.r) / self.c * (L - 1))


class RB:
    name = "RB"

    def act(self, env):
        c_hat, _ = env._c_hat()
        br = env.v["bitrates_kbps"] * 1e3
        ok = np.where(br <= c_hat)[0]
        return int(ok.max()) if len(ok) else 0


class RobustMPC:
    def __init__(self, horizon=5, objective="bitrate", rebuf_pen=4.3, smooth_pen=1.0):
        self.h, self.obj, self.mu, self.lam = horizon, objective, rebuf_pen, smooth_pen
        self.name = f"RobustMPC-{objective}"

    def act(self, env):
        k, N, L = env.k, env.N, env.L
        h = min(self.h, N - k)
        _, c_low = env._c_hat()
        if self.obj == "bitrate":
            util = lambda kk, l: env.v["bitrates_kbps"][l] / 1000.0
            scale_mu = self.mu
        else:  # perceptual: observed quality curve (PQF / oracle / bitrate proxy depending on env)
            util = lambda kk, l: env._quality_curve(kk)[l] / 20.0
            scale_mu = self.mu
        prev = max(env.prev_level, 0)
        best, best_l = -1e18, 0
        # restrict the combinatorial search to levels near the previous one for speed
        cand = range(L) if L <= 6 else sorted(set(range(max(0, prev - 3), min(L, prev + 4))) | {0})
        for seq in itertools.product(cand, repeat=h):
            buf, val, last = env.buffer, 0.0, prev
            for j, l in enumerate(seq):
                kk = k + j
                dl = env.v["sizes"][kk, l] * 8.0 / c_low + env.cfg.rtt
                rebuf = max(dl - buf, 0.0)
                buf = max(buf - dl, 0.0) + env.chunk_len
                val += util(kk, l) - scale_mu * rebuf - self.lam * abs(util(kk, l) - util(kk, last))
                last = l
            if val > best:
                best, best_l = val, seq[0]
        return int(best_l)
