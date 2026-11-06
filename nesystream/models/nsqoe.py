"""Neuro-Symbolic QoE (NS-QoE).

Streaming QoE is expressed as weighted first-order rules over
  * visual predicates   (from PQF concept heads): HighMotion, HighTexture, SceneCut, Salient, Dark
  * playback predicates (from the player):        Rebuffer, LowBuffer, ThroughputDrop, Switch,
                                                  QualityDrop, LowRate, NearEnd, UpSwitch

Rules are relaxed with the product t-norm (a AND b = a*b, NOT a = 1-a) so that
rule weights are learned by gradient descent from subjective scores (or from a
teacher model such as ITU-T P.1203 for the playback rules).

Per chunk k:
    QoE_k = q_k - sum_r  w_r * T(body_r)_k * m_r,k
Session:
    MOS_hat = alpha * mean_k QoE_k + beta

Two rule sets:
  * hand-written rules R_hand (each with literature support, see paper Tab. 1)
  * a template bank R_ind = {visual AND playback-event} with L1-sparse weights
    (sparse rule induction); weights below a threshold are pruned after fitting.

Hard rules are not part of the score: they define the feasible action set used
by the logic-guided diffusion policy (see `feasible_max_level`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VISUAL = ["HighMotion", "HighTexture", "SceneCut", "Salient", "Dark"]
PLAYBACK_EVENTS = ["Rebuffer", "Switch", "QualityDrop", "LowRate"]


@dataclass
class PredicateConfig:
    tau_rebuf: float = 0.5      # s, Rebuffer = 1 - exp(-rebuf / tau)
    switch_thr: float = 3.0     # VMAF points
    switch_temp: float = 1.5
    drop_thr: float = 5.0
    drop_temp: float = 2.0
    low_buffer_s: float = 4.0   # s
    buffer_temp: float = 1.0
    thr_drop_ratio: float = 0.6
    thr_temp: float = 0.1


def playback_predicates(q, prev_q, level01, rebuf, buffer, thr_ratio, pos, cfg: PredicateConfig):
    """All inputs broadcastable tensors. Returns dict of soft truth values in [0,1]
    and magnitudes used by the rules."""
    dq = q - prev_q
    r = rebuf.clamp_min(0)
    ev = 1.0 - torch.exp(-r / cfg.tau_rebuf)              # soft "a stall happened in this chunk"
    first = (pos <= 1e-6).to(r.dtype)                     # chunk 0: initial loading, not a stall
    return {
        "Rebuffer": ev * (1.0 - first),
        "Startup": ev * first,
        "Switch": torch.sigmoid((dq.abs() - cfg.switch_thr) / cfg.switch_temp),
        "QualityDrop": torch.sigmoid((-dq - cfg.drop_thr) / cfg.drop_temp),
        "LowRate": 1.0 - level01,
        "NearEnd": pos,
        "LowBuffer": torch.sigmoid((cfg.low_buffer_s - buffer) / cfg.buffer_temp),
        "ThroughputDrop": torch.sigmoid((cfg.thr_drop_ratio - thr_ratio) / cfg.thr_temp),
        # magnitudes (not truth values)
        # stall impact per event, saturating in duration (P.1203 / QoE literature): 1 + ln(1 + seconds)
        "_rebuf_s": 1.0 + torch.log1p(r),
        "_startup_s": r,
        "_abs_dq": dq.abs(),
        "_drop": (-dq).clamp_min(0),
    }


# name, body literals [(predicate, negated)], magnitude key (None => 10 VMAF points)
HAND_RULES: List[Tuple[str, List[Tuple[str, bool]], str]] = [
    ("rebuffer", [("Rebuffer", False)], "_rebuf_s"),
    ("startup_delay", [("Startup", False)], "_startup_s"),
    ("rebuffer_recency", [("Rebuffer", False), ("NearEnd", False)], "_rebuf_s"),
    ("switch_at_cut", [("Switch", False), ("SceneCut", False)], "_abs_dq"),
    ("switch_mid_shot", [("Switch", False), ("SceneCut", True)], "_abs_dq"),
    ("motion_artifact", [("HighMotion", False), ("LowRate", False)], None),
    ("salient_drop", [("Salient", False), ("QualityDrop", False)], "_drop"),
    ("low_rate", [("LowRate", False)], None),
]


def induced_templates() -> List[Tuple[str, List[Tuple[str, bool]], str]]:
    out = []
    for v in VISUAL:
        for e in PLAYBACK_EVENTS:
            mag = {"Rebuffer": "_rebuf_s", "Switch": "_abs_dq", "QualityDrop": "_drop", "LowRate": None}[e]
            out.append((f"ind:{v}&{e}", [(v, False), (e, False)], mag))
    return out


class NSQoE(nn.Module):
    def __init__(self, use_hand: bool = True, use_induced: bool = True, cfg: PredicateConfig | None = None,
                 init_weights: Dict[str, float] | None = None):
        super().__init__()
        self.cfg = cfg or PredicateConfig()
        self.rules = (HAND_RULES if use_hand else []) + (induced_templates() if use_induced else [])
        self.n_hand = len(HAND_RULES) if use_hand else 0
        # literature-informed initial weights (VMAF points per unit magnitude)
        default = {"rebuffer": 8.0, "startup_delay": 1.0, "rebuffer_recency": 4.0, "switch_at_cut": 0.3, "switch_mid_shot": 0.6,
                   "motion_artifact": 0.5, "salient_drop": 0.5}
        default.update(init_weights or {})
        init = [default.get(n, 0.05) for n, _, _ in self.rules]
        w0 = torch.tensor(init, dtype=torch.float32).clamp_min(1e-3)
        self.raw_w = nn.Parameter(w0 + torch.log(-torch.expm1(-w0)))   # stable softplus^-1 (no overflow for w > 88)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    log_param = False   # fitting: weights = exp(raw_w) (multiplicative steps; weights can span decades)

    @property
    def weights(self):
        return torch.exp(self.raw_w) if self.log_param else F.softplus(self.raw_w)

    def use_log_param(self):
        with torch.no_grad():
            w = self.weights.clone()
            self.log_param = True
            self.raw_w.copy_(torch.log(w.clamp_min(1e-6)))

    def _truth(self, body, preds):
        t = 1.0
        for name, neg in body:
            p = preds[name]
            t = t * ((1.0 - p) if neg else p)
        return t

    def rule_activations(self, preds) -> torch.Tensor:
        """(..., R) penalty per rule before weighting = truth * magnitude."""
        acts = []
        for _, body, mag in self.rules:
            m = preds[mag] if mag is not None else 10.0
            acts.append(self._truth(body, preds) * m)
        return torch.stack(torch.broadcast_tensors(*acts), dim=-1)

    def chunk_qoe(self, q, preds):
        if not self.rules:
            return q
        return q - (self.rule_activations(preds) * self.weights).sum(-1)

    def build_preds(self, q, prev_q, level01, rebuf, buffer, thr_ratio, pos, concepts):
        preds = playback_predicates(q, prev_q, level01, rebuf, buffer, thr_ratio, pos, self.cfg)
        for i, v in enumerate(VISUAL):
            preds[v] = concepts[..., i]
        return preds

    # ------------------------------------------------------------------ #
    def session_score(self, s: Dict[str, torch.Tensor]):
        """s: batched sessions, each tensor (B, N) plus concepts (B, N, C) and mask (B, N)."""
        prev_q = torch.cat([s["q"][:, :1], s["q"][:, :-1]], dim=1)
        preds = self.build_preds(s["q"], prev_q, s["level01"], s["rebuf"], s["buffer"], s["thr_ratio"],
                                 s["pos"], s["concepts"])
        qk = self.chunk_qoe(s["q"], preds)
        m = s["mask"]
        mean = (qk * m).sum(1) / m.sum(1).clamp_min(1)
        return self.alpha * mean + self.beta

    def sparsity(self):
        return self.weights[self.n_hand:].sum() if len(self.rules) > self.n_hand else torch.tensor(0.0)

    def rule_table(self, prune: float = 0.05):
        rows = []
        for (name, body, _), w in zip(self.rules, self.weights.detach().cpu().tolist()):
            if name.startswith("ind:") and w < prune:
                continue
            lits = " AND ".join(("NOT " if n else "") + p for p, n in body)
            rows.append((name, lits, w))
        return rows

    # ------------------------------------------------------------------ #
    def step_qoe_all_levels(self, qhat, sizes_bytes, c_hat_bps, buffer, prev_q, concepts, pos, rtt=0.08):
        """One-step look-ahead QoE for every candidate level (used as -energy for guidance).

        qhat: (B, L)  predicted quality of next chunk per level
        sizes_bytes: (B, L); c_hat_bps: (B,) throughput estimate; buffer: (B,) seconds
        prev_q: (B,) quality of previous chunk; concepts: (B, C); pos: (B,)
        returns (B, L)
        """
        B, L = qhat.shape
        dl = sizes_bytes * 8.0 / c_hat_bps.clamp_min(1.0)[:, None] + rtt
        rebuf = (dl - buffer[:, None]).clamp_min(0)
        level01 = torch.linspace(0, 1, L, device=qhat.device).expand(B, L)
        thr_ratio = torch.ones_like(qhat)
        preds = self.build_preds(qhat, prev_q[:, None].expand(B, L), level01, rebuf,
                                 buffer[:, None].expand(B, L), thr_ratio, pos[:, None].expand(B, L),
                                 concepts[:, None, :].expand(B, L, concepts.shape[-1]))
        return self.chunk_qoe(qhat, preds)


# --------------------------------------------------------------------------- #
# Hard rules -> feasible action set
# --------------------------------------------------------------------------- #
def feasible_max_level(sizes_bytes, c_low_bps, buffer, prev_level, thr_ratio, cfg: PredicateConfig,
                       b_min: float = 0.5, rtt: float = 0.08):
    """Feasible set F(s) = {0, ..., l_max}.  Because chunk size grows with level,
    both hard rules yield a prefix of the ladder, so projection onto F is a clamp.

    H1 (buffer safety):  buffer - (size_l / c_low + rtt) >= b_min   (level 0 always allowed)
    H2 (no up-switch under danger): LowBuffer AND ThroughputDrop -> level <= prev_level
    All inputs are batched tensors: sizes (B, L), others (B,). Returns LongTensor (B,).
    """
    dl = sizes_bytes * 8.0 / c_low_bps.clamp_min(1.0)[:, None] + rtt
    ok = (buffer[:, None] - dl) >= b_min
    ok[:, 0] = True
    # sizes are monotone in level, so the largest ok index is a prefix bound
    idx = torch.arange(ok.shape[1], device=ok.device).expand_as(ok)
    l_max = torch.where(ok, idx, torch.zeros_like(idx)).max(dim=1).values
    danger = (buffer < cfg.low_buffer_s) & (thr_ratio < cfg.thr_drop_ratio)
    l_max = torch.where(danger, torch.minimum(l_max, prev_level.clamp_min(0)), l_max)
    return l_max
