"""Learning-based baselines re-implemented on the same simulator, state and data.

All share one PPO core (discrete levels) and differ in what they observe and optimise:

  Pensieve-PPO   [Mao et al. SIGCOMM'17; PPO variant widely used as reference]
                 obs: bitrate-only (quality_source='bitrate'), reward: linear bitrate QoE
  PPO-VMAF       content-aware RL: obs with PQF curve + descriptors, reward: linear VMAF QoE
  Jade-style     [Huang et al. ACM MM'23, "Optimizing adaptive video streaming with human
                 feedback"]: reward = a neural QoE model learned with a pairwise RANKING loss
                 from session scores (the human-feedback signal); obs as PPO-VMAF
  PPO-Lagrangian safe-RL baseline: stall indicator as a cost with a dual-ascent multiplier,
                 i.e. 'constraints by Lagrangian' instead of our 'constraints by guidance'

These are re-implementations of the core algorithms (documented in the paper's
supplement); where official code exists, results with it should be added as well.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# rewards
# --------------------------------------------------------------------------- #
def reward_linear_bitrate(rec, prev):
    br = rec["bitrate_kbps"] / 1000.0
    pb = br if prev is None else prev["bitrate_kbps"] / 1000.0
    return br - 4.3 * rec["rebuf"] - abs(br - pb)


def reward_linear_vmaf(rec, prev):
    q = rec["vmaf"] / 20.0
    pq = q if prev is None else prev["vmaf"] / 20.0
    return q - 4.3 * rec["rebuf"] - abs(q - pq)


class RankQoE(nn.Module):
    """Jade-style learned QoE: per-chunk MLP over (q, rebuf, |dq|, drop, level) summed over a session,
    trained with a pairwise (Bradley-Terry) ranking loss on session preferences."""

    def __init__(self, hidden=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @staticmethod
    def feats(q, prev_q, rebuf, level01):
        dq = q - prev_q
        return torch.stack([q / 100, rebuf.clamp(max=10) / 10, dq.abs() / 100, (-dq).clamp_min(0) / 100, level01], -1)

    def chunk(self, q, prev_q, rebuf, level01):
        return self.f(self.feats(q, prev_q, rebuf, level01)).squeeze(-1)

    def session(self, s):
        prev = torch.cat([s["q"][:, :1], s["q"][:, :-1]], 1)
        c = self.chunk(s["q"], prev, s["rebuf"], s["level01"]) * s["mask"]
        return c.sum(1) / s["mask"].sum(1).clamp_min(1)


def fit_rank_qoe(sessions_npz, steps=3000, seed=0):
    d = dict(np.load(sessions_npz, allow_pickle=False))
    ok = np.isfinite(d["target"])
    s = {k: torch.as_tensor(d[k][ok]) for k in ("q", "rebuf", "level01", "mask")}
    y = torch.as_tensor(d["target"][ok])
    m = RankQoE(); opt = torch.optim.Adam(m.parameters(), 1e-3); g = torch.Generator().manual_seed(seed)
    n = len(y)
    for _ in range(steps):
        i, j = torch.randint(0, n, (256,), generator=g), torch.randint(0, n, (256,), generator=g)
        keep = (y[i] - y[j]).abs() > 0.1
        if keep.sum() == 0:
            continue
        sc = m.session({k: v for k, v in s.items()})
        logit = sc[i[keep]] - sc[j[keep]]
        loss = F.binary_cross_entropy_with_logits(logit, (y[i[keep]] > y[j[keep]]).float())
        opt.zero_grad(); loss.backward(); opt.step()
    return m


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, L, hidden=256, mask_tail: int = 0):
        """mask_tail > 0: zero the last mask_tail observation entries (the NS-QoE energy table and l_max,
        i.e. our symbolic layer) so that the baselines do not see information from our method."""
        super().__init__()
        m = torch.ones(obs_dim)
        if mask_tail:
            m[-mask_tail:] = 0.0
        self.register_buffer("obs_mask", m)
        self.mask_tail = mask_tail
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, L)
        self.v = nn.Linear(hidden, 1)
        self.vc = nn.Linear(hidden, 1)  # cost value (Lagrangian)

    def forward(self, o):
        h = self.body(o * self.obs_mask)
        return self.pi(h), self.v(h).squeeze(-1), self.vc(h).squeeze(-1)


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    n_envs: int = 8
    rollout: int = 256
    epochs: int = 4
    minibatch: int = 512
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent: float = 0.01
    lr: float = 3e-4
    lagrangian: bool = False
    cost_limit: float = 0.02   # allowed fraction of chunks with a stall
    lr_lambda: float = 0.01
    mask_tail: int = 0          # hide the last entries of the observation (our energy table + l_max)
    val_every: int = 20_000     # evaluate on held-out validation sessions, keep the best checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _gae(r, v, d, v_last, gamma, lam):
    T = len(r); adv = np.zeros_like(r); last = 0
    for t in reversed(range(T)):
        nv = v_last if t == T - 1 else v[t + 1]
        delta = r[t] + gamma * nv * (1 - d[t]) - v[t]
        last = delta + gamma * lam * (1 - d[t]) * last
        adv[t] = last
    return adv, adv + v


def train_ppo(envs, reward_fn, cfg: PPOConfig, logger=print, val_fn=None):
    """reward_fn(rec, prev_rec, env) -> float. val_fn(net) -> score on held-out sessions (higher = better)."""
    dev = cfg.device
    L, D = envs[0].L, envs[0].obs_dim
    net = ActorCritic(D, L, mask_tail=cfg.mask_tail).to(dev)
    best = (-float("inf"), None); next_val = cfg.val_every
    opt = torch.optim.Adam(net.parameters(), cfg.lr)
    lam_mult = 0.0
    obs = [e.reset() for e in envs]; prev = [None] * len(envs)
    steps, ep_r, hist = 0, [0.0] * len(envs), []
    while steps < cfg.total_steps:
        buf = {k: [] for k in ("o", "a", "lp", "r", "c", "d", "v", "vc", "mask")}
        for _ in range(cfg.rollout):
            o = torch.as_tensor(np.stack(obs), device=dev)
            with torch.no_grad():
                logits, v, vc = net(o)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            buf["o"].append(o); buf["a"].append(a); buf["lp"].append(dist.log_prob(a)); buf["v"].append(v); buf["vc"].append(vc)
            rs, cs, ds = [], [], []
            for i, e in enumerate(envs):
                o2, _, done, rec = e.step(int(a[i]))
                r = reward_fn(rec, prev[i], e); c = float(rec["rebuf"] > 0 and rec["k"] > 0)
                rs.append(r - (lam_mult * c if cfg.lagrangian else 0.0)); cs.append(c); ds.append(float(done))
                ep_r[i] += r; prev[i] = rec
                if done:
                    hist.append(ep_r[i]); ep_r[i] = 0.0; prev[i] = None; o2 = e.reset()
                obs[i] = o2
            buf["r"].append(rs); buf["c"].append(cs); buf["d"].append(ds)
            steps += len(envs)
        with torch.no_grad():
            _, v_last, vc_last = net(torch.as_tensor(np.stack(obs), device=dev))
        R = np.array(buf["r"]); C = np.array(buf["c"]); Dn = np.array(buf["d"])
        V = torch.stack(buf["v"]).cpu().numpy(); VC = torch.stack(buf["vc"]).cpu().numpy()
        adv, ret = _gae(R, V, Dn, v_last.cpu().numpy(), cfg.gamma, cfg.lam)
        _, cret = _gae(C, VC, Dn, vc_last.cpu().numpy(), cfg.gamma, cfg.lam)
        if cfg.lagrangian:  # dual ascent on the average stall rate
            lam_mult = max(0.0, lam_mult + cfg.lr_lambda * (C.mean() - cfg.cost_limit) * 100)
        O = torch.cat(buf["o"]); A = torch.cat(buf["a"]); LP = torch.cat(buf["lp"])
        ADV = torch.as_tensor(adv.reshape(-1), device=dev, dtype=torch.float32)
        RET = torch.as_tensor(ret.reshape(-1), device=dev, dtype=torch.float32)
        CRET = torch.as_tensor(cret.reshape(-1), device=dev, dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(A)
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n, device=dev).split(cfg.minibatch):
                logits, v, vc = net(O[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ADV[idx]).mean()
                loss = pg + 0.5 * F.mse_loss(v, RET[idx]) + 0.5 * F.mse_loss(vc, CRET[idx]) - cfg.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if hist:
            logger(f"[ppo] steps {steps}  ep_return {np.mean(hist[-20:]):.3f}  lambda {lam_mult:.3f}")
        if val_fn is not None and steps >= next_val:
            next_val += cfg.val_every
            score = val_fn(net)
            logger(f"[val] steps {steps}  score {score:.3f}" + ("  (best)" if score > best[0] else ""))
            if score > best[0]:
                import copy
                best = (score, copy.deepcopy(net.state_dict()))
    if val_fn is not None:
        score = val_fn(net)
        if score > best[0]:
            best = (score, net.state_dict())
        net.load_state_dict(best[1]); net.val_score = best[0]
    return net


def validation_score(net, val_envs, reward_fn, n_sessions=48, lagrangian=False, cost_limit=0.02):
    """Greedy policy on fixed held-out sessions (validation traces, fixed seeds): mean episode return of the
    method's OWN objective. Lagrangian: the return, minus a large penalty when the stall-rate budget is exceeded."""
    e = val_envs[0]; act = ppo_act_fn(net); R, C, n = [], [], 0
    for j in range(n_sessions):
        e.rng = np.random.default_rng(10_000 + j)
        o = e.reset(j % len(e.videos), j % len(e.traces), random_start=False, playlist_seed=j)
        done, prev, ret = False, None, 0.0
        while not done:
            o, _, done, rec = e.step(act(e, o))
            ret += reward_fn(rec, prev, e); prev = rec
            C.append(float(rec["rebuf"] > 0 and rec["k"] > 0)); n += 1
        R.append(ret)
    score = float(np.mean(R))
    if lagrangian:
        score -= 1000.0 * max(0.0, float(np.mean(C)) - cost_limit)
    return score


def ppo_act_fn(net):
    def f(env, obs):
        with torch.no_grad():
            logits, _, _ = net(torch.as_tensor(obs)[None].to(next(net.parameters()).device))
        return int(logits.argmax(-1)[0])
    return f


class NeuralQoE(nn.Module):
    """Black-box QoE baseline: GRU over per-chunk features (same inputs as NS-QoE incl. concepts).
    Used in E2 to compare against NS-QoE at a similar parameter budget."""

    def __init__(self, n_concepts=5, hidden=16):
        super().__init__()
        self.gru = nn.GRU(6 + n_concepts, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def session_score(self, s):
        prev = torch.cat([s["q"][:, :1], s["q"][:, :-1]], 1)
        x = torch.cat([torch.stack([s["q"] / 100, (s["q"] - prev) / 100, s["rebuf"].clamp(max=10) / 10, s["buffer"] / 30,
                                    s["level01"], s["pos"]], -1), s["concepts"]], -1)
        h, _ = self.gru(x * s["mask"][..., None])
        n = s["mask"].sum(1).long().clamp_min(1) - 1
        return self.out(h[torch.arange(len(n)), n]).squeeze(-1)
