"""Two 2026 learning-based ABR baselines, re-implemented on our simulator, state and data.

EAStream  [Huang, Xiao, Chen et al., "EAStream: An Environment-Aware Adaptive Bitrate Algorithm for Reliable
          Video Streaming Services", IEEE Trans. Services Computing 19(2), 2026]
          Context-based meta-RL (BAMDP): a GRU encoder q_phi(m | tau_{:t}) infers a Gaussian belief m over the
          network environment from (a_{t-1}, r_t, s_t); a decoder reconstructs next states and rewards of the whole
          trajectory from m (ELBO = lambda_s L_state + lambda_r L_reward + beta KL). The PPO policy pi(a | s, m) uses
          the belief; policy gradients do not reach the encoder, which is trained from a separate trajectory replay
          buffer. Entropy weight adapted to a target entropy. No gradient update at test time.
          Paper hyper-parameters: m dim 16, beta 0.1, lambda_s = lambda_r = 1, policy lr 1e-4, belief lr 1e-3,
          replay 1000 trajectories, batch 32, clip 0.2, gamma 0.99, target entropy 0.1.

NeuroBA   [Chen, Xiao, Liu et al., "NeuroBA: Neuro-Symbolic Bitrate Adaptation for IRS-Aided Mobile Video
          Streaming", IEEE/ACM Trans. Networking 34, 2026]
          Neuro-symbolic DRL: observations are grounded into first-order predicates; a Logical Neural Network
          (weighted real-valued AND/OR neurons) over the predicates produces the truth of Select(level), which is the
          policy; trained with a clipped policy-gradient objective (PPO-style) and prioritized experience replay.
          Adaptation to our setting: single client, no IRS (phase-shift action and channel-gain state removed;
          throughput comes from the trace); predicates are grounded on the same observation as PPO-VMAF (PQF
          quality, chunk sizes, throughput / download history, buffer). With fully observed inputs the LNN lower and
          upper bounds coincide, so each neuron carries one truth value.

Both: discrete levels, linear VMAF QoE reward (as PPO-VMAF; the papers' QoE families with VMAF as quality), the
same videos, traces, throughput augmentation, budget (200k decisions), validation-based checkpoint selection and
hidden NS-QoE energy table as the other learned baselines.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nesystream.eval.learned_baselines import _gae


@dataclass
class NBConfig:
    total_steps: int = 200_000
    n_envs: int = 8
    rollout: int = 256
    epochs: int = 4
    minibatch: int = 512
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 1e-4
    ent: float = 0.01            # NeuroBA: fixed entropy coefficient; EAStream: initial value (adapted)
    mask_tail: int = 0
    val_every: int = 20_000
    # EAStream
    m_dim: int = 16
    beta: float = 0.1
    lam_s: float = 1.0
    lam_r: float = 1.0
    belief_lr: float = 1e-3
    replay: int = 1000
    belief_batch: int = 32
    belief_updates: int = 4
    target_entropy: float = 0.1
    ent_lr: float = 0.01
    # NeuroBA
    n_rules: int = 8
    n_thr: int = 3
    per_alpha: float = 0.6
    per_beta: float = 0.4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _layout(env):
    h, L = env.cfg.hist, env.L
    o = 2 * h + 4
    return dict(h=h, L=L, thr=slice(0, h), dl=slice(h, 2 * h), buf=2 * h, prev=2 * h + 1, prevq=2 * h + 2,
                rem=2 * h + 3, sizes=slice(o, o + L), q=slice(o + L, o + 2 * L), log_tput=bool(env.cfg.log_tput))


def _obs_mask(D, mask_tail):
    m = torch.ones(D)
    if mask_tail:
        m[-mask_tail:] = 0.0
    return m


# =========================================================================== #
# EAStream
# =========================================================================== #
class EAStreamNet(nn.Module):
    def __init__(self, D, L, mask_tail=0, m_dim=16, hid=64, body=256):
        super().__init__()
        self.D, self.L, self.m_dim = D, L, m_dim
        self.register_buffer("obs_mask", _obs_mask(D, mask_tail))
        self.gru = nn.GRUCell(D + L + 1, hid)
        self.mu, self.lv = nn.Linear(hid, m_dim), nn.Linear(hid, m_dim)
        self.body = nn.Sequential(nn.Linear(D + m_dim, body), nn.ReLU(), nn.Linear(body, body), nn.ReLU())
        self.pi, self.v = nn.Linear(body, L), nn.Linear(body, 1)
        dec_in = D + L + m_dim
        self.dec_s = nn.Sequential(nn.Linear(dec_in, 128), nn.ReLU(), nn.Linear(128, D))
        self.dec_r = nn.Sequential(nn.Linear(dec_in, 128), nn.ReLU(), nn.Linear(128, 1))
        self.hid = hid

    def enc_in(self, o, a_prev, r_prev):
        oh = F.one_hot(a_prev.clamp_min(0), self.L).float() * (a_prev >= 0).float()[:, None]
        return torch.cat([o * self.obs_mask, oh, r_prev[:, None]], -1)

    def belief(self, h):
        return self.mu(h), self.lv(h).clamp(-8, 4)

    def policy(self, o, m):
        x = self.body(torch.cat([o * self.obs_mask, m], -1))
        return self.pi(x), self.v(x).squeeze(-1)

    def policy_params(self):
        return list(self.body.parameters()) + list(self.pi.parameters()) + list(self.v.parameters())

    def belief_params(self):
        return [p for n, p in self.named_parameters() if n.split(".")[0] in ("gru", "mu", "lv", "dec_s", "dec_r")]


def _belief_loss(net, batch, cfg):
    """VariBAD-style ELBO: from every prefix belief m_t, decode all transitions (s_j, a_j) -> (s_{j+1}, r_j)."""
    dev = cfg.device
    T = max(len(tr["a"]) for tr in batch); B = len(batch); D, L = net.D, net.L
    O = torch.zeros(B, T + 1, D, device=dev); A = torch.full((B, T), -1, device=dev, dtype=torch.long)
    R = torch.zeros(B, T, device=dev); M = torch.zeros(B, T, device=dev)
    for i, tr in enumerate(batch):
        n = len(tr["a"])
        O[i, :n + 1] = torch.as_tensor(np.stack(tr["o"]), device=dev)
        A[i, :n] = torch.as_tensor(tr["a"], device=dev); R[i, :n] = torch.as_tensor(tr["r"], device=dev, dtype=torch.float32)
        M[i, :n] = 1
    h = torch.zeros(B, net.hid, device=dev); mus, lvs = [], []
    a_prev = torch.full((B,), -1, device=dev, dtype=torch.long); r_prev = torch.zeros(B, device=dev)
    for t in range(T):
        h = net.gru(net.enc_in(O[:, t], a_prev, r_prev), h)
        mu, lv = net.belief(h); mus.append(mu); lvs.append(lv)
        a_prev, r_prev = A[:, t], R[:, t]
    MU, LV = torch.stack(mus, 1), torch.stack(lvs, 1)                     # (B, T, m)
    m = MU + torch.randn_like(MU) * torch.exp(0.5 * LV)
    # decode every transition j of the trajectory from every belief m_t
    s_in = (O[:, :T] * net.obs_mask)[:, None].expand(B, T, T, D)
    a_in = F.one_hot(A.clamp_min(0), L).float()[:, None].expand(B, T, T, L)
    m_in = m[:, :, None].expand(B, T, T, net.m_dim)
    x = torch.cat([s_in, a_in, m_in], -1)
    s_hat, r_hat = net.dec_s(x), net.dec_r(x).squeeze(-1)
    s_tgt = (O[:, 1:] * net.obs_mask)[:, None].expand(B, T, T, D)
    pair = M[:, :, None] * M[:, None, :]                                    # valid (t, j)
    L_state = (((s_hat - s_tgt) ** 2).sum(-1) * pair).sum() / pair.sum()
    L_rew = (((r_hat - R[:, None].expand(B, T, T)) ** 2) * pair).sum() / pair.sum()
    kl = (-0.5 * (1 + LV - MU ** 2 - LV.exp()).sum(-1) * M).sum() / M.sum()
    return cfg.lam_s * L_state + cfg.lam_r * L_rew + cfg.beta * kl, (float(L_state), float(L_rew), float(kl))


def train_eastream(envs, reward_fn, cfg: NBConfig, logger=print, val_fn=None):
    dev = cfg.device; L, D, E = envs[0].L, envs[0].obs_dim, len(envs)
    net = EAStreamNet(D, L, cfg.mask_tail, cfg.m_dim).to(dev)
    opt_pi = torch.optim.Adam(net.policy_params(), cfg.lr)
    opt_b = torch.optim.Adam(net.belief_params(), cfg.belief_lr)
    ent_w = cfg.ent; replay = []
    obs = [e.reset() for e in envs]; prev = [None] * E
    h = torch.zeros(E, net.hid, device=dev); a_prev = torch.full((E,), -1, device=dev, dtype=torch.long)
    r_prev = torch.zeros(E, device=dev)
    cur = [{"o": [o], "a": [], "r": []} for o in obs]
    steps, ep_r, hist, best, next_val = 0, [0.0] * E, [], (-float("inf"), None), cfg.val_every
    while steps < cfg.total_steps:
        buf = {k: [] for k in ("o", "m", "a", "lp", "v", "r", "d")}
        for _ in range(cfg.rollout):
            o = torch.as_tensor(np.stack(obs), device=dev)
            with torch.no_grad():
                h = net.gru(net.enc_in(o, a_prev, r_prev), h)
                mu, lv = net.belief(h); m = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                logits, v = net.policy(o, m)
            dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
            for k_, x in (("o", o), ("m", m), ("a", a), ("lp", dist.log_prob(a)), ("v", v)):
                buf[k_].append(x)
            rs, ds = [], []
            for i, e in enumerate(envs):
                o2, _, done, rec = e.step(int(a[i]))
                r = reward_fn(rec, prev[i], e); prev[i] = rec; ep_r[i] += r
                cur[i]["a"].append(int(a[i])); cur[i]["r"].append(r)
                rs.append(r); ds.append(float(done))
                if done:
                    hist.append(ep_r[i]); ep_r[i] = 0.0; prev[i] = None
                    cur[i]["o"].append(np.zeros(D, np.float32))      # terminal: no next observation
                    replay.append(cur[i]); replay = replay[-cfg.replay:]
                    o2 = e.reset(); cur[i] = {"o": [o2], "a": [], "r": []}
                else:
                    cur[i]["o"].append(o2)
                obs[i] = o2
            d_t = torch.as_tensor(ds, device=dev)
            h = h * (1 - d_t)[:, None]                                       # new session: fresh belief
            a_prev = torch.where(d_t > 0, torch.full_like(a, -1), a)
            r_prev = torch.as_tensor(rs, device=dev, dtype=torch.float32) * (1 - d_t)
            buf["r"].append(rs); buf["d"].append(ds); steps += E
        # ---- policy update (PPO), belief fixed
        with torch.no_grad():
            o = torch.as_tensor(np.stack(obs), device=dev)
            hh = net.gru(net.enc_in(o, a_prev, r_prev), h); mu, _ = net.belief(hh)
            _, v_last = net.policy(o, mu)
        Rr, Dn = np.array(buf["r"]), np.array(buf["d"]); V = torch.stack(buf["v"]).cpu().numpy()
        adv, ret = _gae(Rr, V, Dn, v_last.cpu().numpy(), cfg.gamma, cfg.lam)
        O, Mb, A, LP = torch.cat(buf["o"]), torch.cat(buf["m"]), torch.cat(buf["a"]), torch.cat(buf["lp"])
        ADV = torch.as_tensor(adv.reshape(-1), device=dev, dtype=torch.float32); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        RET = torch.as_tensor(ret.reshape(-1), device=dev, dtype=torch.float32); ents = []
        for _ in range(cfg.epochs):
            for idx in torch.randperm(len(A), device=dev).split(cfg.minibatch):
                logits, v = net.policy(O[idx], Mb[idx]); dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx]); ent = dist.entropy().mean(); ents.append(float(ent))
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ADV[idx]).mean()
                loss = pg + 0.5 * F.mse_loss(v, RET[idx]) - ent_w * ent
                opt_pi.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.policy_params(), 0.5); opt_pi.step()
        ent_w = float(np.clip(ent_w - cfg.ent_lr * (np.mean(ents) - cfg.target_entropy), 0.0, 1.0))   # automatic entropy weight
        # ---- belief update (ELBO) from the trajectory replay buffer
        lb = None
        if len(replay) >= cfg.belief_batch:
            for _ in range(cfg.belief_updates):
                idx = np.random.randint(len(replay), size=cfg.belief_batch)
                loss_b, lb = _belief_loss(net, [replay[i] for i in idx], cfg)
                opt_b.zero_grad(); loss_b.backward(); nn.utils.clip_grad_norm_(net.belief_params(), 1.0); opt_b.step()
        if hist:
            logger(f"[eastream] steps {steps}  ep_return {np.mean(hist[-20:]):.3f}  ent {np.mean(ents):.3f} ent_w {ent_w:.4f}"
                   + (f"  belief(s,r,kl) {lb[0]:.3f},{lb[1]:.3f},{lb[2]:.2f}" if lb else ""))
        if val_fn is not None and steps >= next_val:
            next_val += cfg.val_every; score = val_fn(net)
            logger(f"[val] steps {steps}  score {score:.3f}" + ("  (best)" if score > best[0] else ""))
            if score > best[0]:
                best = (score, copy.deepcopy(net.state_dict()))
    if val_fn is not None:
        score = val_fn(net)
        if score > best[0]:
            best = (score, copy.deepcopy(net.state_dict()))
        net.load_state_dict(best[1]); net.val_score = best[0]
    return net


def eastream_act_fn(net, reward_fn):
    """Stateful greedy policy: the belief is updated online from (a_{t-1}, r_t, s_t); reset at a new session."""
    dev = next(net.parameters()).device; st = {}

    def f(env, obs):
        key = id(env)
        if env.k == 0 or key not in st:
            st[key] = {"h": torch.zeros(1, net.hid, device=dev), "a": -1, "r": 0.0}
        s = st[key]
        if env.k > 0 and env.log:
            s["r"] = float(reward_fn(env.log[-1], env.log[-2] if len(env.log) > 1 else None, env))
            s["a"] = int(env.log[-1]["level"])
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=dev)[None]
            s["h"] = net.gru(net.enc_in(o, torch.tensor([s["a"]], device=dev), torch.tensor([s["r"]], device=dev)), s["h"])
            mu, _ = net.belief(s["h"])
            logits, _ = net.policy(o, mu)
        return int(logits.argmax(-1)[0])
    return f


# =========================================================================== #
# NeuroBA
# =========================================================================== #
def _soft_clamp(x):
    """LNN truth values live in [0, 1]; a small leak keeps gradients alive outside (straight-through style)."""
    c = x.clamp(0.0, 1.0)
    return c + 0.01 * (x - c)


class LNNAnd(nn.Module):
    """Weighted real-valued (Lukasiewicz) conjunction of LNN: y = clamp(beta - sum_i w_i (1 - x_i)), w >= 0.
    Every rule may use each predicate or its negation (learned, sparse)."""

    def __init__(self, n_in, n_rules):
        super().__init__()
        # small initial literal weights so that the conjunction starts unsaturated (sum_i w_i (1 - x_i) ~ 0.2-0.3)
        # weights are kept non-negative by projection after each step (as in LNN training), not by a softplus,
        # whose gradient vanishes for the small initial weights
        self.w = nn.Parameter((0.5 / (2 * n_in)) * (1 + 0.5 * torch.randn(n_rules, 2 * n_in)).clamp_min(0.05))
        self.beta = nn.Parameter(torch.ones(n_rules))

    def forward(self, u):                                       # u: (..., n_in) truths
        x = torch.cat([u, 1 - u], -1)                           # literals: p and NOT p
        return _soft_clamp(self.beta - torch.einsum("...i,ri->...r", 1 - x, self.w))


class LNNOr(nn.Module):
    """Weighted real-valued disjunction: y = clamp(1 - beta + sum_r w_r y_r), w >= 0."""

    def __init__(self, n_rules):
        super().__init__()
        self.w = nn.Parameter(torch.full((n_rules,), 1.0 / n_rules)); self.beta = nn.Parameter(torch.ones(1))

    def forward(self, y):
        return _soft_clamp(1 - self.beta + (y * self.w).sum(-1))


class NeuroBANet(nn.Module):
    """Predicate grounding -> LNN rules Select(l) <- AND(literals) ; OR over rules -> policy logits.
    Per-level features: quality q_l, change vs. previous chunk, size, estimated download time, buffer slack, level.
    Global features: buffer, throughput estimate, throughput variability, last download time, remaining chunks,
    previous level. Each feature x is grounded into n_thr learned threshold predicates sigma(s_k (x - theta_k))."""

    N_ACT, N_GLB = 6, 6

    def __init__(self, D, L, lay, mask_tail=0, n_rules=8, n_thr=3, body=256):
        super().__init__()
        self.D, self.L, self.lay = D, L, lay
        self.register_buffer("obs_mask", _obs_mask(D, mask_tail))
        nf = self.N_ACT + self.N_GLB
        self.theta = nn.Parameter(torch.linspace(-1, 1, n_thr).repeat(nf, 1) + 0.1 * torch.randn(nf, n_thr))
        self.raw_s = nn.Parameter(torch.full((nf, n_thr), 1.0))
        n_pred = nf * n_thr
        self.and_ = LNNAnd(n_pred, n_rules); self.or_ = LNNOr(n_rules)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(50.0)))); self.bias = nn.Parameter(torch.zeros(L))
        self.critic = nn.Sequential(nn.Linear(D, body), nn.ReLU(), nn.Linear(body, body), nn.ReLU(), nn.Linear(body, 1))
        self.register_buffer("f_mu", torch.zeros(nf)); self.register_buffer("f_sd", torch.ones(nf))

    def features(self, o):
        y = self.lay; B = o.shape[0]; L = self.L
        thr = o[:, y["thr"]]
        mbps = torch.expm1(thr * np.log1p(1000.0)) if y["log_tput"] else thr * 10.0
        last5 = mbps[:, -5:].clamp_min(0.05)          # no history yet (session start): 50 kbps floor
        c_hat = 5.0 / (1.0 / last5).sum(-1)                                    # harmonic mean (Mbps)
        var = mbps[:, -5:].std(-1) / mbps[:, -5:].mean(-1).clamp_min(1e-3)
        buf = (o[:, y["buf"]] * 10.0).clamp_min(0); dl_last = o[:, y["dl"]][:, -1] * 10.0
        prev = o[:, y["prev"]]; prevq = o[:, y["prevq"]]; rem = o[:, y["rem"]]
        q = o[:, y["q"]]; size_mb = o[:, y["sizes"]]
        t_dl = (size_mb * 8.0 / c_hat[:, None]).clamp(max=60.0)
        lvl = torch.linspace(0, 1, L, device=o.device).expand(B, L)
        fa = torch.stack([q, q - prevq[:, None], torch.log1p(size_mb), torch.log1p(t_dl), ((buf[:, None] - t_dl) / 10.0).clamp(-3, 3), lvl], -1)
        fg = torch.stack([buf / 10.0, torch.log1p(c_hat), var, torch.log1p(dl_last.clamp(0, 60)), rem, prev], -1)
        return torch.cat([fa, fg[:, None].expand(B, L, self.N_GLB)], -1)   # (B, L, nf)

    def calibrate(self, obs_batch):
        """Normalise features once from initial rollouts so that thresholds start in range."""
        with torch.no_grad():
            f = self.features(obs_batch).reshape(-1, self.f_mu.numel())
            self.f_mu.copy_(f.mean(0)); self.f_sd.copy_(f.std(0).clamp_min(1e-3))

    def predicates(self, o):
        f = (self.features(o) - self.f_mu) / self.f_sd                       # (B, L, nf)
        p = torch.sigmoid(F.softplus(self.raw_s) * 4.0 * (f[..., None] - self.theta))   # (B, L, nf, n_thr)
        return p.flatten(-2)

    def forward(self, o):
        sel = self.or_(self.and_(self.predicates(o)))                         # truth of Select(l): (B, L)
        return self.log_temp.exp() * sel + self.bias, self.critic(o * self.obs_mask).squeeze(-1)

    def rules(self, names=None, top=4):
        """Readable rules: the strongest literals of each AND neuron."""
        w = self.and_.w.detach().cpu().numpy(); wo = self.or_.w.detach().cpu().numpy()
        n = w.shape[1] // 2; out = []
        for r in np.argsort(-wo):
            lits = [(("" if j < n else "NOT ") + (names[j % n] if names else f"p{j % n}"), w[r, j]) for j in np.argsort(-w[r])[:top]]
            out.append((float(wo[r]), lits))
        return out


def train_neuroba(envs, reward_fn, cfg: NBConfig, logger=print, val_fn=None):
    dev = cfg.device; L, D, E = envs[0].L, envs[0].obs_dim, len(envs)
    net = NeuroBANet(D, L, _layout(envs[0]), cfg.mask_tail, cfg.n_rules, cfg.n_thr).to(dev)
    obs = [e.reset() for e in envs]
    # feature normalisation from a short random-policy warm-up
    warm = []
    for _ in range(64):
        for i, e in enumerate(envs):
            o2, _, done, _ = e.step(int(np.random.randint(L))); warm.append(obs[i] if o2 is None else o2)
            obs[i] = e.reset() if done else o2
    net.calibrate(torch.as_tensor(np.stack(warm), device=dev))
    opt = torch.optim.Adam(net.parameters(), cfg.lr)
    prev = [None] * E; steps, ep_r, hist, best, next_val = 0, [0.0] * E, [], (-float("inf"), None), cfg.val_every
    while steps < cfg.total_steps:
        buf = {k: [] for k in ("o", "a", "lp", "v", "r", "d")}
        for _ in range(cfg.rollout):
            o = torch.as_tensor(np.stack(obs), device=dev)
            with torch.no_grad():
                logits, v = net(o)
            dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
            for k_, x in (("o", o), ("a", a), ("lp", dist.log_prob(a)), ("v", v)):
                buf[k_].append(x)
            rs, ds = [], []
            for i, e in enumerate(envs):
                o2, _, done, rec = e.step(int(a[i]))
                r = reward_fn(rec, prev[i], e); prev[i] = rec; ep_r[i] += r
                rs.append(r); ds.append(float(done))
                if done:
                    hist.append(ep_r[i]); ep_r[i] = 0.0; prev[i] = None; o2 = e.reset()
                obs[i] = o2
            buf["r"].append(rs); buf["d"].append(ds); steps += E
        with torch.no_grad():
            _, v_last = net(torch.as_tensor(np.stack(obs), device=dev))
        Rr, Dn = np.array(buf["r"]), np.array(buf["d"]); V = torch.stack(buf["v"]).cpu().numpy()
        adv, ret = _gae(Rr, V, Dn, v_last.cpu().numpy(), cfg.gamma, cfg.lam)
        O, A, LP = torch.cat(buf["o"]), torch.cat(buf["a"]), torch.cat(buf["lp"])
        ADV = torch.as_tensor(adv.reshape(-1), device=dev, dtype=torch.float32)
        RET = torch.as_tensor(ret.reshape(-1), device=dev, dtype=torch.float32)
        TD = (RET - torch.as_tensor(V.reshape(-1), device=dev, dtype=torch.float32)).abs()
        ADVn = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        # prioritized experience replay over the collected samples: P(i) ~ (|TD error| + eps)^alpha, IS weights
        n = len(A); pr = (TD + 1e-3) ** cfg.per_alpha; P = pr / pr.sum()
        isw = (n * P) ** (-cfg.per_beta); isw = isw / isw.max()
        for _ in range(cfg.epochs):
            for _mb in range(max(1, n // cfg.minibatch)):
                idx = torch.multinomial(P, cfg.minibatch, replacement=True)
                logits, v = net(O[idx]); dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                pg = -(isw[idx] * torch.min(ratio * ADVn[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ADVn[idx])).mean()
                vl = (isw[idx] * (v - RET[idx]) ** 2).mean()
                loss = pg + 0.5 * vl - cfg.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    net.and_.beta.clamp_(0.5, 3.0); net.or_.beta.clamp_(0.5, 3.0); net.and_.w.clamp_(min=0.0); net.or_.w.clamp_(min=0.0); net.log_temp.clamp_(np.log(0.5), np.log(500.0))
        if hist:
            logger(f"[neuroba] steps {steps}  ep_return {np.mean(hist[-20:]):.3f}  temp {float(net.log_temp.exp()):.2f}")
        if val_fn is not None and steps >= next_val:
            next_val += cfg.val_every; score = val_fn(net)
            logger(f"[val] steps {steps}  score {score:.3f}" + ("  (best)" if score > best[0] else ""))
            if score > best[0]:
                best = (score, copy.deepcopy(net.state_dict()))
    if val_fn is not None:
        score = val_fn(net)
        if score > best[0]:
            best = (score, copy.deepcopy(net.state_dict()))
        net.load_state_dict(best[1]); net.val_score = best[0]
    return net


def neuroba_act_fn(net):
    dev = next(net.parameters()).device

    def f(env, obs):
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(obs, dtype=torch.float32, device=dev)[None])
        return int(logits.argmax(-1)[0])
    return f


# =========================================================================== #
def validation_score_fn(act, val_envs, reward_fn, n_sessions=48):
    """Same protocol as learned_baselines.validation_score, for any act(env, obs)."""
    e = val_envs[0]; R = []
    for j in range(n_sessions):
        e.rng = np.random.default_rng(10_000 + j)
        o = e.reset(j % len(e.videos), j % len(e.traces), random_start=False, playlist_seed=j)
        done, prev, ret = False, None, 0.0
        while not done:
            o, _, done, rec = e.step(act(e, o))
            ret += reward_fn(rec, prev, e); prev = rec
        R.append(ret)
    return float(np.mean(R))


def load_new_baseline(path, map_location="cpu"):
    """-> (name, act_fn, quality_source, log_tput) from a checkpoint written by scripts/train_new_baselines.py."""
    from nesystream.eval.learned_baselines import reward_linear_vmaf
    b = torch.load(path, map_location=map_location, weights_only=False)
    if b["kind"] == "eastream":
        net = EAStreamNet(b["obs_dim"], b["L"], b["mask_tail"], b["m_dim"]); net.load_state_dict(b["state"]); net.eval()
        rf = lambda r, p, e: reward_linear_vmaf(r, p)
        return "EAStream", eastream_act_fn(net, rf), b["quality_source"], b["log_tput"]
    net = NeuroBANet(b["obs_dim"], b["L"], b["layout"], b["mask_tail"], b["n_rules"], b["n_thr"])
    net.load_state_dict(b["state"]); net.eval()
    return "NeuroBA", neuroba_act_fn(net), b["quality_source"], b["log_tput"]
