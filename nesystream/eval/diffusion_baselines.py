"""Two NeurIPS 2025 diffusion-RL baselines, re-implemented on our simulator (same protocol as DIPO / QSM).

GenPO  [Ding et al., "GenPO: Generative Diffusion Models Meet On-Policy Reinforcement Learning", NeurIPS 2025;
       official code github.com/wadx2019/genpo (flow_rsl_rl/rsl_rl/modules/reflow/flow.py, algorithms/flowppo.py)]
       Flow policy with a doubled ("dummy") action [z, y]: N alternating Euler updates z <- z + v(y) dt,
       y <- y + v(z) dt mixed with p, which are exactly invertible; log pi(a|s) = log N(x0) - log|det J| by the
       change of variables, so PPO can use the exact likelihood ratio. Executed action = (z + y) / 2. KL-adaptive
       learning rate, clipped value loss. The flow, its inverse and the Jacobian log-determinant follow the
       official code; hyper-parameters are those of its Ant configuration (5 flow steps, mix 0.95, lr 5e-4
       adaptive to KL 0.01, clip 0.2, 5 epochs x 4 minibatches, entropy coefficient 0, max grad norm 1).
       Discrete levels: the 1-D continuous action is mapped to a level with embed_to_level (as our diffusion
       policies); the rounding is treated as part of the environment.

VDRL   [Hu et al., "Value Diffusion Reinforcement Learning", NeurIPS 2025]
       Off-policy distributional actor-critic: the return distribution Z(.|s,a) is a conditional DDPM (T = 10)
       trained with the denoising loss on target samples T Z = r + gamma * min_{i=1,2} Z_i(s',a'),
       Z_i ~ target networks, a' ~ target policy (double value diffusion with sample selection); the policy
       maximises min_i E[Z_i(s,a)] (Monte Carlo over diffusion samples). Paper hyper-parameters: 2x256 layers,
       Mish (value) / GeLU (policy), Adam 3e-4, batch 256, tau 0.005, gamma 0.99.
       Discrete levels: the policy is categorical, so E_{a~pi}[Q(s,a)] = sum_a pi(a|s) Q(s,a) is computed exactly
       over the 13 levels (no reparameterisation through the sampler is needed); a small entropy bonus keeps the
       categorical policy from collapsing early (the paper's Gaussian policy has its own noise).

Both use the environment reward (NS-QoE fitted to MOS, as DIPO / QSM), 300k decisions, no hard rules, and the
NS-QoE energy table / l_max hidden from the observation.
"""
from __future__ import annotations

import copy, math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.func import jacrev, vmap

from nesystream.models.diffusion_policy import embed_to_level
from nesystream.eval.learned_baselines import _gae


def _obs_mask(D, mask_tail):
    m = torch.ones(D)
    if mask_tail:
        m[-mask_tail:] = 0.0
    return m


class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        e = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        e = x[:, None] * e[None]
        return torch.cat([e.sin(), e.cos()], -1)


# =========================================================================== #
# GenPO
# =========================================================================== #
class GenPOFlow(nn.Module):
    """Port of the official Flow (reflow/flow.py): doubled action, alternating mixed Euler steps, exact inverse."""

    def __init__(self, obs_dim, a_dim=1, N=5, p=0.95, hidden=(400, 200, 100), time_dim=32, time_hidden=(256, 256)):
        super().__init__()
        self.a_dim, self.N, self.dt, self.p = a_dim, N, 1.0 / N, p
        layers, d = [], a_dim + obs_dim + time_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Mish()]; d = h
        self.vec = nn.Sequential(*layers, nn.Linear(d, a_dim))
        tl, d = [SinEmb(time_dim)], time_dim
        for h in time_hidden:
            tl += [nn.Linear(d, h), nn.Mish()]; d = h
        self.time_mlp = nn.Sequential(*tl, nn.Linear(d, time_dim))

    def _v(self, x, obs, t_em):
        return self.vec(torch.cat([x, obs, t_em], -1))

    def flow(self, obs, x):                       # noise (B, 2a) -> action (B, 2a)
        squeeze = x.dim() == 1
        if squeeze:
            obs, x = obs[None], x[None]
        z, y = x[..., :self.a_dim], x[..., self.a_dim:]
        for i in range(self.N):
            t_em = self.time_mlp(torch.full((obs.shape[0],), i / self.N, device=obs.device))
            z_in = z + self._v(y, obs, t_em) * self.dt
            y_in = y + self._v(z_in, obs, t_em) * self.dt
            z = self.p * z_in + (1 - self.p) * y_in
            y = self.p * y_in + (1 - self.p) * z
        out = torch.cat([z, y], -1)
        return out[0] if squeeze else out

    def inverse_flow(self, obs, x):               # action -> noise
        squeeze = x.dim() == 1
        if squeeze:
            obs, x = obs[None], x[None]
        z, y = x[..., :self.a_dim], x[..., self.a_dim:]
        for i in reversed(range(self.N)):
            t_em = self.time_mlp(torch.full((obs.shape[0],), i / self.N, device=obs.device))
            y_in = (y - (1 - self.p) * z) / self.p
            z_in = (z - (1 - self.p) * y_in) / self.p
            y = y_in - self._v(z_in, obs, t_em) * self.dt
            z = z_in - self._v(y, obs, t_em) * self.dt
        out = torch.cat([z, y], -1)
        return out[0] if squeeze else out

    def _std(self, x):
        return Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(x).sum(-1)

    def sample(self, obs):
        x0 = torch.randn(obs.shape[0], 2 * self.a_dim, device=obs.device)
        with torch.no_grad():
            a = self.flow(obs, x0)
            J = vmap(jacrev(self.flow, argnums=1))(obs, x0)
        return a, self._std(x0) - torch.log(torch.abs(torch.linalg.det(J)) + 1e-12)

    def log_prob(self, obs, a):
        x0 = self.inverse_flow(obs, a)
        J = vmap(jacrev(self.inverse_flow, argnums=1))(obs, a)
        return self._std(x0) + torch.log(torch.abs(torch.linalg.det(J)) + 1e-12)

    def mode(self, obs):
        with torch.no_grad():
            return self.flow(obs, torch.zeros(obs.shape[0], 2 * self.a_dim, device=obs.device))


class GenPONet(nn.Module):
    def __init__(self, D, L, mask_tail=0, N=5, p=0.95):
        super().__init__()
        self.L = L
        self.register_buffer("obs_mask", _obs_mask(D, mask_tail))
        self.actor = GenPOFlow(D, 1, N, p)
        self.critic = nn.Sequential(nn.Linear(D, 400), nn.Mish(), nn.Linear(400, 200), nn.Mish(), nn.Linear(200, 100),
                                    nn.Mish(), nn.Linear(100, 1))

    def obs(self, o):
        return o * self.obs_mask

    def value(self, o):
        return self.critic(self.obs(o)).squeeze(-1)

    def to_level(self, a2):
        return embed_to_level(0.5 * (a2[..., 0] + a2[..., 1]), self.L)


@dataclass
class DBConfig:
    total_steps: int = 300_000
    n_envs: int = 8
    mask_tail: int = 0
    gamma: float = 0.99
    val_every: int = 20_000
    # GenPO (official Ant config)
    rollout: int = 256
    lam: float = 0.95
    lr: float = 5e-4
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    flow_steps: int = 5
    mix: float = 0.95
    ent: float = 0.0
    # VDRL (paper Table 3)
    T: int = 10
    batch: int = 256
    tau: float = 0.005
    v_lr: float = 3e-4
    pi_lr: float = 3e-4
    n_mc: int = 4
    z_scale: float = 10.0          # returns are divided by this inside the diffusion model
    replay: int = 500_000
    warmup: int = 5_000
    alpha: float = 0.01            # entropy bonus of the categorical policy
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_genpo(envs, cfg: DBConfig, logger=print, val_fn=None):
    dev = cfg.device; L, D, E = envs[0].L, envs[0].obs_dim, len(envs)
    net = GenPONet(D, L, cfg.mask_tail, cfg.flow_steps, cfg.mix).to(dev)
    opt = torch.optim.Adam(net.parameters(), cfg.lr); lr = cfg.lr
    obs = [e.reset() for e in envs]
    steps, ep_r, hist, best, next_val = 0, [0.0] * E, [], (-float("inf"), None), cfg.val_every
    while steps < cfg.total_steps:
        buf = {k: [] for k in ("o", "a", "lp", "v", "r", "d")}
        for _ in range(cfg.rollout):
            o = torch.as_tensor(np.stack(obs), device=dev)
            a2, lp = net.actor.sample(net.obs(o))
            with torch.no_grad():
                v = net.value(o)
            lv = net.to_level(a2)
            for k_, x in (("o", o), ("a", a2), ("lp", lp), ("v", v)):
                buf[k_].append(x)
            rs, ds = [], []
            for i, e in enumerate(envs):
                o2, r, done, _ = e.step(int(lv[i]))
                ep_r[i] += r; rs.append(r); ds.append(float(done))
                if done:
                    hist.append(ep_r[i]); ep_r[i] = 0.0; o2 = e.reset()
                obs[i] = o2
            buf["r"].append(rs); buf["d"].append(ds); steps += E
        with torch.no_grad():
            v_last = net.value(torch.as_tensor(np.stack(obs), device=dev))
        V = torch.stack(buf["v"]).cpu().numpy()
        adv, ret = _gae(np.array(buf["r"]), V, np.array(buf["d"]), v_last.cpu().numpy(), cfg.gamma, cfg.lam)
        O, A, LP = torch.cat(buf["o"]), torch.cat(buf["a"]), torch.cat(buf["lp"])
        Vo = torch.as_tensor(V.reshape(-1), device=dev, dtype=torch.float32)
        ADV = torch.as_tensor(adv.reshape(-1), device=dev, dtype=torch.float32); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        RET = torch.as_tensor(ret.reshape(-1), device=dev, dtype=torch.float32)
        n = len(A); mb = n // cfg.minibatches; kls = []
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n, device=dev).split(mb):
                lp_new = net.actor.log_prob(net.obs(O[idx]), A[idx])
                kl_hat = LP[idx] - lp_new; kls.append(float(kl_hat.mean()))
                ratio = torch.exp(-kl_hat)
                surr = torch.max(-ADV[idx] * ratio, -ADV[idx] * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v = net.value(O[idx]); v_cl = Vo[idx] + (v - Vo[idx]).clamp(-cfg.clip, cfg.clip)
                vloss = torch.max((v - RET[idx]) ** 2, (v_cl - RET[idx]) ** 2).mean()
                loss = surr + vloss - cfg.ent * (-lp_new.mean())
                # KL-adaptive learning rate (official schedule="adaptive")
                kl = float(kl_hat.mean())
                if kl > cfg.desired_kl * 2.0:
                    lr = max(1e-5, lr / 1.5)
                elif cfg.desired_kl / 2.0 > kl > -1e-4:
                    lr = min(1e-2, lr * 1.5)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm); opt.step()
        if hist:
            logger(f"[genpo] steps {steps}  ep_return {np.mean(hist[-20:]):.3f}  kl {np.mean(kls):.4f}  lr {lr:.2e}")
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


def genpo_act_fn(net):
    dev = next(net.parameters()).device

    def f(env, obs):
        o = torch.as_tensor(obs, dtype=torch.float32, device=dev)[None]
        return int(net.to_level(net.actor.mode(net.obs(o)))[0])        # deterministic: zero initial noise
    return f


# =========================================================================== #
# VDRL
# =========================================================================== #
class ValueDiffusion(nn.Module):
    """epsilon_theta(z_t, s, a, t): conditional DDPM over the scalar return (Mish, 2 x 256)."""

    def __init__(self, D, L, T=10, hidden=256, temb=16):
        super().__init__()
        self.L, self.T = L, T
        self.temb = SinEmb(temb)
        self.net = nn.Sequential(nn.Linear(1 + D + L + temb, hidden), nn.Mish(), nn.Linear(hidden, hidden), nn.Mish(),
                                 nn.Linear(hidden, 1))
        b = torch.linspace(1e-4, 0.1, T)        # linear schedule over T = 10 steps
        a = 1 - b
        self.register_buffer("beta", b); self.register_buffer("alpha", a); self.register_buffer("abar", torch.cumprod(a, 0))

    def eps(self, z_t, s, a_oh, t):
        return self.net(torch.cat([z_t[:, None], s, a_oh, self.temb(t.float())], -1)).squeeze(-1)

    def loss(self, z0, s, a_oh):
        t = torch.randint(0, self.T, (len(z0),), device=z0.device)
        e = torch.randn_like(z0); ab = self.abar[t]
        return F.mse_loss(self.eps(ab.sqrt() * z0 + (1 - ab).sqrt() * e, s, a_oh, t), e)

    @torch.no_grad()
    def sample(self, s, a_oh):
        z = torch.randn(len(s), device=s.device)
        for t in reversed(range(self.T)):
            tt = torch.full((len(s),), t, device=s.device, dtype=torch.long)
            e = self.eps(z, s, a_oh, tt)
            z = (z - self.beta[t] / (1 - self.abar[t]).sqrt() * e) / self.alpha[t].sqrt()
            if t > 0:
                z = z + self.beta[t].sqrt() * torch.randn_like(z)
        return z


class VDRLNet(nn.Module):
    def __init__(self, D, L, mask_tail=0, T=10):
        super().__init__()
        self.L = L
        self.register_buffer("obs_mask", _obs_mask(D, mask_tail))
        self.Z = nn.ModuleList([ValueDiffusion(D, L, T), ValueDiffusion(D, L, T)])
        self.pi = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, L))

    def obs(self, o):
        return o * self.obs_mask

    def logits(self, o):
        return self.pi(self.obs(o))


class _Replay:
    def __init__(self, cap, D):
        self.o = np.zeros((cap, D), np.float32); self.o2 = np.zeros((cap, D), np.float32)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, o, a, r, o2, d):
        j = self.i; self.o[j] = o; self.a[j] = a; self.r[j] = r; self.d[j] = d
        if o2 is not None:
            self.o2[j] = o2
        self.i = (j + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, b, dev):
        k = np.random.randint(self.n, size=b)
        T = lambda x: torch.as_tensor(x[k], device=dev)
        return T(self.o), T(self.a), T(self.r), T(self.o2), T(self.d)


def train_vdrl(envs, cfg: DBConfig, logger=print, val_fn=None):
    dev = cfg.device; L, D, E = envs[0].L, envs[0].obs_dim, len(envs)
    net = VDRLNet(D, L, cfg.mask_tail, cfg.T).to(dev); tgt = copy.deepcopy(net)
    opt_z = torch.optim.Adam(net.Z.parameters(), cfg.v_lr); opt_pi = torch.optim.Adam(net.pi.parameters(), cfg.pi_lr)
    rb = _Replay(cfg.replay, D); eye = torch.eye(L, device=dev)
    obs = [e.reset() for e in envs]
    steps, ep_r, hist, best, next_val, info = 0, [0.0] * E, [], (-float("inf"), None), cfg.val_every, {}
    while steps < cfg.total_steps:
        if steps < cfg.warmup:
            acts = np.random.randint(L, size=E)
        else:
            with torch.no_grad():
                acts = torch.distributions.Categorical(logits=net.logits(torch.as_tensor(np.stack(obs), device=dev))).sample().cpu().numpy()
        for i, e in enumerate(envs):
            o2, r, done, _ = e.step(int(acts[i])); ep_r[i] += r
            rb.add(obs[i], acts[i], r, None if done else o2, float(done))
            if done:
                hist.append(ep_r[i]); ep_r[i] = 0.0; o2 = e.reset()
            obs[i] = o2; steps += 1
        if steps < cfg.warmup:
            continue
        o, a, r, o2, d = rb.sample(cfg.batch, dev); s, s2 = net.obs(o), net.obs(o2)
        # ---- value distribution: T Z = r + gamma * min_i Z_i(s', a'), Z_i ~ target diffusions, a' ~ target policy
        with torch.no_grad():
            a2 = torch.distributions.Categorical(logits=tgt.logits(o2)).sample(); oh2 = eye[a2]
            z2 = torch.minimum(tgt.Z[0].sample(s2, oh2), tgt.Z[1].sample(s2, oh2)) * cfg.z_scale
            tz = (r + cfg.gamma * (1 - d) * z2) / cfg.z_scale
        lz = sum(Zi.loss(tz, s, eye[a]) for Zi in net.Z)
        opt_z.zero_grad(); lz.backward(); nn.utils.clip_grad_norm_(net.Z.parameters(), 1.0); opt_z.step()
        # ---- policy: maximise sum_a pi(a|s) min_i E[Z_i(s,a)] (Monte Carlo over n_mc diffusion samples)
        with torch.no_grad():
            B = len(s); sr = s.repeat_interleave(L * cfg.n_mc, 0)
            ar = eye.repeat(B, 1).repeat_interleave(cfg.n_mc, 0)
            q = torch.stack([tgt.Z[k].sample(sr, ar).view(B, L, cfg.n_mc).mean(-1) for k in range(2)]).min(0).values * cfg.z_scale
        lg = net.logits(o); p = F.softmax(lg, -1)
        lpi = -(p * q).sum(-1).mean() - cfg.alpha * torch.distributions.Categorical(logits=lg).entropy().mean()
        opt_pi.zero_grad(); lpi.backward(); opt_pi.step()
        with torch.no_grad():
            for pt, pp in zip(tgt.parameters(), net.parameters()):
                pt.mul_(1 - cfg.tau).add_(cfg.tau * pp)
        info = {"loss_z": float(lz), "q_mean": float(q.mean())}
        if steps % 2000 < E and hist:
            logger(f"[vdrl] steps {steps}  ep_return {np.mean(hist[-20:]):.3f}  " + " ".join(f"{k} {v:.3f}" for k, v in info.items()))
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


def vdrl_act_fn(net):
    dev = next(net.parameters()).device

    def f(env, obs):
        with torch.no_grad():
            return int(net.logits(torch.as_tensor(obs, dtype=torch.float32, device=dev)[None]).argmax(-1)[0])
    return f


def env_reward_validation(act, val_envs, n_sessions=48):
    """Validation on held-out training traces with the environment reward (NS-QoE), as for DIPO / QSM."""
    e = val_envs[0]; R = []
    for j in range(n_sessions):
        e.rng = np.random.default_rng(10_000 + j)
        o = e.reset(j % len(e.videos), j % len(e.traces), random_start=False, playlist_seed=j)
        done, ret = False, 0.0
        while not done:
            o, r, done, _ = e.step(act(e, o)); ret += r
        R.append(ret)
    return float(np.mean(R))


def load_diffusion_baseline(path, map_location="cpu"):
    b = torch.load(path, map_location=map_location, weights_only=False)
    if b["kind"] == "genpo":
        net = GenPONet(b["obs_dim"], b["L"], b["mask_tail"], b["flow_steps"], b["mix"]); net.load_state_dict(b["state"]); net.eval()
        return "GenPO", genpo_act_fn(net), b["quality_source"], b["log_tput"]
    net = VDRLNet(b["obs_dim"], b["L"], b["mask_tail"], b["T"]); net.load_state_dict(b["state"]); net.eval()
    return "VDRL", vdrl_act_fn(net), b["quality_source"], b["log_tput"]
