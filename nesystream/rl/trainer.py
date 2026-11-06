"""Online training of the logic-guided diffusion policy with a Q-ensemble critic,
and consistency distillation of the trained (guided) teacher."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..models.diffusion_policy import (DiffusionPolicy, QEnsemble, ConsistencyStudent, level_embed,
                                       energy_of_embedding,
                                       consistency_distill_loss)


@dataclass
class TrainConfig:
    total_steps: int = 200_000
    n_envs: int = 8
    warmup: int = 5_000
    batch: int = 256
    gamma: float = 0.99
    lr_pi: float = 3e-4
    lr_q: float = 3e-4
    n_q: int = 10            # ensemble size
    n_q_target: int = 2      # random subset for the clipped target (REDQ-style)
    tau: float = 0.005
    T: int = 20              # diffusion steps of the teacher
    sample_steps: int = 5    # steps used while collecting data
    pi_samples: int = 8      # samples to estimate pi(l|s)
    ent: float = 0.1         # uniform floor (entropy regularisation)
    eta_guid: float = 1.0    # soft-rule guidance strength
    eps_explore: float = 0.05
    horizon: int = 1
    replay: int = 500_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 2_000
    policy: str = "diffusion"  # 'diffusion' | 'categorical' (ablation: same critic, AW cross-entropy, exact tilt)
    actor_update: str = "aw"   # 'aw' (ours) | 'dipo' (Yang et al. 2023) | 'qsm' (Psenka et al. ICML'24)
    dipo_lr: float = 0.05       # action-gradient step for DIPO
    qsm_alpha: float = 0.1      # score scale for QSM
    warmup_policy: str = "random"  # 'random' | 'mpc': warm-up data from RobustMPC(3, quality), clipped to l_max


class Replay:
    def __init__(self, n, obs_dim, L):
        self.o = np.zeros((n, obs_dim), np.float32); self.o2 = np.zeros_like(self.o)
        self.a = np.zeros(n, np.int64); self.r = np.zeros(n, np.float32); self.d = np.zeros(n, np.float32)
        self.e = np.zeros((n, L), np.float32); self.e2 = np.zeros_like(self.e)
        self.m = np.zeros(n, np.int64); self.m2 = np.zeros(n, np.int64)
        self.n, self.i, self.full = n, 0, False

    def add(self, o, a, r, o2, d, e, m, e2, m2):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.d[i], self.e[i], self.m[i] = o, a, r, d, e, m
        if o2 is not None:
            self.o2[i], self.e2[i], self.m2[i] = o2, e2, m2
        self.i = (i + 1) % self.n
        self.full |= self.i == 0

    def sample(self, b, dev):
        idx = np.random.randint(0, self.n if self.full else self.i, b)
        f = lambda x: torch.as_tensor(x[idx], device=dev)
        return f(self.o), f(self.a), f(self.r), f(self.o2), f(self.d), f(self.e), f(self.m), f(self.e2), f(self.m2)


class CategoricalPolicy(torch.nn.Module):
    """Ablation: softmax policy over levels. Guidance = exact tilt logits - eta * E; infeasible levels masked."""

    def __init__(self, obs_dim, L, hidden=256):
        super().__init__()
        self.n_levels, self.H, self.T = L, 1, 1
        self.net = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden), torch.nn.Mish(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.Mish(), torch.nn.Linear(hidden, L))
        self.register_buffer("alpha_bar", torch.ones(1))

    def logits(self, obs, energy_table=None, eta_guid=0.0, l_max=None):
        lg = self.net(obs)
        if energy_table is not None and eta_guid > 0:
            lg = lg - eta_guid * energy_table
        if l_max is not None:
            lg = lg.masked_fill(torch.arange(self.n_levels, device=obs.device)[None] > l_max[:, None], -1e9)
        return lg

    def sample(self, obs, n_steps=None, energy_table=None, eta_guid=0.0, l_max=None, stochastic=1.0,
               return_continuous=False):
        lv = torch.distributions.Categorical(logits=self.logits(obs, energy_table, eta_guid, l_max)).sample()
        return (lv, level_embed(lv, self.n_levels)[:, None]) if return_continuous else lv

    def loss(self, obs, cand_actions, weights):
        return -(weights * torch.log_softmax(self.net(obs), -1)).sum(1).mean()


class Agent:
    def __init__(self, obs_dim, L, cfg: TrainConfig):
        self.cfg, self.L = cfg, L
        dev = cfg.device
        self.pi = (DiffusionPolicy(obs_dim, L, cfg.horizon, cfg.T) if cfg.policy == "diffusion"
                   else CategoricalPolicy(obs_dim, L)).to(dev)
        self.q = QEnsemble(obs_dim, L, cfg.n_q).to(dev)
        self.q_t = copy.deepcopy(self.q)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), cfg.lr_pi)
        self.opt_q = torch.optim.Adam(self.q.parameters(), cfg.lr_q)

    @torch.no_grad()
    def act(self, obs, energy, l_max, n_steps=None, guided=True, explore=0.0):
        dev = self.cfg.device
        o = torch.as_tensor(obs, device=dev)
        e = torch.as_tensor(energy, device=dev)
        m = torch.as_tensor(l_max, device=dev)
        lv = self.pi.sample(o, n_steps or self.cfg.sample_steps, e if guided else None,
                            self.cfg.eta_guid if guided else 0.0, m)
        if explore > 0:
            rnd = torch.rand(len(lv), device=dev) < explore
            uni = (torch.rand(len(lv), device=dev) * (m + 1).float()).long()
            lv = torch.where(rnd, uni, lv)
        return lv.cpu().numpy()

    def _pi_probs(self, o, e, m):
        """Monte-Carlo estimate of pi(l|s) from the guided, projected sampler."""
        K = self.cfg.pi_samples
        B = o.shape[0]
        rep = lambda x: x.repeat_interleave(K, 0)
        lv = self.pi.sample(rep(o), self.cfg.sample_steps, rep(e), self.cfg.eta_guid, rep(m)).view(B, K)
        return torch.nn.functional.one_hot(lv, self.L).float().mean(1)

    def update(self, rb: Replay):
        c, dev = self.cfg, self.cfg.device
        o, a, r, o2, d, e, m, e2, m2 = rb.sample(c.batch, dev)
        # ---------------- critic ----------------
        with torch.no_grad():
            p2 = self._pi_probs(o2, e2, m2)
            idx = torch.randperm(c.n_q, device=dev)[: c.n_q_target]
            q2 = self.q_t(o2)[idx].min(0).values           # (B, L) clipped over a random subset
            y = r + c.gamma * (1 - d) * (p2 * q2).sum(1)
        q_all = self.q(o)                                  # (n_q, B, L)
        q_a = q_all.gather(2, a[None, :, None].expand(c.n_q, -1, 1)).squeeze(2)
        loss_q = ((q_a - y[None]) ** 2).mean()
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()
        # ---------------- actor: advantage-weighted denoising over all levels ----------------
        with torch.no_grad():
            qm = self.q(o).mean(0)                         # (B, L)
            p = self._pi_probs(o, e, m)
            V = (p * qm).sum(1, keepdim=True)
            adv = (qm - V).clamp_min(0)
            feas = torch.arange(self.L, device=dev)[None] <= m[:, None]
            adv = adv * feas
            w = adv / adv.sum(1, keepdim=True).clamp_min(1e-8)
            w = w + c.ent * feas / feas.sum(1, keepdim=True)   # entropy floor on feasible levels
            cand = level_embed(torch.arange(self.L, device=dev), self.L)[None, :, None].expand(len(o), -1, c.horizon)
        if c.actor_update == "aw" or c.policy == "categorical":
            loss_pi = self.pi.loss(o, cand, w)
        elif c.actor_update == "dipo":
            # DIPO: improve replay actions by gradient ascent on Q, then denoising regression onto them
            x = level_embed(a, self.L)[:, None].expand(-1, c.horizon).clone().requires_grad_(True)
            qx = energy_of_embedding(x[:, 0], -self.q(o).mean(0).detach(), self.L)  # = -Q(s, x)
            g = torch.autograd.grad(qx.sum(), x)[0]
            x_new = (x - c.dipo_lr * g).clamp(-1, 1).detach()
            loss_pi = self.pi.loss(o, x_new[:, None], torch.ones(len(o), 1, device=dev))
        elif c.actor_update == "qsm":
            # QSM: match the policy score to alpha * dQ/da at noised actions (eps = -sigma * score)
            x0 = level_embed(a, self.L)[:, None].expand(-1, c.horizon)
            t = torch.randint(0, self.pi.T, (len(o),), device=dev)
            ab = self.pi.alpha_bar[t][:, None]
            xt = (ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)).clamp(-1.5, 1.5).requires_grad_(True)
            negq = energy_of_embedding(xt[:, 0].clamp(-1, 1), -self.q(o).mean(0).detach(), self.L)
            gq = -torch.autograd.grad(negq.sum(), xt)[0]
            target = (-(1 - ab).sqrt() * c.qsm_alpha * gq).detach()
            loss_pi = ((self.pi.model(xt.detach(), t, o) - target) ** 2).mean()
        else:
            raise ValueError(c.actor_update)
        self.opt_pi.zero_grad(); loss_pi.backward(); self.opt_pi.step()
        with torch.no_grad():
            for pt, pp in zip(self.q_t.parameters(), self.q.parameters()):
                pt.mul_(1 - c.tau).add_(c.tau * pp)
        return {"loss_q": loss_q.item(), "loss_pi": loss_pi.item(), "q_mean": qm.mean().item()}


def _ctx_arrays(env):
    return env.ctx["energy"].astype(np.float32), env.ctx["l_max"]


def train(envs, cfg: TrainConfig, logger=print):
    L, obs_dim = envs[0].L, envs[0].obs_dim
    agent = Agent(obs_dim, L, cfg)
    rb = Replay(cfg.replay, obs_dim, L)
    obs = [e.reset() for e in envs]
    ep_ret, ep_rets, step, t0, info = [0.0] * len(envs), [], 0, time.time(), {}
    while step < cfg.total_steps:
        E = np.stack([_ctx_arrays(e)[0] for e in envs]); M = np.array([_ctx_arrays(e)[1] for e in envs])
        if step < cfg.warmup:
            if cfg.warmup_policy == "mpc":
                if not hasattr(train, "_mpc"):
                    from ..eval.baselines import RobustMPC
                    train._mpc = RobustMPC(3, "quality")
                acts = [min(int(train._mpc.act(e)), int(m)) if np.random.rand() > 0.2 else np.random.randint(0, m + 1)
                        for e, m in zip(envs, M)]          # 20% random actions keep coverage of all levels
            else:
                acts = [np.random.randint(0, m + 1) for m in M]
        else:
            acts = agent.act(np.stack(obs), E, M, explore=cfg.eps_explore)
        for i, env in enumerate(envs):
            o2, r, done, _ = env.step(int(acts[i]))
            if done:
                rb.add(obs[i], acts[i], r, None, 1.0, E[i], M[i], None, None)
                ep_rets.append(ep_ret[i] + r); ep_ret[i] = 0.0
                obs[i] = env.reset()
            else:
                e2, m2 = _ctx_arrays(env)
                rb.add(obs[i], acts[i], r, o2, 0.0, E[i], M[i], e2, m2)
                ep_ret[i] += r
                obs[i] = o2
            step += 1
        if step >= cfg.warmup:
            info = agent.update(rb)
        if step % cfg.log_every < len(envs) and ep_rets:
            logger(f"step {step:>8d}  ep_return {np.mean(ep_rets[-20:]):.3f}  "
                   + " ".join(f"{k} {v:.4f}" for k, v in info.items()) + f"  {time.time() - t0:.0f}s")
    return agent, rb


def distill(agent: Agent, rb: Replay, steps=20_000, n_grid=10, batch=256, lr=2e-4, ema=0.995, logger=print):
    """Consistency distillation of the guided teacher into a few-step student."""
    c, dev = agent.cfg, agent.cfg.device
    obs_dim = rb.o.shape[1]
    st = ConsistencyStudent(obs_dim, agent.L, c.horizon, agent.pi.alpha_bar).to(dev)
    st.net.load_state_dict(agent.pi.model.state_dict())  # warm start from the teacher denoiser
    st_ema = copy.deepcopy(st)
    opt = torch.optim.Adam(st.parameters(), lr)
    for it in range(steps):
        o, _, _, _, _, e, m, _, _ = rb.sample(batch, dev)
        with torch.no_grad():
            _, x0 = agent.pi.sample(o, c.T, e, c.eta_guid, m, return_continuous=True)
        loss = consistency_distill_loss(st, st_ema, agent.pi, o, x0, e, c.eta_guid, m, n_grid)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, ps in zip(st_ema.parameters(), st.parameters()):
                pe.mul_(ema).add_((1 - ema) * ps)
        if it % 1000 == 0:
            logger(f"distill {it:>6d}  loss {loss.item():.5f}")
    return st_ema
