"""Logic-guided diffusion policy over bitrate levels.

Action representation: the next H bitrate levels are embedded as
    e_l = -1 + 2 l / (L - 1)   in [-1, 1]^H .
The denoiser is conditioned on the state (playback stats + PQF descriptors +
one-step NS-QoE energy table).

Sampling (Sec. 4.3 of the paper):
  * soft rules  -> energy guidance   eps~ = eps + eta * sqrt(1 - a_bar_t) * dE/dx_t
  * hard rules  -> projection of the predicted clean action onto the feasible set
                   F(s) = {0..l_max}; since F is an interval in embedding space the
                   Euclidean projection is a clamp to [-1, e_{l_max}].
  The final sample is rounded to the nearest feasible level, so hard-constraint
  violations are zero by construction.

Training: advantage-weighted denoising (kept from DiffAVQ):
    L = E_{s, l, t, eps} [ w(s, l) || eps - eps_theta(sqrt(a_bar) e_l + sqrt(1-a_bar) eps, s, t) ||^2 ]
    w(s, l) = normalise( relu(Q(s, l) - V(s)) ) + ent / L      (uniform floor = entropy term)

Few-step inference: consistency distillation of the guided teacher (`ConsistencyStudent`).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def level_embed(levels: torch.Tensor, n_levels: int) -> torch.Tensor:
    return -1.0 + 2.0 * levels.float() / (n_levels - 1)


def embed_to_level(x: torch.Tensor, n_levels: int) -> torch.Tensor:
    return torch.round((x.clamp(-1, 1) + 1.0) * (n_levels - 1) / 2.0).long()


class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=1)


class Denoiser(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, temb: int = 32):
        super().__init__()
        self.temb = SinusoidalEmb(temb)
        self.obs = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Mish(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(nn.Linear(act_dim + temb + hidden, hidden), nn.Mish(),
                                 nn.Linear(hidden, hidden), nn.Mish(), nn.Linear(hidden, hidden), nn.Mish(),
                                 nn.Linear(hidden, act_dim))

    def forward(self, x, t, obs):
        return self.net(torch.cat([x, self.temb(t), self.obs(obs)], dim=1))


def make_schedule(T: int):
    """Cosine schedule (Nichol & Dhariwal)."""
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    betas = (1 - ab[1:] / ab[:-1]).clamp(1e-5, 0.999)
    alpha_bar = torch.cumprod(1 - betas, 0)
    return betas.float(), alpha_bar.float()


def energy_of_embedding(x0_first: torch.Tensor, energy_table: torch.Tensor, n_levels: int, tau: float = 0.02):
    """Differentiable interpolation of a per-level energy table at a continuous action.
    x0_first: (B,), energy_table: (B, L)  ->  (B,)"""
    e = torch.linspace(-1, 1, n_levels, device=x0_first.device)
    w = torch.softmax(-(x0_first[:, None] - e[None]) ** 2 / tau, dim=1)
    return (w * energy_table).sum(1)


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_levels: int, horizon: int = 1, T: int = 20, hidden: int = 256):
        super().__init__()
        self.n_levels, self.H, self.T = n_levels, horizon, T
        self.model = Denoiser(obs_dim, horizon, hidden)
        betas, ab = make_schedule(T)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", ab)

    # ------------------------------------------------------------------ #
    def loss(self, obs, cand_actions, weights):
        """obs: (B, D); cand_actions: (B, K, H) embedded targets; weights: (B, K) >= 0."""
        B, K, H = cand_actions.shape
        o = obs[:, None].expand(B, K, obs.shape[1]).reshape(B * K, -1)
        x0 = cand_actions.reshape(B * K, H)
        t = torch.randint(0, self.T, (B * K,), device=obs.device)
        eps = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        err = ((self.model(xt, t, o) - eps) ** 2).mean(1).view(B, K)
        return (weights * err).sum(1).mean()

    # ------------------------------------------------------------------ #
    def _project(self, x0, l_max):
        if l_max is None:
            return x0.clamp(-1, 1)
        hi = level_embed(l_max, self.n_levels)
        x0 = x0.clamp(-1, 1)
        first = torch.minimum(x0[:, 0], hi)
        return torch.cat([first[:, None], x0[:, 1:]], dim=1)

    def sample(self, obs, n_steps: int | None = None, energy_table=None, eta_guid: float = 0.0, l_max=None,
               stochastic: float = 1.0, return_continuous: bool = False):
        """DDIM-style sampler on a sub-sequence of n_steps timesteps.
        stochastic in [0,1] = DDIM eta (1 ~ DDPM, 0 = deterministic)."""
        B = obs.shape[0]
        n_steps = n_steps or self.T
        ts = torch.linspace(self.T - 1, 0, n_steps, device=obs.device).round().long()
        x = torch.randn(B, self.H, device=obs.device)
        for i, t in enumerate(ts):
            tt = torch.full((B,), int(t), device=obs.device, dtype=torch.long)
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=obs.device)
            if energy_table is not None and eta_guid > 0:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    eps = self.model(xg, tt, obs)
                    x0 = (xg - (1 - ab).sqrt() * eps) / ab.sqrt()
                    E = energy_of_embedding(x0[:, 0], energy_table, self.n_levels).sum()
                    g = torch.autograd.grad(E, xg)[0]
                eps = eps.detach() + eta_guid * (1 - ab).sqrt() * g
            else:
                eps = self.model(x, tt, obs)
            x0 = self._project((x - (1 - ab).sqrt() * eps) / ab.sqrt(), l_max)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            if i + 1 < len(ts):
                sigma = stochastic * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp_min(0).sqrt()
                x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp_min(0).sqrt() * eps \
                    + sigma * torch.randn_like(x)
            else:
                x = x0
        x = self._project(x, l_max)
        lv = embed_to_level(x[:, 0], self.n_levels)
        if l_max is not None:
            lv = torch.minimum(lv, l_max)
        return (lv, x) if return_continuous else lv


# --------------------------------------------------------------------------- #
# Consistency distillation (few-step student)
# --------------------------------------------------------------------------- #
class ConsistencyStudent(nn.Module):
    """f(x_t, t, s) -> x_0 with the boundary condition f(x, t_min) = x."""

    def __init__(self, obs_dim: int, n_levels: int, horizon: int, alpha_bar: torch.Tensor, hidden: int = 256,
                 sigma_data: float = 0.5):
        super().__init__()
        self.n_levels, self.H = n_levels, horizon
        self.net = Denoiser(obs_dim, horizon, hidden)
        self.register_buffer("alpha_bar", alpha_bar.clone())
        self.sd = sigma_data

    def _sigma(self, t):
        ab = self.alpha_bar[t]
        return ((1 - ab) / ab).sqrt()

    def forward(self, x_t, t, obs):
        ab = self.alpha_bar[t][:, None]
        x_ve = x_t / ab.sqrt()                       # VP -> VE: x0 + sigma * z
        s = self._sigma(t)[:, None]
        s0 = self._sigma(torch.zeros_like(t))[:, None]
        c_skip = self.sd ** 2 / ((s - s0) ** 2 + self.sd ** 2)
        c_out = (s - s0) * self.sd / (self.sd ** 2 + s ** 2).sqrt()
        c_in = 1.0 / (s ** 2 + self.sd ** 2).sqrt()
        return c_skip * x_ve + c_out * self.net(c_in * x_ve, t, obs)

    @torch.no_grad()
    def sample(self, obs, n_steps: int = 1, l_max=None, T: int | None = None):
        B = obs.shape[0]
        T = T or self.alpha_bar.shape[0]
        ts = torch.linspace(T - 1, 0, n_steps + 1, device=obs.device).round().long()[:-1]
        x = torch.randn(B, self.H, device=obs.device)
        x0 = None
        for i, t in enumerate(ts):
            tt = torch.full((B,), int(t), device=obs.device, dtype=torch.long)
            if i > 0:  # re-noise the current estimate to the next (smaller) time
                ab = self.alpha_bar[t]
                x = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
            x0 = self.forward(x, tt, obs)
            x0 = DiffusionPolicy._project(self, x0, l_max)
        lv = embed_to_level(x0[:, 0], self.n_levels)
        return torch.minimum(lv, l_max) if l_max is not None else lv


def consistency_distill_loss(student, ema_student, teacher: DiffusionPolicy, obs, x0_data, energy_table,
                             eta_guid: float, l_max, n_grid: int = 10):
    """One CD step on a teacher timestep grid of n_grid points.
    x0_data: (B, H) samples of the (guided) teacher used as data."""
    B = obs.shape[0]
    grid = torch.linspace(0, teacher.T - 1, n_grid, device=obs.device).round().long()
    n = torch.randint(0, n_grid - 1, (B,), device=obs.device)
    t_hi, t_lo = grid[n + 1], grid[n]
    ab_hi = teacher.alpha_bar[t_hi][:, None]
    ab_lo = teacher.alpha_bar[t_lo][:, None]
    x_hi = ab_hi.sqrt() * x0_data + (1 - ab_hi).sqrt() * torch.randn_like(x0_data)
    with torch.no_grad():  # one guided teacher DDIM step t_hi -> t_lo
        if energy_table is not None and eta_guid > 0:
            with torch.enable_grad():
                xg = x_hi.detach().requires_grad_(True)
                eps = teacher.model(xg, t_hi, obs)
                x0 = (xg - (1 - ab_hi).sqrt() * eps) / ab_hi.sqrt()
                E = energy_of_embedding(x0[:, 0], energy_table, teacher.n_levels).sum()
                g = torch.autograd.grad(E, xg)[0]
            eps = eps.detach() + eta_guid * (1 - ab_hi).sqrt() * g
        else:
            eps = teacher.model(x_hi, t_hi, obs)
        x0_t = teacher._project((x_hi - (1 - ab_hi).sqrt() * eps) / ab_hi.sqrt(), l_max)
        eps = (x_hi - ab_hi.sqrt() * x0_t) / (1 - ab_hi).sqrt()
        x_lo = ab_lo.sqrt() * x0_t + (1 - ab_lo).sqrt() * eps
        target = ema_student(x_lo, t_lo, obs)
    pred = student(x_hi, t_hi, obs)
    return F.mse_loss(pred, target)


# --------------------------------------------------------------------------- #
# Q-ensemble critic over discrete first action
# --------------------------------------------------------------------------- #
class QEnsemble(nn.Module):
    def __init__(self, obs_dim: int, n_levels: int, n_q: int = 10, hidden: int = 256):
        super().__init__()
        self.n_q = n_q
        self.qs = nn.ModuleList([nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                               nn.ReLU(), nn.Linear(hidden, n_levels)) for _ in range(n_q)])

    def forward(self, obs):  # (n_q, B, L)
        return torch.stack([q(obs) for q in self.qs], 0)
