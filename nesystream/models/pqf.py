"""Perceptual Quality Field (PQF).

Given cheap visual input for one video chunk (a few low-resolution keyframes
decoded from the lowest rendition plus compressed-domain statistics), PQF
predicts in a single forward pass

  * the full, monotone rate-quality curve  q_hat(l), l = 1..L   (VMAF scale 0..100)
  * visual concept probabilities          p(c),     c = 1..C   (HighMotion, SceneCut, ...)

through a compact quantized descriptor z that is rate-distortion optimised so it
can be shipped in the streaming manifest (tens of bytes per chunk).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Frame encoder
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x))


class FrameEncoder(nn.Module):
    """Light ResNet-style encoder. Replace by a pretrained backbone
    (e.g. a distilled DOVER / FAST-VQA student) via `backbone=` if available."""

    def __init__(self, width: int = 32, out_dim: int = 256):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 5, 2, 2, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.blocks = nn.Sequential(ConvBlock(w, w), ConvBlock(w, 2 * w), ConvBlock(2 * w, 4 * w),
                                    ConvBlock(4 * w, 8 * w))
        self.proj = nn.Linear(8 * w * 2, out_dim)

    def forward(self, x):  # x: (B*F, 3, H, W) in [0,1]
        h = self.blocks(self.stem(x))
        # mean + std pooling keeps texture-energy information useful for quality
        mu = h.mean(dim=(2, 3))
        sd = h.flatten(2).std(dim=2)
        return self.proj(torch.cat([mu, sd], dim=1))


class PretrainedFrameEncoder(nn.Module):
    """ImageNet-pretrained light backbone from torchvision (needs `torchvision`; weights are
    downloaded on first use). name: 'mobilenet_v3_small' | 'mobilenet_v3_large' | 'efficientnet_b0'."""

    def __init__(self, name: str = "mobilenet_v3_small", out_dim: int = 256, pretrained: bool = True,
                 freeze: bool = False, multiscale: bool = True):
        super().__init__()
        import torchvision.models as tvm
        ctor = {"mobilenet_v3_small": (tvm.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
                "mobilenet_v3_large": (tvm.mobilenet_v3_large, "MobileNet_V3_Large_Weights"),
                "efficientnet_b0": (tvm.efficientnet_b0, "EfficientNet_B0_Weights")}[name]
        weights = getattr(tvm, ctor[1]).DEFAULT if pretrained else None
        net = ctor[0](weights=weights)
        self.features = net.features
        # multi-scale: pool mean/std after every stage that changes resolution (shallow stages carry the
        # blur / blocking / texture-loss cues that decide compressed quality, as in LPIPS); else last stage only
        n = len(self.features)
        self.taps = self._stage_ends(n) if multiscale else [n - 1]
        with torch.no_grad():
            x, c = torch.zeros(1, 3, 64, 64), 0
            for i, m in enumerate(self.features):
                x = m(x)
                if i in self.taps:
                    c += x.shape[1]
        self.frozen = freeze
        if freeze:
            for p in self.features.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(2 * c, out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _stage_ends(self, n):
        """indices of the last block before each spatial down-sampling, plus the final block."""
        x, sizes = torch.zeros(1, 3, 64, 64), []
        with torch.no_grad():
            for m in self.features:
                x = m(x); sizes.append(x.shape[-1])
        return [i for i in range(n - 1) if sizes[i + 1] != sizes[i]] + [n - 1]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:  # frozen backbone: keep its BatchNorm statistics fixed too
            self.features.eval()
        return self

    def pooled(self, x):
        """multi-scale mean/std pooled backbone features (before the projection), (B, 2*sum(C_tap))."""
        h, pooled = (x - self.mean) / self.std, []
        for i, m in enumerate(self.features):
            h = m(h)
            if i in self.taps:
                pooled += [h.mean(dim=(2, 3)), h.flatten(2).std(dim=2)]
        return torch.cat(pooled, dim=1)

    def forward(self, x):  # (B*F, 3, H, W) in [0,1]
        return self.proj(self.pooled(x))


# --------------------------------------------------------------------------- #
# Quantised bottleneck with a factorised entropy model
# --------------------------------------------------------------------------- #
class FactorizedEntropyModel(nn.Module):
    """Per-dimension zero-mean Laplacian prior with learned scale.

    bits(z_hat) = -log2( CDF(z_hat + 0.5) - CDF(z_hat - 0.5) ).
    A simplified version of the factorised prior of Balle et al.; sufficient to
    trade descriptor size against prediction accuracy.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def _cdf(self, x, b):
        return 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs() / b))

    def bits(self, z_hat: torch.Tensor) -> torch.Tensor:
        b = F.softplus(self.log_scale) + 1e-3
        p = self._cdf(z_hat + 0.5, b) - self._cdf(z_hat - 0.5, b)
        return (-torch.log2(p.clamp_min(1e-9))).sum(dim=-1)  # (B,)


class QuantizedBottleneck(nn.Module):
    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.enc = nn.Linear(in_dim, z_dim)
        self.entropy = FactorizedEntropyModel(z_dim)

    def forward(self, h, training: bool):
        z = self.enc(h)
        if training:  # additive uniform noise = differentiable proxy of rounding
            z_hat = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_hat = torch.round(z)
        return z_hat, self.entropy.bits(z_hat)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class MonotoneRQHead(nn.Module):
    """q_hat(l) = q_min + sum_{j<=l} softplus(delta_j) , clipped to [0, 100]."""

    def __init__(self, z_dim: int, n_levels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_levels + 1))

    @torch.no_grad()
    def init_prior(self, mean_curve):
        """Start from the content-agnostic prior (mean training curve): the network then only learns a
        content-dependent residual, so with weight decay it can shrink back to the prior but not far below it."""
        m = torch.as_tensor(mean_curve, dtype=torch.float32).clamp(0.5, 99.5)
        last = self.net[-1]
        last.weight.mul_(0.01)
        d = (m[1:] - m[:-1]).clamp_min(1e-2)
        inv_sp = lambda x: x + torch.log(-torch.expm1(-x))        # softplus^-1
        last.bias.copy_(torch.cat([torch.logit(m[:1] / 100), torch.zeros(1), inv_sp(d)]))

    def forward(self, z):
        o = self.net(z)
        q_min = 100 * torch.sigmoid(o[:, :1])
        incr = F.softplus(o[:, 1:])
        q = q_min + torch.cumsum(incr, dim=1) - incr[:, :1]  # level 1 == q_min
        return q.clamp(0.0, 100.0)


class PQF(nn.Module):
    def __init__(self, n_levels: int, n_concepts: int, cmp_dim: int, z_dim: int = 32, width: int = 32,
                 feat_dim: int = 256, backbone: str = "scratch", pretrained: bool = True, inputs: str = "both",
                 teacher_dim: int = 0, dropout: float = 0.0, freeze_backbone: bool = False, pair: int = 0):
        """backbone: 'scratch' (light ResNet) or a torchvision name (see PretrainedFrameEncoder).
        inputs: 'both' | 'frames' | 'bitstream'  (input ablation).
        teacher_dim > 0 adds a head that regresses a frozen teacher feature (offline distillation, e.g. FineVQ).
        pair: number K of distorted rungs whose fragments are stacked after the source fragments (3*(1+K) channels);
              all go through the SAME encoder and the frame feature is [f_src, |f_src - f_r1|, ..., |f_src - f_rK|] -- a learned, fragment-level full-reference
              distortion cue (LPIPS-like) that tells how much this content loses under compression."""
        super().__init__()
        self.n_levels, self.n_concepts, self.z_dim, self.inputs = n_levels, n_concepts, z_dim, inputs
        self.frame_enc = (FrameEncoder(width, feat_dim) if backbone == "scratch"
                          else PretrainedFrameEncoder(backbone, feat_dim, pretrained, freeze=freeze_backbone))
        self.cmp_mlp = nn.Sequential(nn.Linear(cmp_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.n_pair = int(pair)
        self.pair = nn.Sequential(nn.Linear((1 + self.n_pair) * feat_dim, feat_dim), nn.ReLU()) if pair else None
        # temporal: mean, max, and mean absolute difference of frame features
        self.fuse = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * feat_dim + 64, feat_dim), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.bottleneck = QuantizedBottleneck(feat_dim, z_dim)
        self.rq_head = MonotoneRQHead(z_dim, n_levels)
        self.concept_head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, n_concepts))
        self.teacher_head = nn.Linear(feat_dim, teacher_dim) if teacher_dim > 0 else None

    def encode(self, frames, cmp_feats):
        """frames: (B, F, 3, H, W) uint8 or float in [0,1]; cmp_feats: (B, D)"""
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        B, Fn = frames.shape[:2]
        if self.inputs == "bitstream":
            frames = torch.zeros_like(frames)
        if self.inputs == "frames":
            cmp_feats = torch.zeros_like(cmp_feats)
        if self.pair is not None:
            fs = self.frame_enc(frames[:, :, :3].flatten(0, 1))
            dists = [(fs - self.frame_enc(frames[:, :, 3 * (r + 1):3 * (r + 2)].flatten(0, 1))).abs()
                     for r in range(self.n_pair)]
            f = self.pair(torch.cat([fs] + dists, dim=1)).view(B, Fn, -1)
        else:
            f = self.frame_enc(frames.flatten(0, 1)).view(B, Fn, -1)
        diff = (f[:, 1:] - f[:, :-1]).abs().mean(1) if Fn > 1 else torch.zeros_like(f[:, 0])
        h = torch.cat([f.mean(1), f.max(1).values, diff, self.cmp_mlp(cmp_feats)], dim=1)
        return self.fuse(h)

    def forward(self, frames, cmp_feats):
        h = self.encode(frames, cmp_feats)
        z_hat, bits = self.bottleneck(h, self.training)
        out = {"z": z_hat, "bits": bits, "q": self.rq_head(z_hat), "concept_logits": self.concept_head(z_hat)}
        if self.teacher_head is not None:
            out["teacher_pred"] = self.teacher_head(h)
        return out

    # client side: only the decoder heads run on the received descriptor
    @torch.no_grad()
    def decode(self, z_hat):
        return self.rq_head(z_hat), torch.sigmoid(self.concept_head(z_hat))


def pqf_loss(out, q_target, concept_target, lam_c: float = 0.5, lam_r: float = 0.01, teacher=None,
             lam_t: float = 0.0):
    """q_target: (B, L) VMAF; concept_target: (B, C) soft labels in [0,1];
    teacher: (B, Dt) frozen teacher features (offline distillation, cosine loss)."""
    l_q = F.l1_loss(out["q"], q_target)
    l_c = F.binary_cross_entropy_with_logits(out["concept_logits"], concept_target)
    l_r = out["bits"].mean()
    loss = l_q + lam_c * l_c + lam_r * l_r
    info = {"l1_vmaf": l_q.item(), "bce": l_c.item(), "bits": l_r.item()}
    if teacher is not None and lam_t > 0 and "teacher_pred" in out:
        l_t = 1 - F.cosine_similarity(out["teacher_pred"], teacher, dim=1).mean()
        loss = loss + lam_t * l_t
        info["distill"] = l_t.item()
    return loss, info


def descriptor_to_bytes(z_hat: torch.Tensor) -> bytes:
    """Fixed-length int8 serialisation (upper bound on the entropy-coded size)."""
    return z_hat.round().clamp(-128, 127).to(torch.int8).cpu().numpy().tobytes()
