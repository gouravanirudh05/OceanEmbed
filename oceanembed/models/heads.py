"""Decoder heads that read the satellite embedding.

The profile head is where the physics goes in.  Two variants are provided and
compared under identical training:

``free``
    A plain MLP emitting 15 standardised temperatures.  Nothing stops it from
    producing an unstable profile, so stratification has to be enforced by a
    penalty in the loss.

``monotone``
    Emits a surface temperature plus 14 **strictly positive** downward
    decrements, so the backbone of the profile is monotone by construction, and
    adds a bounded inversion term (``|dT| <= max_inversion``).  The bound matters
    for this basin: northern Bay of Bengal winter profiles carry a genuine
    1-2 degC barrier-layer inversion, so a hard monotonicity constraint would be
    physically wrong, while an unconstrained head wastes capacity rediscovering
    that the ocean is stratified.

The decrement biases are initialised from the training climatology, so an
untrained model already emits the climatological mean profile and learns the
anomaly on top of it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of ``softplus``, numerically safe for small positive ``y``."""
    return y + torch.log(-torch.expm1(-y))


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int, dropout: float) -> nn.Sequential:
    """Per-pixel MLP, implemented with 1x1 convolutions."""
    mods: list[nn.Module] = []
    c = in_dim
    for _ in range(max(layers - 1, 0)):
        mods += [nn.Conv2d(c, hidden, 1), nn.GELU()]
        if dropout > 0:
            mods.append(nn.Dropout2d(dropout))
        c = hidden
    mods.append(nn.Conv2d(c, out_dim, 1))
    return nn.Sequential(*mods)


class ProfileHead(nn.Module):
    """Map the embedding to a standardised temperature profile (and its sigma)."""

    def __init__(self, embed_dim: int, n_depth: int, y_mean, y_std,
                 hidden: int = 256, layers: int = 3, dropout: float = 0.0,
                 mode: str = "monotone", max_inversion: float = 1.5,
                 predict_uncertainty: bool = True):
        super().__init__()
        self.K = int(n_depth)
        self.mode = mode
        self.max_inversion = float(max_inversion)
        self.predict_uncertainty = bool(predict_uncertainty)

        self.register_buffer("y_mean", torch.as_tensor(np.asarray(y_mean), dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(np.asarray(y_std), dtype=torch.float32))

        if mode == "free":
            n_out = self.K
        elif mode == "monotone":
            n_out = 1 + (self.K - 1) + self.K       # surface + decrements + inversion
        else:
            raise KeyError(f"unknown profile head mode {mode!r}")
        self.n_mean_out = n_out
        n_total = n_out + (self.K if self.predict_uncertainty else 0)
        self.net = _mlp(embed_dim, hidden, n_total, layers, dropout)

        if mode == "monotone":
            # Initialise decrement biases so that a zero-activation network
            # reproduces the climatological mean profile exactly.
            drops = (self.y_mean[:-1] - self.y_mean[1:]).clamp_min(0.0) + 0.02
            self.register_buffer("dec_bias", _inverse_softplus(drops))
        # Start with sigma ~ 1 standardised unit, i.e. "as uncertain as climatology".
        self.log_sigma_bias = nn.Parameter(torch.zeros(self.K))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw = self.net(z)
        mean_raw = raw[:, : self.n_mean_out]

        if self.mode == "free":
            y = mean_raw
        else:
            ym = self.y_mean.view(1, -1, 1, 1)
            ys = self.y_std.view(1, -1, 1, 1)
            # Surface temperature, predicted as an anomaly on the climatology.
            t0 = ym[:, :1] + ys[:, :1] * mean_raw[:, :1]
            dec = F.softplus(mean_raw[:, 1:self.K] + self.dec_bias.view(1, -1, 1, 1))
            # Cumulative decrease gives a monotone backbone in degrees Celsius.
            t_phys = torch.cat([t0, t0 - torch.cumsum(dec, dim=1)], dim=1)
            inv = self.max_inversion * torch.tanh(mean_raw[:, self.K:self.K + self.K])
            y = (t_phys + inv - ym) / ys

        log_sigma = None
        if self.predict_uncertainty:
            log_sigma = raw[:, self.n_mean_out:] + self.log_sigma_bias.view(1, -1, 1, 1)
            # Keep sigma in [~0.006, ~7.4] standardised units to avoid the NLL
            # collapsing onto a degenerate solution early in training.
            log_sigma = log_sigma.clamp(-5.0, 2.0)
        return y, log_sigma

    def to_celsius(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.y_std.view(1, -1, 1, 1) + self.y_mean.view(1, -1, 1, 1)


class AuxHead(nn.Module):
    """Predict mixed layer depth and 20 degC isotherm depth from the embedding.

    These are trained as auxiliary tasks.  Both are exactly the quantities that
    control the shape of the profile, so supervising them directly gives the
    encoder a much sharper learning signal about thermocline structure than the
    15 temperatures alone.
    """

    def __init__(self, embed_dim: int, n_aux: int = 2, hidden: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.net = _mlp(embed_dim, hidden, n_aux, 2, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SurfaceReconHead(nn.Module):
    """Reconstruct the surface observations from the embedding.

    Acts as an autoencoding regulariser: it forces the embedding to remain a
    faithful compressed representation of the surface state rather than
    collapsing onto whatever few directions the profile loss happens to need.
    """

    def __init__(self, embed_dim: int, n_surface: int = 7, hidden: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.net = _mlp(embed_dim, hidden, n_surface, 2, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
