"""The assembled OceanEmbed model: encoder + embedding + decoder heads."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..grid import N_DEPTH, N_SURFACE
from .embed import build_encoder
from .heads import AuxHead, ProfileHead, SurfaceReconHead


class OceanEmbed(nn.Module):
    """Reconstruct T(z) at 15 standard depths from surface satellite fields.

    ``forward`` returns a dict with

    ``y``          (B, 15, H, W) standardised temperature
    ``log_sigma``  (B, 15, H, W) predictive log-sigma, or ``None``
    ``aux``        (B, 2, H, W)  standardised MLD and D20
    ``recon``      (B, 7, H, W)  reconstructed surface channels
    ``embedding``  (B, D, H, W)  the satellite embedding itself
    """

    def __init__(self, cfg_model, y_mean, y_std, in_channels: int,
                 profile_head: str = "monotone", n_aux: int = 2):
        super().__init__()
        self.cfg = cfg_model
        self.in_channels = int(in_channels)
        self.encoder = build_encoder(cfg_model.encoder, in_channels, cfg_model)
        self.profile = ProfileHead(
            cfg_model.embed_dim, N_DEPTH, y_mean, y_std,
            hidden=cfg_model.decoder_hidden, layers=cfg_model.decoder_layers,
            dropout=cfg_model.dropout, mode=profile_head,
            max_inversion=cfg_model.max_inversion,
            predict_uncertainty=cfg_model.predict_uncertainty)
        self.aux = AuxHead(cfg_model.embed_dim, n_aux, dropout=cfg_model.dropout)
        self.recon = SurfaceReconHead(cfg_model.embed_dim, N_SURFACE, dropout=cfg_model.dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        y, log_sigma = self.profile(z)
        return {"y": y, "log_sigma": log_sigma, "aux": self.aux(z),
                "recon": self.recon(z), "embedding": z}

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Satellite embedding only - useful for transfer to other variables."""
        return self.encoder(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_report(self) -> str:
        parts = [("encoder", self.encoder), ("profile", self.profile),
                 ("aux", self.aux), ("recon", self.recon)]
        lines = [f"  {n:9s} {sum(p.numel() for p in m.parameters()):>10,d}" for n, m in parts]
        return "\n".join(lines + [f"  {'total':9s} {self.n_parameters():>10,d}"])


@torch.no_grad()
def predict_field(model: nn.Module, x: torch.Tensor, tile: int = 0,
                  overlap: int = 8) -> dict[str, torch.Tensor]:
    """Run the model over a whole domain.

    With ``tile = 0`` the field is processed in one pass, which is exact but
    needs attention over every token at once.  With ``tile > 0`` the field is
    covered by overlapping windows blended with a raised-cosine weight, which
    bounds memory and keeps each window's token count equal to the one the model
    was trained on - at the cost of losing context beyond the window.
    """
    model.eval()
    if tile <= 0:
        return model(x)

    B, _, H, W = x.shape
    step = max(tile - overlap, 1)
    starts_i = list(range(0, max(H - tile, 0) + 1, step))
    starts_j = list(range(0, max(W - tile, 0) + 1, step))
    if starts_i[-1] + tile < H:
        starts_i.append(H - tile)
    if starts_j[-1] + tile < W:
        starts_j.append(W - tile)

    # Raised-cosine window, floored so edge pixels still receive weight.
    ramp = 0.5 * (1.0 - torch.cos(torch.linspace(0, np.pi, tile, device=x.device)))
    wgt = torch.clamp(ramp[:, None] * ramp[None, :], min=1e-3)

    acc: dict[str, torch.Tensor] = {}
    norm = torch.zeros(1, 1, H, W, device=x.device)
    for i in starts_i:
        for j in starts_j:
            out = model(x[..., i:i + tile, j:j + tile])
            for k, v in out.items():
                if v is None or k == "embedding":
                    continue
                if k not in acc:
                    acc[k] = torch.zeros(B, v.shape[1], H, W, device=x.device)
                acc[k][..., i:i + tile, j:j + tile] += v * wgt
            norm[..., i:i + tile, j:j + tile] += wgt
    return {k: v / norm for k, v in acc.items()}
