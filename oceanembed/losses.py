"""Training objective.

Terms
-----
``profile``
    Huber loss on the standardised temperature at all 15 levels.  Standardising
    per level before the loss is what keeps the deep levels in play: in degrees
    Celsius the 1000 m level varies by 0.3 degC against 1.6 degC at the
    thermocline, so an unstandardised loss would effectively ignore it.

``nll``
    Gaussian negative log-likelihood used **only** to fit the predictive sigma.
    The residual is detached, so this term calibrates uncertainty without
    letting a large sigma soften the gradient on the mean - the usual failure
    mode of joint heteroscedastic training.

``strat``
    Penalises vertical temperature increases beyond the physically admissible
    inversion.  Active for the ``free`` head; for the ``monotone`` head the
    constraint is already built into the parameterisation and this term stays at
    zero, which is a useful check that the head behaves as intended.

``grad``
    Matches the *horizontal gradient* of the prediction to that of the truth.
    Plain pointwise losses are minimised by blurry fields, which would smear out
    exactly the mesoscale eddy structure the reconstruction is supposed to
    resolve; matching gradients penalises that blur directly.

``recon``
    Surface autoencoding, to keep the embedding a faithful representation of the
    surface state.

``aux``
    Mixed layer depth and 20 degC isotherm depth - the two parameters that set
    the shape of the profile.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .grid import STANDARD_DEPTHS


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over cells where ``mask`` is 1.

    ``x`` is ``(B, C, H, W)`` and ``mask`` is ``(B, H, W)``.  Returns a scalar.
    """
    m = mask.unsqueeze(1)
    denom = m.sum() * x.shape[1]
    return (x * m).sum() / denom.clamp_min(1.0)


def _huber(err: torch.Tensor, delta: float) -> torch.Tensor:
    a = err.abs()
    return torch.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


class ProfileLoss(torch.nn.Module):
    def __init__(self, cfg_train, y_mean, y_std, max_inversion: float = 1.5):
        super().__init__()
        self.c = cfg_train
        self.max_inversion = float(max_inversion)
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        dz = torch.as_tensor(STANDARD_DEPTHS, dtype=torch.float32).diff()
        self.register_buffer("dz", dz)

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        y_hat, y, mask = out["y"], batch["y"], batch["mask"]
        parts: dict[str, torch.Tensor] = {}

        err = y_hat - y
        parts["profile"] = masked_mean(_huber(err, self.c.huber_delta), mask)

        if out.get("log_sigma") is not None:
            ls = out["log_sigma"]
            e2 = err.detach() ** 2
            parts["nll"] = masked_mean(0.5 * (e2 * torch.exp(-2.0 * ls) + 2.0 * ls), mask)

        # Stratification: dT/dz must not exceed the admissible inversion rate.
        if self.c.w_strat > 0:
            # The penalty has to see the *full* temperature, not the anomaly:
            # the climatological profile supplies most of the vertical gradient,
            # so differencing the anomaly alone would never detect an inversion.
            t_phys = y_hat * self.y_std.view(1, -1, 1, 1) + self.y_mean.view(1, -1, 1, 1)
            dT = t_phys.diff(dim=1)                   # (B, K-1, H, W), negative when stable
            excess = F.relu(dT - self.max_inversion)
            parts["strat"] = masked_mean(excess, mask)

        if self.c.w_grad > 0:
            gx = (y_hat.diff(dim=-1) - y.diff(dim=-1)).abs()
            gy = (y_hat.diff(dim=-2) - y.diff(dim=-2)).abs()
            parts["grad"] = (masked_mean(gx, mask[..., 1:]) + masked_mean(gy, mask[..., 1:, :])) * 0.5

        if self.c.w_recon > 0 and "recon" in out:
            n_surf = out["recon"].shape[1]
            parts["recon"] = masked_mean(
                _huber(out["recon"] - batch["x"][:, :n_surf], self.c.huber_delta), mask)

        if "aux" in out and "aux" in batch:
            parts["aux"] = masked_mean(_huber(out["aux"] - batch["aux"], self.c.huber_delta), mask)

        total = (self.c.w_profile * parts["profile"]
                 + self.c.w_strat * parts.get("strat", 0.0)
                 + self.c.w_grad * parts.get("grad", 0.0)
                 + self.c.w_recon * parts.get("recon", 0.0)
                 + self.c.w_aux * parts.get("aux", 0.0)
                 + self.c.w_nll * parts.get("nll", 0.0))
        log = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
               for k, v in parts.items()}
        return total, log | {"total": float(total.detach())}


@torch.no_grad()
def profile_rmse(out: dict, batch: dict, y_std: torch.Tensor) -> torch.Tensor:
    """Per-level RMSE in degrees Celsius, shape ``(15,)``."""
    m = batch["mask"].unsqueeze(1)
    err = (out["y"] - batch["y"]) * y_std.view(1, -1, 1, 1)
    se = ((err ** 2) * m).sum(dim=(0, 2, 3))
    n = m.sum() * torch.ones_like(se)
    return torch.sqrt(se / n.clamp_min(1.0))
