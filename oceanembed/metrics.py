"""Streaming skill metrics.

Predictions for 184 test days at 15 levels over the basin are 2.7 GB, so every
statistic here is accumulated incrementally from running sums rather than by
holding the fields in memory.  For each (region, level) we track enough moments
to recover RMSE, bias, MAE, centred RMSE, anomaly correlation and the skill
score against climatology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import N_DEPTH, NIO, SUBREGIONS, Domain, sea_mask


def region_masks(domain: Domain = NIO, names=("full",)) -> dict[str, np.ndarray]:
    """Boolean ocean masks for the named sub-basins."""
    base = sea_mask(domain)
    LAT, LON = domain.meshgrid()
    out = {}
    for n in names:
        if n == "full":
            out[n] = base.copy()
            continue
        la0, la1, lo0, lo1 = SUBREGIONS[n]
        out[n] = base & (LAT >= la0) & (LAT <= la1) & (LON >= lo0) & (LON <= lo1)
    return out


@dataclass
class LevelStats:
    """Running moments for one (region, level) pair."""

    n: float = 0.0
    se: float = 0.0        # sum (p - t)^2
    ae: float = 0.0        # sum |p - t|
    e: float = 0.0         # sum (p - t)
    pa: float = 0.0        # sum of prediction anomalies
    ta: float = 0.0        # sum of truth anomalies
    paa: float = 0.0
    taa: float = 0.0
    pta: float = 0.0
    tvar_n: float = 0.0    # sum of squared truth anomalies (climatology error)

    def update(self, p: np.ndarray, t: np.ndarray, clim: np.ndarray) -> None:
        good = np.isfinite(p) & np.isfinite(t) & np.isfinite(clim)
        if not good.any():
            return
        p, t, c = p[good], t[good], clim[good]
        d = p - t
        pa, ta = p - c, t - c
        self.n += d.size
        self.se += float((d ** 2).sum())
        self.ae += float(np.abs(d).sum())
        self.e += float(d.sum())
        self.pa += float(pa.sum())
        self.ta += float(ta.sum())
        self.paa += float((pa ** 2).sum())
        self.taa += float((ta ** 2).sum())
        self.pta += float((pa * ta).sum())
        self.tvar_n += float((ta ** 2).sum())

    def summary(self) -> dict[str, float]:
        n = max(self.n, 1.0)
        rmse = float(np.sqrt(self.se / n))
        bias = self.e / n
        # Anomaly correlation, both series taken relative to climatology.
        mp, mt = self.pa / n, self.ta / n
        cov = self.pta / n - mp * mt
        vp = max(self.paa / n - mp ** 2, 0.0)
        vt = max(self.taa / n - mt ** 2, 0.0)
        acc = float(cov / np.sqrt(vp * vt)) if vp > 0 and vt > 0 else float("nan")
        clim_rmse = float(np.sqrt(self.tvar_n / n))
        return {
            "n": int(self.n), "rmse": rmse, "bias": bias, "mae": self.ae / n,
            "crmse": float(np.sqrt(max(self.se / n - bias ** 2, 0.0))),
            "acc": acc, "clim_rmse": clim_rmse,
            # Murphy skill score: fraction of climatological error variance removed.
            "skill_vs_clim": float(1.0 - (rmse ** 2) / clim_rmse ** 2) if clim_rmse > 0 else float("nan"),
        }


class FieldEvaluator:
    """Accumulate gridded skill for one method across regions and levels."""

    def __init__(self, regions: dict[str, np.ndarray], n_depth: int = N_DEPTH):
        self.regions = regions
        self.stats = {r: [LevelStats() for _ in range(n_depth)] for r in regions}

    def update(self, pred: np.ndarray, truth: np.ndarray, clim: np.ndarray) -> None:
        for r, m in self.regions.items():
            for k in range(pred.shape[0]):
                self.stats[r][k].update(pred[k][m], truth[k][m], clim[k][m])

    def summary(self) -> dict:
        out: dict = {}
        for r, levels in self.stats.items():
            per_level = [s.summary() for s in levels]
            rmse = np.array([d["rmse"] for d in per_level])
            crm = np.array([d["clim_rmse"] for d in per_level])
            out[r] = {
                "per_level": per_level,
                "rmse_mean": float(rmse.mean()),
                "clim_rmse_mean": float(crm.mean()),
                # Aggregate skill uses the mean squared errors, not the mean of
                # per-level skill scores, so deep low-variance levels cannot
                # dominate the headline number.
                "skill_vs_clim": float(1.0 - (rmse ** 2).mean() / (crm ** 2).mean()),
                "acc_mean": float(np.nanmean([d["acc"] for d in per_level])),
                "bias_mean": float(np.mean([d["bias"] for d in per_level])),
            }
        return out


class ProfileEvaluator:
    """Skill against in-situ profiles, evaluated at the profile locations."""

    def __init__(self, n_depth: int = N_DEPTH):
        self.stats = [LevelStats() for _ in range(n_depth)]
        self.n_profiles = 0

    def update(self, pred: np.ndarray, obs: np.ndarray, clim: np.ndarray) -> None:
        """``pred``, ``obs``, ``clim`` all shaped ``(n_profiles, n_depth)``."""
        self.n_profiles += pred.shape[0]
        for k in range(pred.shape[1]):
            self.stats[k].update(pred[:, k], obs[:, k], clim[:, k])

    def summary(self) -> dict:
        per_level = [s.summary() for s in self.stats]
        rmse = np.array([d["rmse"] for d in per_level])
        crm = np.array([d["clim_rmse"] for d in per_level])
        return {"per_level": per_level, "n_profiles": self.n_profiles,
                "rmse_mean": float(rmse.mean()),
                "clim_rmse_mean": float(crm.mean()),
                "skill_vs_clim": float(1.0 - (rmse ** 2).mean() / (crm ** 2).mean()),
                "acc_mean": float(np.nanmean([d["acc"] for d in per_level]))}


class CalibrationEvaluator:
    """Is the predicted uncertainty honest?

    Tracks the empirical coverage of the nominal 68% and 95% predictive
    intervals, and the ratio of RMSE to mean predicted sigma.  A ratio above 1
    means the model is overconfident.
    """

    def __init__(self, n_depth: int = N_DEPTH):
        self.n = np.zeros(n_depth)
        self.in68 = np.zeros(n_depth)
        self.in95 = np.zeros(n_depth)
        self.se = np.zeros(n_depth)
        self.sig = np.zeros(n_depth)

    def update(self, err: np.ndarray, sigma: np.ndarray) -> None:
        """``err`` and ``sigma`` shaped ``(n_depth, ...)`` in degrees Celsius."""
        for k in range(err.shape[0]):
            e, s = err[k].ravel(), sigma[k].ravel()
            good = np.isfinite(e) & np.isfinite(s) & (s > 0)
            e, s = e[good], s[good]
            self.n[k] += e.size
            self.in68[k] += float((np.abs(e) <= s).sum())
            self.in95[k] += float((np.abs(e) <= 1.96 * s).sum())
            self.se[k] += float((e ** 2).sum())
            self.sig[k] += float(s.sum())

    def summary(self) -> dict:
        n = np.maximum(self.n, 1.0)
        rmse = np.sqrt(self.se / n)
        msig = self.sig / n
        return {"coverage_68": (self.in68 / n).tolist(),
                "coverage_95": (self.in95 / n).tolist(),
                "rmse": rmse.tolist(), "mean_sigma": msig.tolist(),
                "overconfidence": (rmse / np.maximum(msig, 1e-6)).tolist(),
                "coverage_68_mean": float((self.in68 / n).mean()),
                "coverage_95_mean": float((self.in95 / n).mean())}


class SharpnessEvaluator:
    """Is the reconstruction as sharp as the field it is reconstructing?

    RMSE alone cannot answer this: a pointwise loss is minimised by a smooth
    field, so a method can score well while having smeared out the mesoscale
    eddies that matter for thermocline structure.  This tracks the mean
    horizontal gradient magnitude of the prediction against that of the truth.
    A ratio below 1 means the field is over-smoothed; above 1 means it is noisy.

    Gradients are taken only where both differenced cells are ocean, so the
    coastline does not register as spurious structure.
    """

    def __init__(self, n_depth: int = N_DEPTH):
        self.gp = np.zeros(n_depth)
        self.gt = np.zeros(n_depth)
        self.n = np.zeros(n_depth)

    @staticmethod
    def _grad_mag(f: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = np.abs(np.diff(f, axis=-1))
        mx = mask[:, :-1] & mask[:, 1:]
        dy = np.abs(np.diff(f, axis=-2))
        my = mask[:-1, :] & mask[1:, :]
        return np.concatenate([dx[..., mx], dy[..., my]], axis=-1), None

    def update(self, pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        gp, _ = self._grad_mag(pred, mask)
        gt, _ = self._grad_mag(truth, mask)
        good = np.isfinite(gp) & np.isfinite(gt)
        for k in range(pred.shape[0]):
            g = good[k]
            self.gp[k] += float(gp[k][g].sum())
            self.gt[k] += float(gt[k][g].sum())
            self.n[k] += int(g.sum())

    def summary(self) -> dict:
        n = np.maximum(self.n, 1.0)
        mp, mt = self.gp / n, self.gt / n
        ratio = mp / np.maximum(mt, 1e-9)
        return {"mean_grad_pred": mp.tolist(), "mean_grad_truth": mt.tolist(),
                "sharpness_ratio": ratio.tolist(),
                "sharpness_ratio_mean": float(ratio.mean())}


def bootstrap_rmse_ci(daily_level_mse: np.ndarray, n_boot: int = 200, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the depth-averaged RMSE.

    ``daily_level_mse`` is ``(n_days, n_depth)``: the mean squared error of each
    depth level on each day.

    Resamples **days**, not individual grid cells, because errors within a day
    are strongly spatially correlated - resampling cells would understate the
    interval by orders of magnitude.

    The statistic computed here is ``mean_k sqrt(mean_days MSE[day, k])``, which
    is exactly the headline ``rmse_mean``.  Pooling the levels before taking the
    square root instead would estimate ``sqrt(mean_k MSE_k)``, a quadratic mean
    that is always the larger of the two - which is how an interval can end up
    not containing the point estimate it is supposed to bracket.
    """
    rng = np.random.default_rng(seed)
    m = np.asarray(daily_level_mse, dtype="float64")
    if m.ndim == 1:
        m = m[:, None]
    keep = np.isfinite(m).all(axis=1)
    m = m[keep]
    if m.shape[0] == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, m.shape[0], size=(n_boot, m.shape[0]))
    draws = np.sqrt(m[idx].mean(axis=1)).mean(axis=1)      # (n_boot,)
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))
