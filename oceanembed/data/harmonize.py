"""Preprocessing and harmonisation pipeline.

Two responsibilities:

1. :func:`regrid` / :func:`harmonize_real` - bring heterogeneous real products
   (OSTIA 0.05deg, SMAP/SMOS 0.125deg, DUACS 0.25deg, OSCAR, CCMP/ASCAT) onto the
   single 0.25deg daily analysis grid, with QC, unit normalisation and gap
   filling.  This is the path used when ``data.source = cmems``.

2. :func:`observe` - the **observation operator** for the OSSE twin.  It turns
   the simulator's true surface state into something that looks like a satellite
   product: correlated mapping error plus white instrument noise, and realistic
   missing-data patterns.  The subsurface truth is never touched, so training
   never sees information the satellites would not have.

Per-variable error budgets below are the documented accuracies of the products
named in the problem statement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from ..grid import NIO, SURFACE_VARS, Domain, sea_mask

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErrorBudget:
    """Observation error model for one surface product.

    ``sigma_white`` is pixel-independent instrument noise; ``sigma_corr`` is a
    spatially correlated mapping/representation error with decorrelation scale
    ``corr_px`` grid cells (0.25deg each).  Real L4 products are dominated by the
    correlated term, which is why a purely white noise model would make the
    inverse problem unrealistically easy.
    """

    sigma_white: float
    sigma_corr: float
    corr_px: float


#: Accuracies of the products named in PS#01.
#:   OSTIA L4 SST          ~0.25 degC RMSE
#:   SMAP / SMOS SSS       ~0.2-0.3 psu, degrading near land
#:   DUACS SLA             ~0.03 m mapping error
#:   OSCAR surface current ~0.08-0.12 m/s
#:   CCMP / ASCAT wind     ~0.8-1.0 m/s per component
ERROR_BUDGET: dict[str, ErrorBudget] = {
    "sst":  ErrorBudget(0.12, 0.22, 6.0),
    "sss":  ErrorBudget(0.15, 0.22, 5.0),
    "sla":  ErrorBudget(0.012, 0.028, 8.0),
    "ucur": ErrorBudget(0.04, 0.085, 6.0),
    "vcur": ErrorBudget(0.04, 0.085, 6.0),
    "uwnd": ErrorBudget(0.45, 0.75, 10.0),
    "vwnd": ErrorBudget(0.45, 0.75, 10.0),
}

#: Physically admissible ranges; values outside are flagged and filled.
VALID_RANGE: dict[str, tuple[float, float]] = {
    "sst": (18.0, 35.0), "sss": (25.0, 40.0), "sla": (-1.0, 1.0),
    "ucur": (-3.0, 3.0), "vcur": (-3.0, 3.0), "uwnd": (-30.0, 30.0), "vwnd": (-30.0, 30.0),
}

#: SMAP/SMOS retrievals are contaminated by land within roughly 100 km of the
#: coast, i.e. about one degree at this latitude.
SSS_COAST_BLIND_DEG = 1.0


def _correlated_noise(rng: np.random.Generator, shape: tuple[int, int], corr_px: float) -> np.ndarray:
    w = rng.standard_normal(shape)
    f = gaussian_filter(w, sigma=corr_px, mode="reflect")
    sd = f.std()
    return f / sd if sd > 0 else f


def _cloud_mask(rng: np.random.Generator, domain: Domain, frac: float) -> np.ndarray:
    """Blobby missing-data mask emulating cloud / rain / RFI contamination.

    Real gaps are spatially coherent, not salt-and-pepper, so a smoothed random
    field is thresholded at the quantile that yields the requested fraction.
    """
    if frac <= 0:
        return np.zeros(domain.shape, dtype=bool)
    field = gaussian_filter(rng.standard_normal(domain.shape), sigma=5.0, mode="reflect")
    thr = np.quantile(field, 1.0 - frac)
    return field > thr


def observe(surf: dict[str, np.ndarray], rng: np.random.Generator,
            domain: Domain = NIO, add_noise: bool = True,
            gaps: bool = True, gap_fraction: float = 0.12,
            ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply the satellite observation operator to a true surface state.

    Returns ``(observed, valid)``: the degraded fields (NaN where missing) and a
    per-variable boolean validity mask.  ``surf`` is not modified.
    """
    mask = sea_mask(domain)
    obs: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}

    # One cloud field per day drives SST and SSS gaps together, since both are
    # degraded by the same weather systems.
    clouds = _cloud_mask(rng, domain, gap_fraction) if gaps else np.zeros(domain.shape, bool)
    dist = distance_transform_edt(mask) * domain.resolution

    for v in SURFACE_VARS:
        x = np.asarray(surf[v], dtype="float64").copy()
        if add_noise:
            eb = ERROR_BUDGET[v]
            x = (x + eb.sigma_white * rng.standard_normal(domain.shape)
                 + eb.sigma_corr * _correlated_noise(rng, domain.shape, eb.corr_px))
        ok = mask.copy()
        if gaps:
            if v == "sst":
                # OSTIA is a gap-free L4 analysis, but its uncertainty rises
                # sharply under persistent cloud; emulate as partial dropout.
                ok &= ~(clouds & (rng.random(domain.shape) < 0.35))
            elif v == "sss":
                # Land contamination plus RFI.
                ok &= dist > SSS_COAST_BLIND_DEG
                ok &= ~clouds
            elif v in ("uwnd", "vwnd"):
                # Scatterometer rain flagging.
                ok &= ~(clouds & (rng.random(domain.shape) < 0.25))
        lo, hi = VALID_RANGE[v]
        ok &= np.isfinite(x) & (x >= lo) & (x <= hi)
        obs[v] = np.where(ok, x, np.nan).astype("float32")
        valid[v] = ok
    return obs, valid


def fill_gaps(field: np.ndarray, mask: np.ndarray, max_iter: int = 64) -> np.ndarray:
    """Fill NaNs inside the ocean mask by iterative neighbour diffusion.

    Cheap, edge-preserving and land-aware: each pass replaces missing cells by
    the mean of their valid ocean neighbours, so information spreads inward from
    the gap boundary.  Land stays NaN.
    """
    out = np.asarray(field, dtype="float32").copy()
    holes = mask & ~np.isfinite(out)
    if not holes.any():
        out[~mask] = np.nan
        return out

    # Start from the basin mean so isolated interior holes converge quickly.
    filled = np.where(np.isfinite(out), out, np.nan)
    for _ in range(max_iter):
        missing = mask & ~np.isfinite(filled)
        if not missing.any():
            break
        padded = np.where(np.isfinite(filled), filled, 0.0)
        weight = np.isfinite(filled).astype("float32")
        acc = np.zeros_like(padded)
        wsum = np.zeros_like(weight)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(padded, (di, dj), axis=(0, 1))
            wsum += np.roll(weight, (di, dj), axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            nb = acc / np.where(wsum > 0, wsum, np.nan)
        filled = np.where(missing & (wsum > 0), nb, filled)
    # Any cell still missing (fully enclosed by land) falls back to the basin mean.
    rest = mask & ~np.isfinite(filled)
    if rest.any():
        filled = np.where(rest, np.nanmean(filled[mask]), filled)
    filled[~mask] = np.nan
    return filled.astype("float32")


def preprocess_day(obs: dict[str, np.ndarray], domain: Domain = NIO
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Turn one day of gappy observations into the model input tensor.

    Returns ``(x, gapflag)`` where ``x`` has shape ``(7, nlat, nlon)`` with gaps
    filled, and ``gapflag`` has shape ``(7, nlat, nlon)`` and is 1 where the
    value was reconstructed rather than observed.  Handing the flag to the model
    lets it discount filled pixels instead of trusting them blindly.
    """
    mask = sea_mask(domain)
    x = np.empty((len(SURFACE_VARS),) + domain.shape, dtype="float32")
    flag = np.zeros_like(x)
    for k, v in enumerate(SURFACE_VARS):
        raw = np.asarray(obs[v], dtype="float32")
        missing = mask & ~np.isfinite(raw)
        flag[k] = missing.astype("float32")
        x[k] = fill_gaps(raw, mask)
    return x, flag


# ---------------------------------------------------------------------------
# Real-product harmonisation
# ---------------------------------------------------------------------------
def regrid(da, domain: Domain = NIO, method: str = "linear"):
    """Regrid an :mod:`xarray` DataArray onto the analysis grid.

    Handles the two conventions that differ between the PS#01 products:
    latitude stored north-to-south, and longitude on 0-360 instead of -180-180.
    Conservative area-weighted remapping would be preferable when coarsening
    OSTIA (0.05deg -> 0.25deg); :func:`coarsen_then_regrid` does that.
    """
    lat_name = next(n for n in ("latitude", "lat", "nav_lat") if n in da.coords)
    lon_name = next(n for n in ("longitude", "lon", "nav_lon") if n in da.coords)

    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})
    lons = da[lon_name].values
    if lons.max() > 180.0:
        da = da.assign_coords({lon_name: ((lons + 180.0) % 360.0) - 180.0}).sortby(lon_name)

    return da.interp({lat_name: domain.lat, lon_name: domain.lon},
                     method=method, kwargs={"fill_value": np.nan})


def coarsen_then_regrid(da, domain: Domain = NIO):
    """Area-average a finer product to ~0.25deg before interpolating.

    Straight bilinear interpolation from 0.05deg to 0.25deg discards 96% of the
    source pixels and aliases mesoscale structure; block-averaging first is the
    conservative choice for SST.
    """
    lat_name = next(n for n in ("latitude", "lat") if n in da.coords)
    lon_name = next(n for n in ("longitude", "lon") if n in da.coords)
    src_res = float(abs(np.diff(da[lat_name].values[:2])[0]))
    factor = max(1, int(round(domain.resolution / src_res)))
    if factor > 1:
        da = da.coarsen({lat_name: factor, lon_name: factor}, boundary="trim").mean()
    return regrid(da, domain)
