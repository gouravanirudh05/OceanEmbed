"""Build the analysis-ready dataset.

Produces, for the configured period:

``inputs.npy``   (T, 7, nlat, nlon) float32 - gap-filled surface observations
``gapflag.npy``  (T, 7, nlat, nlon) uint8   - 1 where a value was reconstructed
``target.npy``   (T, 15, nlat, nlon) float32 - subsurface temperature truth
``aux.npy``      (T, 2, nlat, nlon) float32 - MLD and D20, auxiliary targets
``argo.npz``     withheld in-situ profiles
``manifest.json`` dates, splits, normalisation statistics, provenance

Arrays are written as memory-mapped ``.npy`` so training never loads the whole
period into RAM, and a CF-compliant NetCDF copy of the harmonised product is
written per year for inspection and for delivery.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from ..config import Config
from ..grid import (N_DEPTH, STANDARD_DEPTHS, SURFACE_VARS, VAR_ATTRS,
                    Domain, sea_mask)
from . import argo as argo_mod
from .harmonize import observe, preprocess_day
from .synthetic import NIOSimulator, as_date, date_range

log = logging.getLogger(__name__)

AUX_VARS = ("mld", "d20")


def _splits(days: list[date], cfg: Config) -> dict[str, list[int]]:
    train_end = as_date(cfg.data.train_end)
    val_end = as_date(cfg.data.val_end)
    idx = {"train": [], "val": [], "test": []}
    for t, d in enumerate(days):
        if d <= train_end:
            idx["train"].append(t)
        elif d <= val_end:
            idx["val"].append(t)
        else:
            idx["test"].append(t)
    return idx


def _norm_stats(arr: np.memmap, train_idx: list[int], mask: np.ndarray,
                chunk: int = 60) -> tuple[list[float], list[float]]:
    """Per-channel mean/std over ocean points of the training split only.

    Computed in chunks with a running sum so the statistics never require the
    full array in memory, and derived from the training period alone so that
    validation and test data leak nothing into preprocessing.
    """
    n_ch = arr.shape[1]
    count = 0
    s1 = np.zeros(n_ch, dtype="float64")
    s2 = np.zeros(n_ch, dtype="float64")
    for start in range(0, len(train_idx), chunk):
        sel = train_idx[start:start + chunk]
        block = np.asarray(arr[sel], dtype="float64")          # (n, C, H, W)
        vals = block[:, :, mask]                               # (n, C, P)
        good = np.isfinite(vals)
        s1 += np.where(good, vals, 0.0).sum(axis=(0, 2))
        s2 += np.where(good, vals ** 2, 0.0).sum(axis=(0, 2))
        count += int(good[:, 0].sum())
    mean = s1 / max(count, 1)
    var = np.maximum(s2 / max(count, 1) - mean ** 2, 1e-8)
    return mean.tolist(), np.sqrt(var).tolist()


def _write_year_netcdf(path: Path, days: list[date], inputs, target, aux,
                       domain: Domain) -> None:
    """CF-1.8 compliant NetCDF of the harmonised product for one year."""
    import xarray as xr

    coords = {
        "time": ("time", np.array([np.datetime64(d) for d in days])),
        "depth": ("depth", np.asarray(STANDARD_DEPTHS, dtype="float32")),
        "latitude": ("latitude", domain.lat),
        "longitude": ("longitude", domain.lon),
    }
    data = {}
    for k, v in enumerate(SURFACE_VARS):
        data[v] = (("time", "latitude", "longitude"), inputs[:, k], VAR_ATTRS[v])
    data["thetao"] = (("time", "depth", "latitude", "longitude"), target,
                      VAR_ATTRS["thetao"])
    data["mlotst"] = (("time", "latitude", "longitude"), aux[:, 0],
                      {"long_name": "ocean mixed layer thickness", "units": "m",
                       "standard_name": "ocean_mixed_layer_thickness"})
    data["d20"] = (("time", "latitude", "longitude"), aux[:, 1],
                   {"long_name": "depth of 20 degC isotherm", "units": "m"})

    ds = xr.Dataset({k: xr.DataArray(v[1], dims=v[0], attrs=v[2]) for k, v in data.items()},
                    coords=coords)
    ds["depth"].attrs = {"long_name": "depth below sea surface", "units": "m",
                         "positive": "down", "axis": "Z"}
    ds["latitude"].attrs = {"units": "degrees_north", "axis": "Y", "standard_name": "latitude"}
    ds["longitude"].attrs = {"units": "degrees_east", "axis": "X", "standard_name": "longitude"}
    ds.attrs = {
        "Conventions": "CF-1.8",
        "title": "OceanEmbed harmonised North Indian Ocean surface + subsurface dataset",
        "institution": "Prototype for INCOIS SIH-2026 PS#01",
        "source": "OSSE twin (oceanembed.data.synthetic)",
        "geospatial_lat_min": float(domain.lat_min), "geospatial_lat_max": float(domain.lat_max),
        "geospatial_lon_min": float(domain.lon_min), "geospatial_lon_max": float(domain.lon_max),
        "geospatial_lat_resolution": float(domain.resolution),
        "geospatial_lon_resolution": float(domain.resolution),
        "time_coverage_start": days[0].isoformat(), "time_coverage_end": days[-1].isoformat(),
        "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    ds.close()


def build(cfg: Config, write_netcdf: bool = True) -> Path:
    """Generate (or harmonise) the dataset and return the output directory."""
    domain = Domain(cfg.domain.lat_min, cfg.domain.lat_max,
                    cfg.domain.lon_min, cfg.domain.lon_max, cfg.domain.resolution)
    mask = sea_mask(domain)
    days = date_range(cfg.data.start, cfg.data.end)
    T = len(days)
    H, W = domain.shape
    out = cfg.paths.resolve("processed")
    log.info("building %d days on %dx%d grid -> %s", T, H, W, out)

    if cfg.data.source != "synthetic":
        raise NotImplementedError(
            "source='cmems' requires Copernicus Marine / Earthdata credentials; "
            "see scripts/download_real_data.sh and oceanembed.data.cmems")

    mk = lambda name, shape, dtype: np.lib.format.open_memmap(
        out / name, mode="w+", dtype=dtype, shape=shape)
    X = mk("inputs.npy", (T, len(SURFACE_VARS), H, W), "float32")
    Gf = mk("gapflag.npy", (T, len(SURFACE_VARS), H, W), "uint8")
    Y = mk("target.npy", (T, N_DEPTH, H, W), "float32")
    A = mk("aux.npy", (T, len(AUX_VARS), H, W), "float32")

    sim = NIOSimulator(domain=domain, seed=cfg.data.seed)
    rng = np.random.default_rng(cfg.data.seed + 7)
    sampler = argo_mod.ArgoSampler(domain=domain, seed=cfg.data.seed + 11)
    profiles: list[argo_mod.ProfileSet] = []

    t0 = time.time()
    for t, (day, surf, thetao) in enumerate(sim.run(days)):
        aux = surf.pop("_aux")
        obs, _ = observe(surf, rng, domain=domain,
                         add_noise=cfg.data.obs_noise,
                         gaps=cfg.data.cloud_gaps,
                         gap_fraction=cfg.data.gap_fraction)
        x, flag = preprocess_day(obs, domain=domain)
        X[t] = np.nan_to_num(x, nan=0.0)
        Gf[t] = flag.astype("uint8")
        Y[t] = thetao
        A[t] = np.stack([aux[v] for v in AUX_VARS])
        ps = sampler.sample_day(t, thetao)
        if ps is not None:
            profiles.append(ps)
        if (t + 1) % 100 == 0:
            rate = (t + 1) / (time.time() - t0)
            log.info("  %d/%d days (%.1f day/s, eta %.0fs)", t + 1, T, rate, (T - t - 1) / rate)

    for arr in (X, Gf, Y, A):
        arr.flush()

    argo = argo_mod.concat(profiles)
    argo.save(out / "argo.npz")
    log.info("withheld %d ARGO-like profiles (%.1f/day)", len(argo), len(argo) / T)

    idx = _splits(days, cfg)
    log.info("splits: train=%d val=%d test=%d days",
             len(idx["train"]), len(idx["val"]), len(idx["test"]))
    x_mean, x_std = _norm_stats(X, idx["train"], mask)
    y_mean, y_std = _norm_stats(Y, idx["train"], mask)
    a_mean, a_std = _norm_stats(A, idx["train"], mask)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": cfg.data.source,
        "domain": {"lat_min": domain.lat_min, "lat_max": domain.lat_max,
                   "lon_min": domain.lon_min, "lon_max": domain.lon_max,
                   "resolution": domain.resolution, "shape": [H, W]},
        "depths": list(STANDARD_DEPTHS),
        "surface_vars": list(SURFACE_VARS),
        "aux_vars": list(AUX_VARS),
        "dates": [d.isoformat() for d in days],
        "splits": idx,
        "norm": {"x_mean": x_mean, "x_std": x_std,
                 "y_mean": y_mean, "y_std": y_std,
                 "a_mean": a_mean, "a_std": a_std},
        "n_argo": len(argo),
        "config": cfg.to_dict(),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    np.save(out / "seamask.npy", mask)

    if write_netcdf:
        nc_dir = cfg.paths.resolve("interim")
        years = sorted({d.year for d in days})
        for yr in years:
            sel = [t for t, d in enumerate(days) if d.year == yr]
            path = nc_dir / f"oceanembed_nio_{yr}.nc"
            log.info("writing %s (%d days)", path.name, len(sel))
            _write_year_netcdf(path, [days[t] for t in sel], X[sel], Y[sel], A[sel], domain)

    log.info("dataset built in %.1f s", time.time() - t0)
    return out
