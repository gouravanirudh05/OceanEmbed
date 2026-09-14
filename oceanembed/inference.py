"""Produce the operational product: daily 3-D temperature on the standard grid.

Output is a CF-1.8 NetCDF per day (or one file for a date range) carrying

* ``thetao``          reconstructed temperature at the 15 standard depths
* ``thetao_stderr``   predictive standard deviation, when the model provides it
* ``mlotst``, ``d20``, ``d26``, ``ohc300``, ``tchp`` derived diagnostics

so downstream users get both the field and a statement of how much to trust it.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.dataset import OceanStore
from .diagnostics import DIAG_UNITS, DIAGNOSTICS
from .grid import STANDARD_DEPTHS, VAR_ATTRS
from .train import load_model, pick_device

log = logging.getLogger(__name__)


@torch.no_grad()
def reconstruct(model, store: OceanStore, t: int, state: dict,
                tile: int = 0) -> dict[str, np.ndarray]:
    """Reconstruct one day; returns temperature, sigma and the diagnostics."""
    from .models.oceanembed import predict_field

    device = next(model.parameters()).device
    x = torch.from_numpy(store.x_full(t))[None].to(device)
    out = predict_field(model, x, tile=tile)
    y_mean = np.asarray(state["norm"]["y_mean"], dtype="float32")[:, None, None]
    y_std = np.asarray(state["norm"]["y_std"], dtype="float32")[:, None, None]

    T = out["y"][0].cpu().numpy() * y_std + y_mean
    T = np.where(store.mask[None], T, np.nan).astype("float32")
    res: dict[str, np.ndarray] = {"thetao": T}
    if out.get("log_sigma") is not None:
        sig = np.exp(out["log_sigma"][0].cpu().numpy()) * y_std
        res["thetao_stderr"] = np.where(store.mask[None], sig, np.nan).astype("float32")
    for name, fn in DIAGNOSTICS.items():
        res[name] = np.where(store.mask, fn(T), np.nan).astype("float32")
    return res


def write_product(path: Path, days: list[date], fields: list[dict[str, np.ndarray]],
                  store: OceanStore, source: str) -> Path:
    """Write a CF-compliant NetCDF covering ``days``."""
    import xarray as xr

    dom = store.domain
    stack = lambda k: np.stack([f[k] for f in fields])
    coords = {
        "time": ("time", np.array([np.datetime64(d) for d in days])),
        "depth": ("depth", np.asarray(STANDARD_DEPTHS, dtype="float32")),
        "latitude": ("latitude", dom.lat),
        "longitude": ("longitude", dom.lon),
    }
    dv: dict[str, xr.DataArray] = {
        "thetao": xr.DataArray(stack("thetao"), dims=("time", "depth", "latitude", "longitude"),
                               attrs=VAR_ATTRS["thetao"] | {"cell_methods": "time: mean"}),
    }
    if "thetao_stderr" in fields[0]:
        dv["thetao_stderr"] = xr.DataArray(
            stack("thetao_stderr"), dims=("time", "depth", "latitude", "longitude"),
            attrs={"long_name": "predictive standard deviation of sea water potential temperature",
                   "units": "degree_Celsius"})
    diag_long = {
        "mld": ("ocean mixed layer thickness", "ocean_mixed_layer_thickness"),
        "d20": ("depth of 20 degC isotherm", None),
        "d26": ("depth of 26 degC isotherm", None),
        "ohc300": ("ocean heat content, upper 300 m", None),
        "tchp": ("tropical cyclone heat potential", None),
    }
    for name in DIAGNOSTICS:
        long_name, std_name = diag_long[name]
        attrs = {"long_name": long_name, "units": DIAG_UNITS[name]}
        if std_name:
            attrs["standard_name"] = std_name
        dv[name] = xr.DataArray(stack(name), dims=("time", "latitude", "longitude"), attrs=attrs)

    ds = xr.Dataset(dv, coords=coords)
    ds["depth"].attrs = {"long_name": "depth below sea surface", "units": "m",
                         "positive": "down", "axis": "Z"}
    ds["latitude"].attrs = {"units": "degrees_north", "axis": "Y", "standard_name": "latitude"}
    ds["longitude"].attrs = {"units": "degrees_east", "axis": "X", "standard_name": "longitude"}
    ds.attrs = {
        "Conventions": "CF-1.8",
        "title": "OceanEmbed reconstructed subsurface temperature, North Indian Ocean",
        "summary": ("Daily 0.25 degree three-dimensional ocean temperature reconstructed "
                    "from surface satellite observations by a satellite-embedding "
                    "deep learning framework."),
        "institution": "Prototype for INCOIS SIH-2026 PS#01",
        "source": source,
        "geospatial_lat_min": float(dom.lat_min), "geospatial_lat_max": float(dom.lat_max),
        "geospatial_lon_min": float(dom.lon_min), "geospatial_lon_max": float(dom.lon_max),
        "geospatial_lat_resolution": float(dom.resolution),
        "geospatial_lon_resolution": float(dom.resolution),
        "geospatial_vertical_min": float(min(STANDARD_DEPTHS)),
        "geospatial_vertical_max": float(max(STANDARD_DEPTHS)),
        "geospatial_vertical_positive": "down",
        "time_coverage_start": days[0].isoformat(),
        "time_coverage_end": days[-1].isoformat(),
        "time_coverage_resolution": "P1D",
        "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32",
               "_FillValue": np.float32(np.nan)} for v in ds.data_vars}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=enc)
    ds.close()
    log.info("wrote %s (%d days, %.1f MB)", path, len(days), path.stat().st_size / 1e6)
    return path


def predict_range(cfg: Config, ckpt: str | Path, start: str | None = None,
                  end: str | None = None, data_dir: str | Path | None = None,
                  tile: int = 0, out_name: str | None = None) -> Path:
    """Reconstruct every day in ``[start, end]`` and write one NetCDF."""
    store = OceanStore(data_dir or cfg.paths.resolve("processed"))
    device = pick_device(cfg.train.device)
    model, state = load_model(ckpt, device)

    from .data.synthetic import as_date
    d0 = as_date(start) if start else store.dates[store.splits["test"][0]]
    d1 = as_date(end) if end else store.dates[store.splits["test"][-1]]
    idx = [t for t, d in enumerate(store.dates) if d0 <= d <= d1]
    if not idx:
        raise ValueError(f"no days in the dataset between {d0} and {d1}")
    log.info("reconstructing %d days (%s..%s)", len(idx), d0, d1)

    fields = []
    for n, t in enumerate(idx):
        fields.append(reconstruct(model, store, t, state, tile=tile))
        if (n + 1) % 20 == 0:
            log.info("  %d/%d", n + 1, len(idx))

    name = out_name or f"oceanembed_thetao_{d0:%Y%m%d}_{d1:%Y%m%d}.nc"
    return write_product(cfg.paths.resolve("predictions") / name,
                         [store.dates[t] for t in idx], fields, store,
                         source=f"OceanEmbed {Path(ckpt).name}")
