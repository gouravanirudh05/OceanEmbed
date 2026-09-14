"""In-situ profile handling for independent validation.

Two entry points:

* :func:`sample_argo` draws ARGO-like profiles from the OSSE truth, at the real
  float density and with a realistic error budget.
* :func:`load_argo_netcdf` reads gridded/profile ARGO from the INCOIS Live
  Access Server or an Argo GDAC file into the same in-memory structure, so the
  evaluation code is identical for synthetic and real in-situ data.

Why profiles are not simply grid points
---------------------------------------
An ARGO float measures a single vertical cast, not a 0.25deg x 0.25deg daily
average.  Comparing a gridded reconstruction against a point cast therefore
carries a **representativeness error** that is irreducible and, in the
thermocline, larger than the instrument error by two orders of magnitude.  It is
included here explicitly (see :data:`REPR_ERROR_FRAC`) so that reported
validation numbers are directly comparable with published ARGO-vs-reanalysis
statistics instead of being flatteringly clean.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..grid import NIO, STANDARD_DEPTHS, Domain, sea_mask

log = logging.getLogger(__name__)

#: ARGO CTD temperature accuracy after delayed-mode QC.
ARGO_INSTRUMENT_SIGMA = 0.002
#: Representativeness error, modelled as an effective **vertical offset** (m)
#: between the cast and the daily 0.25deg cell mean, converted to a temperature
#: error through the local gradient.  A 0.25deg (~28 km) and 1-day mismatch
#: corresponds to a few metres of thermocline heave, so 6 m gives ~0.6 degC in
#: the thermocline and ~0.01 degC in the mixed layer - the observed shape of
#: ARGO-minus-gridded-product differences.
REPR_ERROR_M = 6.0
#: Number of active ARGO floats typically reporting in the North Indian Ocean.
#: Each float profiles on a 10-day cycle, giving ~15-25 profiles per day.
NIO_FLOAT_COUNT = 200
ARGO_CYCLE_DAYS = 10


@dataclass
class ProfileSet:
    """A bundle of in-situ profiles, all on the standard depth levels.

    Attributes
    ----------
    day_index : (N,) int32     index into the dataset time axis
    lat, lon  : (N,) float32   true position (not snapped to the grid)
    temp      : (N, 15) float32 observed temperature, NaN where the cast is short
    """

    day_index: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    temp: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.day_index.size)

    def grid_index(self, domain: Domain = NIO) -> tuple[np.ndarray, np.ndarray]:
        """Nearest grid indices for each profile."""
        i = np.clip(np.round((self.lat - domain.lat_min) / domain.resolution), 0,
                    domain.n_lat - 1).astype("int32")
        j = np.clip(np.round((self.lon - domain.lon_min) / domain.resolution), 0,
                    domain.n_lon - 1).astype("int32")
        return i, j

    def subset(self, keep: np.ndarray) -> "ProfileSet":
        return ProfileSet(self.day_index[keep], self.lat[keep], self.lon[keep],
                          self.temp[keep], dict(self.meta))

    def save(self, path) -> None:
        np.savez_compressed(path, day_index=self.day_index, lat=self.lat,
                            lon=self.lon, temp=self.temp)

    @classmethod
    def load(cls, path) -> "ProfileSet":
        z = np.load(path)
        return cls(z["day_index"], z["lat"], z["lon"], z["temp"])


class ArgoSampler:
    """Draw ARGO-like casts from the true 3-D field, one day at a time.

    A fixed population of floats drifts slowly and surfaces on a staggered
    10-day cycle, which reproduces the clustered, non-uniform sampling of the
    real array far better than drawing fresh uniform-random points each day.
    """

    def __init__(self, domain: Domain = NIO, seed: int = 0,
                 n_floats: int = NIO_FLOAT_COUNT):
        self.domain = domain
        self.rng = np.random.default_rng(seed)
        self.mask = sea_mask(domain)
        ocean_i, ocean_j = np.nonzero(self.mask)
        pick = self.rng.choice(ocean_i.size, size=n_floats, replace=False)
        self.lat = domain.lat[ocean_i[pick]] + self.rng.uniform(-0.12, 0.12, n_floats)
        self.lon = domain.lon[ocean_j[pick]] + self.rng.uniform(-0.12, 0.12, n_floats)
        # Stagger the surfacing day so profiles arrive steadily.
        self.phase = self.rng.integers(0, ARGO_CYCLE_DAYS, n_floats)
        self.z = np.asarray(STANDARD_DEPTHS, dtype="float64")

    def _drift(self) -> None:
        """Random-walk the floats at a few cm/s and keep them over water."""
        step = 0.045   # degrees/day ~ 5 cm/s
        new_lat = self.lat + self.rng.normal(0, step, self.lat.size)
        new_lon = self.lon + self.rng.normal(0, step, self.lon.size)
        i = np.clip(np.round((new_lat - self.domain.lat_min) / self.domain.resolution),
                    0, self.domain.n_lat - 1).astype(int)
        j = np.clip(np.round((new_lon - self.domain.lon_min) / self.domain.resolution),
                    0, self.domain.n_lon - 1).astype(int)
        ok = self.mask[i, j]
        self.lat = np.where(ok, np.clip(new_lat, self.domain.lat_min, self.domain.lat_max), self.lat)
        self.lon = np.where(ok, np.clip(new_lon, self.domain.lon_min, self.domain.lon_max), self.lon)

    def sample_day(self, day_idx: int, thetao: np.ndarray) -> ProfileSet | None:
        """Profiles surfacing on ``day_idx``, read off the true field ``thetao``."""
        self._drift()
        surfacing = np.nonzero((day_idx % ARGO_CYCLE_DAYS) == self.phase)[0]
        if surfacing.size == 0:
            return None

        lat = self.lat[surfacing]
        lon = self.lon[surfacing]
        i = np.clip(np.round((lat - self.domain.lat_min) / self.domain.resolution),
                    0, self.domain.n_lat - 1).astype(int)
        j = np.clip(np.round((lon - self.domain.lon_min) / self.domain.resolution),
                    0, self.domain.n_lon - 1).astype(int)
        prof = thetao[:, i, j].T.astype("float64")        # (n, 15)

        # Representativeness error.  One offset is drawn per cast and applied
        # coherently down the whole column - a float samples a single displaced
        # water column, it does not get independent errors at each level.
        dtdz = np.gradient(prof, self.z, axis=1)
        offset_m = REPR_ERROR_M * self.rng.standard_normal((prof.shape[0], 1))
        prof = (prof + dtdz * offset_m
                + ARGO_INSTRUMENT_SIGMA * self.rng.standard_normal(prof.shape))

        # Roughly one profile in six is a shallow cast that stops above 1000 m.
        shallow = self.rng.random(prof.shape[0]) < 0.17
        cut = self.rng.choice([10, 11, 12, 13], size=prof.shape[0])
        for n in np.nonzero(shallow)[0]:
            prof[n, cut[n]:] = np.nan
        prof[~np.isfinite(thetao[:, i, j].T)] = np.nan     # land / no data

        return ProfileSet(
            day_index=np.full(surfacing.size, day_idx, dtype="int32"),
            lat=lat.astype("float32"), lon=lon.astype("float32"),
            temp=prof.astype("float32"),
            meta={"float_id": surfacing.astype("int32")},
        )


def concat(sets: list[ProfileSet]) -> ProfileSet:
    sets = [s for s in sets if s is not None and len(s)]
    if not sets:
        raise ValueError("no profiles to concatenate")
    return ProfileSet(
        np.concatenate([s.day_index for s in sets]),
        np.concatenate([s.lat for s in sets]),
        np.concatenate([s.lon for s in sets]),
        np.concatenate([s.temp for s in sets]),
    )


def sample_argo(thetao_stack, days: list[date], seed: int = 0,
                domain: Domain = NIO, n_floats: int = NIO_FLOAT_COUNT) -> ProfileSet:
    """Convenience wrapper: sample the whole period from a stacked truth array."""
    sampler = ArgoSampler(domain=domain, seed=seed, n_floats=n_floats)
    out = [sampler.sample_day(t, thetao_stack[t]) for t in range(len(days))]
    return concat(out)


# ---------------------------------------------------------------------------
# Real in-situ data
# ---------------------------------------------------------------------------
def load_argo_netcdf(path, days: list[date], domain: Domain = NIO,
                     temp_var: str = "TEMP", depth_var: str = "PRES") -> ProfileSet:
    """Read an Argo GDAC / INCOIS LAS file into a :class:`ProfileSet`.

    Profiles are interpolated onto :data:`oceanembed.grid.STANDARD_DEPTHS` and
    subset to the domain and period.  Pressure in dbar is treated as depth in
    metres, which is accurate to better than 1% in the upper 1000 m.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    lat = np.asarray(ds["LATITUDE"].values, dtype="float64")
    lon = np.asarray(ds["LONGITUDE"].values, dtype="float64")
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    times = np.asarray(ds["JULD"].values).astype("datetime64[D]")

    day0 = np.datetime64(days[0])
    day_index = (times - day0).astype("int64")

    keep = ((lat >= domain.lat_min) & (lat <= domain.lat_max)
            & (lon >= domain.lon_min) & (lon <= domain.lon_max)
            & (day_index >= 0) & (day_index < len(days)))
    if not keep.any():
        raise ValueError(f"no profiles from {path} fall inside the domain/period")

    temp_raw = np.asarray(ds[temp_var].values, dtype="float64")[keep]
    pres_raw = np.asarray(ds[depth_var].values, dtype="float64")[keep]
    z = np.asarray(STANDARD_DEPTHS, dtype="float64")

    out = np.full((int(keep.sum()), z.size), np.nan)
    for n in range(out.shape[0]):
        pr, tp = pres_raw[n], temp_raw[n]
        good = np.isfinite(pr) & np.isfinite(tp)
        if good.sum() < 4:
            continue
        pr, tp = pr[good], tp[good]
        order = np.argsort(pr)
        pr, tp = pr[order], tp[order]
        # Only interpolate within the observed pressure span; never extrapolate.
        inside = (z >= pr[0] - 5.0) & (z <= pr[-1])
        out[n, inside] = np.interp(z[inside], pr, tp)

    log.info("loaded %d ARGO profiles from %s", out.shape[0], path)
    return ProfileSet(day_index[keep].astype("int32"), lat[keep].astype("float32"),
                      lon[keep].astype("float32"), out.astype("float32"),
                      meta={"source": str(path)})
