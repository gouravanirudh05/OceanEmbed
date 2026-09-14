"""Physics-based North Indian Ocean simulator used as an OSSE twin.

Why this exists
---------------
The operational inputs named in the problem statement (OSTIA, SMAP/SMOS, DUACS,
OSCAR, CCMP) and the GLORYS target all sit behind Copernicus Marine / NASA
Earthdata credentials and run to hundreds of gigabytes.  To make the framework
runnable, testable and reviewable *today*, this module synthesises a
self-consistent "twin" of the basin: surface fields and the matching
three-dimensional temperature field come from one shared physical state, so the
surface -> subsurface mapping the network must learn is real, nonlinear and
noise-limited rather than circular.

This is a standard Observing System Simulation Experiment (OSSE) set-up: skill
measured here is skill at the *inverse problem*, and the identical pipeline
consumes real products through :mod:`oceanembed.data.cmems`.

Physical content
----------------
* Mesoscale eddy population with beta-plane westward propagation, finite
  lifetimes, and birth/death - drives sea level and thermocline displacement.
* Sea level split into a **steric** part (thermal expansion, tied to seasonal
  SST) and a **dynamic** part (eddies + Rossby waves).  Only the dynamic part
  displaces the thermocline, so the network cannot simply regress T(z) on total
  SLA - it has to disentangle the two using season and SST.  This is the main
  source of genuine difficulty in the inverse problem.
* Monsoon wind reversal with the Findlater jet, coastal upwelling off Somalia,
  Oman and SW India, Ekman + geostrophic surface currents.
* Bay of Bengal freshwater cap, and the winter barrier-layer **temperature
  inversion** in the northern Bay - which is why the decoder must be able to
  represent non-monotonic profiles.

References for the parameter values are given inline; they are typical
climatological magnitudes for the basin rather than a fit to any one product.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter

from ..grid import NIO, STANDARD_DEPTHS, Domain, sea_mask

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G = 9.81                      # m s-2
OMEGA = 7.2921e-5             # rad s-1, Earth rotation
RHO_AIR = 1.225               # kg m-3
RHO_SW = 1025.0               # kg m-3
DEG2KM = 111.195              # km per degree of latitude

#: Regression of 20 degC isotherm depth onto *dynamic* sea level.  Observed
#: tropical values fall in 100-170 m m-1; 150 is mid-range.
D20_PER_SLA = 150.0
#: SST imprint of a mesoscale eddy, degC per metre of dynamic sea level
#: (a 0.25 m warm-core eddy warms the surface by ~0.45 degC).
SST_PER_SLA = 1.8

#: Reference (climatological) vertical temperature profile for the basin,
#: representative of the North Indian Ocean annual mean.  Interpolated with a
#: monotone cubic (PCHIP) so that vertical gradients are smooth - a linear
#: interpolant would put artificial kinks at the table nodes.
_Z_REF = np.array([0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300,
                   400, 500, 600, 700, 850, 1000], dtype="float64")
_T_REF = np.array([28.6, 28.5, 28.3, 28.0, 26.6, 24.2, 21.4, 19.2, 17.5, 14.9,
                   13.4, 12.2, 10.7, 9.6, 8.5, 7.6, 6.4, 5.2], dtype="float64")

_PCHIP = PchipInterpolator(_Z_REF, _T_REF, extrapolate=True)


def _t_reference(z):
    """Climatological temperature (degC) at depth ``z`` (m); monotone decreasing."""
    return _PCHIP(np.clip(z, 0.0, 1000.0))


def _d20_reference() -> float:
    """Depth of the 20 degC isotherm in the reference profile (~116 m)."""
    zz = np.arange(0.0, 400.0, 0.1)
    return float(zz[np.argmin(np.abs(_t_reference(zz) - 20.0))])


#: Reference 20 degC isotherm depth, the pivot for vertical stretching.
D20_REF = _d20_reference()

#: Depth below which thermocline-displacement stretching is progressively
#: relaxed (the deep ocean is far less variable than the thermocline).
STRETCH_FULL_M = 250.0
STRETCH_TAPER_M = 400.0
#: e-folding scale over which the mixed-layer temperature offset is blended into
#: the stratified interior below the mixed-layer base.
ML_BLEND_M = 32.0

#: Amplitude (degC) of slowly-varying deep water-mass variability that leaves
#: *no* surface signature.  Without it the deepest levels would be exactly
#: constant and trivially predictable; with it, deep levels carry an honest
#: irreducible error floor set by water-mass variability rather than by the
#: skill of the inverse method.
DEEP_NOISE_AMP = 0.30
DEEP_NOISE_Z0 = 350.0
DEEP_NOISE_DZ = 150.0


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale_px: float) -> np.ndarray:
    """Unit-variance spatially correlated noise with decorrelation ``scale_px``."""
    w = rng.standard_normal(shape)
    f = gaussian_filter(w, sigma=scale_px, mode="reflect")
    s = f.std()
    return f / s if s > 0 else f


# ---------------------------------------------------------------------------
# Monthly surface/upper-ocean climatology for five archetype sub-basins.
# ---------------------------------------------------------------------------
# A two-harmonic (annual + semiannual) fit cannot reproduce the North Indian
# Ocean seasonal cycle: the basin has a sharp May maximum, a broad warm
# post-monsoon plateau and a narrow February minimum, which forces a spurious
# ~1.5 degC winter cold bias.  Instead we tabulate monthly climatological values
# for five archetypes and interpolate in time with a *periodic* cubic spline and
# in space with smooth basin/latitude weights.
#
# Values are representative climatological means (WOA/OSTIA-era magnitudes) for:
#   EQ     equatorial band, 5-10degN
#   AS_S   southern/central Arabian Sea, 10-18degN
#   AS_N   northern Arabian Sea, 18-25degN
#   BOB_S  southern/central Bay of Bengal, 10-18degN
#   BOB_N  northern Bay of Bengal, 18-22degN
_ARCHETYPES = ("EQ", "AS_S", "AS_N", "BOB_S", "BOB_N")

#: Sea surface temperature, degC
_CLIM_SST = {
    "EQ":    [28.2, 28.5, 29.2, 29.7, 29.8, 29.0, 28.5, 28.4, 28.6, 28.9, 29.0, 28.7],
    "AS_S":  [27.3, 27.3, 28.3, 29.4, 29.9, 28.9, 27.8, 27.4, 27.8, 28.5, 28.4, 27.9],
    "AS_N":  [24.8, 24.5, 26.0, 28.0, 29.6, 29.3, 28.2, 27.6, 27.9, 28.2, 27.0, 25.6],
    "BOB_S": [27.6, 28.1, 29.1, 30.0, 30.2, 29.4, 28.9, 28.8, 28.9, 28.9, 28.4, 27.9],
    "BOB_N": [25.3, 26.3, 28.2, 29.7, 30.0, 29.6, 29.1, 29.0, 29.1, 28.9, 27.7, 26.0],
}
#: Sea surface salinity (PSS-78).  The Bay carries the Ganges-Brahmaputra plume;
#: the Arabian Sea is evaporation-dominated.
_CLIM_SSS = {
    "EQ":    [34.9, 34.9, 34.8, 34.8, 34.8, 34.9, 34.9, 34.8, 34.7, 34.6, 34.7, 34.8],
    "AS_S":  [35.8, 35.9, 35.9, 36.0, 36.0, 35.9, 35.8, 35.7, 35.6, 35.6, 35.7, 35.8],
    "AS_N":  [36.5, 36.6, 36.6, 36.7, 36.8, 36.7, 36.5, 36.4, 36.3, 36.3, 36.4, 36.5],
    "BOB_S": [34.0, 34.1, 34.2, 34.3, 34.2, 33.9, 33.6, 33.3, 33.0, 32.9, 33.3, 33.7],
    "BOB_N": [31.5, 32.0, 32.4, 32.5, 32.3, 31.6, 30.8, 30.2, 30.0, 30.1, 30.6, 31.0],
}
#: Mixed layer depth, m.  Deep under the SW monsoon jet and in the northern
#: Arabian Sea winter convection; shallow year-round in the fresh-capped Bay.
_CLIM_MLD = {
    "EQ":    [40, 40, 30, 25, 35, 50, 55, 55, 50, 45, 45, 45],
    "AS_S":  [55, 60, 40, 25, 25, 45, 60, 65, 55, 40, 45, 50],
    "AS_N":  [75, 85, 55, 25, 20, 35, 50, 55, 45, 35, 45, 65],
    "BOB_S": [35, 30, 25, 20, 25, 35, 40, 40, 35, 30, 30, 35],
    "BOB_N": [25, 22, 18, 15, 18, 22, 22, 20, 18, 16, 18, 22],
}
#: Depth of the 20 degC isotherm, m - the upper-ocean heat content proxy.
_CLIM_D20 = {
    "EQ":    [110, 110, 115, 120, 125, 130, 130, 125, 120, 115, 110, 110],
    "AS_S":  [100, 100, 105, 110, 115, 110, 100,  95,  95, 100, 100, 100],
    "AS_N":  [ 85,  80,  85,  90,  95,  95,  90,  85,  80,  80,  85,  85],
    "BOB_S": [110, 110, 115, 120, 120, 115, 110, 110, 115, 120, 115, 110],
    "BOB_N": [ 95,  90,  90,  95, 100, 100, 100, 105, 110, 110, 105, 100],
}

#: Day-of-year of each month midpoint, used as the spline abscissa.
_MONTH_MID = np.array([15.5, 45.0, 74.5, 105.0, 135.5, 166.0,
                       196.5, 227.5, 258.0, 288.5, 319.0, 349.5])


def _periodic_spline(monthly: np.ndarray):
    """Periodic cubic spline through 12 monthly fields, indexed by day-of-year.

    ``monthly`` has shape ``(12, ...)``; the returned callable maps a scalar
    day-of-year to an array of shape ``monthly.shape[1:]``.
    """
    x = np.concatenate([_MONTH_MID, [_MONTH_MID[0] + 365.25]])
    y = np.concatenate([monthly, monthly[:1]], axis=0)
    return CubicSpline(x, y, axis=0, bc_type="periodic")


# ---------------------------------------------------------------------------
# Mesoscale eddy population
# ---------------------------------------------------------------------------
@dataclass
class _Eddy:
    lon: float
    lat: float
    radius: float      # e-folding radius, degrees
    amp: float         # peak sea-level anomaly, m (signed)
    age: int
    life: int          # days
    c_zonal: float     # deg/day, negative = westward
    c_merid: float

    @property
    def envelope(self) -> float:
        """Growth/decay envelope over the eddy lifetime (0 at birth and death)."""
        return float(np.sin(np.pi * np.clip(self.age / self.life, 0.0, 1.0)) ** 0.7)


class _EddyField:
    """Population of propagating mesoscale eddies over the domain.

    Westward phase speed follows the long-Rossby-wave estimate
    ``c = -beta * Ld**2`` which gives ~0.10-0.25 m/s in this latitude band; the
    values below are drawn around that range.
    """

    def __init__(self, domain: Domain, rng: np.random.Generator, n_eddies: int = 90):
        self.domain, self.rng, self.n = domain, rng, n_eddies
        self.eddies: list[_Eddy] = []
        self._mask = sea_mask(domain)
        for _ in range(n_eddies):
            e = self._spawn()
            e.age = int(rng.integers(0, e.life))    # de-synchronise initial ages
            self.eddies.append(e)

    def _spawn(self, at_east_edge: bool = False) -> _Eddy:
        d, rng = self.domain, self.rng
        lat = float(rng.uniform(d.lat_min + 0.5, d.lat_max - 0.5))
        lon = float(d.lon_max - rng.uniform(0.0, 2.0)) if at_east_edge else float(
            rng.uniform(d.lon_min + 0.5, d.lon_max - 0.5))
        # Rossby speed grows towards the equator; 0.08-0.20 deg/day ~ 0.10-0.26 m/s
        c = -(0.055 + 0.16 * np.exp(-((lat - 5.0) / 14.0) ** 2)) * rng.uniform(0.6, 1.4)
        return _Eddy(
            lon=lon, lat=lat,
            radius=float(rng.uniform(1.2, 3.6)),
            amp=float(rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.21)),
            age=0, life=int(rng.integers(45, 200)),
            c_zonal=float(c), c_merid=float(rng.normal(0.0, 0.012)),
        )

    def advance(self) -> None:
        for i, e in enumerate(self.eddies):
            e.lon += e.c_zonal
            e.lat += e.c_merid
            e.age += 1
            dead = e.age >= e.life or e.lon < self.domain.lon_min - 2.0 or not (
                self.domain.lat_min - 1.0 < e.lat < self.domain.lat_max + 1.0)
            if dead:
                self.eddies[i] = self._spawn(at_east_edge=self.rng.random() < 0.7)

    def sla(self) -> np.ndarray:
        """Render the eddy sea-level field, evaluating each eddy on a local window."""
        d = self.domain
        out = np.zeros(d.shape, dtype="float64")
        lat, lon = d.lat, d.lon
        for e in self.eddies:
            env = e.envelope
            if env < 1e-3:
                continue
            reach = 3.0 * e.radius
            i0, i1 = np.searchsorted(lat, [e.lat - reach, e.lat + reach])
            j0, j1 = np.searchsorted(lon, [e.lon - reach, e.lon + reach])
            if i0 >= i1 or j0 >= j1:
                continue
            dy = (lat[i0:i1] - e.lat)[:, None]
            dx = ((lon[j0:j1] - e.lon) * np.cos(np.deg2rad(e.lat)))[None, :]
            out[i0:i1, j0:j1] += e.amp * env * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * e.radius ** 2))
        return out


# ---------------------------------------------------------------------------
# Basin simulator
# ---------------------------------------------------------------------------
class NIOSimulator:
    """Generate daily surface fields and the matching 3-D temperature field.

    Usage::

        sim = NIOSimulator(seed=0)
        for day, surf, thetao in sim.run(dates):
            ...

    ``surf`` is a dict of 2-D arrays keyed by :data:`oceanembed.grid.SURFACE_VARS`
    holding the *true* state; ``thetao`` has shape ``(15, nlat, nlon)``.
    Observation error and satellite gaps are applied separately by
    :func:`oceanembed.data.harmonize.observe` so that the truth stays clean.
    """

    def __init__(self, domain: Domain = NIO, seed: int = 0, n_eddies: int = 90):
        self.domain = domain
        self.rng = np.random.default_rng(seed)
        self.mask = sea_mask(domain)
        self.eddies = _EddyField(domain, self.rng, n_eddies)
        self._build_static()
        # AR(1) synoptic noise states: weather-band variability, ~5-10 day memory.
        self._ar = {
            "sst":  np.zeros(domain.shape),
            "sss":  np.zeros(domain.shape),
            "mld":  np.zeros(domain.shape),
            "wind": np.zeros((2,) + domain.shape),
            "sla":  np.zeros(domain.shape),
            "deep": np.zeros(domain.shape),
        }
        self._ar_rho = {"sst": 0.90, "sss": 0.95, "mld": 0.85, "wind": 0.72,
                        "sla": 0.94, "deep": 0.988}
        self._ar_scale_px = {"sst": 9.0, "sss": 12.0, "mld": 7.0, "wind": 14.0,
                             "sla": 10.0, "deep": 20.0}

    # -- static geography ---------------------------------------------------
    def _build_static(self) -> None:
        d = self.domain
        LAT, LON = d.meshgrid()
        self.LAT, self.LON = LAT, LON

        # Coriolis parameter, magnitude clamped at its 5degN value so the
        # geostrophic balance stays finite at the southern boundary.
        f = 2.0 * OMEGA * np.sin(np.deg2rad(LAT))
        f_min = 2.0 * OMEGA * np.sin(np.deg2rad(5.0))
        self.f = np.sign(f) * np.maximum(np.abs(f), f_min)

        # Basin membership weights (smooth, so fields blend across the tip of India).
        self.w_arabian = 1.0 / (1.0 + np.exp((LON - 74.0) / 2.5))
        self.w_bengal = 1.0 / (1.0 + np.exp((78.0 - LON) / 2.5))
        self.w_north_bob = self.w_bengal * 1.0 / (1.0 + np.exp((14.0 - LAT) / 2.0))
        self.w_north_as = self.w_arabian * 1.0 / (1.0 + np.exp((18.0 - LAT) / 2.5))

        # Distance to the nearest land point, in degrees - controls upwelling and
        # is handed to the model as a static predictor.
        self.dist_coast = self._distance_to_land()

        # --- archetype blending weights ------------------------------------
        # South->north ramp, Arabian Sea->Bay ramp, and an equatorial override.
        w_north = 1.0 / (1.0 + np.exp((16.5 - LAT) / 2.2))
        w_bob = self.w_bengal / np.maximum(self.w_bengal + self.w_arabian, 1e-6)
        w_eq = 1.0 / (1.0 + np.exp((LAT - 9.0) / 1.4))
        weights = {
            "EQ":    w_eq,
            "AS_S":  (1.0 - w_eq) * (1.0 - w_bob) * (1.0 - w_north),
            "AS_N":  (1.0 - w_eq) * (1.0 - w_bob) * w_north,
            "BOB_S": (1.0 - w_eq) * w_bob * (1.0 - w_north),
            "BOB_N": (1.0 - w_eq) * w_bob * w_north,
        }
        wsum = sum(weights.values())
        self._weights = {k: (v / wsum) for k, v in weights.items()}

        # Monthly climatology stacks -> periodic splines in day-of-year.
        self._clim = {
            name: _periodic_spline(self._blend(table))
            for name, table in (("sst", _CLIM_SST), ("sss", _CLIM_SSS),
                                ("mld", _CLIM_MLD), ("d20", _CLIM_D20))
        }

        # --- coastal upwelling cells ----------------------------------------
        # The archetype climatology already carries the basin-scale monsoon
        # cooling; these cells add the sharp coastal signal on top of it.
        # The Somali cell is given a wide offshore reach because the Great Whirl
        # and its filaments advect upwelled water several hundred km offshore.
        coastal = np.exp(-(self.dist_coast / 3.0) ** 2)
        coastal_narrow = np.exp(-(self.dist_coast / 1.5) ** 2)
        self.up_somali = coastal * np.exp(-((LAT - 8.0) / 4.0) ** 2) * np.exp(-((LON - 50.0) / 4.5) ** 2)
        self.up_oman = coastal * np.exp(-((LAT - 19.5) / 3.5) ** 2) * np.exp(-((LON - 58.0) / 3.5) ** 2)
        self.up_swindia = coastal_narrow * np.exp(-((LAT - 11.0) / 3.0) ** 2) * np.exp(-((LON - 75.0) / 1.6) ** 2)

        # Northern Bay winter barrier-layer / temperature-inversion footprint.
        self.w_inversion = self.w_north_bob * 1.0 / (1.0 + np.exp((16.0 - LAT) / 1.5))

    def _blend(self, table: dict[str, list[float]]) -> np.ndarray:
        """Combine per-archetype monthly values into a ``(12, nlat, nlon)`` stack."""
        out = np.zeros((12,) + self.domain.shape, dtype="float64")
        for name in _ARCHETYPES:
            vals = np.asarray(table[name], dtype="float64")
            out += vals[:, None, None] * self._weights[name][None]
        return out

    def _distance_to_land(self) -> np.ndarray:
        """Great-circle distance (degrees) from each node to the nearest land node."""
        from scipy.ndimage import distance_transform_edt
        # Cell diagonal in degrees is anisotropic in km, but a degree metric is
        # adequate for shaping coastal cells at 0.25deg.
        d_px = distance_transform_edt(self.mask)
        out = (d_px * self.domain.resolution).astype("float64")
        out[~self.mask] = 0.0
        return out

    # -- seasonal helpers ---------------------------------------------------
    @staticmethod
    def _season(doy: int) -> dict[str, float]:
        """Scalar seasonal indices for day-of-year ``doy``."""
        th = 2.0 * np.pi * doy / 365.25
        # SW monsoon window: ramps up in June, peaks late July, decays by late September.
        sw = float(np.exp(-((doy - 205) / 42.0) ** 2))
        # NE monsoon: December-February.
        ne = float(np.exp(-((((doy - 20) + 182) % 365 - 182) / 45.0) ** 2))
        return {"th": th, "sw": sw, "ne": ne}

    def _winds(self, doy: int) -> tuple[np.ndarray, np.ndarray]:
        """Monsoonal 10 m wind with the Findlater (Somali) jet."""
        s = self._season(doy)
        LAT, LON = self.LAT, self.LON
        # South-westerlies during the SW monsoon, north-easterlies in winter.
        u = 7.6 * s["sw"] - 4.6 * s["ne"]
        v = 4.8 * s["sw"] - 3.3 * s["ne"]
        # Findlater jet: intense low-level jet along the Somali/Omani coast.
        jet = np.exp(-((LAT - 12.0) / 7.0) ** 2) * np.exp(-((LON - 55.0) / 9.0) ** 2)
        u = u + 9.0 * s["sw"] * jet
        v = v + 7.0 * s["sw"] * jet
        # Weaker, more easterly flow over the Bay.
        u = u - 2.0 * s["sw"] * self.w_bengal
        nz = self._ar["wind"]
        return (u + 2.6 * nz[0]), (v + 2.2 * nz[1])

    def _advance_noise(self) -> None:
        for k, rho in self._ar_rho.items():
            sc = self._ar_scale_px[k]
            if k == "wind":
                new = np.stack([_smooth_noise(self.rng, self.domain.shape, sc) for _ in range(2)])
            else:
                new = _smooth_noise(self.rng, self.domain.shape, sc)
            self._ar[k] = rho * self._ar[k] + np.sqrt(1.0 - rho ** 2) * new

    # -- one day ------------------------------------------------------------
    def step(self, day: date) -> tuple[dict[str, np.ndarray], np.ndarray]:
        doy = day.timetuple().tm_yday
        s = self._season(doy)
        self._advance_noise()
        self.eddies.advance()

        LAT = self.LAT
        upwell_season = s["sw"]

        # ---------------- sea level -----------------------------------------
        sla_eddy = self.eddies.sla()
        # Basin-scale annual Rossby/Kelvin signal with westward phase propagation.
        sla_wave = 0.045 * np.sin(s["th"] - np.deg2rad(self.LON - 45.0) * 0.9
                                  - 0.04 * (LAT - 5.0))
        sla_dyn = sla_eddy + sla_wave + 0.022 * self._ar["sla"]

        # ---------------- SST ------------------------------------------------
        sst_clim = self._clim["sst"](doy)
        sst = (sst_clim
               - upwell_season * (4.5 * self.up_somali + 3.0 * self.up_oman + 1.5 * self.up_swindia)
               + SST_PER_SLA * sla_eddy
               + 0.42 * self._ar["sst"])
        sst = np.clip(sst, 20.0, 32.5)

        # Steric sea level: thermal expansion of the warm upper layer.  Tied to
        # the SST anomaly, and deliberately *not* coupled to the thermocline.
        sst_anom = sst - sst_clim
        sla_steric = 0.022 * np.sin(s["th"] - 2.0 * np.pi * 70.0 / 365.25) + 0.012 * sst_anom
        sla = sla_dyn + sla_steric

        # ---------------- SSS ------------------------------------------------
        # Bay freshening peaks in the post-monsoon discharge season (Aug-Nov).
        # The archetype tables already carry the seasonal Bay freshening, so only
        # the sharp coastal plume and upwelling salinity signals are added here.
        fresh = float(np.exp(-((doy - 280) / 50.0) ** 2))
        sss = (self._clim["sss"](doy)
               - 0.9 * fresh * self.w_north_bob
               + 0.25 * upwell_season * (self.up_somali + self.up_oman)
               + 0.16 * self._ar["sss"])
        sss = np.clip(sss, 28.0, 37.5)

        # ---------------- winds and surface currents --------------------------
        uwnd, vwnd = self._winds(doy)
        wspd = np.hypot(uwnd, vwnd)

        # Geostrophy from the sea-level gradient.
        dy_m = self.domain.resolution * DEG2KM * 1000.0
        dx_m = dy_m * np.cos(np.deg2rad(LAT))
        deta_dy = np.gradient(gaussian_filter(sla, 1.0), axis=0) / dy_m
        deta_dx = np.gradient(gaussian_filter(sla, 1.0), axis=1) / dx_m
        ug = -(G / self.f) * deta_dy
        vg = (G / self.f) * deta_dx

        # Ekman drift: ~2.5% of the wind, rotated 45deg to the right (NH).
        ek = 0.025 * wspd
        ang = np.arctan2(vwnd, uwnd) - np.pi / 4.0
        ue, ve = ek * np.cos(ang), ek * np.sin(ang)

        # Seasonal reversing monsoon currents (Summer/Winter Monsoon Current).
        smc = np.exp(-((LAT - 7.0) / 3.0) ** 2)
        ucur = ug + ue + smc * (0.35 * s["sw"] - 0.30 * s["ne"])
        vcur = vg + ve
        ucur = np.clip(ucur, -2.0, 2.0)
        vcur = np.clip(vcur, -2.0, 2.0)

        # ---------------- mixed layer and thermocline -------------------------
        # Wind-stress driven deepening: tau = rho_a * Cd * U^2, h ~ tau^(1/2).
        tau = RHO_AIR * 1.3e-3 * wspd ** 2
        tau_ref = RHO_AIR * 1.3e-3 * 6.0 ** 2
        # Wind-driven departure from the climatological mixed layer.  The
        # climatology already encodes the monsoon and winter-convection cycle,
        # so only the synoptic wind anomaly acts here.
        mld = self._clim["mld"](doy) * np.clip(tau / tau_ref, 0.4, 2.5) ** 0.22
        mld = mld + 30.0 * sla_dyn + 4.0 * self._ar["mld"]
        mld = np.clip(mld, 10.0, 140.0)

        d20 = self._clim["d20"](doy) + D20_PER_SLA * sla_dyn - 22.0 * upwell_season * (
            self.up_somali + self.up_oman)
        d20 = np.clip(d20, np.maximum(mld + 18.0, 55.0), 255.0)

        # Winter barrier-layer temperature inversion in the northern Bay.
        inv_season = float(np.exp(-((((doy - 15) + 182) % 365 - 182) / 40.0) ** 2))
        inv_amp = 2.10 * inv_season * self.w_inversion

        thetao = self._profile(sst, mld, d20, inv_amp)

        # Deep water-mass variability: large scale, long memory, and invisible
        # from the surface, so it bounds achievable skill below ~300 m.
        z = np.asarray(STANDARD_DEPTHS, dtype="float64")
        w_deep = 1.0 / (1.0 + np.exp(-(z - DEEP_NOISE_Z0) / DEEP_NOISE_DZ))
        thetao = thetao + DEEP_NOISE_AMP * w_deep[:, None, None] * self._ar["deep"][None]

        surf = {
            "sst": sst, "sss": sss, "sla": sla,
            "ucur": ucur, "vcur": vcur, "uwnd": uwnd, "vwnd": vwnd,
        }
        # Land is undefined everywhere.
        for k in surf:
            surf[k] = np.where(self.mask, surf[k], np.nan).astype("float32")
        thetao = np.where(self.mask[None], thetao, np.nan).astype("float32")

        aux = {"mld": np.where(self.mask, mld, np.nan).astype("float32"),
               "d20": np.where(self.mask, d20, np.nan).astype("float32")}
        return surf | {"_aux": aux}, thetao

    # -- vertical structure -------------------------------------------------
    def _profile(self, sst: np.ndarray, mld: np.ndarray, d20: np.ndarray,
                 inv_amp: np.ndarray) -> np.ndarray:
        """Assemble T(z) on the standard levels from (SST, MLD, D20).

        The interior is built by **vertical coordinate stretching**: a shallower
        or deeper 20 degC isotherm squeezes or stretches the whole reference
        thermocline about the surface, which is how isotherm displacement
        actually works and keeps the vertical gradient structure physical
        (maximum just below the mixed layer, weakening downwards).  A linear
        superposition of an anomaly onto the reference profile, by contrast,
        puts a spurious kink at the isotherm depth.

            z* = z * [1 + (D20_ref/D20 - 1) * w(z)]

        ``w`` is 1 above 250 m and tapers to 0 by ~900 m, because
        thermocline-displacement anomalies are surface-intensified and the deep
        ocean varies by only a few tenths of a degree.  ``z*`` is forced
        monotone so the resulting profile can never invert numerically.

        On top of the interior:

        * the mixed layer is held at SST, blended into the interior with a
          32 m e-folding scale so both value and gradient stay continuous;
        * a Gaussian bump below the mixed-layer base represents the northern Bay
          of Bengal winter **barrier-layer inversion** - the one place where real
          profiles are genuinely non-monotonic.
        """
        z = np.asarray(STANDARD_DEPTHS, dtype="float64")

        # Stretching factor, clipped to keep the mapping well conditioned.
        ratio = np.clip(D20_REF / np.maximum(d20, 40.0), 0.44, 2.05)

        # Depth taper w(z): full stretching in the thermocline, none in the abyss.
        taper = np.exp(-(np.maximum(z - STRETCH_FULL_M, 0.0) / STRETCH_TAPER_M) ** 2)

        # z_star[k] has the shape of the horizontal grid for each level k.
        z_star = z[:, None, None] * (1.0 + (ratio[None] - 1.0) * taper[:, None, None])
        # Guarantee a monotone vertical coordinate (cheap safety net on top of
        # the analytic bound provided by the clip above).
        z_star = np.maximum.accumulate(z_star, axis=0)

        interior = _t_reference(z_star)

        # Mixed-layer offset, blended downwards from the mixed-layer base.
        # MLD never exceeds 140 m, so the taper there is exactly 1 and the
        # stretched mixed-layer base is simply ``mld * ratio``.
        offset = sst - _t_reference(mld * ratio)

        below_ml = np.maximum(z[:, None, None] - mld[None], 0.0)
        blend = np.exp(-below_ml / ML_BLEND_M)
        prof = np.where(z[:, None, None] <= mld[None], sst[None], interior + offset[None] * blend)

        # Barrier-layer temperature inversion just beneath the mixed layer.
        if np.any(inv_amp > 0.0):
            bump = np.exp(-((z[:, None, None] - (mld[None] + 22.0)) / 26.0) ** 2)
            prof = prof + inv_amp[None] * bump

        return prof

    # -- driver -------------------------------------------------------------
    def run(self, days: list[date]):
        """Yield ``(day, surface_dict, thetao)`` for each day, in order.

        The simulator is sequential (eddies propagate, AR(1) noise has memory),
        so days must be consumed in chronological order.
        """
        for i, day in enumerate(days):
            surf, thetao = self.step(day)
            if i % 200 == 0:
                log.info("simulated %s (%d/%d)", day.isoformat(), i + 1, len(days))
            yield day, surf, thetao


def as_date(value: str | date) -> date:
    """Coerce a config value to ``datetime.date``.

    YAML parses an unquoted ``2021-01-01`` into a ``date`` but a quoted one into
    a ``str``, and CLI overrides go through the same parser - so both forms
    reach us and both have to work.
    """
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def date_range(start: str | date, end: str | date) -> list[date]:
    d0, d1 = as_date(start), as_date(end)
    if d1 < d0:
        raise ValueError(f"end date {d1} precedes start date {d0}")
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
