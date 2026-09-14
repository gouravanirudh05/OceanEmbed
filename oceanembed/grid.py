
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard depth levels (metres).  Mandated by the problem statement; these are
# also the classic WOA/Levitus standard levels for the upper 1000 m.
# ---------------------------------------------------------------------------
STANDARD_DEPTHS: tuple[int, ...] = (
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000,
)
N_DEPTH = len(STANDARD_DEPTHS)

#: Surface predictor channels, in canonical order.  The model always receives
#: channels in this order; renaming/reordering here propagates everywhere.
SURFACE_VARS: tuple[str, ...] = ("sst", "sss", "sla", "ucur", "vcur", "uwnd", "vwnd")
N_SURFACE = len(SURFACE_VARS)

#: Long names + CF units used when writing NetCDF.
VAR_ATTRS: dict[str, dict[str, str]] = {
    "sst":  {"long_name": "sea surface temperature",            "units": "degree_Celsius", "standard_name": "sea_surface_temperature"},
    "sss":  {"long_name": "sea surface salinity",               "units": "1e-3",           "standard_name": "sea_surface_salinity"},
    "sla":  {"long_name": "sea level anomaly",                  "units": "m",              "standard_name": "sea_surface_height_above_sea_level"},
    "ucur": {"long_name": "eastward surface current",           "units": "m s-1",          "standard_name": "eastward_sea_water_velocity"},
    "vcur": {"long_name": "northward surface current",          "units": "m s-1",          "standard_name": "northward_sea_water_velocity"},
    "uwnd": {"long_name": "eastward wind at 10 m",              "units": "m s-1",          "standard_name": "eastward_wind"},
    "vwnd": {"long_name": "northward wind at 10 m",             "units": "m s-1",          "standard_name": "northward_wind"},
    "thetao": {"long_name": "sea water potential temperature",  "units": "degree_Celsius", "standard_name": "sea_water_potential_temperature"},
}


@dataclass(frozen=True)
class Domain:
    """Regular lat/lon analysis grid.

    Bounds are inclusive of both endpoints, matching the convention used by the
    CMEMS/DUACS 0.25deg products (cell centres on the 0.125deg offset grid are
    resampled onto these nodes during harmonisation).
    """

    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0
    resolution: float = 0.25
    name: str = "north_indian_ocean"

    # ---- axes ------------------------------------------------------------
    @property
    def lat(self) -> np.ndarray:
        n = int(round((self.lat_max - self.lat_min) / self.resolution)) + 1
        return (self.lat_min + self.resolution * np.arange(n)).astype("float32")

    @property
    def lon(self) -> np.ndarray:
        n = int(round((self.lon_max - self.lon_min) / self.resolution)) + 1
        return (self.lon_min + self.resolution * np.arange(n)).astype("float32")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lat.size, self.lon.size)

    @property
    def n_lat(self) -> int:
        return self.lat.size

    @property
    def n_lon(self) -> int:
        return self.lon.size

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(LAT, LON)`` 2-D arrays of shape :attr:`shape`."""
        return np.meshgrid(self.lat, self.lon, indexing="ij")

    def subset(self, name: str) -> "Domain":
        """Named sub-basins used for the Bay of Bengal / Arabian Sea PoC."""
        try:
            b = SUBREGIONS[name]
        except KeyError as exc:  # pragma: no cover - guard for typos
            raise KeyError(f"unknown subregion {name!r}; have {sorted(SUBREGIONS)}") from exc
        return Domain(*b, resolution=self.resolution, name=name)


#: (lat_min, lat_max, lon_min, lon_max) for regions of operational interest.
SUBREGIONS: dict[str, tuple[float, float, float, float]] = {
    "bay_of_bengal":   (5.0, 22.0, 80.0, 95.0),
    "arabian_sea":     (5.0, 25.0, 55.0, 78.0),
    "equatorial":      (5.0, 10.0, 50.0, 100.0),
    "somali_upwelling":(5.0, 12.0, 45.0, 55.0),
    "full":            (5.0, 30.0, 45.0, 105.0),
}

NIO = Domain()


# ---------------------------------------------------------------------------
# Land / sea mask
# ---------------------------------------------------------------------------
# The mask is rasterised once from a coastline polygon set and cached to .npy.
# Preference order:
#   1. Natural Earth 50 m land polygons (accurate, fetched by scripts/fetch_coastline.py)
#   2. Analytic fallback polygons digitised at ~1deg (keeps the repo runnable offline)

_MASK_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"

# Coarse hand-digitised coastlines (lon, lat), closed rings, used only when the
# Natural Earth download is unavailable.  Accurate to roughly one grid cell,
# which is sufficient for masking a 0.25deg basin-scale analysis.
_FALLBACK_LAND: dict[str, list[tuple[float, float]]] = {
    # Horn of Africa + East African coast
    "africa": [
        (40.0, 0.0), (41.5, 2.0), (44.0, 4.5), (48.5, 8.0), (51.4, 10.5),
        (51.0, 12.0), (48.0, 11.5), (44.0, 10.4), (43.0, 12.7), (40.0, 15.0),
        (37.0, 18.0), (35.0, 23.0), (32.0, 31.0), (32.0, 0.0),
    ],
    # Arabian peninsula
    "arabia": [
        (43.0, 12.7), (44.0, 12.8), (47.0, 14.0), (52.2, 15.6), (55.0, 17.0),
        (56.3, 18.0), (59.8, 22.5), (58.0, 23.8), (56.4, 25.0), (56.3, 26.4),
        (54.0, 24.3), (51.5, 24.3), (50.8, 25.5), (48.5, 28.5), (47.5, 30.0),
        (40.0, 30.0), (40.0, 12.0),
    ],
    # Iran / Pakistan / NW India margin
    "makran": [
        (47.5, 30.0), (50.0, 30.0), (57.0, 25.4), (61.6, 25.0), (66.5, 25.0),
        (67.5, 23.8), (70.0, 22.8), (72.6, 21.5), (72.9, 20.0), (73.5, 16.0),
        (76.0, 30.0), (47.5, 30.0),
    ],
    # Peninsular India + Gangetic plain
    "india": [
        (72.9, 20.0), (73.5, 16.0), (74.9, 12.5), (76.5, 8.9), (77.5, 8.1),
        (79.9, 10.3), (80.3, 13.1), (81.2, 16.3), (84.0, 19.0), (86.5, 20.7),
        (88.1, 21.6), (89.1, 22.0), (90.6, 22.3), (91.5, 22.9), (92.3, 25.0),
        (95.0, 27.0), (97.0, 30.0), (72.0, 30.0), (72.6, 21.5),
    ],
    "sri_lanka": [
        (79.8, 9.4), (81.2, 8.5), (81.9, 7.2), (81.7, 6.4), (80.4, 5.9),
        (79.7, 6.9), (79.8, 9.4),
    ],
    # Myanmar / Thai-Malay peninsula
    "indochina": [
        (92.3, 25.0), (94.0, 21.0), (94.5, 18.0), (96.5, 16.5), (98.0, 14.0),
        (99.5, 11.0), (100.3, 7.5), (103.5, 5.5), (104.3, 1.4), (105.0, 1.4),
        (105.0, 25.0), (92.3, 25.0),
    ],
    # Sumatra (only its northern tip enters the 5degN cut-off)
    "sumatra": [
        (95.2, 5.6), (97.5, 5.2), (99.5, 3.6), (102.0, 1.5), (105.0, -1.0),
        (105.0, 5.0), (98.0, 5.0), (95.2, 5.6),
    ],
    # Andaman & Nicobar chain, approximated as a narrow ridge
    "andaman": [
        (92.5, 13.7), (93.1, 13.5), (93.0, 10.5), (92.8, 6.7), (93.6, 7.0),
        (93.9, 10.6), (93.6, 13.8), (92.5, 13.7),
    ],
    "lakshadweep": [(72.6, 11.6), (73.1, 11.6), (73.1, 10.8), (72.6, 10.8)],
}


def _rasterise(polygons: dict[str, list[tuple[float, float]]], domain: Domain) -> np.ndarray:
    """Point-in-polygon test of every grid node against every land ring."""
    from matplotlib.path import Path as MplPath

    LAT, LON = domain.meshgrid()
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    land = np.zeros(pts.shape[0], dtype=bool)
    for ring in polygons.values():
        land |= MplPath(np.asarray(ring)).contains_points(pts)
    return land.reshape(domain.shape)


def _rasterise_natural_earth(shp: Path, domain: Domain) -> np.ndarray:
    """Rasterise a Natural Earth land shapefile onto ``domain``."""
    import shapefile  # pyshp
    from matplotlib.path import Path as MplPath

    LAT, LON = domain.meshgrid()
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    land = np.zeros(pts.shape[0], dtype=bool)
    reader = shapefile.Reader(str(shp))
    # Only test polygons whose bounding box intersects the domain.
    for shape in reader.shapes():
        x0, y0, x1, y1 = shape.bbox
        if x1 < domain.lon_min or x0 > domain.lon_max or y1 < domain.lat_min or y0 > domain.lat_max:
            continue
        parts = list(shape.parts) + [len(shape.points)]
        for i in range(len(parts) - 1):
            ring = np.asarray(shape.points[parts[i]:parts[i + 1]])
            if ring.shape[0] < 3:
                continue
            land |= MplPath(ring).contains_points(pts)
    return land.reshape(domain.shape)


@lru_cache(maxsize=8)
def sea_mask(domain: Domain = NIO, use_cache: bool = True) -> np.ndarray:
    """Boolean array, ``True`` over ocean.

    Cached to ``data/cache/seamask_<name>_<res>.npy`` so the polygon raster is
    only computed once.
    """
    tag = f"{domain.name}_{domain.lat_min}_{domain.lat_max}_{domain.lon_min}_{domain.lon_max}_{domain.resolution}"
    cache = _MASK_CACHE / f"seamask_{tag}.npy"
    if use_cache and cache.exists():
        return np.load(cache)

    shp = _MASK_CACHE / "ne_50m_land" / "ne_50m_land.shp"
    if shp.exists():
        log.info("rasterising land mask from Natural Earth 50m polygons")
        try:
            land = _rasterise_natural_earth(shp, domain)
        except ImportError:
            log.warning("pyshp not installed - falling back to analytic coastline")
            land = _rasterise(_FALLBACK_LAND, domain)
    else:
        log.info("Natural Earth polygons not found - using analytic coastline")
        land = _rasterise(_FALLBACK_LAND, domain)

    mask = ~land
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, mask)
    log.info("sea mask: %d/%d ocean points (%.1f%%)", mask.sum(), mask.size, 100 * mask.mean())
    return mask


def cell_area_km2(domain: Domain = NIO) -> np.ndarray:
    """Area of each grid cell in km^2 (cos-latitude weighting)."""
    r_earth = 6371.0
    dlat = np.deg2rad(domain.resolution)
    dlon = np.deg2rad(domain.resolution)
    lat = np.deg2rad(domain.lat)[:, None]
    return (r_earth ** 2 * dlon * dlat * np.cos(lat) * np.ones((1, domain.n_lon))).astype("float32")


def depth_layer_thickness() -> np.ndarray:
    """Thickness (m) attributed to each standard level, for heat-content integrals."""
    z = np.asarray(STANDARD_DEPTHS, dtype="float64")
    edges = np.concatenate([[z[0]], 0.5 * (z[1:] + z[:-1]), [z[-1]]])
    return np.diff(edges).astype("float32")
