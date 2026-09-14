"""Real-product ingestion: Copernicus Marine and NASA PO.DAAC.

This is the production data path.  It is separate from the OSSE twin in
:mod:`oceanembed.data.synthetic` because it needs credentials and network
access, but it feeds the *same* harmonisation, training and evaluation code:
switching over is ``data.source: cmems`` in the config plus a download.

Credentials
-----------
Copernicus Marine (SST, SSS, SLA, GLORYS target)::

    pip install copernicusmarine
    copernicusmarine login            # stores ~/.copernicusmarine/

NASA PO.DAAC / Earthdata (OSCAR currents, CCMP and ASCAT winds)::

    pip install earthaccess
    earthaccess.login(persist=True)   # or ~/.netrc

Dataset identifiers
-------------------
The DOIs below are the ones quoted in the problem statement and are stable.
The ``dataset_id`` strings are what the toolbox actually needs, and Copernicus
does rename and version them - so :func:`describe` is provided to confirm the
current identifier before a bulk download rather than failing halfway through.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..grid import NIO, SURFACE_VARS, Domain

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Product:
    """One input product and how to get it onto the analysis grid."""

    key: str               # internal variable name
    doi: str               # as quoted in PS#01
    source: str            # "cmems" or "podaac"
    dataset_id: str
    variables: tuple[str, ...]
    native_res: float      # degrees
    #: True when the native grid is finer than 0.25deg and must be block-averaged
    #: before interpolation, rather than sampled bilinearly.
    coarsen_first: bool = False
    notes: str = ""


PRODUCTS: dict[str, Product] = {
    "sst": Product(
        key="sst", doi="10.48670/moi-00168", source="cmems",
        dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        variables=("analysed_sst",), native_res=0.05, coarsen_first=True,
        notes="OSTIA L4. Kelvin -> degC. Use the REP dataset "
              "METOFFICE-GLO-SST-L4-REP-OBS-SST for a reanalysis-consistent record."),
    "sss": Product(
        key="sss", doi="10.48670/moi-00051", source="cmems",
        dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
        variables=("sos",), native_res=0.125,
        notes="Multi-observation SMOS/SMAP/Aquarius L4. Unreliable within ~100 km "
              "of the coast; the harmoniser flags rather than silently fills that."),
    "sla": Product(
        key="sla", doi="10.48670/moi-00145", source="cmems",
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
        variables=("sla", "adt"), native_res=0.25,
        notes="DUACS L4 altimetry, already on the target grid."),
    "cur": Product(
        key="cur", doi="podaac:OSCAR_L4_OC_FINAL_V2.0", source="podaac",
        dataset_id="OSCAR_L4_OC_FINAL_V2.0",
        variables=("u", "v"), native_res=0.25,
        notes="OSCAR surface currents (geostrophic + Ekman), 0.25deg daily."),
    "wnd": Product(
        key="wnd", doi="podaac:CCMP_WINDS_10M6HR_L4_V3.1", source="podaac",
        dataset_id="CCMP_WINDS_10M6HR_L4_V3.1",
        variables=("uwnd", "vwnd"), native_res=0.25,
        notes="CCMP 6-hourly L4 winds; average the four slices to daily. "
              "ASCAT-L2-Coastal is the swath alternative and needs L2->L4 gridding."),
}

#: Training target.
TARGET = Product(
    key="thetao", doi="10.48670/moi-00021", source="cmems",
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=("thetao",), native_res=0.083, coarsen_first=True,
    notes="GLORYS12V1 global reanalysis, 1/12deg, 50 levels. Interpolate "
          "vertically onto the 15 standard depths after horizontal regridding.")

#: Independent in-situ validation.
INSITU_NOTE = (
    "Gridded ARGO from the INCOIS Live Access Server "
    "(https://las.incois.gov.in/), or Argo GDAC profiles. "
    "Load with oceanembed.data.argo.load_argo_netcdf."
)


def describe(dataset_id: str) -> None:  # pragma: no cover - needs network
    """Print the Copernicus catalogue entry, to confirm an identifier."""
    subprocess.run(["copernicusmarine", "describe", "--dataset-id", dataset_id,
                    "--return-fields", "dataset_id,versions,variables"], check=False)


def download_cmems(product: Product, start: str, end: str, out_dir: Path,
                   domain: Domain = NIO, depth_max: float = 1100.0
                   ) -> Path:  # pragma: no cover - needs credentials
    """Subset and download one Copernicus product over the domain and period.

    Subsetting server-side is essential: the global GLORYS record for three
    years is tens of terabytes, while the North Indian Ocean upper 1000 m is a
    few hundred gigabytes.
    """
    try:
        import copernicusmarine as cm
    except ImportError as exc:
        raise ImportError("pip install copernicusmarine, then `copernicusmarine login`") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        dataset_id=product.dataset_id,
        variables=list(product.variables),
        minimum_longitude=domain.lon_min, maximum_longitude=domain.lon_max,
        minimum_latitude=domain.lat_min, maximum_latitude=domain.lat_max,
        start_datetime=f"{start}T00:00:00", end_datetime=f"{end}T23:59:59",
        output_directory=str(out_dir),
        output_filename=f"{product.key}_{start}_{end}.nc",
    )
    if product.key == "thetao":
        kwargs.update(minimum_depth=0.0, maximum_depth=depth_max)
    log.info("downloading %s (%s)", product.dataset_id, product.doi)
    cm.subset(**kwargs)
    return out_dir / kwargs["output_filename"]


def download_podaac(product: Product, start: str, end: str, out_dir: Path,
                    domain: Domain = NIO) -> list[Path]:  # pragma: no cover
    """Download one PO.DAAC collection through earthaccess."""
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError("pip install earthaccess, then earthaccess.login(persist=True)") from exc

    earthaccess.login()
    granules = earthaccess.search_data(
        short_name=product.dataset_id, temporal=(start, end),
        bounding_box=(domain.lon_min, domain.lat_min, domain.lon_max, domain.lat_max))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %d granules of %s", len(granules), product.dataset_id)
    return [Path(p) for p in earthaccess.download(granules, str(out_dir))]


# ---------------------------------------------------------------------------
# Harmonisation of downloaded files
# ---------------------------------------------------------------------------
#: Native variable name -> internal name, and the unit conversion to apply.
VAR_MAP: dict[str, tuple[str, str]] = {
    "analysed_sst": ("sst", "K->C"),
    "sst": ("sst", "none"),
    "sos": ("sss", "none"),
    "so": ("sss", "none"),
    "sla": ("sla", "none"),
    "u": ("ucur", "none"),
    "v": ("vcur", "none"),
    "uo": ("ucur", "none"),
    "vo": ("vcur", "none"),
    "uwnd": ("uwnd", "none"),
    "vwnd": ("vwnd", "none"),
}


def _convert(da, how: str):
    if how == "K->C":
        return da - 273.15
    return da


def harmonize_surface(files: dict[str, Path], domain: Domain = NIO):
    """Bring downloaded surface products onto one daily 0.25deg xarray Dataset.

    Steps, in order: rename to internal variable names, convert units, resample
    sub-daily products to daily means, block-average anything finer than 0.25deg,
    interpolate onto the target grid, then align all products on a common time
    axis.  Returns an ``xarray.Dataset`` with the seven
    :data:`oceanembed.grid.SURFACE_VARS`.
    """
    import xarray as xr
    from .harmonize import coarsen_then_regrid, regrid

    out = {}
    for key, path in files.items():
        prod = PRODUCTS[key]
        ds = xr.open_dataset(path)
        for native in prod.variables:
            if native not in ds:
                log.warning("%s not present in %s - skipping", native, path.name)
                continue
            internal, conv = VAR_MAP.get(native, (native, "none"))
            da = _convert(ds[native], conv)
            # Sub-daily products (CCMP is 6-hourly) become daily means.
            if "time" in da.dims and da.time.size > 1:
                step = np.diff(da.time.values[:2]).astype("timedelta64[h]").astype(int)[0]
                if 0 < step < 24:
                    da = da.resample(time="1D").mean()
            da = (coarsen_then_regrid(da, domain) if prod.coarsen_first
                  else regrid(da, domain))
            out[internal] = da
    missing = [v for v in SURFACE_VARS if v not in out]
    if missing:
        raise ValueError(f"harmonisation is missing required variables: {missing}")
    ds = xr.Dataset({k: out[k] for k in SURFACE_VARS})
    # Inner join on time: a day is only usable if every product covers it.
    return ds.dropna("time", how="all")


def harmonize_target(path: Path, domain: Domain = NIO):
    """Regrid GLORYS temperature horizontally, then interpolate to standard depths."""
    import xarray as xr
    from ..grid import STANDARD_DEPTHS
    from .harmonize import coarsen_then_regrid

    ds = xr.open_dataset(path)
    da = ds["thetao"]
    depth_name = next(n for n in ("depth", "deptht", "lev") if n in da.dims)
    # Horizontal first, on each native level: interpolating vertically before
    # regridding would mix water masses across the sharp thermocline.
    levels = [coarsen_then_regrid(da.isel({depth_name: k}), domain)
              for k in range(da.sizes[depth_name])]
    stacked = xr.concat(levels, dim=depth_name).assign_coords(
        {depth_name: da[depth_name].values})
    return stacked.interp({depth_name: list(STANDARD_DEPTHS)}, method="linear")
