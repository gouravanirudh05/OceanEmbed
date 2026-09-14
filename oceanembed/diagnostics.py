"""Derived upper-ocean diagnostics computed from a reconstructed profile.

These are the quantities operational users actually act on, so the
reconstruction is scored on them as well as on raw temperature:

``mld``   mixed layer depth, de Boyer Montegut 0.2 degC criterion referenced to 10 m
``d20``   depth of the 20 degC isotherm - the standard thermocline/heat-content proxy
``d26``   depth of the 26 degC isotherm - the layer that can sustain a cyclone
``ohc``   ocean heat content of the upper 300 m
``tchp``  tropical cyclone heat potential, the heat stored above the 26 degC
          isotherm.  This is the single most used subsurface predictor of
          cyclone rapid intensification in the Bay of Bengal, and it cannot be
          obtained from SST alone - which is precisely the operational argument
          for reconstructing the subsurface at all.

All functions take ``T`` of shape ``(K, ...)`` on :data:`STANDARD_DEPTHS` and
return fields shaped like ``T[0]``.
"""
from __future__ import annotations

import numpy as np

from .grid import STANDARD_DEPTHS

RHO_CP = 1025.0 * 3985.0     # J m-3 K-1, seawater volumetric heat capacity
Z = np.asarray(STANDARD_DEPTHS, dtype="float64")


def isotherm_depth(T: np.ndarray, value: float) -> np.ndarray:
    """Depth of the deepest crossing of the ``value`` isotherm, by linear interpolation.

    Two degenerate cases are resolved deliberately, because both occur in this
    basin and both would otherwise poison the statistics:

    * **The isotherm outcrops** - the surface is already colder than ``value``,
      which happens for the 26 degC isotherm in the northern Arabian Sea in
      winter. The depth is reported as **0**, the continuous limit as
      SST approaches ``value`` from above, rather than NaN. NaN would silently
      drop exactly those points from the error statistics, hiding the cases
      where the reconstruction put the surface on the wrong side of the
      threshold.
    * **The column never cools past** ``value`` - the depth is reported as the
      deepest level resolved (1000 m).
    """
    t = np.asarray(T, dtype="float64")
    above = t >= value                                   # (K, ...)
    # Crossing between level k and k+1 where above[k] and not above[k+1].
    cross = above[:-1] & ~above[1:]
    out = np.full(t.shape[1:], np.nan)
    # Walk downwards so the deepest crossing wins.
    for k in range(t.shape[0] - 1):
        t0, t1 = t[k], t[k + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (t0 - value) / (t0 - t1)
        z = Z[k] + frac * (Z[k + 1] - Z[k])
        out = np.where(cross[k], z, out)
    # Column entirely warmer than `value`: isotherm is below the deepest level.
    out = np.where(above.all(axis=0), Z[-1], out)
    # Isotherm outcrops: zero-thickness layer.
    out = np.where(~np.isfinite(out) & ~above[0], 0.0, out)
    # Anything still unset had no valid data at all (land).
    return np.where(np.isfinite(t[0]), out, np.nan).astype("float32")


def mixed_layer_depth(T: np.ndarray, delta: float = 0.2) -> np.ndarray:
    """de Boyer Montegut MLD: first depth where T drops ``delta`` below T(10 m).

    Referenced to 10 m rather than the surface so the estimate is not thrown off
    by the diurnal warm layer, following the standard criterion.
    """
    t = np.asarray(T, dtype="float64")
    k_ref = int(np.argmin(np.abs(Z - 10.0)))
    return _threshold_depth(t, t[k_ref] - delta, k_ref)


def _threshold_depth(t: np.ndarray, target: np.ndarray, k_start: int) -> np.ndarray:
    """First depth below ``k_start`` where the column falls to ``target``."""
    out = np.full(t.shape[1:], np.nan)
    found = np.zeros(t.shape[1:], dtype=bool)
    for k in range(k_start, t.shape[0] - 1):
        t0, t1 = t[k], t[k + 1]
        hit = (~found) & (t0 >= target) & (t1 < target)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (t0 - target) / (t0 - t1)
        z = Z[k] + frac * (Z[k + 1] - Z[k])
        out = np.where(hit, z, out)
        found |= hit
    # Never reached: the mixed layer is deeper than the profile we resolve.
    out = np.where(found, out, Z[-1])
    out = np.where(np.isfinite(t[k_start]), out, np.nan)
    return out.astype("float32")


def _layer_integral(T: np.ndarray, z_top: float, z_bot) -> np.ndarray:
    """Trapezoidal integral of ``T`` dz from ``z_top`` to ``z_bot`` (array or scalar)."""
    t = np.asarray(T, dtype="float64")
    zb = np.broadcast_to(np.asarray(z_bot, dtype="float64"), t.shape[1:])
    acc = np.zeros(t.shape[1:])
    for k in range(t.shape[0] - 1):
        z0, z1 = Z[k], Z[k + 1]
        if z1 <= z_top:
            continue
        lo = max(z0, z_top)
        hi = np.minimum(z1, zb)
        thick = np.clip(hi - lo, 0.0, None)
        # Values of T at the (possibly clipped) sub-layer edges.
        f0 = t[k] + (t[k + 1] - t[k]) * (lo - z0) / (z1 - z0)
        f1 = t[k] + (t[k + 1] - t[k]) * (hi - z0) / (z1 - z0)
        acc = acc + 0.5 * (f0 + f1) * thick
    return acc


def ocean_heat_content(T: np.ndarray, z_bot: float = 300.0) -> np.ndarray:
    """Heat content of the upper ``z_bot`` metres, in GJ m-2."""
    return (RHO_CP * _layer_integral(T, 0.0, z_bot) / 1e9).astype("float32")


def cyclone_heat_potential(T: np.ndarray) -> np.ndarray:
    """Tropical cyclone heat potential above the 26 degC isotherm, in kJ cm-2.

    Zero where SST is already below 26 degC, since there is then no layer capable
    of sustaining a cyclone.
    """
    t = np.asarray(T, dtype="float64")
    d26 = isotherm_depth(t, 26.0).astype("float64")
    d26 = np.where(np.isfinite(d26), d26, 0.0)
    integral = _layer_integral(t - 26.0, 0.0, d26)
    tchp = RHO_CP * np.clip(integral, 0.0, None) / 1e7      # J m-2 -> kJ cm-2
    return np.where(t[0] >= 26.0, tchp, 0.0).astype("float32")


DIAGNOSTICS = {
    "mld": mixed_layer_depth,
    "d20": lambda T: isotherm_depth(T, 20.0),
    "d26": lambda T: isotherm_depth(T, 26.0),
    "ohc300": lambda T: ocean_heat_content(T, 300.0),
    "tchp": cyclone_heat_potential,
}

DIAG_UNITS = {"mld": "m", "d20": "m", "d26": "m", "ohc300": "GJ m-2", "tchp": "kJ cm-2"}


def all_diagnostics(T: np.ndarray) -> dict[str, np.ndarray]:
    return {k: f(T) for k, f in DIAGNOSTICS.items()}
