"""Physical and numerical invariants that must hold regardless of training.

These are the checks that would catch a silent regression in the science: a
profile that inverts, a diagnostic that disagrees with its own definition, a
normalisation that leaks test data into training, or a coordinate convention
that flips.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from oceanembed.diagnostics import (Z, cyclone_heat_potential, isotherm_depth,
                                    mixed_layer_depth, ocean_heat_content)
from oceanembed.grid import (NIO, STANDARD_DEPTHS, cell_area_km2,
                             depth_layer_thickness, sea_mask)


# --- grid -----------------------------------------------------------------
def test_grid_matches_problem_statement():
    assert NIO.lat[0] == 5.0 and NIO.lat[-1] == 30.0
    assert NIO.lon[0] == 45.0 and NIO.lon[-1] == 105.0
    assert NIO.resolution == 0.25
    assert NIO.shape == (101, 241)
    assert STANDARD_DEPTHS == (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000)


def test_layer_thickness_sums_to_range():
    assert depth_layer_thickness().sum() == pytest.approx(1000.0)


@pytest.mark.parametrize("lat,lon,is_sea", [
    (15.0, 88.0, True),    # central Bay of Bengal
    (15.0, 65.0, True),    # central Arabian Sea
    (22.0, 78.0, False),   # central India
    (22.0, 45.0, False),   # Saudi Arabia
    (7.5, 80.75, False),   # Sri Lanka
    (7.0, 47.0, False),    # Somalia
])
def test_land_mask_known_points(lat, lon, is_sea):
    m = sea_mask()
    i = int(round((lat - 5.0) / 0.25))
    j = int(round((lon - 45.0) / 0.25))
    assert bool(m[i, j]) is is_sea


def test_ocean_area_plausible():
    # North Indian Ocean north of 5degN is roughly 9-11 million km^2.
    area = cell_area_km2()[sea_mask()].sum()
    assert 7.5e6 < area < 1.2e7


# --- diagnostics ----------------------------------------------------------
def _linear_column(t_top=30.0, t_bot=10.0):
    return (np.linspace(t_top, t_bot, len(STANDARD_DEPTHS))[:, None, None]
            * np.ones((1, 2, 2)))


def test_isotherm_depth_exact_on_linear_column():
    T = _linear_column()
    got = isotherm_depth(T, 20.0)
    want = np.interp(20.0, T[::-1, 0, 0], Z[::-1])
    assert got[0, 0] == pytest.approx(want, abs=1e-3)


def test_isotherm_depth_saturates_when_never_crossed():
    T = np.full((len(STANDARD_DEPTHS), 2, 2), 28.0)
    assert np.all(isotherm_depth(T, 20.0) == Z[-1])


def test_isotherm_depth_zero_when_outcropping():
    """Northern Arabian Sea winter: the 26 degC isotherm outcrops.

    It must report 0, not NaN - a NaN would drop these points from the error
    statistics and hide reconstructions that put the surface on the wrong side
    of the threshold.
    """
    T = np.linspace(24.0, 8.0, len(STANDARD_DEPTHS))[:, None, None] * np.ones((1, 2, 2))
    d26 = isotherm_depth(T, 26.0)
    assert np.all(d26 == 0.0)
    assert np.isfinite(d26).all()


def test_isotherm_depth_is_nan_on_land():
    T = np.full((len(STANDARD_DEPTHS), 2, 2), np.nan)
    assert np.isnan(isotherm_depth(T, 20.0)).all()


def test_mixed_layer_depth_on_two_layer_column():
    # Uniform 29 degC to 50 m, then a sharp drop: MLD must land near 50 m.
    T = np.empty((len(STANDARD_DEPTHS), 1, 1))
    for k, z in enumerate(STANDARD_DEPTHS):
        T[k, 0, 0] = 29.0 if z <= 50 else 29.0 - 0.15 * (z - 50)
    mld = mixed_layer_depth(T, delta=0.2)[0, 0]
    assert 50.0 <= mld <= 56.0


def test_heat_content_matches_analytic_integral():
    T = np.full((len(STANDARD_DEPTHS), 1, 1), 20.0)
    ohc = ocean_heat_content(T, 300.0)[0, 0]
    expected = 1025.0 * 3985.0 * 20.0 * 300.0 / 1e9
    assert ohc == pytest.approx(expected, rel=1e-4)


def test_tchp_zero_when_surface_below_26():
    T = np.full((len(STANDARD_DEPTHS), 1, 1), 24.0)
    assert cyclone_heat_potential(T)[0, 0] == 0.0


def test_tchp_positive_and_ordered():
    warm = _linear_column(t_top=30.0, t_bot=10.0)
    warmer = _linear_column(t_top=31.5, t_bot=10.0)
    a = cyclone_heat_potential(warm)[0, 0]
    b = cyclone_heat_potential(warmer)[0, 0]
    assert 0.0 < a < b


# --- simulator ------------------------------------------------------------
@pytest.fixture(scope="module")
def sim_day():
    from oceanembed.data.synthetic import NIOSimulator
    sim = NIOSimulator(seed=3)
    # Step a few days so the eddy field and AR(1) noise are spun up.
    for d in range(5):
        surf, thetao = sim.step(dt.date(2021, 8, 11) + dt.timedelta(days=d))
    return surf, thetao


def test_simulator_shapes_and_finiteness(sim_day):
    surf, thetao = sim_day
    m = sea_mask()
    assert thetao.shape == (len(STANDARD_DEPTHS), 101, 241)
    assert np.isfinite(thetao[:, m]).all()
    assert not np.isfinite(thetao[:, ~m]).any()


def test_simulator_temperature_ranges(sim_day):
    surf, thetao = sim_day
    m = sea_mask()
    assert 20.0 < np.nanmin(surf["sst"][m]) and np.nanmax(surf["sst"][m]) < 33.0
    assert 27.0 < np.nanmean(surf["sst"][m]) < 30.5
    assert 28.0 < np.nanmin(surf["sss"][m]) and np.nanmax(surf["sss"][m]) < 37.5
    # 1000 m water in the North Indian Ocean is 4-7 degC.
    assert 4.0 < np.nanmean(thetao[-1][m]) < 7.0


def test_profiles_are_stratified(sim_day):
    """Inversions must be rare and bounded - they occur only in the winter Bay."""
    _, thetao = sim_day
    m = sea_mask()
    dT = np.diff(thetao[:, m], axis=0)
    inverted = (dT > 0.05).mean()
    assert inverted < 0.02, f"{inverted:.3%} of level pairs inverted"
    assert dT.max() < 2.5


def test_deep_levels_less_variable_than_thermocline():
    """A monotone decrease of variance with depth is the key realism check:
    it is what stops the deepest levels from being trivially predictable."""
    from oceanembed.data.synthetic import NIOSimulator, date_range
    sim = NIOSimulator(seed=5)
    days = date_range("2021-01-01", "2021-03-31")
    stack = np.stack([th for _, _, th in sim.run(days)])
    m = sea_mask()
    sd = np.array([np.nanstd(stack[:, k][:, m]) for k in range(len(STANDARD_DEPTHS))])
    k100 = STANDARD_DEPTHS.index(100)
    assert sd[k100] > sd[-1] * 3, "thermocline should vary far more than 1000 m"
    assert sd[-1] > 0.01, "1000 m must not be exactly constant"


# --- metrics ---------------------------------------------------------------
def test_bootstrap_ci_brackets_the_point_estimate():
    """The interval must bracket the statistic it reports.

    Pooling depth levels before taking the square root estimates a quadratic
    mean, which is always larger than the mean of per-level RMSEs the headline
    reports - producing an interval that sits entirely above its own point
    estimate.
    """
    from oceanembed.metrics import bootstrap_rmse_ci

    rng = np.random.default_rng(0)
    per_level_rmse = np.array([0.25, 0.25, 0.25, 0.25, 0.29, 0.56, 0.78, 0.66,
                               0.61, 0.58, 0.47, 0.39, 0.41, 0.37, 0.37])
    mse = (per_level_rmse ** 2)[None] * rng.gamma(20, 1 / 20, size=(184, 15))
    point = np.sqrt(mse.mean(axis=0)).mean()
    lo, hi = bootstrap_rmse_ci(mse, n_boot=2000)
    assert lo <= point <= hi, f"CI [{lo:.4f}, {hi:.4f}] excludes point {point:.4f}"
    assert lo < hi


def test_bootstrap_ci_resamples_days_not_cells():
    """Widening with fewer days is the signature of day-level resampling."""
    from oceanembed.metrics import bootstrap_rmse_ci

    rng = np.random.default_rng(1)
    mse = 0.25 * rng.gamma(4, 1 / 4, size=(400, 15))
    wide = bootstrap_rmse_ci(mse[:20], n_boot=2000)
    narrow = bootstrap_rmse_ci(mse, n_boot=2000)
    assert (wide[1] - wide[0]) > 2 * (narrow[1] - narrow[0])


def test_skill_score_is_zero_for_climatology_itself():
    """A method that *is* the climatology must score exactly zero skill."""
    from oceanembed.metrics import LevelStats

    rng = np.random.default_rng(2)
    clim = rng.normal(20.0, 1.0, 500)
    truth = clim + rng.normal(0.0, 1.5, 500)
    st = LevelStats()
    st.update(clim, truth, clim)
    s = st.summary()
    assert abs(s["skill_vs_clim"]) < 1e-9
    assert abs(s["rmse"] - s["clim_rmse"]) < 1e-9
