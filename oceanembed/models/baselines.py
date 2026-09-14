"""Reference methods the deep model has to beat.

These are not strawmen.  ``sla_regression`` in particular is close to what is
actually used operationally to produce synthetic subsurface profiles - the
"Gravest Empirical Mode" / synthetic-profile family of methods regress
temperature at each level onto sea level and SST anomalies, point by point and
season by season.  Reporting skill against it is the only honest way to show
that the learned embedding contributes something beyond established statistics.

============================  ===============================================
``climatology``               Harmonic (annual + semiannual + terannual) fit per
                              grid point and level.  Its RMSE is the natural
                              zero-skill reference.
``linear``                    One global multiple linear regression per level on
                              all surface variables plus static and seasonal
                              predictors.
``sla_regression``            Per-grid-point regression of T(z) on SLA and SST
                              anomaly with a seasonal cycle - the operational
                              synthetic-profile approach.
``mlp``                       Point-wise neural network on the same predictors,
                              *without* spatial context.  Isolates how much of
                              OceanEmbed's skill comes from seeing the
                              surrounding field rather than from nonlinearity.
============================  ===============================================

All estimators expose ``fit(store, days)`` and ``predict(t)`` returning
``(15, H, W)`` in degrees Celsius, so :mod:`oceanembed.evaluate` treats them
interchangeably with the neural model.
"""
from __future__ import annotations

import logging

import numpy as np

from ..grid import N_DEPTH

log = logging.getLogger(__name__)

N_HARMONIC = 3          # annual + semiannual + terannual


def seasonal_design(doy: np.ndarray, n_harm: int = N_HARMONIC) -> np.ndarray:
    """Design matrix ``[1, cos, sin, cos2, sin2, ...]`` of shape ``(n, 1+2*n_harm)``."""
    doy = np.asarray(doy, dtype="float64")
    cols = [np.ones_like(doy)]
    for h in range(1, n_harm + 1):
        ang = 2.0 * np.pi * h * doy / 365.25
        cols += [np.cos(ang), np.sin(ang)]
    return np.stack(cols, axis=1)


def _doy(store, days: list[int]) -> np.ndarray:
    return np.asarray([store.dates[t].timetuple().tm_yday for t in days], dtype="float64")


class Baseline:
    name = "baseline"

    def fit(self, store, days: list[int]) -> "Baseline":  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, store, t: int) -> np.ndarray:       # pragma: no cover - interface
        raise NotImplementedError


class Climatology(Baseline):
    """Per-grid-point, per-level harmonic seasonal climatology.

    Fitted by accumulating the normal equations over the training days, so the
    whole training period never has to be held in memory at once.
    """

    name = "climatology"

    def __init__(self, n_harm: int = N_HARMONIC):
        self.n_harm = n_harm
        self.coef: np.ndarray | None = None

    def fit(self, store, days: list[int], chunk: int = 60) -> "Climatology":
        P = 1 + 2 * self.n_harm
        H, W = store.shape
        ata = np.zeros((P, P))
        aty = np.zeros((P, N_DEPTH * H * W))
        for s in range(0, len(days), chunk):
            sel = days[s:s + chunk]
            A = seasonal_design(_doy(store, sel), self.n_harm)          # (n, P)
            Y = np.asarray(store.target[sel], dtype="float64").reshape(len(sel), -1)
            Y = np.nan_to_num(Y, nan=0.0)
            ata += A.T @ A
            aty += A.T @ Y
        self.coef = np.linalg.solve(ata + 1e-6 * np.eye(P), aty).reshape(P, N_DEPTH, H, W)
        log.info("climatology fitted on %d days (%d harmonics)", len(days), self.n_harm)
        return self

    def predict(self, store, t: int) -> np.ndarray:
        A = seasonal_design(_doy(store, [t]), self.n_harm)[0]            # (P,)
        return np.tensordot(A, self.coef, axes=(0, 0)).astype("float32")


class _PointwiseFeatures:
    """Shared feature construction for the point-wise estimators.

    Features per ocean cell: the seven surface observations, three static fields
    (latitude, longitude, log distance to coast) and the seasonal harmonics.
    The surface values are anomalies relative to the fitted surface climatology,
    which removes the seasonal cycle the harmonics already explain and leaves
    the regression to work on the dynamically meaningful part.
    """

    def __init__(self, n_harm: int = N_HARMONIC):
        self.n_harm = n_harm
        self.surf_clim: np.ndarray | None = None

    def fit_surface_climatology(self, store, days: list[int], chunk: int = 60) -> None:
        P = 1 + 2 * self.n_harm
        n_ch = store.inputs.shape[1]
        H, W = store.shape
        ata = np.zeros((P, P))
        aty = np.zeros((P, n_ch * H * W))
        for s in range(0, len(days), chunk):
            sel = days[s:s + chunk]
            A = seasonal_design(_doy(store, sel), self.n_harm)
            X = np.asarray(store.inputs[sel], dtype="float64").reshape(len(sel), -1)
            ata += A.T @ A
            aty += A.T @ np.nan_to_num(X, nan=0.0)
        self.surf_clim = np.linalg.solve(ata + 1e-6 * np.eye(P), aty).reshape(P, n_ch, H, W)

    def features(self, store, t: int, mask_idx: tuple) -> np.ndarray:
        """``(n_points, n_features)`` design matrix for day ``t``."""
        A = seasonal_design(_doy(store, [t]), self.n_harm)[0]
        clim = np.tensordot(A, self.surf_clim, axes=(0, 0))              # (C, H, W)
        anom = np.asarray(store.inputs[t], dtype="float64") - clim
        cols = [anom[:, mask_idx[0], mask_idx[1]].T]                     # (n, 7)
        cols.append(store.static[:, mask_idx[0], mask_idx[1]].T)         # (n, 3)
        n = cols[0].shape[0]
        cols.append(np.tile(A[1:], (n, 1)))                              # harmonics
        cols.append(np.ones((n, 1)))
        return np.concatenate(cols, axis=1)


class LinearRegression(Baseline):
    """One global ridge regression per depth level on the point-wise features."""

    name = "linear"

    def __init__(self, alpha: float = 1.0, n_harm: int = N_HARMONIC):
        self.alpha = alpha
        self.feat = _PointwiseFeatures(n_harm)
        self.coef: np.ndarray | None = None

    def fit(self, store, days: list[int], chunk: int = 30) -> "LinearRegression":
        self.feat.fit_surface_climatology(store, days)
        idx = np.nonzero(store.mask)
        F = None
        ata = aty = None
        for s in range(0, len(days), chunk):
            for t in days[s:s + chunk]:
                X = self.feat.features(store, t, idx)
                Y = np.asarray(store.target[t], dtype="float64")[:, idx[0], idx[1]].T
                if ata is None:
                    F = X.shape[1]
                    ata = np.zeros((F, F))
                    aty = np.zeros((F, N_DEPTH))
                ata += X.T @ X
                aty += X.T @ np.nan_to_num(Y, nan=0.0)
        self.coef = np.linalg.solve(ata + self.alpha * np.eye(F), aty)   # (F, K)
        log.info("linear baseline fitted: %d features, %d days", F, len(days))
        return self

    def predict(self, store, t: int) -> np.ndarray:
        idx = np.nonzero(store.mask)
        X = self.feat.features(store, t, idx)
        pred = X @ self.coef                                             # (n, K)
        out = np.full((N_DEPTH,) + store.shape, np.nan, dtype="float32")
        out[:, idx[0], idx[1]] = pred.T
        return out


class SLARegression(Baseline):
    """Per-grid-point regression of T(z) on SLA and SST anomaly, plus a seasonal cycle.

    This reproduces the operational synthetic-profile technique.  Because the
    predictors do not depend on depth, one ``(P, P)`` normal-equation system per
    grid point is shared across all 15 levels, which makes the fit cheap despite
    being local.
    """

    name = "sla_regression"

    def __init__(self, n_harm: int = 2, ridge: float = 1e-2):
        self.n_harm = n_harm
        self.ridge = ridge
        self.feat = _PointwiseFeatures(n_harm)
        self.coef: np.ndarray | None = None

    def _local_features(self, store, t: int) -> np.ndarray:
        """``(P, H, W)``: intercept, harmonics, SLA, SST anomaly, SSS anomaly."""
        A = seasonal_design(_doy(store, [t]), self.n_harm)[0]
        clim = np.tensordot(A, self.feat.surf_clim, axes=(0, 0))
        anom = np.asarray(store.inputs[t], dtype="float64") - clim
        H, W = store.shape
        cols = [np.ones((H, W))]
        cols += [np.full((H, W), a) for a in A[1:]]
        cols += [anom[2], anom[0], anom[1]]        # sla, sst', sss'
        return np.stack(cols)

    def fit(self, store, days: list[int]) -> "SLARegression":
        self.feat.fit_surface_climatology(store, days)
        H, W = store.shape
        P = 1 + 2 * self.n_harm + 3
        ata = np.zeros((H * W, P, P))
        aty = np.zeros((H * W, P, N_DEPTH))
        for t in days:
            X = self._local_features(store, t).reshape(P, -1).T          # (HW, P)
            Y = np.nan_to_num(np.asarray(store.target[t], dtype="float64"), nan=0.0)
            Y = Y.reshape(N_DEPTH, -1).T                                 # (HW, K)
            ata += X[:, :, None] * X[:, None, :]
            aty += X[:, :, None] * Y[:, None, :]
        ata += self.ridge * np.eye(P)[None]
        self.coef = np.linalg.solve(ata, aty).reshape(H, W, P, N_DEPTH)
        log.info("sla_regression fitted: %d local systems of size %d", H * W, P)
        return self

    def predict(self, store, t: int) -> np.ndarray:
        H, W = store.shape
        X = self._local_features(store, t)                               # (P, H, W)
        pred = np.einsum("phw,hwpk->khw", X, self.coef)
        return np.where(store.mask[None], pred, np.nan).astype("float32")


class PointwiseMLP(Baseline):
    """Point-wise neural network - same information, no spatial context."""

    name = "mlp"

    def __init__(self, hidden=(128, 128), max_iter: int = 60, n_samples: int = 400_000,
                 seed: int = 0, n_harm: int = N_HARMONIC):
        self.hidden, self.max_iter, self.n_samples, self.seed = hidden, max_iter, n_samples, seed
        self.feat = _PointwiseFeatures(n_harm)
        self.net = None
        self.x_mu = self.x_sd = self.y_mu = self.y_sd = None

    def fit(self, store, days: list[int]) -> "PointwiseMLP":
        from sklearn.neural_network import MLPRegressor
        self.feat.fit_surface_climatology(store, days)
        rng = np.random.default_rng(self.seed)
        idx = np.nonzero(store.mask)
        n_ocean = idx[0].size
        # Subsample days and cells: 400k rows is ample for a 14-feature model and
        # keeps the fit to a couple of minutes.
        per_day = max(int(self.n_samples / max(len(days), 1)), 32)
        Xs, Ys = [], []
        for t in days:
            take = rng.choice(n_ocean, size=min(per_day, n_ocean), replace=False)
            sub = (idx[0][take], idx[1][take])
            Xs.append(self.feat.features(store, t, sub))
            Ys.append(np.asarray(store.target[t], dtype="float64")[:, sub[0], sub[1]].T)
        X = np.concatenate(Xs)
        Y = np.nan_to_num(np.concatenate(Ys), nan=0.0)
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-6
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0) + 1e-6
        self.net = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=self.max_iter,
                                random_state=self.seed, early_stopping=True,
                                n_iter_no_change=8, learning_rate_init=1e-3)
        self.net.fit((X - self.x_mu) / self.x_sd, (Y - self.y_mu) / self.y_sd)
        log.info("mlp baseline fitted on %d samples, %d features", X.shape[0], X.shape[1])
        return self

    def predict(self, store, t: int) -> np.ndarray:
        idx = np.nonzero(store.mask)
        X = self.feat.features(store, t, idx)
        pred = self.net.predict((X - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu
        out = np.full((N_DEPTH,) + store.shape, np.nan, dtype="float32")
        out[:, idx[0], idx[1]] = pred.T
        return out


BASELINES = {b.name: b for b in (Climatology, LinearRegression, SLARegression, PointwiseMLP)}


def build_baseline(name: str) -> Baseline:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINES)}")
    return BASELINES[name]()
