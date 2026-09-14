"""Figures for the proof of concept.

All plotting is matplotlib-only (Agg backend) so the figures render on a
headless machine with no cartopy dependency; coastlines come from the same
land/sea mask the model uses, drawn as a contour.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .diagnostics import DIAG_UNITS
from .grid import STANDARD_DEPTHS, Domain

log = logging.getLogger(__name__)

# A perceptually ordered pair: sequential for fields, diverging for errors.
CMAP_T = "RdYlBu_r"
CMAP_ERR = "RdBu_r"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def _basemap(ax, domain: Domain, mask: np.ndarray) -> None:
    """Draw the coastline and label the axes in degrees."""
    ax.contour(domain.lon, domain.lat, mask.astype(float), levels=[0.5],
               colors="k", linewidths=0.6)
    ax.contourf(domain.lon, domain.lat, (~mask).astype(float), levels=[0.5, 1.5],
                colors=["0.85"])
    ax.set_xlabel("longitude (degE)")
    ax.set_ylabel("latitude (degN)")
    ax.set_aspect("equal")


def map_panel(fig, ax, field: np.ndarray, domain: Domain, mask: np.ndarray,
              title: str, cmap: str = CMAP_T, vmin=None, vmax=None,
              center: float | None = None, units: str = "degC"):
    data = np.where(mask, field, np.nan)
    norm = None
    if center is not None:
        lim = np.nanpercentile(np.abs(data - center), 99) or 1.0
        norm = TwoSlopeNorm(vcenter=center, vmin=center - lim, vmax=center + lim)
        vmin = vmax = None
    im = ax.pcolormesh(domain.lon, domain.lat, data, cmap=cmap, shading="auto",
                       vmin=vmin, vmax=vmax, norm=norm)
    _basemap(ax, domain, mask)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(units)
    return im


def figure_reconstruction(store, truth: np.ndarray, pred: np.ndarray, day,
                          level_m: int = 100, sigma: np.ndarray | None = None,
                          out: Path | None = None):
    """Inputs, truth, reconstruction and error at one depth on one day."""
    k = STANDARD_DEPTHS.index(level_m)
    dom, mask = store.domain, store.mask
    t = store.dates.index(day) if not isinstance(day, int) else day
    surf = np.asarray(store.inputs[t])

    ncol = 3
    nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 7.2), constrained_layout=True)
    map_panel(fig, axes[0, 0], surf[0], dom, mask, "Input: SST (OSTIA-like)", units="degC")
    map_panel(fig, axes[0, 1], surf[2], dom, mask, "Input: sea level anomaly",
              cmap="RdBu_r", center=0.0, units="m")
    spd = np.hypot(surf[3], surf[4])
    map_panel(fig, axes[0, 2], spd, dom, mask, "Input: surface current speed",
              cmap="viridis", units="m s$^{-1}$")

    vmin = float(np.nanpercentile(np.where(mask, truth[k], np.nan), 1))
    vmax = float(np.nanpercentile(np.where(mask, truth[k], np.nan), 99))
    map_panel(fig, axes[1, 0], truth[k], dom, mask, f"Truth: T at {level_m} m",
              vmin=vmin, vmax=vmax)
    map_panel(fig, axes[1, 1], pred[k], dom, mask,
              f"OceanEmbed: T at {level_m} m", vmin=vmin, vmax=vmax)
    err = pred[k] - truth[k]
    rmse = float(np.sqrt(np.nanmean(err[mask] ** 2)))
    map_panel(fig, axes[1, 2], err, dom, mask,
              f"Error (RMSE {rmse:.2f} degC)", cmap=CMAP_ERR, center=0.0)

    ds = store.dates[t]
    fig.suptitle(f"OceanEmbed reconstruction - {ds:%d %b %Y} - {level_m} m", fontsize=12)
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_profiles(store, truth: np.ndarray, preds: dict[str, np.ndarray],
                    day_idx: int, points: list[tuple[float, float, str]],
                    sigma: np.ndarray | None = None, out: Path | None = None):
    """Vertical profiles at named locations, with the predictive interval."""
    z = np.asarray(STANDARD_DEPTHS, dtype="float64")
    dom = store.domain
    fig, axes = plt.subplots(1, len(points), figsize=(3.3 * len(points), 5.2),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (lat, lon, name) in zip(axes, points):
        i = int(round((lat - dom.lat_min) / dom.resolution))
        j = int(round((lon - dom.lon_min) / dom.resolution))
        ax.plot(truth[:, i, j], z, "k-", lw=2.2, label="truth", zorder=5)
        for nm, p in preds.items():
            ax.plot(p[:, i, j], z, lw=1.4, ls="--", label=nm)
        if sigma is not None:
            m = preds.get("OceanEmbed")
            if m is not None:
                ax.fill_betweenx(z, m[:, i, j] - 1.96 * sigma[:, i, j],
                                 m[:, i, j] + 1.96 * sigma[:, i, j],
                                 alpha=0.18, color="tab:blue", label="95% interval")
        ax.invert_yaxis()
        ax.set_yscale("symlog", linthresh=100)
        ax.set_title(f"{name}\n{lat:.1f}degN {lon:.1f}degE")
        ax.set_xlabel("temperature (degC)")
    axes[0].set_ylabel("depth (m)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Reconstructed profiles - {store.dates[day_idx]:%d %b %Y}", fontsize=12)
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_rmse_profile(results: dict, out: Path | None = None, region: str = "full"):
    """RMSE and skill score as a function of depth, all methods."""
    z = np.asarray(results["meta"]["depths"], dtype="float64")
    methods = sorted(results["methods"],
                     key=lambda m: results["methods"][m]["field"][region]["rmse_mean"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)

    for m in methods:
        f = results["methods"][m]["field"][region]
        rmse = [d["rmse"] for d in f["per_level"]]
        skill = [100 * d["skill_vs_clim"] for d in f["per_level"]]
        acc = [d["acc"] for d in f["per_level"]]
        style = dict(lw=2.4, marker="o", ms=4) if m == "oceanembed" else dict(lw=1.3, ls="--", ms=3, marker="s")
        axes[0].plot(rmse, z, label=m, **style)
        axes[1].plot(skill, z, label=m, **style)
        axes[2].plot(acc, z, label=m, **style)

    clim = [d["clim_rmse"] for d in results["methods"][methods[0]]["field"][region]["per_level"]]
    axes[0].plot(clim, z, color="0.5", lw=1.0, ls=":", label="climatological std")
    for ax, xl, tt in zip(axes,
                          ["RMSE (degC)", "skill vs climatology (%)", "anomaly correlation"],
                          ["Error by depth", "Variance explained", "Anomaly correlation"]):
        ax.invert_yaxis()
        ax.set_yscale("symlog", linthresh=100)
        ax.set_xlabel(xl)
        ax.set_title(tt)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[0].set_ylabel("depth (m)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Reconstruction skill by depth - {region} - "
                 f"{results['meta']['test_days']} held-out days", fontsize=12)
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_diagnostics(store, truth: np.ndarray, pred: np.ndarray, day_idx: int,
                       names=("d26", "tchp", "ohc300"), out: Path | None = None):
    """Truth / reconstruction / error for the derived operational diagnostics."""
    from .diagnostics import DIAGNOSTICS
    dom, mask = store.domain, store.mask
    fig, axes = plt.subplots(len(names), 3, figsize=(14, 3.6 * len(names)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, nm in enumerate(names):
        ft, fp = DIAGNOSTICS[nm](truth), DIAGNOSTICS[nm](pred)
        lo = float(np.nanpercentile(ft[mask], 2))
        hi = float(np.nanpercentile(ft[mask], 98))
        map_panel(fig, axes[r, 0], ft, dom, mask, f"Truth: {nm}", cmap="magma",
                  vmin=lo, vmax=hi, units=DIAG_UNITS[nm])
        map_panel(fig, axes[r, 1], fp, dom, mask, f"OceanEmbed: {nm}", cmap="magma",
                  vmin=lo, vmax=hi, units=DIAG_UNITS[nm])
        e = fp - ft
        rm = float(np.sqrt(np.nanmean(e[mask] ** 2)))
        map_panel(fig, axes[r, 2], e, dom, mask, f"Error ({rm:.2f} {DIAG_UNITS[nm]})",
                  cmap=CMAP_ERR, center=0.0, units=DIAG_UNITS[nm])
    fig.suptitle(f"Derived operational diagnostics - {store.dates[day_idx]:%d %b %Y}",
                 fontsize=12)
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_embedding(store, embedding: np.ndarray, day_idx: int, out: Path | None = None):
    """Visualise the satellite embedding.

    The leading three principal components of the per-pixel embedding are mapped
    to red, green and blue.  Coherent patches of colour mean the embedding has
    organised the basin into distinct dynamical regimes rather than memorising
    position - which is the qualitative claim the framework rests on.
    """
    dom, mask = store.domain, store.mask
    D = embedding.shape[0]
    flat = embedding.reshape(D, -1)[:, mask.ravel()].T          # (P, D)
    flat = flat - flat.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    pcs = flat @ Vt[:3].T                                      # (P, 3)
    var = (S ** 2 / (S ** 2).sum())[:3]

    rgb = np.full(mask.shape + (3,), np.nan)
    for c in range(3):
        v = pcs[:, c]
        lo, hi = np.percentile(v, [2, 98])
        rgb[mask, c] = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    axes[0].imshow(np.where(np.isnan(rgb), 0.85, rgb), origin="lower",
                   extent=[dom.lon_min, dom.lon_max, dom.lat_min, dom.lat_max])
    _basemap(axes[0], dom, mask)
    axes[0].set_title(f"Embedding, PC1-3 as RGB\n({100*var.sum():.0f}% of variance)")
    for c in range(3):
        f = np.full(mask.shape, np.nan)
        f[mask] = pcs[:, c]
        map_panel(fig, axes[c + 1], f, dom, mask,
                  f"PC{c+1} ({100*var[c]:.0f}% variance)", cmap="Spectral_r",
                  center=0.0, units="")
    fig.suptitle(f"Satellite embedding structure - {store.dates[day_idx]:%d %b %Y} "
                 f"- {D} latent dimensions", fontsize=12)
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def load_history(path: Path) -> dict:
    """Read a training history file, tolerating the older bare-list format."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return {"run": Path(path).stem.replace("_history", ""), "epochs": raw,
                "dataset": {}}
    return raw


def figure_training(history_files: dict[str, Path], out: Path | None = None):
    """Loss and validation RMSE curves for one or more runs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for name, path in history_files.items():
        h = load_history(Path(path))["epochs"]
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train"]["total"] for r in h], lw=1.6, label=f"{name} train")
        axes[0].plot(ep, [r["val"]["total"] for r in h], lw=1.6, ls="--", label=f"{name} val")
        axes[1].plot(ep, [r["val_rmse_mean"] for r in h], lw=1.8, marker="o", ms=3, label=name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Objective")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation RMSE (degC)")
    axes[1].set_title("Depth-mean validation RMSE"); axes[1].legend(fontsize=8)
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_timeseries(store, truth: np.ndarray, pred: np.ndarray, days,
                      lat: float, lon: float, out: Path | None = None):
    """Depth-time section of truth and reconstruction at a single location."""
    dom = store.domain
    i = int(round((lat - dom.lat_min) / dom.resolution))
    j = int(round((lon - dom.lon_min) / dom.resolution))
    z = np.asarray(STANDARD_DEPTHS, dtype="float64")
    tt = truth[:, :, i, j].T
    pp = pred[:, :, i, j].T
    x = np.arange(len(days))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
    lo, hi = np.nanpercentile(tt, [1, 99])
    for ax, f, ttl in ((axes[0], tt, "Truth"), (axes[1], pp, "OceanEmbed")):
        im = ax.pcolormesh(x, z, f, cmap=CMAP_T, shading="auto", vmin=lo, vmax=hi)
        ax.invert_yaxis(); ax.set_yscale("symlog", linthresh=100)
        ax.set_ylabel("depth (m)"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax, shrink=0.9, label="degC")
    e = pp - tt
    lim = float(np.nanpercentile(np.abs(e), 99)) or 1.0
    im = axes[2].pcolormesh(x, z, e, cmap=CMAP_ERR, shading="auto", vmin=-lim, vmax=lim)
    axes[2].invert_yaxis(); axes[2].set_yscale("symlog", linthresh=100)
    axes[2].set_ylabel("depth (m)"); axes[2].set_title("Error")
    fig.colorbar(im, ax=axes[2], shrink=0.9, label="degC")
    ticks = np.linspace(0, len(days) - 1, min(8, len(days))).astype(int)
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([f"{days[t]:%d %b}" for t in ticks])
    fig.suptitle(f"Depth-time section at {lat:.1f}degN {lon:.1f}degE", fontsize=12)
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig
