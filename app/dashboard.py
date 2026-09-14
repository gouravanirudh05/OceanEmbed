"""OceanEmbed proof-of-concept dashboard.

Run with::

    streamlit run app/dashboard.py -- --data data/processed --ckpt outputs/checkpoints/<name>_best.pt

The app reconstructs a chosen day on demand and shows, side by side, the surface
observations that went in, the reconstructed subsurface field, the truth, the
error, the vertical profile with its predictive interval, and the derived
operational diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from oceanembed.data.argo import ProfileSet                            # noqa: E402
from oceanembed.data.dataset import OceanStore                         # noqa: E402
from oceanembed.diagnostics import DIAG_UNITS, DIAGNOSTICS             # noqa: E402
from oceanembed.grid import STANDARD_DEPTHS, SUBREGIONS                # noqa: E402

st.set_page_config(page_title="OceanEmbed - subsurface reconstruction",
                   layout="wide", initial_sidebar_state="expanded")


def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "data" / "processed"))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--results", default=str(REPO / "outputs" / "reports" / "evaluation.json"))
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return p.parse_known_args(argv)[0]


ARGS = cli_args()


@st.cache_resource(show_spinner="loading dataset...")
def get_store(path: str) -> OceanStore:
    return OceanStore(path)


@st.cache_resource(show_spinner="loading model...")
def get_predictor(ckpt: str):
    from oceanembed.evaluate import ModelPredictor
    return ModelPredictor(ckpt)


@st.cache_resource(show_spinner="fitting climatology (once)...")
def get_climatology(path: str):
    from oceanembed.models.baselines import Climatology
    store = get_store(path)
    return Climatology().fit(store, store.splits["train"])


@st.cache_data(show_spinner="reconstructing...")
def reconstruct_day(ckpt: str, path: str, t: int):
    """Returns (prediction, sigma, truth, climatology) in degrees Celsius."""
    store = get_store(path)
    pred_obj = get_predictor(ckpt)
    pred = pred_obj.predict(store, t)
    sigma = pred_obj.last_sigma
    truth = np.where(store.mask[None], np.asarray(store.target[t]), np.nan)
    clim = np.where(store.mask[None], get_climatology(path).predict(store, t), np.nan)
    return pred, sigma, truth, clim


def find_checkpoint() -> str | None:
    if ARGS.ckpt:
        return ARGS.ckpt
    cands = sorted((REPO / "outputs" / "checkpoints").glob("*_best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else None


def heatmap(field, store, title, cmap="RdYlBu_r", zmin=None, zmax=None,
            units="degC", view: tuple[float, float, float, float] | None = None):
    """Map of ``field``. ``view`` is ``(lat0, lat1, lon0, lon1)`` to zoom."""
    import plotly.graph_objects as go
    dom = store.domain
    fig = go.Figure(go.Heatmap(
        z=np.where(store.mask, field, np.nan), x=dom.lon, y=dom.lat,
        colorscale=cmap, zmin=zmin, zmax=zmax, zsmooth="best",
        colorbar=dict(title=units, thickness=12),
        hovertemplate="%{y:.2f}degN %{x:.2f}degE<br>%{z:.2f} " + units + "<extra></extra>"))
    # Land drawn as a grey overlay so the coastline reads clearly.
    land = np.where(store.mask, np.nan, 1.0)
    fig.add_trace(go.Heatmap(z=land, x=dom.lon, y=dom.lat, showscale=False,
                             colorscale=[[0, "#d9d9d9"], [1, "#d9d9d9"]],
                             hoverinfo="skip"))
    xaxis = dict(title="longitude")
    yaxis = dict(title="latitude", scaleanchor="x", scaleratio=1)
    if view is not None:
        la0, la1, lo0, lo1 = view
        xaxis["range"] = [lo0, lo1]
        yaxis["range"] = [la0, la1]
    fig.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=38, b=10),
                      xaxis=xaxis, yaxis=yaxis)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("OceanEmbed")
st.sidebar.caption("Subsurface temperature from surface satellite observations "
                   "- INCOIS SIH-2026 PS#01")

ckpt = find_checkpoint()
if ckpt is None:
    st.error("No checkpoint found. Train a model first:\n\n"
             "`python -m oceanembed.cli train -c configs/nio_full.yaml`")
    st.stop()

store = get_store(ARGS.data)
st.sidebar.markdown(f"**Checkpoint**\n\n`{Path(ckpt).name}`")
st.sidebar.markdown(f"**Grid** {store.shape[0]}x{store.shape[1]} @ "
                    f"{store.domain.resolution}deg  \n"
                    f"**Period** {store.dates[0]} - {store.dates[-1]}")

split = st.sidebar.selectbox("Period", ["test", "val", "train"], index=0,
                             help="'test' is the fully held-out period")
days = store.splits[split]
labels = [f"{store.dates[t]:%Y-%m-%d}" for t in days]
pick = st.sidebar.select_slider("Date", options=labels, value=labels[len(labels) // 2])
t = days[labels.index(pick)]

level = st.sidebar.selectbox("Depth level (m)", STANDARD_DEPTHS,
                             index=STANDARD_DEPTHS.index(100))
k = STANDARD_DEPTHS.index(level)

st.sidebar.markdown("---")
st.sidebar.subheader("Profile location")
region = st.sidebar.selectbox(
    "Zoom to", ["full"] + [r for r in SUBREGIONS if r != "full"],
    help="Restricts the map view and the headline metrics to one sub-basin")
lat = st.sidebar.slider("latitude (degN)", float(store.domain.lat_min),
                        float(store.domain.lat_max), 15.0, 0.25)
lon = st.sidebar.slider("longitude (degE)", float(store.domain.lon_min),
                        float(store.domain.lon_max), 88.0, 0.25)

pred, sigma, truth, clim = reconstruct_day(ckpt, ARGS.data, t)

# The sub-basin selection restricts both the map view and the scoring mask, so
# the headline numbers always describe the region actually on screen.
VIEW = None if region == "full" else SUBREGIONS[region]
mask = store.mask.copy()
if VIEW is not None:
    LAT, LON = store.domain.meshgrid()
    la0, la1, lo0, lo1 = VIEW
    mask &= (LAT >= la0) & (LAT <= la1) & (LON >= lo0) & (LON <= lo1)

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------
st.title("Reconstructed subsurface ocean temperature")
st.caption(f"{region.replace('_', ' ').title() if region != 'full' else 'North Indian Ocean'}"
           f" - {store.dates[t]:%d %B %Y} - held-out **{split}** period")

err = (pred - truth)[:, mask]
clim_err = (clim - truth)[:, mask]
rmse_all = float(np.sqrt(np.nanmean(err ** 2)))
rmse_clim = float(np.sqrt(np.nanmean(clim_err ** 2)))
c = st.columns(5)
c[0].metric("RMSE, all levels", f"{rmse_all:.3f} degC")
c[1].metric("Climatology RMSE", f"{rmse_clim:.3f} degC")
c[2].metric("Variance removed", f"{100*(1-rmse_all**2/rmse_clim**2):.1f} %")
c[3].metric(f"RMSE at {level} m", f"{float(np.sqrt(np.nanmean((pred-truth)[k][mask]**2))):.3f} degC")
c[4].metric("Bias, all levels", f"{float(np.nanmean(err)):+.3f} degC")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("1. Surface satellite observations (model input)")
surf = np.asarray(store.inputs[t])
gaps = np.asarray(store.gapflag[t])
cols = st.columns(4)
cols[0].plotly_chart(heatmap(surf[0], store, "SST", "RdYlBu_r", view=VIEW), use_container_width=True)
cols[1].plotly_chart(heatmap(surf[2], store, "Sea level anomaly", "RdBu_r",
                             units="m", view=VIEW), use_container_width=True)
cols[2].plotly_chart(heatmap(surf[1], store, "Sea surface salinity", "Viridis",
                             units="psu", view=VIEW), use_container_width=True)
cols[3].plotly_chart(heatmap(np.hypot(surf[3], surf[4]), store, "Surface current speed",
                             "Turbo", units="m/s", view=VIEW), use_container_width=True)
st.caption("Observed fraction of the basin this day: "
           + ", ".join(f"{v} {100*(1-gaps[i][mask].mean()):.0f}%"
                       for i, v in enumerate(["SST", "SSS", "SLA", "U", "V", "Uwnd", "Vwnd"])))

# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
st.subheader(f"2. Subsurface temperature at {level} m")
lo = float(np.nanpercentile(truth[k][mask], 1))
hi = float(np.nanpercentile(truth[k][mask], 99))
cols = st.columns(3)
cols[0].plotly_chart(heatmap(truth[k], store, "Truth", zmin=lo, zmax=hi, view=VIEW),
                     use_container_width=True)
cols[1].plotly_chart(heatmap(pred[k], store, "OceanEmbed reconstruction",
                             zmin=lo, zmax=hi, view=VIEW), use_container_width=True)
elim = float(np.nanpercentile(np.abs((pred - truth)[k][mask]), 99)) or 1.0
cols[2].plotly_chart(heatmap((pred - truth)[k], store, "Error (reconstruction - truth)",
                             "RdBu_r", zmin=-elim, zmax=elim, view=VIEW), use_container_width=True)

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
st.subheader("3. Vertical profile")
import plotly.graph_objects as go
i = int(round((lat - store.domain.lat_min) / store.domain.resolution))
j = int(round((lon - store.domain.lon_min) / store.domain.resolution))
z = np.asarray(STANDARD_DEPTHS, dtype=float)

left, right = st.columns([2, 1])
if not mask[i, j]:
    left.warning(f"{lat:.2f}degN {lon:.2f}degE is on land - move the sliders.")
else:
    fig = go.Figure()
    if sigma is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([pred[:, i, j] - 1.96 * sigma[:, i, j],
                              (pred[:, i, j] + 1.96 * sigma[:, i, j])[::-1]]),
            y=np.concatenate([z, z[::-1]]), fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0), name="95% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=truth[:, i, j], y=z, name="truth",
                             line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=pred[:, i, j], y=z, name="OceanEmbed",
                             line=dict(color="#1f77b4", width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=clim[:, i, j], y=z, name="climatology",
                             line=dict(color="#999", width=1.5, dash="dot")))
    argo = ProfileSet.load(Path(store.root) / "argo.npz")
    sel = argo.day_index == t
    if sel.any():
        ai, aj = argo.grid_index(store.domain)
        d = np.hypot(ai[sel] - i, aj[sel] - j)
        n = int(np.argmin(d))
        if d[n] <= 6:
            fig.add_trace(go.Scatter(x=argo.temp[sel][n], y=z, mode="markers",
                                     name=f"in-situ cast ({d[n]*0.25:.1f}deg away)",
                                     marker=dict(color="crimson", size=7, symbol="x")))
    fig.update_layout(height=520, yaxis=dict(autorange="reversed", title="depth (m)",
                                             type="log"),
                      xaxis_title="temperature (degC)",
                      title=f"{lat:.2f}degN {lon:.2f}degE",
                      legend=dict(y=0.02, x=0.02))
    left.plotly_chart(fig, use_container_width=True)

    rows = []
    for kk, zz in enumerate(STANDARD_DEPTHS):
        rows.append({"depth (m)": zz, "truth": round(float(truth[kk, i, j]), 2),
                     "OceanEmbed": round(float(pred[kk, i, j]), 2),
                     "error": round(float(pred[kk, i, j] - truth[kk, i, j]), 2),
                     "sigma": round(float(sigma[kk, i, j]), 2) if sigma is not None else None})
    right.dataframe(rows, hide_index=True, height=520)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
st.subheader("4. Derived operational diagnostics")
st.caption("Tropical cyclone heat potential and the depth of the 26 degC isotherm "
           "control cyclone intensification in the Bay of Bengal, and neither can "
           "be obtained from SST alone.")
show = st.multiselect("Diagnostics", list(DIAGNOSTICS), default=["tchp", "d26", "mld"])
for name in show:
    ft = DIAGNOSTICS[name](truth)
    fp = DIAGNOSTICS[name](pred)
    dlo = float(np.nanpercentile(ft[mask], 2))
    dhi = float(np.nanpercentile(ft[mask], 98))
    r = float(np.sqrt(np.nanmean((fp - ft)[mask] ** 2)))
    cc = st.columns(3)
    cc[0].plotly_chart(heatmap(ft, store, f"{name} - truth", "Magma", dlo, dhi,
                               DIAG_UNITS[name], view=VIEW), use_container_width=True)
    cc[1].plotly_chart(heatmap(fp, store, f"{name} - OceanEmbed", "Magma", dlo, dhi,
                               DIAG_UNITS[name], view=VIEW), use_container_width=True)
    lim = float(np.nanpercentile(np.abs((fp - ft)[mask]), 99)) or 1.0
    cc[2].plotly_chart(heatmap(fp - ft, store, f"{name} - error (RMSE {r:.2f} "
                               f"{DIAG_UNITS[name]})", "RdBu_r", -lim, lim,
                               DIAG_UNITS[name], view=VIEW), use_container_width=True)

# ---------------------------------------------------------------------------
# Skill summary
# ---------------------------------------------------------------------------
res_path = Path(ARGS.results)
if res_path.exists():
    st.subheader("5. Held-out skill against reference methods")
    res = json.loads(res_path.read_text())
    order = sorted(res["methods"],
                   key=lambda m: res["methods"][m]["field"]["full"]["rmse_mean"])
    st.dataframe([{
        "method": m,
        "RMSE degC": round(res["methods"][m]["field"]["full"]["rmse_mean"], 3),
        "skill vs clim %": round(100 * res["methods"][m]["field"]["full"]["skill_vs_clim"], 1),
        "anomaly corr": round(res["methods"][m]["field"]["full"]["acc_mean"], 3),
        "in-situ RMSE degC": round(res["methods"][m]["argo"]["rmse_mean"], 3),
    } for m in order], hide_index=True)

    import plotly.graph_objects as go
    fig = go.Figure()
    for m in order:
        pl = res["methods"][m]["field"]["full"]["per_level"]
        fig.add_trace(go.Scatter(x=[d["rmse"] for d in pl], y=list(res["meta"]["depths"]),
                                 name=m, line=dict(width=3 if m == "oceanembed" else 1.5)))
    fig.add_trace(go.Scatter(
        x=[d["clim_rmse"] for d in res["methods"][order[0]]["field"]["full"]["per_level"]],
        y=list(res["meta"]["depths"]), name="climatological std",
        line=dict(color="#bbb", dash="dot")))
    fig.update_layout(height=470, yaxis=dict(autorange="reversed", type="log",
                                             title="depth (m)"),
                      xaxis_title="RMSE (degC)", title="Error by depth, held-out test period")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run `python -m oceanembed.cli evaluate --checkpoint <ckpt>` to populate "
            "the skill comparison panel.")
