"""Generate the proof-of-concept figure set from a trained checkpoint."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.dataset import OceanStore
from .evaluate import ModelPredictor
from .models.baselines import Climatology
from .train import pick_device
from . import viz

log = logging.getLogger(__name__)

#: Locations used for the profile panel: one per dynamical regime of the basin.
POC_POINTS = [
    (15.0, 88.0, "Central Bay of Bengal"),
    (19.0, 89.5, "Head of the Bay"),
    (15.0, 65.0, "Central Arabian Sea"),
    (10.0, 52.0, "Somali upwelling"),
]


def make_figures(cfg: Config, ckpt: str | Path, data_dir: str | Path | None = None,
                 results: str | Path | None = None, day: str | None = None,
                 section_days: int = 120) -> list[Path]:
    store = OceanStore(data_dir or cfg.paths.resolve("processed"))
    fig_dir = cfg.paths.resolve("figures")
    device = pick_device(cfg.train.device)
    pred = ModelPredictor(ckpt, device=device)
    out: list[Path] = []

    test = store.splits["test"]
    if day:
        from .data.synthetic import as_date
        target = as_date(day)
        t = store.dates.index(target)
    else:
        t = test[len(test) // 2]
    log.info("illustrating %s", store.dates[t])

    truth = np.where(store.mask[None], np.asarray(store.target[t]), np.nan)
    p_model = pred.predict(store, t)
    sigma = pred.last_sigma

    p = fig_dir / "01_reconstruction_100m.png"
    viz.figure_reconstruction(store, truth, p_model, t, level_m=100, out=p)
    out.append(p)

    p = fig_dir / "02_reconstruction_50m.png"
    viz.figure_reconstruction(store, truth, p_model, t, level_m=50, out=p)
    out.append(p)

    log.info("fitting climatology for the profile comparison")
    clim = Climatology().fit(store, store.splits["train"])
    p_clim = np.where(store.mask[None], clim.predict(store, t), np.nan)

    p = fig_dir / "03_profiles.png"
    viz.figure_profiles(store, truth, {"OceanEmbed": p_model, "climatology": p_clim},
                        t, POC_POINTS, sigma=sigma, out=p)
    out.append(p)

    p = fig_dir / "04_diagnostics.png"
    viz.figure_diagnostics(store, truth, p_model, t, out=p)
    out.append(p)

    with torch.no_grad():
        x = torch.from_numpy(store.x_full(t))[None].to(device)
        emb = pred.model.embed(x)[0].cpu().numpy()
    p = fig_dir / "05_embedding.png"
    viz.figure_embedding(store, emb, t, out=p)
    out.append(p)

    # Depth-time section over the test period at the Bay of Bengal point.
    sel = test[:section_days]
    log.info("building depth-time section over %d days", len(sel))
    tt = np.stack([np.where(store.mask[None], np.asarray(store.target[s]), np.nan) for s in sel])
    pp = np.stack([pred.predict(store, s) for s in sel])
    p = fig_dir / "06_section_bob.png"
    viz.figure_timeseries(store, tt, pp, [store.dates[s] for s in sel], 15.0, 88.0, out=p)
    out.append(p)

    hist_dir = cfg.paths.resolve("reports")
    hists = {f.stem.replace("_history", ""): f for f in sorted(hist_dir.glob("*_history.json"))}
    if hists:
        p = fig_dir / "07_training.png"
        viz.figure_training(hists, out=p)
        out.append(p)

    if results:
        res = json.loads(Path(results).read_text())
        for region in res["meta"]["regions"]:
            p = fig_dir / f"08_skill_{region}.png"
            viz.figure_rmse_profile(res, out=p, region=region)
            out.append(p)

    log.info("wrote %d figures to %s", len(out), fig_dir)
    return out
