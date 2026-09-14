"""Evaluate OceanEmbed and the reference methods on the held-out test period.

Produces, for every method:

* gridded skill per depth level and per sub-basin (RMSE, bias, MAE, anomaly
  correlation, Murphy skill score against climatology) with day-resampled
  bootstrap confidence intervals;
* skill against the withheld in-situ profiles;
* skill on the derived operational diagnostics (MLD, D20, D26, OHC, TCHP);
* calibration of the predictive uncertainty, for the neural model.

Results are written as JSON plus a Markdown summary table.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.argo import ProfileSet
from .data.dataset import OceanStore
from .diagnostics import DIAG_UNITS, DIAGNOSTICS
from .grid import STANDARD_DEPTHS
from .metrics import (CalibrationEvaluator, FieldEvaluator, ProfileEvaluator,
                      SharpnessEvaluator, bootstrap_rmse_ci, region_masks)
from .models.baselines import Climatology, build_baseline
from .train import load_model, pick_device

log = logging.getLogger(__name__)


class ModelPredictor:
    """Wrap a trained network so it looks like a baseline estimator."""

    name = "oceanembed"

    def __init__(self, ckpt: str | Path, device=None, tile: int = 0, overlap: int = 8):
        self.device = device or pick_device()
        self.model, self.state = load_model(ckpt, self.device)
        self.tile, self.overlap = tile, overlap
        self.y_std = np.asarray(self.state["norm"]["y_std"], dtype="float32")
        self.y_mean = np.asarray(self.state["norm"]["y_mean"], dtype="float32")
        self.last_sigma: np.ndarray | None = None

    def fit(self, store, days):        # nothing to do - already trained
        return self

    @torch.no_grad()
    def predict(self, store, t: int) -> np.ndarray:
        from .models.oceanembed import predict_field
        x = torch.from_numpy(store.x_full(t))[None].to(self.device)
        out = predict_field(self.model, x, tile=self.tile, overlap=self.overlap)
        y = out["y"][0].cpu().numpy()
        pred = y * self.y_std[:, None, None] + self.y_mean[:, None, None]
        if out.get("log_sigma") is not None:
            # sigma is predicted in standardised units; convert to degrees Celsius.
            self.last_sigma = (np.exp(out["log_sigma"][0].cpu().numpy())
                               * self.y_std[:, None, None])
        return np.where(store.mask[None], pred, np.nan).astype("float32")


def _profile_clim(clim_field: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return clim_field[:, i, j].T


def evaluate(cfg: Config, checkpoints: dict[str, str | Path],
             data_dir: str | Path | None = None,
             methods: tuple[str, ...] | None = None,
             tile: int = 0, max_days: int | None = None) -> dict:
    """Run the full evaluation and return the results dictionary."""
    store = OceanStore(data_dir or cfg.paths.resolve("processed"))
    test_days = store.splits["test"]
    if max_days:
        test_days = test_days[:max_days]
    train_days = store.splits["train"]
    regions = region_masks(store.domain, cfg.eval.regions)
    log.info("evaluating on %d test days, regions %s", len(test_days), list(regions))

    # The climatology is both a competitor and the reference for anomaly
    # correlation and skill scores, so it is always fitted.
    t0 = time.time()
    clim = Climatology().fit(store, train_days)
    log.info("climatology fitted in %.0fs", time.time() - t0)

    predictors: dict[str, object] = {}
    for name, path in checkpoints.items():
        predictors[name] = ModelPredictor(path, tile=tile)
    for name in (methods if methods is not None else cfg.eval.baselines):
        if name == "climatology":
            predictors[name] = clim
            continue
        t0 = time.time()
        predictors[name] = build_baseline(name).fit(store, train_days)
        log.info("%s fitted in %.0fs", name, time.time() - t0)

    field_ev = {n: FieldEvaluator(regions) for n in predictors}
    prof_ev = {n: ProfileEvaluator() for n in predictors}
    diag_ev = {n: {d: FieldEvaluator({"full": regions["full"]}, n_depth=1) for d in DIAGNOSTICS}
               for n in predictors}
    calib = {n: CalibrationEvaluator() for n in predictors
             if isinstance(predictors[n], ModelPredictor)}
    sharp = {n: SharpnessEvaluator() for n in predictors}
    # Per-day, per-level MSE, so the bootstrap estimates the same statistic
    # the headline reports.
    daily_level_mse = {n: [] for n in predictors}

    argo = ProfileSet.load(Path(store.root) / "argo.npz")
    argo_keep = np.isin(argo.day_index, test_days)
    argo = argo.subset(argo_keep)
    ai, aj = argo.grid_index(store.domain)
    log.info("%d in-situ profiles fall in the test period", len(argo))

    # Diagnostics climatology, needed for their skill scores.
    for n_day, t in enumerate(test_days):
        truth = np.asarray(store.target[t], dtype="float32")
        truth = np.where(store.mask[None], truth, np.nan)
        clim_f = clim.predict(store, t)
        clim_f = np.where(store.mask[None], clim_f, np.nan)
        truth_diag = {d: f(truth) for d, f in DIAGNOSTICS.items()}
        clim_diag = {d: f(clim_f) for d, f in DIAGNOSTICS.items()}

        sel = argo.day_index == t
        if sel.any():
            pi, pj = ai[sel], aj[sel]
            obs_prof = argo.temp[sel]
            clim_prof = _profile_clim(clim_f, pi, pj)

        for name, pr in predictors.items():
            pred = pr.predict(store, t)
            field_ev[name].update(pred, truth, clim_f)
            m = regions["full"]
            d = (pred - truth)[:, m]
            daily_level_mse[name].append(np.nanmean(d ** 2, axis=1))

            for dname, fn in DIAGNOSTICS.items():
                diag_ev[name][dname].update(fn(pred)[None], truth_diag[dname][None],
                                            clim_diag[dname][None])
            if sel.any():
                prof_ev[name].update(pred[:, pi, pj].T, obs_prof, clim_prof)
            sharp[name].update(pred, truth, regions["full"])
            if name in calib and getattr(pr, "last_sigma", None) is not None:
                calib[name].update((pred - truth)[:, m], pr.last_sigma[:, m])

        if (n_day + 1) % 20 == 0:
            log.info("  %d/%d test days", n_day + 1, len(test_days))

    results = {
        "meta": {"test_days": len(test_days), "n_argo": len(argo),
                 "depths": list(STANDARD_DEPTHS), "regions": list(regions),
                 "config": cfg.to_dict()},
        "methods": {},
    }
    for name in predictors:
        lo, hi = bootstrap_rmse_ci(np.asarray(daily_level_mse[name]),
                                   n_boot=cfg.eval.bootstrap)
        results["methods"][name] = {
            "field": field_ev[name].summary(),
            "argo": prof_ev[name].summary(),
            "diagnostics": {d: diag_ev[name][d].summary()["full"]["per_level"][0]
                            for d in DIAGNOSTICS},
            "rmse_ci95": [lo, hi],
            "sharpness": sharp[name].summary(),
        }
        if name in calib:
            results["methods"][name]["calibration"] = calib[name].summary()

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def markdown_report(res: dict) -> str:
    depths = res["meta"]["depths"]
    methods = list(res["methods"])
    L: list[str] = []
    L.append("# OceanEmbed evaluation report\n")
    L.append(f"Held-out test period: **{res['meta']['test_days']} days**, "
             f"**{res['meta']['n_argo']:,} independent in-situ profiles**.\n")

    L.append("## Headline skill (whole basin, gridded truth)\n")
    L.append("| method | RMSE degC | 95% CI | skill vs climatology | anomaly corr | bias degC |")
    L.append("|---|---|---|---|---|---|")
    order = sorted(methods, key=lambda m: res["methods"][m]["field"]["full"]["rmse_mean"])
    for m in order:
        f = res["methods"][m]["field"]["full"]
        lo, hi = res["methods"][m]["rmse_ci95"]
        acc = "n/a" if not np.isfinite(f["acc_mean"]) else f"{f['acc_mean']:.3f}"
        L.append(f"| `{m}` | **{f['rmse_mean']:.3f}** | {lo:.3f}-{hi:.3f} | "
                 f"{f['skill_vs_clim']*100:.1f}% | {acc} | {f['bias_mean']:+.3f} |")

    L.append("\n## RMSE by depth (degC, whole basin)\n")
    head = "| depth (m) | " + " | ".join(f"`{m}`" for m in order) + " |"
    L.append(head)
    L.append("|" + "---|" * (len(order) + 1))
    for k, z in enumerate(depths):
        row = [f"| {z} "]
        vals = [res["methods"][m]["field"]["full"]["per_level"][k]["rmse"] for m in order]
        best = min(vals)
        for v in vals:
            row.append(f"| **{v:.3f}** " if abs(v - best) < 1e-9 else f"| {v:.3f} ")
        L.append("".join(row) + "|")

    L.append("\n## Skill against withheld in-situ profiles\n")
    L.append("| method | RMSE degC | skill vs climatology | anomaly corr | n profiles |")
    L.append("|---|---|---|---|---|")
    for m in order:
        a = res["methods"][m]["argo"]
        acc = "n/a" if not np.isfinite(a["acc_mean"]) else f"{a['acc_mean']:.3f}"
        L.append(f"| `{m}` | {a['rmse_mean']:.3f} | {a['skill_vs_clim']*100:.1f}% | "
                 f"{acc} | {a['n_profiles']:,} |")

    L.append("\n## Derived operational diagnostics (RMSE)\n")
    dnames = list(res["methods"][order[0]]["diagnostics"])
    L.append("| method | " + " | ".join(f"{d} ({DIAG_UNITS[d]})" for d in dnames) + " |")
    L.append("|" + "---|" * (len(dnames) + 1))
    for m in order:
        d = res["methods"][m]["diagnostics"]
        L.append(f"| `{m}` | " + " | ".join(f"{d[x]['rmse']:.2f}" for x in dnames) + " |")

    L.append("\n## Skill by sub-basin (RMSE degC)\n")
    regions = res["meta"]["regions"]
    L.append("| method | " + " | ".join(regions) + " |")
    L.append("|" + "---|" * (len(regions) + 1))
    for m in order:
        f = res["methods"][m]["field"]
        L.append(f"| `{m}` | " + " | ".join(f"{f[r]['rmse_mean']:.3f}" for r in regions) + " |")

    L.append("\n## Effective sharpness (mean |horizontal gradient| of reconstruction / truth)\n")
    L.append("Below 1 means the field is over-smoothed - RMSE alone cannot detect this.\n")
    L.append("| method | " + " | ".join(f"{z} m" for z in depths) + " | mean |")
    L.append("|" + "---|" * (len(depths) + 2))
    for m in order:
        sh = res["methods"][m]["sharpness"]
        L.append(f"| `{m}` | " + " | ".join(f"{v:.2f}" for v in sh["sharpness_ratio"])
                 + f" | {sh['sharpness_ratio_mean']:.2f} |")

    for m in order:
        cal = res["methods"][m].get("calibration")
        if not cal:
            continue
        L.append(f"\n## Uncertainty calibration - `{m}`\n")
        L.append(f"Mean coverage of the nominal 68% interval: "
                 f"**{cal['coverage_68_mean']*100:.1f}%**; of the 95% interval: "
                 f"**{cal['coverage_95_mean']*100:.1f}%**.\n")
        L.append("| depth (m) | RMSE degC | mean sigma degC | 68% coverage | 95% coverage |")
        L.append("|---|---|---|---|---|")
        for k, z in enumerate(depths):
            L.append(f"| {z} | {cal['rmse'][k]:.3f} | {cal['mean_sigma'][k]:.3f} | "
                     f"{cal['coverage_68'][k]*100:.1f}% | {cal['coverage_95'][k]*100:.1f}% |")
    return "\n".join(L) + "\n"


def save_results(res: dict, out_dir: Path, tag: str = "evaluation") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{tag}.json"
    mp = out_dir / f"{tag}.md"
    jp.write_text(json.dumps(res, indent=2))
    mp.write_text(markdown_report(res))
    log.info("wrote %s and %s", jp, mp)
    return jp, mp
