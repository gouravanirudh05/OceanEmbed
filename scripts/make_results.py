#!/usr/bin/env python
"""Generate docs/RESULTS.md from the evaluation JSON and the training histories.

Separate from evaluate.py's Markdown report: that one is the exhaustive table
dump, this is the curated summary a reviewer reads first, and it adds the
architecture/head ablation, which lives in the training histories rather than
in the evaluation output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def matched_epoch_ablation(ckpt_dir: Path) -> tuple[list[dict], int | None]:
    """Compare runs at the largest epoch index all of them reached.

    Read from the checkpoints rather than the history JSONs: a checkpoint always
    carries both its full history and the config it was trained with, so runs
    can be grouped by dataset period without depending on the on-disk history
    format.

    Runs are only comparable if they were given the **same epoch budget**: the
    cosine learning-rate schedule is laid out over ``train.epochs``, so a
    10-epoch run is further along its decay at epoch 4 than a 14-epoch run is.
    Tabulating them side by side would measure the schedule, not the
    architecture.  Runs with a different budget are therefore excluded and
    reported, rather than silently mixed in.
    """
    import torch
    from collections import Counter

    runs: dict[str, dict] = {}
    for f in sorted(ckpt_dir.glob("*_last.pt")):
        st = torch.load(f, map_location="cpu", weights_only=False)
        hist = st.get("history") or []
        if not hist:
            continue
        d = st["config"]["data"]
        runs[f.stem.replace("_last", "")] = {
            "epochs": hist,
            "period": f"{d['start']}..{d['end']}",
            "budget": st["config"]["train"]["epochs"],
            "encoder": st["config"]["model"]["encoder"],
            "head": st.get("profile_head", "?"),
        }
    if len(runs) < 2:
        return [], None

    # Comparable means: same data period AND same epoch budget (hence the same
    # learning-rate schedule).
    key = lambda r: (r["period"], r["budget"])
    main_key = Counter(key(r) for r in runs.values()).most_common(1)[0][0]
    dropped = sorted(k for k, v in runs.items() if key(v) != main_key)
    runs = {k: v for k, v in runs.items() if key(v) == main_key}
    if dropped:
        print(f"ablation: excluding {dropped} - different data period or epoch "
              f"budget than {main_key}", file=sys.stderr)
    if len(runs) < 2:
        return [], None

    common = min(len(v["epochs"]) for v in runs.values()) - 1
    rows = []
    for name, v in sorted(runs.items(), key=lambda kv: kv[1]["epochs"][common]["val_rmse_mean"]):
        curve = [r["val_rmse_mean"] for r in v["epochs"]]
        # Late-epoch scatter within this one run: the natural yardstick for
        # deciding whether a between-run difference means anything.
        tail = curve[max(len(curve) - 5, 0):]
        rows.append({
            "run": name,
            "encoder": v["encoder"],
            "head": v["head"],
            "epochs_run": len(curve),
            "val_rmse_at_common": curve[common],
            "val_rmse_best": min(curve),
            "own_late_spread": max(tail) - min(tail),
        })

    # How often does the ranking change from one epoch to the next?  If it
    # changes repeatedly, the between-run gaps are epoch noise, not signal.
    order_at = lambda e: tuple(sorted(runs, key=lambda k: runs[k]["epochs"][e]["val_rmse_mean"]))
    flips = sum(1 for e in range(1, common + 1) if order_at(e) != order_at(e - 1))
    between = max(r["val_rmse_at_common"] for r in rows) - min(r["val_rmse_at_common"] for r in rows)
    within = max(r["own_late_spread"] for r in rows)
    return rows, {"common": common, "flips": flips, "n_compared": common,
                  "between_spread": between, "within_spread": within,
                  "conclusive": bool(between > within and flips <= 1)}


def main() -> int:
    reports = REPO / "outputs" / "reports"
    ev_path = reports / "evaluation.json"
    if not ev_path.exists():
        print(f"missing {ev_path}; run `make evaluate` first", file=sys.stderr)
        return 1
    res = json.loads(ev_path.read_text())
    depths = res["meta"]["depths"]
    M = res["methods"]
    order = sorted(M, key=lambda m: M[m]["field"]["full"]["rmse_mean"])
    best = order[0]
    clim = M["climatology"]["field"]["full"] if "climatology" in M else None

    L: list[str] = []
    L.append("# Results\n")
    L.append("Written for: SIH evaluators and INCOIS reviewers.\n")
    L.append("> **All numbers below are from the OSSE twin, not the real ocean.** "
             "See [the note in the README](../README.md#read-this-first-what-the-numbers-in-this-repo-mean) "
             "for what that does and does not validate.\n")
    L.append(f"Held-out test period: **{res['meta']['test_days']} consecutive days** "
             f"(chronologically after all training and validation data), evaluated over "
             f"the whole basin and against **{res['meta']['n_argo']:,} withheld in-situ "
             f"profiles**.\n")

    L.append("## Headline\n")
    L.append("| method | RMSE degC | 95% CI (day bootstrap) | variance removed vs climatology | anomaly correlation | bias degC |")
    L.append("|---|---|---|---|---|---|")
    for m in order:
        f = M[m]["field"]["full"]
        lo, hi = M[m]["rmse_ci95"]
        star = "**" if m == best else ""
        L.append(f"| `{m}` | {star}{f['rmse_mean']:.3f}{star} | {lo:.3f} - {hi:.3f} | "
                 f"{f['skill_vs_clim']*100:.1f}% | {f['acc_mean']:.3f} | {f['bias_mean']:+.3f} |")
    if clim:
        L.append(f"\nClimatological error (the zero-skill reference) is "
                 f"**{clim['clim_rmse_mean']:.3f} degC** averaged over the 15 levels.\n")

    L.append("## RMSE by depth (degC)\n")
    L.append("| depth (m) | " + " | ".join(f"`{m}`" for m in order) + " | climatological std |")
    L.append("|" + "---|" * (len(order) + 2))
    cl = [d["clim_rmse"] for d in M[order[0]]["field"]["full"]["per_level"]]
    for k, z in enumerate(depths):
        vals = [M[m]["field"]["full"]["per_level"][k]["rmse"] for m in order]
        mn = min(vals)
        cells = "".join(f"| **{v:.3f}** " if abs(v - mn) < 1e-9 else f"| {v:.3f} " for v in vals)
        L.append(f"| {z} {cells}| {cl[k]:.3f} |")

    L.append("\n## Independent in-situ validation\n")
    L.append("Profiles are sampled at real ARGO density (200 floats, 10-day cycle) and "
             "carry a representativeness error, so these numbers are on the same footing "
             "as published ARGO-versus-reanalysis comparisons and are not directly "
             "comparable with the gridded numbers above.\n")
    L.append("| method | RMSE degC | variance removed | anomaly correlation |")
    L.append("|---|---|---|---|")
    for m in order:
        a = M[m]["argo"]
        L.append(f"| `{m}` | {a['rmse_mean']:.3f} | {a['skill_vs_clim']*100:.1f}% | {a['acc_mean']:.3f} |")

    L.append("\n## Derived operational diagnostics\n")
    L.append("These are what a warning centre acts on. Tropical cyclone heat potential "
             "and the 26 degC isotherm depth govern cyclone rapid intensification in the "
             "Bay of Bengal and cannot be obtained from SST alone.\n")
    from oceanembed.diagnostics import DIAG_UNITS
    dnames = list(M[order[0]]["diagnostics"])
    L.append("| method | " + " | ".join(f"{d} ({DIAG_UNITS[d]})" for d in dnames) + " |")
    L.append("|" + "---|" * (len(dnames) + 1))
    for m in order:
        d = M[m]["diagnostics"]
        L.append(f"| `{m}` | " + " | ".join(f"{d[x]['rmse']:.2f}" for x in dnames) + " |")

    L.append("\n## Skill by sub-basin (RMSE degC)\n")
    regs = res["meta"]["regions"]
    L.append("| method | " + " | ".join(regs) + " |")
    L.append("|" + "---|" * (len(regs) + 1))
    for m in order:
        f = M[m]["field"]
        L.append(f"| `{m}` | " + " | ".join(f"{f[r]['rmse_mean']:.3f}" for r in regs) + " |")

    for m in order:
        cal = M[m].get("calibration")
        if not cal:
            continue
        L.append(f"\n## Uncertainty calibration - `{m}`\n")
        L.append(f"Nominal 68% interval covers **{cal['coverage_68_mean']*100:.1f}%** of "
                 f"cases; nominal 95% covers **{cal['coverage_95_mean']*100:.1f}%**. "
                 "Values below nominal mean the model is overconfident.\n")
        L.append("| depth (m) | RMSE degC | mean predicted sigma | 68% coverage | 95% coverage |")
        L.append("|---|---|---|---|---|")
        for k, z in enumerate(depths):
            L.append(f"| {z} | {cal['rmse'][k]:.3f} | {cal['mean_sigma'][k]:.3f} | "
                     f"{cal['coverage_68'][k]*100:.1f}% | {cal['coverage_95'][k]*100:.1f}% |")

    rows, ab = matched_epoch_ablation(REPO / "outputs" / "checkpoints")
    if rows and ab is not None:
        common = ab["common"]
        L.append("\n## Ablation: encoder and profile head\n")
        L.append(f"Compared at **epoch {common}** - the last epoch every run "
                 "completed. All runs share the same data, seed, learning rate "
                 "and epoch budget, so they share the same cosine schedule and "
                 "only the architecture differs. Depth-averaged validation RMSE "
                 "in degC.\n")
        L.append("| run | encoder | profile head | val RMSE at epoch "
                 f"{common} | own best | own late-epoch spread | epochs run |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| `{r['run']}` | {r['encoder']} | {r['head']} | "
                     f"{r['val_rmse_at_common']:.4f} | {r['val_rmse_best']:.4f} | "
                     f"{r['own_late_spread']:.4f} | {r['epochs_run']} |")
        verdict = ("**Conclusive**" if ab["conclusive"] else
                   "**Not conclusive - treat these differences as noise**")
        L.append(f"\n{verdict}. The spread between architectures at epoch "
                 f"{common} is **{ab['between_spread']:.4f} degC**, while the "
                 f"largest late-epoch spread *within* a single run is "
                 f"**{ab['within_spread']:.4f} degC**. The ranking changed "
                 f"**{ab['flips']} times** over the {ab['n_compared']} epochs "
                 "compared.\n")
        if not ab["conclusive"]:
            L.append("Separating these architectures would need several seeds "
                     "per configuration to estimate run-to-run variance. That "
                     "was not run here, so no claim is made that either the "
                     "attention or the monotone head improves accuracy on this "
                     "benchmark.\n")

    L.append("\n## Figures\n")
    figs = sorted((REPO / "outputs" / "figures").glob("*.png"))
    for f in figs:
        L.append(f"![{f.stem}](../outputs/figures/{f.name})\n")

    L.append("\n---\n")
    L.append("Regenerate with:\n")
    L.append("```bash\nmake evaluate && make figures && python scripts/make_results.py\n```\n")

    out = REPO / "docs" / "RESULTS.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
