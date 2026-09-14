"""Command line entry point.

    python -m oceanembed.cli build      # generate / harmonise the dataset
    python -m oceanembed.cli train      # train the reconstruction model
    python -m oceanembed.cli evaluate   # skill against baselines and in-situ data
    python -m oceanembed.cli predict    # write the daily 3-D NetCDF product
    python -m oceanembed.cli figures    # regenerate the proof-of-concept figures
    python -m oceanembed.cli info       # summarise a dataset or checkpoint

Every subcommand accepts ``-c/--config`` and repeated ``-o key=value`` overrides,
so any field of :class:`oceanembed.config.Config` can be set from the shell::

    python -m oceanembed.cli train -c configs/nio_full.yaml -o train.lr=5.0e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, parse_overrides

LOG_FORMAT = "%(asctime)s %(levelname).1s %(name)s: %(message)s"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", default="configs/default.yaml")
    p.add_argument("-o", "--override", action="append", metavar="KEY=VALUE",
                   help="override a config field, repeatable")
    p.add_argument("-d", "--data-dir", default=None,
                   help="dataset directory (default: paths.processed)")
    p.add_argument("-v", "--verbose", action="store_true")


def _load(args) -> Config:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=LOG_FORMAT, datefmt="%H:%M:%S")
    return Config.load(args.config, parse_overrides(args.override))


def cmd_build(args) -> int:
    from .data.build import build
    cfg = _load(args)
    out = build(cfg, write_netcdf=not args.no_netcdf)
    print(f"dataset written to {out}")
    return 0


def cmd_train(args) -> int:
    from .train import train
    cfg = _load(args)
    ckpt = train(cfg, data_dir=args.data_dir, profile_head=args.head, resume=args.resume)
    print(f"best checkpoint: {ckpt}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate, save_results
    cfg = _load(args)
    ckpts = {}
    for spec in args.checkpoint or []:
        name, _, path = spec.partition("=")
        if not path:
            name, path = "oceanembed", name
        ckpts[name] = path
    res = evaluate(cfg, ckpts, data_dir=args.data_dir,
                   methods=tuple(args.baselines) if args.baselines else None,
                   tile=args.tile, max_days=args.max_days)
    jp, mp = save_results(res, cfg.paths.resolve("reports"), tag=args.tag)
    print(Path(mp).read_text())
    return 0


def cmd_predict(args) -> int:
    from .inference import predict_range
    cfg = _load(args)
    path = predict_range(cfg, args.checkpoint, start=args.start, end=args.end,
                         data_dir=args.data_dir, tile=args.tile, out_name=args.out)
    print(f"product written to {path}")
    return 0


def cmd_figures(args) -> int:
    from .report import make_figures
    cfg = _load(args)
    paths = make_figures(cfg, args.checkpoint, data_dir=args.data_dir,
                         results=args.results, day=args.day)
    for p in paths:
        print(p)
    return 0


def cmd_info(args) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    target = Path(args.target)
    if target.is_dir():
        m = json.loads((target / "manifest.json").read_text())
        print(f"dataset   : {target}")
        print(f"source    : {m['source']}")
        print(f"domain    : {m['domain']['lat_min']}-{m['domain']['lat_max']}degN, "
              f"{m['domain']['lon_min']}-{m['domain']['lon_max']}degE @ {m['domain']['resolution']}deg "
              f"-> {m['domain']['shape']}")
        print(f"period    : {m['dates'][0]} .. {m['dates'][-1]} ({len(m['dates'])} days)")
        print("splits    : " + ", ".join(f"{k}={len(v)}" for k, v in m["splits"].items()))
        print(f"depths    : {m['depths']}")
        print(f"in-situ   : {m['n_argo']:,} profiles withheld")
        print(f"created   : {m['created']}")
        return 0

    import torch
    st = torch.load(target, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(st["config"])
    print(f"checkpoint: {target}")
    print(f"run       : {cfg.run_name}")
    print(f"encoder   : {cfg.model.encoder} | head: {st.get('profile_head')} | "
          f"embed_dim: {cfg.model.embed_dim}")
    print(f"epoch     : {st['epoch']} | best val RMSE: {st.get('best', float('nan')):.4f} degC")
    if st.get("history"):
        h = st["history"][-1]
        print(f"last val  : loss {h['val']['total']:.4f}, RMSE {h['val_rmse_mean']:.4f} degC")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oceanembed", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate or harmonise the dataset")
    _common(b)
    b.add_argument("--no-netcdf", action="store_true", help="skip the NetCDF copy")
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train", help="train the reconstruction model")
    _common(t)
    t.add_argument("--head", default=None, choices=["monotone", "free"])
    t.add_argument("--resume", default=None)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate against baselines and in-situ profiles")
    _common(e)
    e.add_argument("--checkpoint", action="append",
                   help="NAME=path, repeatable; bare path is named 'oceanembed'")
    e.add_argument("--baselines", nargs="*", default=None)
    e.add_argument("--tile", type=int, default=0, help="tile size for inference (0 = whole field)")
    e.add_argument("--max-days", type=int, default=None)
    e.add_argument("--tag", default="evaluation")
    e.set_defaults(fn=cmd_evaluate)

    r = sub.add_parser("predict", help="write the daily 3-D product")
    _common(r)
    r.add_argument("checkpoint")
    r.add_argument("--start", default=None)
    r.add_argument("--end", default=None)
    r.add_argument("--tile", type=int, default=0)
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_predict)

    f = sub.add_parser("figures", help="regenerate the proof-of-concept figures")
    _common(f)
    f.add_argument("checkpoint")
    f.add_argument("--results", default=None, help="evaluation JSON, for the skill plots")
    f.add_argument("--day", default=None, help="ISO date to illustrate")
    f.set_defaults(fn=cmd_figures)

    i = sub.add_parser("info", help="summarise a dataset directory or a checkpoint")
    i.add_argument("target")
    i.set_defaults(fn=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
