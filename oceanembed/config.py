"""Typed configuration for the OceanEmbed framework.

A single YAML file drives data generation, training, evaluation and inference so
that every run is reproducible from one artefact.  ``configs/default.yaml`` holds
the shipped defaults; ``Config.load`` deep-merges a user file on top of them.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from datetime import date as _date, datetime as _datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class PathsConfig:
    root: str = str(REPO_ROOT)
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    cache: str = "data/cache"
    checkpoints: str = "outputs/checkpoints"
    figures: str = "outputs/figures"
    predictions: str = "outputs/predictions"
    reports: str = "outputs/reports"

    def resolve(self, key: str) -> Path:
        p = Path(getattr(self, key))
        out = p if p.is_absolute() else Path(self.root) / p
        out.mkdir(parents=True, exist_ok=True)
        return out


@dataclass
class DomainConfig:
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0
    resolution: float = 0.25


@dataclass
class DataConfig:
    """Which dataset to build, over what period, and how to split it.

    Date fields are normalised to ISO strings on construction: YAML turns an
    unquoted ``2021-01-01`` into a ``datetime.date``, which would then make the
    config unserialisable when it is embedded in the dataset manifest and in
    every checkpoint.
    """

    source: str = "synthetic"          # synthetic | cmems
    start: str = "2021-01-01"
    end: str = "2023-12-31"
    train_end: str = "2022-12-31"      # train  = [start, train_end]
    val_end: str = "2023-06-30"        # val    = (train_end, val_end]
                                        # test   = (val_end, end]
    seed: int = 20260914

    # Observing-system realism applied to the *inputs* only.
    obs_noise: bool = True
    cloud_gaps: bool = True            # microwave/IR style gaps in SST & SSS
    gap_fraction: float = 0.12         # mean fraction of masked SST pixels/day

    # Patch sampling for training.
    patch: int = 32                    # 32 px @ 0.25deg = 8deg x 8deg window
    patches_per_day: int = 12
    min_sea_fraction: float = 0.55     # reject patches that are mostly land

    # ARGO-like validation profiles withheld from training entirely.
    argo_profiles_per_day: int = 28

    def __post_init__(self) -> None:
        for f in ("start", "end", "train_end", "val_end"):
            v = getattr(self, f)
            if isinstance(v, (_date, _datetime)):
                setattr(self, f, v.isoformat()[:10])
            elif not isinstance(v, str):
                setattr(self, f, str(v))


@dataclass
class ModelConfig:
    encoder: str = "cnn_vit"           # cnn | cnn_vit | unet | autoencoder
    embed_dim: int = 128               # dimensionality of the satellite embedding
    stem_width: int = 64
    depth_blocks: int = 4              # transformer blocks
    n_heads: int = 4
    patch_size: int = 4                # ViT tokenisation over the CNN feature map
    dropout: float = 0.1
    decoder_hidden: int = 256
    decoder_layers: int = 3
    max_inversion: float = 1.5         # degC of permitted temperature inversion
    predict_uncertainty: bool = True   # heteroscedastic sigma per depth level
    use_static: bool = True            # lat/lon/day-of-year/land-distance features
    profile_head: str = "monotone"     # monotone (physics-constrained) | free


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    amp: bool = False                  # CPU default; enable on CUDA
    num_workers: int = 4
    device: str = "auto"
    threads: int = 0                   # 0 = let torch decide
    log_every: int = 25
    early_stop_patience: int = 8
    # Loss weights
    w_profile: float = 1.0             # Huber / Gaussian-NLL on T(z)
    w_strat: float = 0.05              # thermocline stratification penalty
    w_grad: float = 0.15               # horizontal gradient matching (anti-blur)
    w_recon: float = 0.10              # surface auto-encoding (embedding regulariser)
    w_aux: float = 0.20                # MLD / D20 auxiliary supervision
    w_nll: float = 0.05                # predictive-sigma calibration
    huber_delta: float = 1.0
    seed: int = 1337


@dataclass
class EvalConfig:
    baselines: tuple[str, ...] = ("climatology", "linear", "sla_regression", "mlp")
    metrics_depths: tuple[int, ...] = tuple()   # empty -> all standard depths
    regions: tuple[str, ...] = ("full", "bay_of_bengal", "arabian_sea")
    bootstrap: int = 200


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "oceanembed_v1"

    # ---- (de)serialisation ------------------------------------------------
    @staticmethod
    def _merge(base: dict, over: dict) -> dict:
        out = copy.deepcopy(base)
        for k, v in (over or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Config._merge(out[k], v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        sections = {
            "paths": PathsConfig, "domain": DomainConfig, "data": DataConfig,
            "model": ModelConfig, "train": TrainConfig, "eval": EvalConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            sub = d.get(name, {}) or {}
            valid = {f.name for f in dataclasses.fields(klass)}
            unknown = set(sub) - valid
            if unknown:
                raise ValueError(f"unknown keys in config section {name!r}: {sorted(unknown)}")
            kwargs[name] = klass(**sub)
        if "run_name" in d:
            kwargs["run_name"] = d["run_name"]
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict | None = None) -> "Config":
        base: dict = {}
        if DEFAULT_CONFIG.exists():
            base = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
        if path is not None and Path(path) != DEFAULT_CONFIG:
            base = cls._merge(base, yaml.safe_load(Path(path).read_text()) or {})
        if overrides:
            base = cls._merge(base, overrides)
        return cls.from_dict(base)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def __str__(self) -> str:  # pragma: no cover - human readable dump
        return json.dumps(self.to_dict(), indent=2, default=str)


def parse_overrides(items: list[str] | None) -> dict:
    """Turn ``["train.lr=1e-3", "model.embed_dim=256"]`` into a nested dict."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            val = yaml.safe_load(raw)
        except yaml.YAMLError:
            val = raw
        # YAML 1.1 does not recognise "1e-3" as a float (it wants "1.0e-3"),
        # which would silently pass a string into a float field.
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out
