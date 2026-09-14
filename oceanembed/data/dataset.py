"""PyTorch datasets over the memory-mapped analysis arrays.

Two access patterns share one backing store (:class:`OceanStore`):

* :class:`PatchDataset` - random 8deg x 8deg windows, used for training.
* Full-field access via :meth:`OceanStore.field` - used for evaluation and for
  producing the daily gridded product.

Input channels handed to the network (19 by default):

======  ==========================================================
 0-6    gap-filled surface observations, standardised
 7-13   gap flags (1 = value was reconstructed, not observed)
14-15   sin/cos of day-of-year
16-17   normalised latitude and longitude
   18   normalised log distance to the nearest coast
======  ==========================================================

The gap flags matter: without them the network cannot tell an observation from
an interpolation and ends up over-trusting the filled pixels, which in this
basin means over-trusting SSS within a degree of the coast.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..grid import SURFACE_VARS, Domain

log = logging.getLogger(__name__)

N_STATIC = 5
N_INPUT_CHANNELS = 2 * len(SURFACE_VARS) + N_STATIC


class OceanStore:
    """Memory-mapped access to a built dataset, with normalisation applied."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.inputs = np.load(self.root / "inputs.npy", mmap_mode="r")
        self.gapflag = np.load(self.root / "gapflag.npy", mmap_mode="r")
        self.target = np.load(self.root / "target.npy", mmap_mode="r")
        self.aux = np.load(self.root / "aux.npy", mmap_mode="r")
        self.mask = np.load(self.root / "seamask.npy")
        self.dates = [date.fromisoformat(d) for d in self.manifest["dates"]]
        self.splits = {k: list(v) for k, v in self.manifest["splits"].items()}
        self.depths = np.asarray(self.manifest["depths"], dtype="float32")

        n = self.manifest["norm"]
        f32 = lambda k: np.asarray(n[k], dtype="float32")
        self.x_mean, self.x_std = f32("x_mean"), f32("x_std")
        self.y_mean, self.y_std = f32("y_mean"), f32("y_std")
        self.a_mean, self.a_std = f32("a_mean"), f32("a_std")

        d = self.manifest["domain"]
        self.domain = Domain(d["lat_min"], d["lat_max"], d["lon_min"], d["lon_max"],
                             d["resolution"])
        self.shape = tuple(d["shape"])
        self._build_static()

    # -- static channels ----------------------------------------------------
    def _build_static(self) -> None:
        from scipy.ndimage import distance_transform_edt
        LAT, LON = self.domain.meshgrid()
        dom = self.domain
        lat_n = (LAT - dom.lat_min) / (dom.lat_max - dom.lat_min) * 2.0 - 1.0
        lon_n = (LON - dom.lon_min) / (dom.lon_max - dom.lon_min) * 2.0 - 1.0
        dist = distance_transform_edt(self.mask) * dom.resolution
        dist_n = np.log1p(dist) / np.log1p(dist.max())
        # (3, H, W): the two day-of-year channels are prepended per sample.
        self.static = np.stack([lat_n, lon_n, dist_n]).astype("float32")

    def _doy_channels(self, t: int) -> np.ndarray:
        doy = self.dates[t].timetuple().tm_yday
        ang = 2.0 * np.pi * doy / 365.25
        return np.stack([
            np.full(self.shape, np.sin(ang), dtype="float32"),
            np.full(self.shape, np.cos(ang), dtype="float32"),
        ])

    # -- accessors ----------------------------------------------------------
    def x_full(self, t: int) -> np.ndarray:
        """Normalised input tensor ``(19, H, W)`` for day index ``t``."""
        surf = (np.asarray(self.inputs[t]) - self.x_mean[:, None, None]) / self.x_std[:, None, None]
        flag = np.asarray(self.gapflag[t], dtype="float32")
        return np.concatenate([surf, flag, self._doy_channels(t), self.static]).astype("float32")

    def y_full(self, t: int, normalise: bool = True) -> np.ndarray:
        y = np.asarray(self.target[t], dtype="float32")
        if normalise:
            y = (y - self.y_mean[:, None, None]) / self.y_std[:, None, None]
        return np.nan_to_num(y, nan=0.0)

    def a_full(self, t: int, normalise: bool = True) -> np.ndarray:
        a = np.asarray(self.aux[t], dtype="float32")
        if normalise:
            a = (a - self.a_mean[:, None, None]) / self.a_std[:, None, None]
        return np.nan_to_num(a, nan=0.0)

    def field(self, t: int) -> dict[str, torch.Tensor]:
        """Whole-domain sample, batch dimension included."""
        return {
            "x": torch.from_numpy(self.x_full(t))[None],
            "y": torch.from_numpy(self.y_full(t))[None],
            "aux": torch.from_numpy(self.a_full(t))[None],
            "mask": torch.from_numpy(self.mask.astype("float32"))[None],
            "t": torch.tensor([t]),
        }

    def denormalise(self, y_norm: np.ndarray) -> np.ndarray:
        """Map a normalised profile stack back to degrees Celsius."""
        return y_norm * self.y_std[:, None, None] + self.y_mean[:, None, None]


class PatchDataset(Dataset):
    """Random spatial windows from a fixed set of days.

    Valid window origins are precomputed from the sea mask using a summed-area
    table, so sampling never rejects-and-retries and every epoch covers the
    basin evenly.
    """

    def __init__(self, store: OceanStore, split: str, patch: int = 32,
                 per_day: int = 12, min_sea_fraction: float = 0.55,
                 seed: int = 0, augment: bool = True):
        self.store = store
        self.days = store.splits[split]
        self.patch = int(patch)
        self.per_day = int(per_day)
        self.augment = augment and split == "train"
        self.seed = seed
        self.epoch = 0
        self.origins = self._valid_origins(min_sea_fraction)
        if self.origins.size == 0:
            raise ValueError(f"no patch of size {patch} reaches {min_sea_fraction:.0%} ocean")
        log.info("%s split: %d days x %d patches, %d valid origins",
                 split, len(self.days), self.per_day, len(self.origins))

    def _valid_origins(self, min_frac: float) -> np.ndarray:
        m = self.store.mask.astype("float64")
        ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)   # summed-area table
        p, (H, W) = self.patch, self.store.shape
        if p > H or p > W:
            raise ValueError(f"patch {p} exceeds grid {H}x{W}")
        counts = (ii[p:, p:] - ii[:-p, p:] - ii[p:, :-p] + ii[:-p, :-p])
        frac = counts / (p * p)
        oi, oj = np.nonzero(frac >= min_frac)
        return np.stack([oi, oj], axis=1).astype("int32")

    def set_epoch(self, epoch: int) -> None:
        """Reseed patch selection so each epoch sees different windows."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.days) * self.per_day

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        day = self.days[idx // self.per_day]
        # Deterministic given (epoch, idx): reproducible yet varied across epochs.
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        oi, oj = self.origins[rng.integers(len(self.origins))]
        p = self.patch
        sl = (slice(oi, oi + p), slice(oj, oj + p))

        x = self.store.x_full(day)[(slice(None),) + sl]
        y = self.store.y_full(day)[(slice(None),) + sl]
        a = self.store.a_full(day)[(slice(None),) + sl]
        m = self.store.mask[sl].astype("float32")

        if self.augment:
            # Reflections, but never transposition: the basin is not isotropic
            # (the Coriolis parameter and the monsoon both pick out north).
            #
            # A reflection is only physically meaningful if everything that
            # carries orientation is transformed with it.  Reversing the window
            # east-west therefore also negates the zonal velocity and wind
            # components - the mirrored flow really does run the other way - and
            # negates the normalised longitude channel, which maps the window
            # onto the mirror-image position in the basin so that longitude
            # still increases from left to right.  Omitting either negation
            # would teach the network an inconsistent coordinate frame.
            if rng.random() < 0.5:                      # flip east-west
                x, y, a, m = (np.ascontiguousarray(v[..., ::-1]) for v in (x, y, a, m))
                for c in (3, 5, 17):                    # ucur, uwnd, lon
                    x[c] *= -1.0
            if rng.random() < 0.5:                      # flip north-south
                x, y, a, m = (np.ascontiguousarray(v[..., ::-1, :]) for v in (x, y, a, m))
                for c in (4, 6, 16):                    # vcur, vwnd, lat
                    x[c] *= -1.0

        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y),
                "aux": torch.from_numpy(a), "mask": torch.from_numpy(m),
                "t": torch.tensor(day)}


def build_loaders(store: OceanStore, cfg) -> dict[str, "torch.utils.data.DataLoader"]:
    """Training and validation loaders from a config."""
    from torch.utils.data import DataLoader
    out = {}
    for split in ("train", "val"):
        ds = PatchDataset(store, split, patch=cfg.data.patch,
                          per_day=cfg.data.patches_per_day,
                          min_sea_fraction=cfg.data.min_sea_fraction,
                          seed=cfg.train.seed, augment=(split == "train"))
        out[split] = DataLoader(
            ds, batch_size=cfg.train.batch_size, shuffle=(split == "train"),
            num_workers=cfg.train.num_workers, drop_last=(split == "train"),
            persistent_workers=cfg.train.num_workers > 0,
        )
    return out
