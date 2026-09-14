"""End-to-end pipeline tests on a small synthetic dataset.

A 40-day dataset is built once per session in a temporary directory, then the
observation operator, dataset plumbing, model, loss and training loop are all
exercised against it.  Slow but it is the only test that would catch a break in
the wiring between stages.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from oceanembed.config import Config, parse_overrides
from oceanembed.grid import N_DEPTH, SURFACE_VARS, sea_mask


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from oceanembed.data.build import build
    root = tmp_path_factory.mktemp("ds")
    cfg = Config.load("configs/default.yaml", parse_overrides([
        "data.start=2021-01-01", "data.end=2021-02-09",
        "data.train_end=2021-01-24", "data.val_end=2021-02-01",
        f"paths.root={root}",
    ]))
    out = build(cfg, write_netcdf=False)
    return cfg, out


# --- observation operator -------------------------------------------------
def test_observation_operator_respects_error_budget():
    import datetime as dt
    from oceanembed.data.harmonize import ERROR_BUDGET, observe
    from oceanembed.data.synthetic import NIOSimulator

    sim = NIOSimulator(seed=2)
    truth, _ = sim.step(dt.date(2021, 6, 15))
    obs, valid = observe(truth, np.random.default_rng(0))
    m = sea_mask()
    for v in SURFACE_VARS:
        err = np.nanstd((obs[v] - truth[v])[m])
        eb = ERROR_BUDGET[v]
        budget = np.hypot(eb.sigma_white, eb.sigma_corr)
        # The realised error must match the declared budget within 35%.
        assert 0.65 * budget < err < 1.35 * budget, f"{v}: {err:.3f} vs {budget:.3f}"


def test_sss_is_blanked_near_the_coast():
    """SMAP/SMOS cannot retrieve salinity within ~1 degree of land."""
    import datetime as dt
    from scipy.ndimage import distance_transform_edt

    from oceanembed.data.harmonize import SSS_COAST_BLIND_DEG, observe
    from oceanembed.data.synthetic import NIOSimulator

    sim = NIOSimulator(seed=2)
    truth, _ = sim.step(dt.date(2021, 6, 15))
    obs, valid = observe(truth, np.random.default_rng(0))
    dist = distance_transform_edt(sea_mask()) * 0.25
    near = sea_mask() & (dist <= SSS_COAST_BLIND_DEG)
    assert not valid["sss"][near].any()


def test_gap_filling_leaves_no_holes_in_the_ocean():
    import datetime as dt
    from oceanembed.data.harmonize import observe, preprocess_day
    from oceanembed.data.synthetic import NIOSimulator

    sim = NIOSimulator(seed=2)
    truth, _ = sim.step(dt.date(2021, 6, 15))
    obs, _ = observe(truth, np.random.default_rng(0))
    x, flag = preprocess_day(obs)
    m = sea_mask()
    assert np.isfinite(x[:, m]).all()
    assert flag[:, m].max() == 1.0 and flag[:, m].min() == 0.0


# --- dataset --------------------------------------------------------------
def test_build_produces_all_artefacts(tiny_dataset):
    cfg, out = tiny_dataset
    for f in ("inputs.npy", "gapflag.npy", "target.npy", "aux.npy",
              "argo.npz", "manifest.json", "seamask.npy"):
        assert (out / f).exists(), f
    m = json.loads((out / "manifest.json").read_text())
    assert len(m["dates"]) == 40
    assert sum(len(v) for v in m["splits"].values()) == 40


def test_splits_are_chronological_and_disjoint(tiny_dataset):
    cfg, out = tiny_dataset
    m = json.loads((out / "manifest.json").read_text())
    tr, va, te = (m["splits"][k] for k in ("train", "val", "test"))
    assert set(tr).isdisjoint(va) and set(va).isdisjoint(te) and set(tr).isdisjoint(te)
    # Temporal order matters: a random split would leak the mesoscale field,
    # which is autocorrelated over weeks, from train into test.
    assert max(tr) < min(va) and max(va) < min(te)


def test_normalisation_uses_training_days_only(tiny_dataset):
    """Recomputing the statistics from the training slice must reproduce them."""
    from oceanembed.data.dataset import OceanStore
    cfg, out = tiny_dataset
    store = OceanStore(out)
    m = store.mask
    tr = store.splits["train"]
    y = np.asarray(store.target[tr])[:, :, m]
    assert np.allclose(store.y_mean, np.nanmean(y, axis=(0, 2)), atol=1e-3)
    assert np.allclose(store.y_std, np.nanstd(y, axis=(0, 2)), atol=1e-3)


def test_store_channels_and_finiteness(tiny_dataset):
    from oceanembed.data.dataset import N_INPUT_CHANNELS, OceanStore
    cfg, out = tiny_dataset
    store = OceanStore(out)
    f = store.field(3)
    assert f["x"].shape == (1, N_INPUT_CHANNELS) + store.shape
    assert f["y"].shape == (1, N_DEPTH) + store.shape
    assert torch.isfinite(f["x"]).all() and torch.isfinite(f["y"]).all()


def test_patch_augmentation_keeps_the_coordinate_frame_consistent(tiny_dataset):
    """A mirrored window must still describe a physically coherent patch.

    Reversing a window east-west and negating the longitude channel maps a patch
    in the eastern basin onto a plausible patch in the western basin: longitude
    still increases left to right, and the zonal velocity components have the
    sign the mirrored flow would have.  If the negation were omitted the network
    would be shown windows whose longitude runs backwards.
    """
    from oceanembed.data.dataset import PatchDataset, OceanStore
    cfg, out = tiny_dataset
    store = OceanStore(out)
    plain = PatchDataset(store, "train", patch=16, per_day=1, augment=False)
    aug = PatchDataset(store, "train", patch=16, per_day=1, augment=True)

    LON_CH, LAT_CH = 17, 16
    flipped = 0
    for i in range(24):
        a = aug[i]
        # Coordinate channels must stay monotone in the expected direction.
        assert np.all(np.diff(a["x"][LON_CH, 0].numpy()) > 0), "longitude runs backwards"
        assert np.all(np.diff(a["x"][LAT_CH, :, 0].numpy()) > 0), "latitude runs backwards"
        # Same index, same window (the origin is drawn before any flip), so any
        # difference in the target is the augmentation at work.
        if not torch.allclose(a["y"], plain[i]["y"]):
            flipped += 1
    assert 0 < flipped < 24, f"expected a mix of flipped and unflipped windows, got {flipped}"


def test_patch_augmentation_negates_zonal_velocity(tiny_dataset):
    """The east-west mirror must flip the sign of u, not just the array order."""
    from oceanembed.data.dataset import PatchDataset, OceanStore
    cfg, out = tiny_dataset
    store = OceanStore(out)
    plain = PatchDataset(store, "train", patch=16, per_day=4, augment=False)
    aug = PatchDataset(store, "train", patch=16, per_day=4, augment=True)
    UCUR = 3
    checked = 0
    for i in range(len(aug)):
        a, b = aug[i]["x"], plain[i]["x"]
        # Detect a pure east-west flip via the (flip-invariant) SST channel.
        ew = torch.allclose(a[0], b[0].flip(-1)) and not torch.allclose(a[0], b[0])
        if ew:
            assert torch.allclose(a[UCUR], -b[UCUR].flip(-1), atol=1e-5)
            checked += 1
    assert checked > 0, "no pure east-west flip occurred in any window"


def test_denormalise_roundtrip(tiny_dataset):
    from oceanembed.data.dataset import OceanStore
    cfg, out = tiny_dataset
    store = OceanStore(out)
    raw = np.asarray(store.target[2])
    back = store.denormalise(store.y_full(2))
    m = store.mask
    assert np.allclose(back[:, m], raw[:, m], atol=1e-3)


# --- model and loss -------------------------------------------------------
@pytest.mark.parametrize("encoder", ["cnn", "cnn_vit", "unet", "autoencoder"])
def test_encoders_accept_odd_sizes(tiny_dataset, encoder):
    from oceanembed.data.dataset import N_INPUT_CHANNELS, OceanStore
    from oceanembed.models.oceanembed import OceanEmbed
    cfg, out = tiny_dataset
    store = OceanStore(out)
    cfg.model.encoder = encoder
    cfg.model.embed_dim, cfg.model.stem_width = 32, 16
    model = OceanEmbed(cfg.model, store.y_mean, store.y_std, N_INPUT_CHANNELS)
    with torch.no_grad():
        o = model(torch.randn(1, N_INPUT_CHANNELS, 23, 37))
    assert o["y"].shape == (1, N_DEPTH, 23, 37)
    assert torch.isfinite(o["y"]).all()


def test_monotone_head_starts_at_climatology(tiny_dataset):
    from oceanembed.data.dataset import N_INPUT_CHANNELS, OceanStore
    from oceanembed.models.oceanembed import OceanEmbed
    cfg, out = tiny_dataset
    store = OceanStore(out)
    cfg.model.embed_dim, cfg.model.stem_width = 32, 16
    model = OceanEmbed(cfg.model, store.y_mean, store.y_std, N_INPUT_CHANNELS,
                       profile_head="monotone")
    with torch.no_grad():
        y, _ = model.profile(torch.zeros(1, cfg.model.embed_dim, 4, 4))
        t = model.profile.to_celsius(y)[0, :, 0, 0].numpy()
    assert np.allclose(t, store.y_mean, atol=1.0)


def test_monotone_head_respects_the_inversion_bound(tiny_dataset):
    """However extreme the activations, the profile may not invert by more than
    max_inversion - this is the constraint the parameterisation exists to give."""
    from oceanembed.data.dataset import N_INPUT_CHANNELS, OceanStore
    from oceanembed.models.oceanembed import OceanEmbed
    cfg, out = tiny_dataset
    store = OceanStore(out)
    cfg.model.embed_dim, cfg.model.stem_width = 32, 16
    cfg.model.max_inversion = 1.5
    model = OceanEmbed(cfg.model, store.y_mean, store.y_std, N_INPUT_CHANNELS,
                       profile_head="monotone")
    rng = torch.Generator().manual_seed(0)
    for scale in (1.0, 20.0, 200.0):
        z = torch.randn(4, cfg.model.embed_dim, 8, 8, generator=rng) * scale
        with torch.no_grad():
            y, _ = model.profile(z)
            t = model.profile.to_celsius(y)
        worst = float(t.diff(dim=1).max())
        # Two neighbouring inversion terms can differ by at most 2*max_inversion.
        assert worst <= 2 * cfg.model.max_inversion + 1e-3, f"scale {scale}: {worst:.3f}"


def test_stratification_penalty_detects_inversion(tiny_dataset):
    from oceanembed.data.dataset import OceanStore
    from oceanembed.losses import ProfileLoss
    cfg, out = tiny_dataset
    store = OceanStore(out)
    crit = ProfileLoss(cfg.train, store.y_mean, store.y_std, 1.5)
    shape = (2, N_DEPTH, 8, 8)
    batch = {"y": torch.zeros(shape), "mask": torch.ones(2, 8, 8),
             "x": torch.zeros(2, 7, 8, 8), "aux": torch.zeros(2, 2, 8, 8)}
    ym, ys = store.y_mean, store.y_std
    to_norm = lambda T: torch.as_tensor((T - ym) / ys, dtype=torch.float32
                                        ).view(1, -1, 1, 1).expand(shape).contiguous()
    assert crit({"y": to_norm(ym), "log_sigma": None}, batch)[1]["strat"] == pytest.approx(0.0)
    reversed_col = crit({"y": to_norm(ym[::-1].copy()), "log_sigma": None}, batch)[1]["strat"]
    assert reversed_col > 0.1


def test_predict_field_tiling_matches_single_pass(tiny_dataset):
    from oceanembed.data.dataset import N_INPUT_CHANNELS, OceanStore
    from oceanembed.models.oceanembed import OceanEmbed, predict_field
    cfg, out = tiny_dataset
    store = OceanStore(out)
    cfg.model.encoder = "cnn"          # fully convolutional: tiling is near-exact
    cfg.model.embed_dim, cfg.model.stem_width = 32, 16
    model = OceanEmbed(cfg.model, store.y_mean, store.y_std, N_INPUT_CHANNELS).eval()
    x = torch.from_numpy(store.x_full(1))[None]
    whole = predict_field(model, x, tile=0)["y"]
    tiled = predict_field(model, x, tile=48, overlap=16)["y"]
    assert whole.shape == tiled.shape
    assert (whole - tiled).abs().mean() < 0.2


# --- training -------------------------------------------------------------
def test_training_reduces_error_and_writes_checkpoint(tiny_dataset):
    from oceanembed.train import load_model, train
    cfg, out = tiny_dataset
    cfg.run_name = "pytest"
    cfg.model.embed_dim, cfg.model.stem_width, cfg.model.depth_blocks = 48, 24, 2
    cfg.data.patches_per_day, cfg.train.batch_size = 8, 16
    cfg.train.epochs, cfg.train.num_workers, cfg.train.threads = 3, 0, 2
    ckpt = train(cfg, data_dir=out)
    assert ckpt.exists()
    model, state = load_model(ckpt)
    hist = state["history"]
    assert len(hist) >= 2
    assert hist[-1]["val_rmse_mean"] < hist[0]["val_rmse_mean"], "validation RMSE must improve"
    # Three epochs on 16 training days will not be good, but it must at least
    # not be worse than predicting the climatological mean everywhere.
    from oceanembed.data.dataset import OceanStore
    assert hist[-1]["val_rmse_mean"] < float(np.mean(OceanStore(out).y_std)) * 1.2
