"""Training loop for OceanEmbed."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.dataset import N_INPUT_CHANNELS, OceanStore, PatchDataset
from .grid import STANDARD_DEPTHS
from .losses import ProfileLoss
from .models.oceanembed import OceanEmbed

log = logging.getLogger(__name__)


def pick_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(step: int, total: int, warmup: int) -> float:
    """Linear warm-up then cosine decay to 1% of the peak learning rate."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def evaluate_split(model, loader, crit, y_std: np.ndarray, device) -> dict:
    """Mean loss and per-level RMSE (degC) over a loader."""
    model.eval()
    ys = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    se = torch.zeros(len(STANDARD_DEPTHS), device=device)
    n_cells = torch.zeros((), device=device)
    agg: dict[str, float] = {}
    nb = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch["x"])
        _, parts = crit(out, batch)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
        m = batch["mask"].unsqueeze(1)
        err = (out["y"] - batch["y"]) * ys
        se += ((err ** 2) * m).sum(dim=(0, 2, 3))
        n_cells += m.sum()
    rmse = torch.sqrt(se / n_cells.clamp_min(1.0)).cpu().numpy()
    return {"loss": {k: v / max(nb, 1) for k, v in agg.items()},
            "rmse": rmse, "rmse_mean": float(rmse.mean())}


def train(cfg: Config, data_dir: str | Path | None = None,
          profile_head: str | None = None, resume: str | Path | None = None) -> Path:
    """Train a model and return the path to the best checkpoint."""
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    profile_head = profile_head or cfg.model.profile_head
    store = OceanStore(data_dir or cfg.paths.resolve("processed"))
    device = pick_device(cfg.train.device)
    if cfg.train.threads > 0:
        torch.set_num_threads(cfg.train.threads)
    log.info("device: %s | threads: %d", device, torch.get_num_threads())

    from torch.utils.data import DataLoader
    train_ds = PatchDataset(store, "train", patch=cfg.data.patch,
                            per_day=cfg.data.patches_per_day,
                            min_sea_fraction=cfg.data.min_sea_fraction,
                            seed=cfg.train.seed, augment=True)
    val_ds = PatchDataset(store, "val", patch=cfg.data.patch,
                          per_day=max(cfg.data.patches_per_day // 2, 2),
                          min_sea_fraction=cfg.data.min_sea_fraction,
                          seed=cfg.train.seed + 1, augment=False)
    mk = lambda ds, sh: DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=sh,
                                   num_workers=cfg.train.num_workers, drop_last=sh,
                                   persistent_workers=cfg.train.num_workers > 0)
    train_dl, val_dl = mk(train_ds, True), mk(val_ds, False)

    model = OceanEmbed(cfg.model, store.y_mean, store.y_std,
                       N_INPUT_CHANNELS, profile_head=profile_head).to(device)
    log.info("model %s / head %s\n%s", cfg.model.encoder, profile_head, model.parameter_report())

    crit = ProfileLoss(cfg.train, store.y_mean, store.y_std, cfg.model.max_inversion).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    total_steps = max(cfg.train.epochs * len(train_dl), 1)
    warmup = int(cfg.train.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, warmup))
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    ckpt_dir = cfg.paths.resolve("checkpoints")
    tag = f"{cfg.run_name}_{cfg.model.encoder}_{profile_head}"
    best_path = ckpt_dir / f"{tag}_best.pt"
    start_epoch, best, bad_epochs = 0, float("inf"), 0
    history: list[dict] = []

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"] + 1
        best = state.get("best", float("inf"))
        history = state.get("history", [])
        log.info("resumed from %s at epoch %d", resume, start_epoch)

    for epoch in range(start_epoch, cfg.train.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        run: dict[str, float] = {}
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type,
                                enabled=cfg.train.amp and device.type == "cuda"):
                out = model(batch["x"])
                loss, parts = crit(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            if (i + 1) % cfg.train.log_every == 0:
                log.info("  e%02d %4d/%d loss %.4f (profile %.4f) lr %.2e",
                         epoch, i + 1, len(train_dl), run["total"] / (i + 1),
                         run["profile"] / (i + 1), sched.get_last_lr()[0])

        tr = {k: v / max(len(train_dl), 1) for k, v in run.items()}
        va = evaluate_split(model, val_dl, crit, store.y_std, device)
        dt = time.time() - t0
        log.info("epoch %02d | train %.4f | val %.4f | val RMSE %.3f degC | %.0fs",
                 epoch, tr["total"], va["loss"]["total"], va["rmse_mean"], dt)
        history.append({"epoch": epoch, "train": tr, "val": va["loss"],
                        "val_rmse": va["rmse"].tolist(),
                        "val_rmse_mean": va["rmse_mean"], "seconds": dt,
                        "lr": sched.get_last_lr()[0]})

        state = {"model": model.state_dict(), "optim": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "best": best,
                 "history": history, "config": cfg.to_dict(),
                 "profile_head": profile_head,
                 "norm": {"y_mean": store.y_mean.tolist(), "y_std": store.y_std.tolist(),
                          "x_mean": store.x_mean.tolist(), "x_std": store.x_std.tolist(),
                          "a_mean": store.a_mean.tolist(), "a_std": store.a_std.tolist()},
                 "in_channels": N_INPUT_CHANNELS}
        torch.save(state, ckpt_dir / f"{tag}_last.pt")

        if va["rmse_mean"] < best - 1e-4:
            best, bad_epochs = va["rmse_mean"], 0
            state["best"] = best
            torch.save(state, best_path)
            log.info("  new best: %.4f degC -> %s", best, best_path.name)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stop_patience:
                log.info("early stop after %d epochs without improvement", bad_epochs)
                break

    # Self-describing: the dataset fingerprint lets downstream comparison group
    # only runs that were trained on the same data, so a run against a different
    # period can never be silently tabulated next to this one.
    (cfg.paths.resolve("reports") / f"{tag}_history.json").write_text(json.dumps({
        "run": tag,
        "run_name": cfg.run_name,
        "encoder": cfg.model.encoder,
        "profile_head": profile_head,
        "dataset": {
            "root": str(store.root),
            "period": [store.manifest["dates"][0], store.manifest["dates"][-1]],
            "created": store.manifest.get("created"),
            "n_train_days": len(store.splits["train"]),
        },
        "best_val_rmse": best,
        "epochs": history,
    }, indent=2))
    log.info("best val RMSE %.4f degC | checkpoint %s", best, best_path)
    return best_path


def load_model(ckpt_path: str | Path, device=None) -> tuple[OceanEmbed, dict]:
    """Rebuild a model from a checkpoint, together with its stored metadata."""
    device = device or pick_device()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config.from_dict(state["config"])
    norm = state["norm"]
    model = OceanEmbed(cfg.model, np.asarray(norm["y_mean"], dtype="float32"),
                       np.asarray(norm["y_std"], dtype="float32"),
                       state["in_channels"],
                       profile_head=state.get("profile_head", "monotone")).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state
