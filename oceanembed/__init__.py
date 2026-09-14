"""OceanEmbed - subsurface ocean temperature from surface satellite observations.

Satellite-embedding deep learning framework for the North Indian Ocean
(5-30 degN, 45-105 degE) at 0.25 degree daily resolution, reconstructing
temperature at 15 standard depth levels between the surface and 1000 m.

Prototype for INCOIS problem statement #01, Smart India Hackathon 2026.

Typical use::

    from oceanembed.config import Config
    from oceanembed.data.build import build
    from oceanembed.train import train, load_model

    cfg = Config.load("configs/nio_full.yaml")
    build(cfg)
    ckpt = train(cfg)
    model, state = load_model(ckpt)

or from the shell::

    python -m oceanembed.cli build
    python -m oceanembed.cli train -c configs/nio_full.yaml
"""

__version__ = "0.1.0"

from .grid import NIO, N_DEPTH, N_SURFACE, STANDARD_DEPTHS, SURFACE_VARS, Domain

__all__ = [
    "__version__",
    "NIO", "Domain", "STANDARD_DEPTHS", "SURFACE_VARS", "N_DEPTH", "N_SURFACE",
]
