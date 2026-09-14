# Mapping to INCOIS problem statement #01

Written for: SIH evaluators and INCOIS reviewers checking coverage of the stated
requirements. Each row points at the code that implements it.

## Mandated specification

| Requirement | Where | Notes |
|---|---|---|
| Domain 5-30 degN, 45-105 degE | [`oceanembed/grid.py`](../oceanembed/grid.py) `Domain` / `NIO` | 101 x 241 grid, asserted in `tests/test_physics.py::test_grid_matches_problem_statement` |
| Spatial resolution 0.25 deg x 0.25 deg | `grid.Domain.resolution` | single analysis grid; every product is co-registered to it |
| Temporal resolution daily | `data/build.py`, `inference.py` | `time_coverage_resolution = P1D` in the output NetCDF |
| 15 standard depths 0-1000 m | `grid.STANDARD_DEPTHS` | `0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000` |

## Required capabilities

### 1. Preprocessing and harmonisation pipeline for multi-source data

[`oceanembed/data/harmonize.py`](../oceanembed/data/harmonize.py),
[`oceanembed/data/cmems.py`](../oceanembed/data/cmems.py)

* `regrid` handles the two conventions that actually differ between these
  products: latitude stored north-to-south, and longitude on 0-360.
* `coarsen_then_regrid` **block-averages before interpolating** for products
  finer than 0.25 deg (OSTIA at 0.05 deg, GLORYS at 1/12 deg). Bilinear
  interpolation alone would discard 96% of OSTIA's pixels and alias the
  mesoscale into the analysis.
* `harmonize_target` regrids GLORYS horizontally level by level *first*, then
  interpolates vertically, so water masses are never mixed across the
  thermocline.
* `VALID_RANGE` range checks, `fill_gaps` land-aware diffusion infilling, and a
  per-variable **gap flag** carried into the model as an input channel so that
  interpolated pixels are distinguishable from real retrievals.
* Sub-daily products (CCMP is 6-hourly) are resampled to daily means.

### 2. Standardisation of all datasets

`data/build.py` writes one set of co-registered memory-mapped arrays plus a
CF-1.8 NetCDF copy per year, and a `manifest.json` recording the domain, depths,
splits, normalisation statistics, configuration and environment.

**Normalisation statistics are computed from the training period only**
(`build._norm_stats`), asserted in
`tests/test_pipeline.py::test_normalisation_uses_training_days_only`. Computing
them over the whole record is a common and silent leak.

### 3. Surface observations as input variables

All seven are used, in `grid.SURFACE_VARS`: SST, SSS, SLA, surface currents
(u, v), surface winds (u, v). Product-specific error budgets and missing-data
behaviour are in `harmonize.ERROR_BUDGET`.

### 4. Compact satellite embeddings via deep learning architectures

[`oceanembed/models/embed.py`](../oceanembed/models/embed.py) - four
interchangeable encoders, selected by `model.encoder`:

| Requested by PS | Provided |
|---|---|
| Convolutional Neural Networks | `cnn` - dilated residual CNN, fully convolutional |
| Vision Transformers | `cnn_vit` - CNN stem then global self-attention over 4x4-cell tokens (**default**) |
| Autoencoders | `autoencoder` - hard bottleneck, plus a surface-reconstruction head on *every* encoder as an autoencoding regulariser |
| Attention-based hybrid architectures | `cnn_vit` is exactly this; `unet` is the multi-scale alternative |
| Graph Neural Networks | **not implemented.** On a regular, complete 0.25 deg lattice a GNN reduces to a CNN with learned edge weights and buys nothing. A GNN would be the right choice for assimilating *scattered* in-situ profiles, which is future work (see Limitations in the README). |

`figure_embedding` in `viz.py` renders the leading three principal components of
the embedding as RGB, to show the latent space organising the basin into
dynamical regimes.

### 5. Reconstruction model learning surface state to temperature profiles

[`oceanembed/models/heads.py`](../oceanembed/models/heads.py) `ProfileHead`,
assembled in `models/oceanembed.py`. The default `monotone` head is
physics-constrained: monotone backbone plus a bounded inversion term, with
decrement biases initialised from the training climatology. The unconstrained
`free` head is provided for a controlled comparison.

### 6. Reconstruction at the standard depth levels

`ProfileHead` emits all 15 levels; `inference.predict_range` writes them as
`thetao(time, depth, latitude, longitude)` with a `thetao_stderr` companion
field.

### 7. Evaluation with independent observations and standard skill metrics

[`oceanembed/metrics.py`](../oceanembed/metrics.py),
[`oceanembed/evaluate.py`](../oceanembed/evaluate.py)

* **Correlation** - anomaly correlation against a fitted harmonic climatology.
* **RMSE**, **centred RMSE**, **MAE**, **bias** - per level, per sub-basin.
* **Murphy skill score** against climatology.
* Independent in-situ validation against ~21,900 withheld profiles.
* Skill on derived operational diagnostics, and uncertainty calibration.
* Confidence intervals bootstrap over **days**, not grid cells.

### 8. Interpolation / regridding when a product is not at the required resolution

`harmonize.regrid`, `harmonize.coarsen_then_regrid`,
`cmems.harmonize_surface`, `cmems.harmonize_target` - see item 1.

## Expected solution checklist

| Deliverable | Status |
|---|---|
| End-to-end preprocessing pipeline | `data/harmonize.py`, `data/cmems.py`, `data/build.py` |
| Satellite embedding engine | `models/embed.py`, four architectures |
| Deep learning reconstruction model | `models/oceanembed.py`, physics-constrained decoder |
| Standardised output, daily, 0.25 deg | `inference.py`, CF-1.8 NetCDF with diagnostics and uncertainty |
| Validation framework using independent ARGO | `data/argo.py`, `metrics.py`, `evaluate.py` |
| Working proof of concept over the Bay of Bengal / Arabian Sea | `app/dashboard.py`, `report.py`; both basins are separate evaluation regions and the profile panel samples four dynamical regimes |

## Theme: Disaster Management

The reconstruction is scored not only on temperature but on the subsurface
quantities that drive ocean hazard warning:

* **Tropical cyclone heat potential** and the **depth of the 26 degC isotherm** -
  the dominant subsurface controls on cyclone rapid intensification in the Bay of
  Bengal, and neither is obtainable from SST alone.
* **0-300 m ocean heat content** - marine heatwave and seasonal-outlook input.
* **Mixed layer depth** and **D20** - stratification and thermocline state.

These are computed in [`oceanembed/diagnostics.py`](../oceanembed/diagnostics.py),
evaluated per method in `evaluate.py`, written into the operational NetCDF, and
displayed in the dashboard.

## Not claimed

* Skill figures are from an Observing System Simulation Experiment, not the real
  ocean. The rationale and the honest limits of that choice are stated up front
  in the README.
* Salinity reconstruction, data assimilation of in-situ profiles, and a GNN
  encoder are not implemented; each is noted with the reason.
