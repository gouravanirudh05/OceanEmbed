# OceanEmbed

**Satellite-embedding deep learning framework for reconstructing subsurface ocean
temperature from surface satellite observations.**

Prototype for **INCOIS problem statement #01**, Smart India Hackathon 2026
(Indian National Centre for Ocean Information Services, Ministry of Earth Sciences).

Reconstructs daily three-dimensional ocean temperature at **15 standard depth
levels (0-1000 m)** on a **0.25 degree** grid over the **North Indian Ocean**
(5-30 degN, 45-105 degE) using only surface fields that satellites can observe:
SST, SSS, sea level anomaly, surface currents and surface winds.

---

## What this repository contains

A complete, runnable pipeline, not a sketch:

| Stage | Module | Status |
|---|---|---|
| Preprocessing & harmonisation to 0.25 deg daily | [`oceanembed/data/harmonize.py`](oceanembed/data/harmonize.py) | working |
| Real-product ingestion (OSTIA, SMOS/SMAP, DUACS, OSCAR, CCMP, GLORYS) | [`oceanembed/data/cmems.py`](oceanembed/data/cmems.py) | working, needs credentials |
| Physics-based OSSE twin so the pipeline runs offline | [`oceanembed/data/synthetic.py`](oceanembed/data/synthetic.py) | working |
| Satellite embedding engine (CNN, CNN+ViT, U-Net, autoencoder) | [`oceanembed/models/embed.py`](oceanembed/models/embed.py) | working |
| Physics-constrained profile decoder | [`oceanembed/models/heads.py`](oceanembed/models/heads.py) | working |
| Training loop, losses | [`oceanembed/train.py`](oceanembed/train.py), [`oceanembed/losses.py`](oceanembed/losses.py) | working |
| Reference methods incl. the operational SLA-regression technique | [`oceanembed/models/baselines.py`](oceanembed/models/baselines.py) | working |
| Validation against withheld in-situ profiles | [`oceanembed/data/argo.py`](oceanembed/data/argo.py), [`oceanembed/metrics.py`](oceanembed/metrics.py) | working |
| Derived operational diagnostics (MLD, D20, D26, OHC, TCHP) | [`oceanembed/diagnostics.py`](oceanembed/diagnostics.py) | working |
| CF-1.8 NetCDF daily product | [`oceanembed/inference.py`](oceanembed/inference.py) | working |
| Interactive proof-of-concept dashboard | [`app/dashboard.py`](app/dashboard.py) | working |

---

## Read this first: what the numbers in this repo mean

The products named in the problem statement (OSTIA, SMAP/SMOS, DUACS, OSCAR,
CCMP) and the GLORYS reanalysis target all sit behind Copernicus Marine and NASA
Earthdata credentials and run to tens of terabytes globally. So that the
framework is reviewable and reproducible today, the default configuration runs
as an **Observing System Simulation Experiment (OSSE)**:

* [`oceanembed/data/synthetic.py`](oceanembed/data/synthetic.py) simulates the
  basin from one shared physical state - a propagating mesoscale eddy field,
  monsoon winds, coastal upwelling, the Bay of Bengal freshwater cap - and emits
  both the surface fields *and* the matching 3-D temperature field.
* [`oceanembed/data/harmonize.py`](oceanembed/data/harmonize.py) then degrades
  the surface fields through an **observation operator** carrying each real
  product's documented error budget and missing-data pattern.

The surface-to-subsurface mapping the network has to learn is therefore genuine,
nonlinear and noise-limited, not circular. Three design choices deliberately
keep it hard:

1. **Sea level is split into a steric and a dynamic part.** Only the dynamic
   part displaces the thermocline. The network sees only their sum, so it has to
   disentangle them using season and SST - it cannot just regress *T(z)* on total
   SLA.
2. **Deep water-mass variability has no surface signature at all.** This puts a
   floor of about 0.2 degC on achievable skill below 500 m, so deep levels are
   not trivially predictable.
3. **Observation error is spatially correlated, not white**, and salinity is
   blanked within one degree of the coast exactly as SMAP/SMOS are.

**What this validates:** the architecture, the losses, the physics constraints,
the harmonisation, the evaluation protocol, and the end-to-end engineering.
**What it does not validate:** skill on the real ocean. The same code path runs
on real products via `data.source: cmems` - see
[Running on real data](#running-on-real-data) - and the numbers will differ.
Every reported figure below is on the OSSE twin and is labelled as such.

The simulator is tuned against published climatological magnitudes for the
basin, verified in [`tests/test_physics.py`](tests/test_physics.py):

| Quantity | Simulated | Observed |
|---|---|---|
| Basin-mean SST, February / May / November | 27.2 / 29.8 / 28.2 degC | ~27.2 / 29.3 / 28.3 |
| Northern Arabian Sea SSS | 36.2 | 36.3-36.8 |
| Northern Bay of Bengal SSS (September) | 29.9 | 29-32 |
| SLA standard deviation | 0.10 m | 0.08-0.13 |
| Peak thermocline gradient | 0.10 degC/m | 0.08-0.15 |
| Subsurface T variability at 100 m | 1.6 degC | 1.5-2.5 |
| Bay of Bengal winter inversion | up to 1.8 degC | 1-2 |
| Findlater jet peak wind | 16.7 m/s | 15-20 |

---

## Approach

### 1. Satellite embedding engine

Nineteen input channels per grid cell - seven surface observations, seven gap
flags, and five static/seasonal channels - are mapped to a dense per-pixel
latent embedding.

The default encoder (`cnn_vit`) combines a **dilated residual CNN** for local
texture with **global self-attention** over 4x4-cell tokens. The attention
matters physically: the temperature 150 m below a point depends on the eddy it
sits inside, which is 200-400 km across, so a purely local receptive field
cannot see the structure that sets the answer. Three alternatives (`cnn`,
`unet`, `autoencoder`) are interchangeable for ablation.

The **gap flags are inputs, not a preprocessing detail.** Without them the
network has no way to distinguish an observation from an interpolation, so it
would be forced to treat gap-filled salinity along the entire Indian coastline
as equally trustworthy as a real retrieval. Supplying the flag is a design
choice on those grounds; its contribution is not separately ablated here.

### 2. Physics-constrained profile decoder

The default head does not emit 15 free numbers. It emits a surface temperature
plus 14 **strictly positive downward decrements**, making the profile backbone
monotone by construction, and adds a **bounded inversion term** of at most
1.5 degC.

The bound is there for a specific reason: northern Bay of Bengal winter profiles
carry a genuine 1-2 degC barrier-layer temperature inversion, so *hard*
monotonicity would be physically wrong - while an unconstrained head spends
capacity rediscovering that the ocean is stratified. The decrement biases are
initialised from the training climatology, so an untrained model already emits
the climatological mean profile and learns only the anomaly.

A `free` head is provided and trained under identical conditions so the
constraint can be measured rather than asserted. It turns out to make **no
measurable difference to accuracy** on this benchmark - the free head learns to
stratify perfectly well (see
[Results](#the-encoder-and-head-ablations-are-a-null-result)). What the
constraint buys is a *guarantee*: the monotone head cannot emit an unphysical
column whatever the input, which matters when extrapolating outside the training
distribution, not a better score inside it.

### 3. Objective

| Term | Purpose |
|---|---|
| Huber on standardised *T(z)* | Per-level standardisation keeps the 1000 m level (0.3 degC variability) in play against the thermocline (1.6 degC) |
| Horizontal gradient matching | Pointwise losses are minimised by blurry fields; this penalises the blur that would smear out the eddies directly |
| Stratification penalty | Penalises inversions beyond the admissible bound |
| MLD and D20 auxiliary heads | Supervising the two parameters that *set* the profile shape gives a far sharper signal than 15 temperatures alone |
| Surface autoencoding | Keeps the embedding a faithful representation of the surface state |
| Gaussian NLL on a detached residual | Calibrates the predictive sigma without letting a large sigma soften the gradient on the mean |

### 4. Evaluation

Splits are **chronological**, never random: the mesoscale field is autocorrelated
over weeks, so a random split leaks eddies from training into test.

* **Gridded skill** per level and sub-basin: RMSE, bias, MAE, anomaly
  correlation, Murphy skill score against a fitted harmonic climatology.
* **In-situ skill** against ~21,900 withheld profiles sampled at real ARGO float
  density (200 floats, 10-day cycle) and carrying a representativeness error, so
  numbers are comparable with published ARGO-versus-reanalysis statistics.
* **Operational diagnostics**: mixed layer depth, 20 and 26 degC isotherm depths,
  0-300 m heat content, and **tropical cyclone heat potential** - the dominant
  subsurface predictor of cyclone rapid intensification in the Bay of Bengal,
  and one that cannot be obtained from SST alone. This is the operational
  argument for reconstructing the subsurface at all.
* **Uncertainty calibration**: empirical coverage of the nominal 68% and 95%
  predictive intervals.
* Confidence intervals **bootstrap over days**, not grid cells, because errors
  within a day are strongly spatially correlated.

Reference methods include `sla_regression` - a per-grid-point regression of
*T(z)* on sea level and SST anomaly with a seasonal cycle. This is close to the
**synthetic-profile / Gravest Empirical Mode family used operationally**, and is
the honest bar to clear.

---

## Quick start

```bash
make setup                      # virtualenv + dependencies (~2 min, downloads torch)
make quickdata                  # 1-year dataset (~80 s)
.venv/bin/python -m oceanembed.cli train -c configs/quick.yaml -d data/processed_quick
make test                       # 39 tests
```

Full run:

```bash
make data                       # 3-year dataset, 1095 days (~5 min, ~3.4 GB)
make train                      # ~75 min on 8 CPU threads; minutes on a GPU
make evaluate                   # all baselines + in-situ validation
make figures                    # figure set in outputs/figures/
make predict                    # CF-1.8 NetCDF product in outputs/predictions/
make dashboard                  # interactive demo at localhost:8501
```

Any config field can be overridden from the shell:

```bash
python -m oceanembed.cli train -o model.encoder=unet -o train.lr=5.0e-4
python -m oceanembed.cli info data/processed
```

---

## Repository layout

```
oceanembed/
  grid.py            analysis grid, standard depths, Natural Earth land mask
  config.py          typed configuration, YAML + CLI overrides
  diagnostics.py     MLD, D20, D26, heat content, cyclone heat potential
  losses.py          training objective
  metrics.py         streaming skill statistics
  train.py           training loop
  evaluate.py        skill against baselines and in-situ profiles
  inference.py       CF-1.8 daily 3-D product writer
  viz.py, report.py  figures
  cli.py             command line entry point
  data/
    synthetic.py     physics-based OSSE twin of the basin
    harmonize.py     regridding, QC, gap filling, observation operator
    cmems.py         real Copernicus Marine / PO.DAAC ingestion
    argo.py          in-situ profile sampling and loading
    dataset.py       memory-mapped PyTorch datasets
    build.py         dataset builder
  models/
    embed.py         four interchangeable embedding encoders
    heads.py         profile / auxiliary / reconstruction decoders
    oceanembed.py    assembled model, tiled full-field inference
    baselines.py     climatology, linear, SLA regression, pointwise MLP
app/dashboard.py     Streamlit proof-of-concept demo
configs/             default.yaml, quick.yaml, nio_full.yaml
scripts/             download_real_data.sh
tests/               physics invariants + end-to-end pipeline
```

---

## Running on real data

```bash
pip install copernicusmarine earthaccess
copernicusmarine login
python -c "import earthaccess; earthaccess.login(persist=True)"

./scripts/download_real_data.sh 2021-01-01 2023-12-31
python -m oceanembed.cli build -o data.source=cmems
python -m oceanembed.cli train -c configs/nio_full.yaml
```

Products, DOIs and dataset identifiers are declared in
[`oceanembed/data/cmems.py`](oceanembed/data/cmems.py). Two notes:

* Copernicus renames and re-versions datasets. Confirm an identifier with
  `oceanembed.data.cmems.describe(dataset_id)` before a bulk download.
* OSTIA (0.05 deg) and GLORYS (1/12 deg) are **block-averaged before**
  interpolation, not sampled bilinearly - straight bilinear interpolation from
  0.05 deg discards 96% of the source pixels and aliases the mesoscale. GLORYS is
  regridded horizontally *first*, level by level, then interpolated vertically,
  so water masses are never mixed across the thermocline.

Switching to real data changes `data.source` and nothing else: the harmoniser
emits the same arrays, so training, evaluation and the product writer are
untouched.

---

## Results (OSSE twin, held-out test period)

184 consecutive held-out days, 3,678 withheld in-situ profiles. Full tables in
[`docs/RESULTS.md`](docs/RESULTS.md) and
[`outputs/reports/evaluation.md`](outputs/reports/evaluation.md).

| method | RMSE degC | skill vs climatology | anomaly correlation | field sharpness | TCHP RMSE kJ/cm2 |
|---|---|---|---|---|---|
| `sla_regression` (operational technique) | **0.403** | 70.4% | 0.760 | 2.44 | 7.14 |
| `mlp` (point-wise, no spatial context) | 0.424 | 69.4% | 0.748 | 2.52 | 7.05 |
| **`oceanembed`** | 0.432 | 67.1% | **0.800** | **0.98** | **6.63** |
| `climatology` | 0.741 | 0.0% | n/a | 0.59 | 12.83 |
| `linear` (one global regression) | 0.743 | 1.1% | 0.601 | 2.36 | 12.37 |

*Field sharpness is the mean horizontal gradient magnitude of the reconstruction
divided by that of the truth. 1.0 is correct; below 1 is over-smoothed, above 1
is spuriously noisy.*

`linear` scores near-zero skill not because linear methods are hopeless here -
`sla_regression` is also linear and wins - but because it is a *single global*
regression. Its only way to represent the local mean state is a linear trend in
latitude and longitude, so it cannot reproduce a per-grid-point climatology at
all. The informative comparison is `sla_regression`, which fits an independent
regression at every grid cell.

### The deep model does not win on RMSE, and that result is informative

**OceanEmbed is third of five on depth-averaged RMSE.** The operational
SLA-regression technique beats it by 0.03 degC, and so does a point-wise MLP with
no spatial context at all. This is reported rather than buried because the
reason is structural and it says something useful about the experiment.

**Why the local linear method wins here.** In the OSSE twin, the 20 degC isotherm
depth is *by construction* a linear function of dynamic sea level,
`d20 = d20_clim + 150 * sla_dyn` ([`synthetic.py`](oceanembed/data/synthetic.py)),
and the profile is then a deterministic function of (SST, MLD, D20). A
per-grid-point linear regression of *T(z)* on sea level and SST anomaly with a
seasonal cycle is therefore close to the *optimal* estimator for this generative
process. The OSSE is structurally favourable to `sla_regression`, and no amount
of architecture can beat the Bayes estimator of the process that made the data.
**This is a limitation of the OSSE design, not evidence about the real ocean**,
where the surface-to-subsurface relation is neither local nor linear. Making the
twin a fair test of nonlinear methods means giving the thermocline a nonlinear,
advective, eddy-shape-dependent response to the surface - concrete future work,
and the single most valuable change to make next.

**What the deep model does win, even so:**

* **Anomaly correlation 0.800 against 0.760** - it recovers the *pattern* of the
  subsurface anomaly best, while being slightly worse on amplitude.
* **Field realism: sharpness 0.98 against 2.44 and 2.52.** The two point-wise
  competitors emit fields roughly two and a half times noisier than the truth,
  because every grid cell is fitted independently. OceanEmbed's field has
  essentially the right amount of structure. For an operational product this is
  not cosmetic: a field that noisy cannot be differentiated for gradients, and a
  warning centre cannot use it to locate an eddy.
* **The deepest levels, where the point-wise methods break down entirely.** At
  1000 m `sla_regression` and `mlp` score *negative* skill against climatology
  (-8% and -6%): with no local surface signal to exploit they fit noise and do
  worse than simply predicting the seasonal mean. OceanEmbed keeps +18% skill and
  an anomaly correlation of 0.48 there, against -0.1 and 0.05. Spatial context is
  what is left to work with once the local signal is gone.
* **Tropical cyclone heat potential** (6.63 against 7.14 kJ/cm2) and **mixed
  layer depth** (6.40 m against 8.00 m) - the two diagnostics most used
  operationally.
* **Bay of Bengal**: 0.346 against 0.342 degC, a statistical tie in the basin the
  proof of concept targets.

### The encoder and head ablations are a null result

Three configurations were trained with identical data, seed, learning rate and
14-epoch schedule, so only the architecture differs. At epoch 9, the last epoch
all three completed:

| encoder | profile head | val RMSE degC |
|---|---|---|
| CNN + ViT | monotone | 0.3309 |
| CNN + ViT | free | 0.3318 |
| dilated CNN | monotone | 0.3328 |

**These differences are noise.** The spread between architectures is 0.0019 degC,
while the spread *within* a single run over its last five epochs reaches
0.0400 degC - twenty times larger. The ranking changed at 9 of the 9 epochs
compared. Separating these configurations would need several seeds each to
estimate run-to-run variance, which was not run, so **no claim is made that
either the attention or the monotone head improves accuracy here.** The
`scripts/make_results.py` ablation table computes and prints this verdict
itself rather than leaving the reader to eyeball four decimal places.

What *does* separate methods by a wide margin is whether they model space at
all: both spatial models produce coherent fields and retain skill at 1000 m,
while both point-wise methods produce fields 2.5x too noisy and lose to
climatology in the deep ocean. The useful distinction on this benchmark is
spatial versus point-wise, not attention versus convolution.

### Honest summary

On this synthetic benchmark the learned embedding buys spatial coherence,
pattern skill, better derived diagnostics and deep-ocean skill - but not lower
pointwise RMSE than a well-tuned local regression, and the specific choice of
encoder or decoder head within the spatial family makes no measurable
difference. Ranking these methods on the real ocean requires the real data.

## Known limitations

* Skill is measured on the OSSE twin, not the real ocean. See
  [the note above](#read-this-first-what-the-numbers-in-this-repo-mean).
* **The predictive uncertainty is overconfident, increasingly so with depth.**
  The nominal 68% interval covers 54% of cases overall: 66% at the surface but
  only 38% at 700-1000 m, where the mean predicted sigma (0.22 degC) understates
  the actual RMSE (0.37 degC) by about 1.7x. The sigma head is fitted on a
  detached residual, which keeps training stable but means it is calibrated
  against the *training* error distribution and does not know about the
  generalisation gap. Treat `thetao_stderr` as a relative confidence map, not an
  absolute error bar, and recalibrate per level on a held-out split before using
  it quantitatively.
* **The barrier-layer inversion amplitude is systematically damped.** Over the
  northern Bay of Bengal in late December, the truth inverts by +0.62 degC on
  average (95th percentile +0.92) while the reconstruction gives +0.23 degC
  (95th percentile +0.29), and the spatial pattern is uncorrelated. Both the
  monotone and the free head *can* represent an inversion and both produce one,
  so this is not the parameterisation - it is the objective. The inversion is a
  few tenths of a degree against a per-level standard deviation of 1.2-1.6 degC,
  so a Huber loss weighted equally across levels has almost no incentive to fit
  it. Fixing it properly means either weighting the mixed-layer and
  barrier-layer levels up, or supervising inversion strength directly as a third
  auxiliary target alongside MLD and D20. Anyone using this for barrier-layer
  work should treat the inversion field as unreliable.
* Only temperature is reconstructed. Salinity uses the same machinery - add a
  second profile head - but is not implemented.
* The geostrophic relation in the simulator clamps the Coriolis parameter at its
  5 degN value, so equatorial dynamics (where geostrophy fails) are not
  represented. The domain's southern boundary is at 5 degN, which limits the
  consequences but does not remove them.
* Training ran on CPU. The default configuration in `configs/default.yaml` is
  larger than `configs/nio_full.yaml` and is what should be used on a GPU.
* No data assimilation: the reconstruction is purely diagnostic and does not
  ingest the in-situ profiles it is validated against.

---

## Licence

MIT. The Natural Earth coastline in `data/cache/` is public domain.
