# OceanEmbed architecture

Written for: engineers extending or reviewing the implementation.

## Data flow

```
                      Copernicus Marine / PO.DAAC          OSSE twin
                      (data/cmems.py)                      (data/synthetic.py)
                               |                                   |
                               |  native grids, units, cadences    |  true surface + true T(z)
                               v                                   v
                   +-----------------------------------------------------------+
                   |  data/harmonize.py                                        |
                   |   regrid / block-average -> 0.25 deg daily                |
                   |   range QC, land-aware gap filling, gap flags             |
                   |   observation operator (OSSE path only)                   |
                   +-----------------------------------------------------------+
                               |
                               v
                   +-----------------------------------------------------------+
                   |  data/build.py                                            |
                   |   inputs.npy  (T, 7,  101, 241)  float32  memmap          |
                   |   gapflag.npy (T, 7,  101, 241)  uint8                    |
                   |   target.npy  (T, 15, 101, 241)  float32                  |
                   |   aux.npy     (T, 2,  101, 241)  float32  MLD, D20        |
                   |   argo.npz    withheld in-situ profiles                   |
                   |   manifest.json  splits + train-only normalisation        |
                   |   + CF-1.8 NetCDF per year                                |
                   +-----------------------------------------------------------+
                               |
              data/dataset.py  |  OceanStore (memmap + normalisation + statics)
                               |  PatchDataset (random 32x32 windows, reflections)
                               v
                   +-----------------------------------------------------------+
                   |  models/oceanembed.py                                     |
                   |                                                           |
                   |   x (B, 19, H, W)                                         |
                   |        |                                                  |
                   |   models/embed.py     encoder -> z (B, D, H, W)            |
                   |        |                  the satellite embedding          |
                   |        +--------------+--------------+--------------+      |
                   |        v              v              v              v      |
                   |   ProfileHead      AuxHead     SurfaceRecon    (embed)     |
                   |   T(z), sigma      MLD, D20    7 surface ch                |
                   +-----------------------------------------------------------+
                               |
                 losses.py     |  Huber + NLL + stratification + gradient
                               |  matching + reconstruction + auxiliary
                               v
                         train.py  ->  checkpoint
                               |
          +--------------------+--------------------+
          v                                         v
   evaluate.py / metrics.py                  inference.py
   vs climatology, linear, SLA regression,   CF-1.8 daily product:
   pointwise MLP, and withheld profiles      thetao, thetao_stderr,
                                             mld, d20, d26, ohc300, tchp
```

## Input channels (19)

| Index | Content | Why |
|---|---|---|
| 0-6 | SST, SSS, SLA, u_cur, v_cur, u_wnd, v_wnd, standardised | the observable surface state |
| 7-13 | gap flags, one per surface variable | lets the network discount interpolated pixels; without the flag it has no way to tell a retrieval from an interpolation, which matters most for SSS within a degree of the coast. Not separately ablated |
| 14-15 | sin, cos of day-of-year | the seasonal cycle is a large part of the signal and is cheap to supply |
| 16-17 | normalised latitude, longitude | the surface-to-subsurface relation is regionally distinct (Arabian Sea versus Bay of Bengal) |
| 18 | normalised log distance to coast | proxy for shelf/coastal dynamics and for where SSS is unreliable |

Normalisation statistics come from the **training period only**.

## Encoders

All produce `(B, embed_dim, H, W)` and are selected by `model.encoder`.

**`cnn_vit` (default).** A dilated residual CNN stem (dilations 1, 2, 4, 8, giving
a ~45-cell receptive field without downsampling) followed by global self-attention
over 4x4-cell tokens, then a fusion convolution that recombines the global and
local paths at full resolution.

The attention is not decoration. Temperature at 150 m depends on the mesoscale
eddy the point sits inside, and eddies here are 200-400 km across - a purely
local receptive field cannot see the structure that determines the answer.

The positional embedding is stored on an 8x8 reference token grid and bilinearly
resized to whatever the input needs, so a model trained on 32x32 windows also
runs on the whole 101x241 basin in one pass. Inputs are replicate-padded up to a
whole number of tokens and cropped afterwards, so odd sizes work.

**`cnn`** is the same stem without attention: fully convolutional, cheapest, and
the natural control for measuring what attention contributes.
**`unet`** gets context from downsampling instead.
**`autoencoder`** bottlenecks to 16 channels for self-supervised pretraining.

## Profile decoder

`models/heads.py::ProfileHead`, per-pixel MLP implemented as 1x1 convolutions.

`mode="monotone"` (default) emits

```
T_0  = clim_0 + std_0 * a_0                      surface, as an anomaly
T_k  = T_{k-1} - softplus(r_k + b_k)             strictly decreasing backbone
T_k += max_inversion * tanh(v_k)                 bounded inversion, |dT| <= 1.5 degC
```

`b_k` is initialised to `softplus^-1(clim_k-1 - clim_k + 0.02)`, so a
zero-activation network emits the climatological mean profile and training only
has to learn the anomaly.

The inversion term is bounded rather than absent because northern Bay of Bengal
winter profiles carry a genuine 1-2 degC barrier-layer inversion: hard
monotonicity would be physically wrong. `tests/test_pipeline.py::
test_monotone_head_respects_the_inversion_bound` asserts the bound holds even
for activations scaled by 200.

`mode="free"` emits 15 unconstrained standardised values, for comparison. In
practice the free head also learns to stratify (worst inversion +0.35 degC over
five test days, against the monotone head's +0.17 degC and a 1.5 degC bound), so
on densely supervised data the constraint buys correctness guarantees rather
than accuracy. Its value is the guarantee: the monotone head *cannot* emit an
unphysical column whatever the input, which matters when extrapolating outside
the training distribution.

Both heads underestimate the amplitude of the real barrier-layer inversion by
roughly a factor of three - see Known limitations in the README. That is a
property of the objective, not of either parameterisation.

Uncertainty: `log_sigma` per level, clamped to `[-5, 2]` standardised units. It
is fitted by a Gaussian NLL on a **detached** residual, so calibrating sigma
cannot soften the gradient on the mean - the usual failure mode of joint
heteroscedastic training.

## Full-field inference

`models/oceanembed.py::predict_field`. With `tile=0` the basin is processed in
one pass (exact, and what the reported numbers use). With `tile>0` it is covered
by overlapping windows blended with a raised-cosine weight, bounding memory at
the cost of losing context beyond the window.

## Evaluation

All statistics stream from running moments (`metrics.py::LevelStats`) because
predictions for 184 test days at 15 levels are ~2.7 GB. Splits are chronological.
Bootstrap confidence intervals resample **days**, not grid cells, because errors
within a day are strongly spatially correlated - resampling cells would
understate the interval by orders of magnitude.

## Extending

**Add an encoder:** implement a `nn.Module` returning `(B, embed_dim, H, W)`,
register it in `embed.ENCODERS`. Nothing else changes.

**Add salinity:** add a second `ProfileHead` in `OceanEmbed.__init__`, a target
array in `build.py`, and a loss term. The decoder is variable-agnostic.

**Swap in real data:** set `data.source: cmems`. `cmems.harmonize_surface` and
`harmonize_target` emit the same arrays `build.py` writes, so training,
evaluation and the product writer are untouched.

**Assimilate in-situ profiles:** this is where a graph neural network belongs -
scattered ARGO casts as nodes over the gridded embedding. Not implemented; a GNN
over the regular 0.25 deg lattice would just be a CNN with learned edge weights.
