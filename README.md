# ForeTrack4D

3D object motion forecasting from a single image. Given one frame of a
hand-object interaction and a set of query points on the object, a
conditional diffusion model predicts world-space 3D trajectories (X, Y, Z
per point per timestep) over a long horizon, capturing the multimodality
of possible futures. The output is 3D point tracks, not pixels.

The model transplants the ForeHand4D recipe (single-image diffusion
forecasting of hand motion) to objects, and uses TAPIP3D (feed-forward 3D
point tracking in persistent world space) to pseudo-label in-the-wild
video for training.

## Method

- **Output**: (T, N, 3) point tracks, N=64 farthest-point-sampled query
  points, T=128 forecast steps, world frame = camera frame at t=0.
- **Conditioning**: a single RGB image (ViT encoder, HaMeR init) plus one
  token per query point: Fourier-embedded 3D position at t=0 concatenated
  with the ViT patch feature at its 2D projection. No motion history.
- **Denoiser**: MDM-style transformer encoder over per-timestep tokens,
  x0 prediction, cosine schedule, depth-scaled L2 loss.
- **Training**: Stage 1 on ground-truth tracks generated from lab dataset
  meshes and 6DoF poses (DexYCB, ARCTIC including articulated objects,
  H2O); Stage 2 finetune adds TAPIP3D-imputed pseudo-labels from
  HoloAssist video, gated by jerk / depth-variance / visibility quality
  filters. A same-architecture regression baseline is trained alongside.
- **Evaluation**: ADE/FDE (plus first-frame- and globally-aligned
  variants), APD3D with TAPVid-3D's depth-adaptive thresholds, sample
  diversity, and error-vs-horizon curves, always against static and
  constant-velocity baselines. EgoExo4D is held out entirely (no training
  data, no pseudo-labels) for zero-shot evaluation.

## Results

DexYCB held-out test split (n=1200), Stage 2 checkpoint:

| model | ADE (cm) |
|---|---|
| static | 12.46 |
| regressor | 7.85 |
| **diffusion** | **7.52** |

On DexYCB val the mixed-dataset diffusion model reaches 6.63cm ADE with
9.23cm sample diversity (5 samples per input). Error-vs-horizon curves
show the gap concentrating at long horizons: static climbs past 30cm by
the end of a clip while diffusion plateaus around 14-16cm.

Stage 2 (pseudo-labels from 108 HoloAssist sessions, 213 accepted clips)
improves zero-shot EgoExo4D ADE over Stage 1 for both model types
(regressor -10.0%, diffusion -7.5% with the OOD gate below).

Known limitation: on fully out-of-distribution scenes (EgoExo4D
zero-shot) plain diffusion loses to the static baseline and its sample
diversity collapses — the conditioning pathway does not meaningfully
distinguish scenes it was never trained on. An inference-time gate
(`--ood_gate`) detects this by comparing conditioned and unconditional
predictions and falls back to static, recovering most of the gap
(23.83cm -> 20.87cm vs static 19.76cm).

## Demo

`demo/` is a "forecast vs. reality" web demo: upload a short single-shot
hand-object video, pick a conditioning frame, and the backend tracks
observed reality with TAPIP3D+MegaSaM, samples K forecasts from that
frame, and renders a side-by-side video — the left panel plays the real
clip while the right freezes on the conditioning frame and animates the
predicted trajectories against what actually happened, alongside a
metric 3D view and a live error-vs-baseline chart. No pixels are
synthesized. See `demo/README.md`.

Per-clip error at the final predicted frame, in cm, against the "object
never moves" baseline. Mean is over the 5 sampled futures (single-sample
deployment); best is the closest of the 5:

| clip | final: mean | final: best | static |
|---|---|---|---|
| DexYCB gelatin box (test) | 10.5 | 5.8 | 46.7 |
| DexYCB foam brick (val) | 10.0 | 8.7 | 48.5 |
| DexYCB wood block (test) | 24.5 | 17.6 | 48.5 |
| H2O milk, egocentric (test) | 10.1 | 5.2 | 12.9 |
| ARCTIC phone, articulated (val) | 25.1 | 20.0 | 39.8 |
| HoloAssist cart, in-the-wild ego | 35.9 | 35.2 | 33.8 |

Error is not uniform over the horizon: the model is worse than the
baseline early (the object has not moved yet, and the model commits to
motion slightly too soon, overshooting by up to 11cm mid-clip) and
better late, where the baseline diverges and the model stays flat.
Sampling is stochastic; repeated runs vary by roughly ±25%.

The last row is the open gap: on in-the-wild egocentric video, where
reality itself comes from TAPIP3D rather than mesh ground truth, the
model does not beat the baseline.

## Setup

Training stack: Python 3.10, torch 2.4.1 / cu124.

```
pip install -e ".[dev]"
```

Three separate environments are required; do not unify them:

1. **Training env** (this repo's `pyproject.toml`): training, eval, and
   forecaster inference (`scripts/forecast_infer.py`).
2. **tapip3d env**: built per TAPIP3D's README (compiles `pointops2` and
   MegaSaM's extensions). Runs TAPIP3D `inference.py` and MegaSaM.
   MegaSaM's checkpoints (Depth-Anything V1, RAFT) must be downloaded
   separately per its README.
3. **labeling env** (`pip install -e ".[labeling]"`, plus SAM2 from the
   `third_party/sam2` clone): hand detection, SAM2 masking, and the demo
   app/worker.

Cross-environment calls go through `subprocess` with `.npz` files on
disk, never cross-environment imports (`src/foretrack/labeling/run_tapip3d.py`,
`scripts/forecast_infer.py`).

`scripts/setup_third_party.sh` clones TAPIP3D and SAM2 into
`third_party/` (not committed). MANO model files are required by the
dataloaders: register at the MANO site, place them under
`downloads/model/body_models/mano`, and set `DOWNLOADS_DIR`.

## Data

DexYCB, ARCTIC (ego split), and H2O for ground-truth tracks (mixed via
the dataset-string config, e.g. `dexycb+arctic_ego+h2o`); HoloAssist for
pseudo-labels; EgoExo4D for zero-shot eval only. See `configs/` and
`src/foretrack/data/`.

## Train

```
python scripts/train.py --config configs/mixed_stage1.yaml --model_type diffusion
python scripts/train.py --config configs/mixed_stage2.yaml --model_type diffusion
```

## Eval

```
python scripts/eval.py --config configs/mixed_stage2.yaml \
    --diffusion_ckpt <path> --regressor_ckpt <path> --split test \
    --horizon_plot horizon.png
```

`--ood_gate` adds a `diffusion_gated` row that falls back to the static
baseline whenever conditioned and unconditional predictions nearly agree
(see `src/foretrack/eval/ood_gate.py`).

## License

CC BY-NC 4.0 (see LICENSE), required by code adapted from forehand4d.
Per-module provenance is in NOTICE.md.

## Acknowledgements

Built on:

- ForeHand4D (Prakash, Forsyth, Gupta) — forecasting architecture and
  training recipe. https://github.com/ap229997/forehand4d
- TAPIP3D (Zhang, Ke, Harley, Fragkiadaki) — 3D point tracking used for
  pseudo-labeling. https://github.com/zbw001/TAPIP3D
- MDM (Tevet et al.) — diffusion backbone. https://github.com/GuyTevet/motion-diffusion-model
- SAM 2 (Meta) — object mask extraction. https://github.com/facebookresearch/sam2
