# NOTICE

This file tracks provenance of code adapted from other repositories. Every
file copied or adapted from an external project should have a one-line
provenance comment at its top and a matching entry here.

## forehand4d (CC BY-NC 4.0)
Source: https://github.com/ap229997/forehand4d

- `src/foretrack/models/denoiser.py` -- adapted from `src/models/mdm/model/mdm.py`'s `MDM`
  class (arch='trans_enc'): per-timestep token embedding (`InputProcess`), `TimestepEmbedder`'s
  reuse of the sinusoidal position table as a timestep lookup, and the 2D-projection Fourier
  term (derived from `model_.py`'s `compute_kpe_enc`, simplified since we already have the 3D
  point rather than 2D keypoints -- see the comment in `project_angle_fourier`). MANO/rotation
  parameterization removed (this project predicts object point tracks, not hands).
- `src/foretrack/models/conditioning.py` -- adapted from `src/models/mdm/model/encoder.py`'s
  `ImageEncoder` (ViT backbone + HaMeR checkpoint loading, kept close to verbatim) and
  `SpatialConditioner` (query-token construction reworked: one token per query
  point via bilinear sampling at its 2D projection, instead of one token per ViT patch).
- `src/foretrack/models/diffusion.py` -- adapted from
  `src/models/mdm/diffusion/gaussian_diffusion.py`'s `get_named_beta_schedule` (cosine branch)
  and `training_losses` (`ModelMeanType.START_X` / x0-prediction path). Loss function itself
  (`depth_scaled_l2`) is not from forehand4d -- see the TAPIP3D entry below.
- `src/foretrack/models/vit.py` -- vendored verbatim from `src/models/mdm/model/vit.py`.

## OpenMMLab / ViTPose (as vendored by forehand4d)
Source: forehand4d's `src/models/mdm/model/vit.py` header attributes this to OpenMMLab's
ViTPose backbone.

- `src/foretrack/models/vit.py` -- the ViT-H architecture itself (`ViT`, `Block`, `Attention`,
  `PatchEmbed`, etc.) originates here, not from forehand4d; forehand4d vendored it unmodified
  and we did the same, keeping it byte-for-byte identical so HaMeR's pretrained checkpoint
  (loaded in `conditioning.py`) matches the architecture exactly.

## TAPIP3D (Apache-2.0)
Source: https://github.com/zbw001/TAPIP3D

- `models/diffusion.py`'s `depth_scaled_l2` implements the *design* of TAPIP3D's
  depth-adaptive loss weighting (1/depth weighting so far points
  don't dominate), independently written -- not literal code copied from TAPIP3D.
- `src/foretrack/labeling/run_tapip3d.py` -- CLI args (`--input_path`, `--output_dir`,
  `--checkpoint`, no `--output`/exact-path flag) and the input/result npz schema
  (`video`/`depths`/`intrinsics`/`extrinsics`/`query_point`, `query_point`'s
  `(N,4)=[query_time,x,y,z]` world-coordinate format, world mode via omitted `extrinsics`)
  verified directly against `inference.py`'s and `utils/inference_utils.py`'s source, not
  guessed from documentation. No TAPIP3D code is copied; this wrapper's own logic (the
  two-pass depth-then-query-point invocation, `lift_query_points`) is independently written.

## microsoft/psi (MIT)
Source: https://github.com/microsoft/psi

- `src/foretrack/data/holoassist.py` -- the `CoordinateSystem`/`ICameraIntrinsics` field
  ordering in `Sources/MixedReality/HoloLensCapture/HoloLensCaptureInterop/Operators.cs` and
  the MathNet basis convention (confirmed via `Sources/Calibration/Microsoft.Psi.Calibration/
  ICameraIntrinsics.cs`'s own docstring) were read from this repo's source to derive our
  parser. Note: this generic exporter's own multi-line, per-frame `Intrinsics.txt` shape does
  NOT match HoloAssist's real released files (a single whole-session line instead -- see the
  taeinkwon/PyHoloAssist entry below and checkpoints.md); this source was useful for the
  MathNet-basis confirmation and the "_sync" file byte offsets, not for `Intrinsics.txt`'s
  layout. No code is copied, the format itself (a Microsoft-authored capture pipeline, not
  part of the CC BY-NC/Apache-2.0 codebases above) is what's being interoperated with.

## Ember-HoloAssist/holoassist-release (license not stated in the repo as checked)
Source: https://github.com/Ember-HoloAssist/holoassist-release (HoloAssist's own official
dataset release repo; the dataset itself is CDLAv2, ICCV 2023, Wang et al.)

- `src/foretrack/data/holoassist.py` -- `list_annotated_clips`/`read_split`'s file locations
  and schema (`{subset}_0724.txt` split lists, `labels_*_classes.json`'s
  `{event_key: {video_name: [[t_start,t_end,[task_id,label_id]]]}}` structure, the
  `int(t*30)` frame-indexing scheme) and the corrected "_sync" file byte offset (one extra
  leading field versus the generic psi exporter, caught via this repo's own `line[2:]`
  slicing and its zero-fallback shape) were derived by reading `src/data_loader/
  raw_loader_index.py`, the official release's own consumer of these exact files. No code is
  copied; our parsing/shot-selection logic is independently written against the format this
  revealed.

## taeinkwon/PyHoloAssist (license not stated in the repo as checked)
Source: https://github.com/taeinkwon/PyHoloAssist (a community loader for HoloAssist's raw
released files, referenced from the official project page)

- `src/foretrack/data/holoassist.py` -- `parse_intrinsics_file`'s corrected format (a single
  whole-session line, not per-frame) matches this repo's own `read_intrinsics_txt` exactly:
  `data[:9].reshape(3,3)` for the K matrix, `data[-2:]` for width/height. Also confirmed
  `parse_hand_file`'s column offset (`line_data[3:-52]`, no StereoKit/OpenXR branching needed
  in practice) via this repo's `read_hand_pose_txt`. No code is copied; both parsers were
  independently rewritten (and, for intrinsics, actively wrong before this cross-check) once
  the real format was understood.

## motion-diffusion-model (MIT)
Source: https://github.com/GuyTevet/motion-diffusion-model

No files adapted yet. (forehand4d's own `mdm/` subtree is itself an adaptation of MDM; our
`denoiser.py`/`diffusion.py` provenance is tracked against forehand4d above, one level removed
from the original MDM code.)

## sam2 (Apache-2.0, BSD-3-Clause for some components)
Source: https://github.com/facebookresearch/sam2

No files adapted; `src/foretrack/labeling/segment.py`'s `Sam2ObjectSegmenter` calls sam2's
public `build_sam2`/`SAM2ImagePredictor` API (a pip-installed dependency, not vendored code)
per its documented interactive-segmentation interface.
