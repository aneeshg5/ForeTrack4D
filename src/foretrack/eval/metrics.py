# ADE/FDE/diversity are the track-forecasting analogs of ForeHand4D's M/M-G/M-F metrics.
# All distances are per-point-per-frame L2 in meters, converted to cm for reporting.

import numpy as np


def ade(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """mean per-point-per-frame L2 displacement error, in cm. pred, gt: (T, N, 3).
    mask: optional (T,) bool, excludes padded frames (e.g. sequences shorter than T)."""
    err = np.linalg.norm(pred - gt, axis=-1)  # (T, N)
    if mask is not None:
        err = err[mask]
    return float(err.mean() * 100)


def fde(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """L2 displacement error at the final (valid) frame, averaged over points, in cm."""
    if mask is not None:
        last = np.where(mask)[0][-1]
    else:
        last = pred.shape[0] - 1
    err = np.linalg.norm(pred[last] - gt[last], axis=-1)  # (N,)
    return float(err.mean() * 100)


def ade_first_frame_aligned(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """M-F analog: rigidly translate pred so its first frame exactly matches gt's first frame,
    then compute ADE. Isolates motion-shape error from the model's t=0 placement error, which
    is expected to be scale-ambiguous from a single image."""
    offset = gt[0].mean(axis=0) - pred[0].mean(axis=0)  # (3,)
    return ade(pred + offset, gt, mask)


def ade_global_aligned(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """M-G analog: rigidly translate pred so its centroid over ALL (valid) frames/points matches
    gt's centroid, then compute ADE. Removes overall placement bias across the whole trajectory,
    not just at t=0 (translation-only alignment; does not correct for rotation/scale)."""
    p, g = (pred[mask], gt[mask]) if mask is not None else (pred, gt)
    offset = g.reshape(-1, 3).mean(axis=0) - p.reshape(-1, 3).mean(axis=0)
    return ade(pred + offset, gt, mask)


def apd3d(pred: np.ndarray, gt: np.ndarray, fx: float, fy: float, visibility: np.ndarray = None, mask: np.ndarray = None, deltas_2d=(1, 2, 4, 8, 16)) -> float:
    """TAPVid-3D's depth-adaptive position accuracy (Koppula et al., NeurIPS 2024, eq. 5):
    a point counts as correct at pixel-threshold delta_2d if its 3D L2 error is within delta_2d
    converted to a metric distance at that point's GT depth via the pinhole model
    (delta_3d = depth * delta_2d / f), averaged over delta_2d in {1,2,4,8,16}px (matching 2D
    TAP-Vid's convention) and over visible points. pred, gt: (T, N, 3), metric (meters), NOT
    normalized. f = mean(fx, fy) -- fine as a single scalar since square-ish pixels are assumed
    elsewhere in this codebase too. Returns a percentage (0-100), unlike ADE/FDE's cm units --
    APD3D is a fraction-correct metric by definition, not a distance."""
    f = (fx + fy) / 2
    err = np.linalg.norm(pred - gt, axis=-1)  # (T, N) meters
    depth = gt[..., 2]  # (T, N) meters
    valid = np.ones_like(err, dtype=bool)
    if visibility is not None:
        valid &= visibility
    if mask is not None:
        valid &= mask[:, None]
    n_valid = max(int(valid.sum()), 1)
    fracs = [((err < depth * d2d / f) & valid).sum() / n_valid for d2d in deltas_2d]
    return float(np.mean(fracs) * 100)


def ade_per_timestep(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """per-timestep mean L2 error over points, in cm -- the error-vs-horizon curve ADE
    aggregates over (ForeHand4D's Fig. 8 analog). pred, gt: (T, N, 3).
    Returns (T,) with NaN at masked-out (padded) timesteps so callers can nanmean-aggregate
    across sequences of different valid length without padded frames pulling the average down."""
    err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1) * 100  # (T,)
    if mask is not None:
        err = np.where(mask, err, np.nan)
    return err


def diversity(samples: np.ndarray) -> float:
    """mean pairwise L2 distance between S sampled futures for the same input, per ForeHand4D's
    protocol -- measures how different the diffusion model's samples are from each other, not
    accuracy against GT. samples: (S, T, N, 3). in cm."""
    s = samples.shape[0]
    assert s >= 2, "diversity needs at least 2 samples"
    total = 0.0
    count = 0
    for i in range(s):
        for j in range(i + 1, s):
            total += np.linalg.norm(samples[i] - samples[j], axis=-1).mean()
            count += 1
    return float(total / count * 100)
