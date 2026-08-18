# ADE/FDE and the aligned variants are the track analogs of ForeHand4D's M/M-G/M-F;
# APD3D follows TAPVid-3D (Koppula et al., NeurIPS 2024). See NOTICE.md.

import numpy as np


def ade(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    err = np.linalg.norm(pred - gt, axis=-1)  # (T, N)
    if mask is not None:
        err = err[mask]
    return float(err.mean() * 100)


def fde(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is not None:
        last = np.where(mask)[0][-1]
    else:
        last = pred.shape[0] - 1
    err = np.linalg.norm(pred[last] - gt[last], axis=-1)  # (N,)
    return float(err.mean() * 100)


def ade_first_frame_aligned(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    offset = gt[0].mean(axis=0) - pred[0].mean(axis=0)  # (3,)
    return ade(pred + offset, gt, mask)


def ade_global_aligned(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    p, g = (pred[mask], gt[mask]) if mask is not None else (pred, gt)
    offset = g.reshape(-1, 3).mean(axis=0) - p.reshape(-1, 3).mean(axis=0)
    return ade(pred + offset, gt, mask)


def apd3d(pred: np.ndarray, gt: np.ndarray, fx: float, fy: float, visibility: np.ndarray = None, mask: np.ndarray = None, deltas_2d=(1, 2, 4, 8, 16)) -> float:
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
    err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1) * 100  # (T,)
    if mask is not None:
        err = np.where(mask, err, np.nan)
    return err


def dataset_diversity(motions: np.ndarray, n_pairs: int = 500, seed: int = 0) -> float:
    k = motions.shape[0]
    assert k >= 2, "dataset diversity needs at least 2 motions"
    rng = np.random.default_rng(seed)
    i = rng.integers(0, k, size=n_pairs)
    j = rng.integers(0, k, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return float(np.linalg.norm(motions[i] - motions[j], axis=-1).mean() * 100)


def diversity(samples: np.ndarray) -> float:
    s = samples.shape[0]
    assert s >= 2, "diversity needs at least 2 samples"
    total = 0.0
    count = 0
    for i in range(s):
        for j in range(i + 1, s):
            total += np.linalg.norm(samples[i] - samples[j], axis=-1).mean()
            count += 1
    return float(total / count * 100)
