import json

import numpy as np


def load_calibration(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def horizon_weights(t: int, fps: float, calib: dict, tol_cm: float = 5.0, floor: float = 0.0) -> np.ndarray:
    # Pseudo-labels are near-sensor-quality early and drift badly late (measured in
    # scripts/exp_depth_ablation.py), so supervise each timestep in proportion to how much
    # the label can be trusted there: full weight while expected label error stays within
    # tol_cm, falling off as tol/err beyond it.
    seconds = np.asarray(calib["seconds"], dtype=np.float64)
    err = np.asarray(calib["label_err_cm"], dtype=np.float64)
    grid = np.arange(t, dtype=np.float64) / fps
    err_t = np.interp(grid, seconds, err, left=err[0])
    # the calibration only reaches as far as the clips it was measured on, and drift keeps
    # growing past that, so extend the last measured slope rather than holding the curve flat
    beyond = grid > seconds[-1]
    if beyond.any():
        slope = (err[-1] - err[-2]) / max(seconds[-1] - seconds[-2], 1e-6)
        err_t[beyond] = err[-1] + slope * (grid[beyond] - seconds[-1])
    w = np.clip(tol_cm / np.maximum(err_t, tol_cm), 0.0, 1.0)
    return np.maximum(w, floor).astype(np.float32)
