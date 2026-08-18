import numpy as np

DEFAULT_JERK_THRESHOLD = 0.084663  # m / frame^3
DEFAULT_DEPTH_VAR_THRESHOLD = 0.031315  # m^2, TAPIP3D's stated depth-flicker failure mode

DEFAULT_MIN_VISIBLE_FRAC = 0.15


def jerk_gate(tracks: np.ndarray, threshold: float) -> bool:
    if tracks.shape[0] < 4:
        return True
    velocity = np.diff(tracks, axis=0)
    accel = np.diff(velocity, axis=0)
    jerk = np.diff(accel, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=-1)  # (T-3, N)
    per_point_peak = jerk_mag.max(axis=0)  # (N,)
    return float(np.median(per_point_peak)) <= threshold


def depth_variance_gate(tracks: np.ndarray, threshold: float) -> bool:
    depth = tracks[..., 2]  # (T, N)
    return float(np.median(depth.var(axis=0))) <= threshold


def visibility_fraction_gate(visibility: np.ndarray, min_frac: float = DEFAULT_MIN_VISIBLE_FRAC) -> bool:
    return visibility.mean(axis=0).mean() >= min_frac


def accept_clip(tracks: np.ndarray, visibility: np.ndarray, cfg: dict) -> bool:
    return (
        jerk_gate(tracks, cfg.get("jerk_threshold", DEFAULT_JERK_THRESHOLD))
        and depth_variance_gate(tracks, cfg.get("depth_var_threshold", DEFAULT_DEPTH_VAR_THRESHOLD))
        and visibility_fraction_gate(visibility, cfg.get("min_visible_frac", DEFAULT_MIN_VISIBLE_FRAC))
    )
