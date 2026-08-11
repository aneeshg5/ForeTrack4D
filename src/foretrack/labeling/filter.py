import numpy as np

# Calibrated against real lab GT, not placeholders:
# scripts/calibrate_filter_thresholds.py run against all 6381 DexYCB+ARCTIC+H2O train-split
# npz files on glacier, using the 99th percentile of each metric's distribution over that
# clean GT (i.e. a threshold that would reject only the noisiest ~1% of genuinely-clean lab
# tracks, calibrating "clearly worse than real motion looks" rather than an arbitrary round
# number). Re-run that script if the underlying GT changes.
DEFAULT_JERK_THRESHOLD = 0.084663  # m / frame^3
DEFAULT_DEPTH_VAR_THRESHOLD = 0.031315  # m^2, TAPIP3D's stated depth-flicker failure mode

# 0.5 is a generic starting point, not a calibrated value
# (unlike the two thresholds above). Real HoloAssist data showed 0.5 rejects 100% of clips: a
# 56-candidate batch spanning the full length of all 3 downloaded sessions
# found TAPIP3D's own visibility signal genuinely concentrated well below 0.5 for close-range
# egocentric hand-manipulation footage -- confirmed NOT a thresholding artifact (a direct sweep
# of TAPIP3D's own --vis_threshold from 0.9 down to 0.01 on a real clip only reached 0.41 mean
# visible fraction; the model is confidently marking points occluded, not merely uncertain).
# Of 38 scored clips in that batch, the real distribution was p1=0.095 p5=0.147 p10=0.164
# median=0.255 max=0.345, with exactly one outlier (0.073) that also had ~45x the sample's
# median depth variance -- a correlated real-failure signal, not just low visibility. 0.15
# (~p5) rejects that one genuine failure plus one borderline clip while accepting 35/38 (92%)
# of the real distribution, mirroring the reject-the-worst-tail philosophy the other two
# thresholds use, adapted to a lower-bound gate.
DEFAULT_MIN_VISIBLE_FRAC = 0.15


def jerk_gate(tracks: np.ndarray, threshold: float) -> bool:
    """True (pass) if the clip's motion is smooth enough. tracks: (T, N, 3). Jerk is the 3rd
    discrete derivative of position -- sudden direction/speed changes (TAPIP3D tracking
    failures, not real object motion) spike here far more than in a genuinely smooth grasp.
    Per point, take the peak jerk over time (a single bad frame should not be averaged away by
    the other T-3 mostly-smooth frames), then the median across points -- "median per-point
    frame-to-frame jerk"."""
    if tracks.shape[0] < 4:
        return True
    velocity = np.diff(tracks, axis=0)
    accel = np.diff(velocity, axis=0)
    jerk = np.diff(accel, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=-1)  # (T-3, N)
    per_point_peak = jerk_mag.max(axis=0)  # (N,)
    return float(np.median(per_point_peak)) <= threshold


def depth_variance_gate(tracks: np.ndarray, threshold: float) -> bool:
    """True (pass) unless per-point depth is flickering (TAPIP3D's stated failure mode on
    small/distant elements) rather than smoothly tracking real depth change."""
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
