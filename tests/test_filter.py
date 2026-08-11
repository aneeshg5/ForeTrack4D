import numpy as np

from foretrack.labeling.filter import (
    accept_clip,
    depth_variance_gate,
    jerk_gate,
    visibility_fraction_gate,
)


def test_jerk_gate_passes_smooth_motion():
    t = np.linspace(0, 1, 20)
    tracks = np.zeros((20, 4, 3), dtype=np.float32)
    tracks[..., 0] = t[:, None]  # constant velocity, zero jerk
    assert jerk_gate(tracks, threshold=1e-3)


def test_jerk_gate_rejects_spiky_motion():
    tracks = np.zeros((20, 4, 3), dtype=np.float32)
    tracks[10, :, 0] = 5.0  # single-frame spike
    assert not jerk_gate(tracks, threshold=0.05)


def test_jerk_gate_short_clip_passes():
    tracks = np.random.randn(2, 4, 3).astype(np.float32)
    assert jerk_gate(tracks, threshold=0.0)


def test_depth_variance_gate():
    stable = np.ones((10, 4, 3), dtype=np.float32)
    assert depth_variance_gate(stable, threshold=0.01)

    flickering = np.ones((10, 4, 3), dtype=np.float32)
    flickering[::2, :, 2] = 2.0
    assert not depth_variance_gate(flickering, threshold=0.01)


def test_visibility_fraction_gate():
    visibility = np.ones((10, 4), dtype=bool)
    assert visibility_fraction_gate(visibility, min_frac=0.5)
    visibility[:8] = False
    assert not visibility_fraction_gate(visibility, min_frac=0.5)


def test_accept_clip_requires_all_gates():
    t = np.linspace(0, 1, 20)
    tracks = np.ones((20, 4, 3), dtype=np.float32)
    tracks[..., 0] = t[:, None]
    visibility = np.ones((20, 4), dtype=bool)
    cfg = {"jerk_threshold": 1e-3, "depth_var_threshold": 0.01, "min_visible_frac": 0.5}
    assert accept_clip(tracks, visibility, cfg)

    visibility[:] = False
    assert not accept_clip(tracks, visibility, cfg)
