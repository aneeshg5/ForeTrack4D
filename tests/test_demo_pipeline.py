import numpy as np

from foretrack.demo_pipeline import detect_likely_cuts, estimate_intrinsics


def test_estimate_intrinsics_uses_max_dim_as_focal():
    k = estimate_intrinsics(h=480, w=640)
    assert k[0, 0] == 640.0
    assert k[1, 1] == 640.0
    assert k[0, 2] == 320.0
    assert k[1, 2] == 240.0


def test_detect_likely_cuts_no_cuts_in_smooth_video():
    frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    for i in range(10):
        frames[i] = min(i * 5, 255)  # gentle, gradual brightness ramp
    assert detect_likely_cuts(frames) == []


def test_detect_likely_cuts_flags_a_hard_cut():
    frames = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    frames[:3] = 10
    frames[3:] = 250  # abrupt scene change at frame 3
    cuts = detect_likely_cuts(frames)
    assert 3 in cuts
