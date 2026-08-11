import numpy as np

from foretrack.eval.metrics import ade, ade_first_frame_aligned, ade_global_aligned, ade_per_timestep, apd3d, diversity, fde


def test_ade_zero_for_identical_tracks():
    gt = np.random.randn(10, 64, 3).astype(np.float32)
    assert ade(gt, gt) == 0.0
    assert fde(gt, gt) == 0.0


def test_ade_known_offset():
    gt = np.zeros((5, 4, 3), dtype=np.float32)
    pred = gt.copy()
    pred[..., 0] = 0.1  # 10cm offset on every point, every frame
    assert np.isclose(ade(pred, gt), 10.0, atol=1e-3)
    assert np.isclose(fde(pred, gt), 10.0, atol=1e-3)


def test_ade_respects_mask():
    gt = np.zeros((4, 2, 3), dtype=np.float32)
    pred = gt.copy()
    pred[2:] = 100.0  # garbage in the padded (invalid) frames
    mask = np.array([True, True, False, False])
    assert ade(pred, gt, mask) == 0.0
    assert fde(pred, gt, mask) == 0.0  # last VALID frame is index 1, not the garbage index 3


def test_first_frame_alignment_removes_constant_offset():
    gt = np.random.randn(8, 16, 3).astype(np.float32)
    pred = gt + np.array([1.0, 2.0, 3.0], dtype=np.float32)  # pred is gt + constant offset
    assert np.isclose(ade_first_frame_aligned(pred, gt), 0.0, atol=1e-3)
    assert np.isclose(ade_global_aligned(pred, gt), 0.0, atol=1e-3)
    assert ade(pred, gt) > 0  # unaligned ADE should NOT be zero


def test_diversity_zero_for_identical_samples():
    samples = np.repeat(np.random.randn(1, 8, 16, 3).astype(np.float32), 5, axis=0)
    assert diversity(samples) == 0.0


def test_diversity_positive_for_different_samples():
    base = np.random.randn(8, 16, 3).astype(np.float32)
    samples = np.stack([base + i * 0.1 for i in range(5)])
    assert diversity(samples) > 0


def test_apd3d_all_thresholds_pass():
    gt = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)  # (T=1, N=1, 3), depth=1m
    pred = gt.copy()
    pred[..., 0] = 0.001  # 1mm error, well under even the tightest threshold (2mm at f=500)
    assert apd3d(pred, gt, fx=500, fy=500) == 100.0


def test_apd3d_partial_thresholds_pass():
    # thresholds at f=500, depth=1m: {0.002, 0.004, 0.008, 0.016, 0.032}m for delta_2d={1,2,4,8,16}
    gt = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    pred = gt.copy()
    pred[..., 0] = 0.01  # 1cm error: passes only the 0.016 and 0.032 thresholds -> 2/5
    assert np.isclose(apd3d(pred, gt, fx=500, fy=500), 40.0)


def test_apd3d_excludes_invisible_points():
    gt = np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=np.float32)  # (T=1, N=2, 3)
    pred = gt.copy()
    pred[:, 0, 0] = 0.001  # visible point: tiny error, always correct
    pred[:, 1, 0] = 10.0  # invisible point: huge error, would fail every threshold
    visibility = np.array([[True, False]])
    assert apd3d(pred, gt, fx=500, fy=500, visibility=visibility) == 100.0


def test_ade_per_timestep_known_offset():
    gt = np.zeros((5, 4, 3), dtype=np.float32)
    pred = gt.copy()
    pred[..., 0] = 0.1
    np.testing.assert_allclose(ade_per_timestep(pred, gt), np.full(5, 10.0), atol=1e-3)


def test_ade_per_timestep_masks_as_nan():
    gt = np.zeros((4, 2, 3), dtype=np.float32)
    pred = gt.copy()
    mask = np.array([True, True, False, False])
    result = ade_per_timestep(pred, gt, mask)
    assert not np.isnan(result[0]) and not np.isnan(result[1])
    assert np.isnan(result[2]) and np.isnan(result[3])
