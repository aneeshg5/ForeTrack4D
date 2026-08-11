import numpy as np
import pytest

from foretrack.labeling.run_tapip3d import _latest_result_npz, lift_query_points


def test_lift_query_points_center_pixel_at_known_depth():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]], dtype=np.float32)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    query_uv = np.array([[320.0, 240.0]], dtype=np.float32)  # principal point
    out = lift_query_points(query_uv, depth, intrinsics)
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0, 2.0], atol=1e-4)


def test_lift_query_points_off_center_pixel():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]], dtype=np.float32)
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    query_uv = np.array([[420.0, 240.0]], dtype=np.float32)  # 100px right of center
    out = lift_query_points(query_uv, depth, intrinsics)
    expected_x = (420.0 - 320.0) / 500.0 * 4.0
    np.testing.assert_allclose(out[0], [0.0, expected_x, 0.0, 4.0], atol=1e-4)


def test_lift_query_points_query_time_is_always_zero():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]], dtype=np.float32)
    depth = np.full((480, 640), 1.0, dtype=np.float32)
    query_uv = np.stack(
        [np.random.uniform(0, 639, size=10), np.random.uniform(0, 479, size=10)], axis=-1
    ).astype(np.float32)
    out = lift_query_points(query_uv, depth, intrinsics)
    np.testing.assert_allclose(out[:, 0], 0.0)


def test_latest_result_npz_picks_most_recent(tmp_path):
    import time

    d1 = tmp_path / "2024-01-01_00-00-00"
    d1.mkdir()
    (d1 / "input.result.npz").write_bytes(b"old")
    time.sleep(0.01)
    d2 = tmp_path / "2024-01-02_00-00-00"
    d2.mkdir()
    (d2 / "input.result.npz").write_bytes(b"new")

    latest = _latest_result_npz(tmp_path)
    assert latest == d2 / "input.result.npz"


def test_latest_result_npz_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        _latest_result_npz(tmp_path)
