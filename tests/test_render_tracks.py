import cv2
import numpy as np

from foretrack.viz.render_tracks import project_points


def test_project_points_center_pixel_at_known_depth():
    intrinsics = np.array([[500.0, 0, 100], [0, 500.0, 80], [0, 0, 1]], dtype=np.float32)
    points = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)  # on-axis, at 2m depth
    uv = project_points(points, intrinsics)
    np.testing.assert_allclose(uv[0], [100.0, 80.0], atol=1e-3)


def test_project_points_offset_matches_pinhole_formula():
    intrinsics = np.array([[500.0, 0, 100], [0, 500.0, 80], [0, 0, 1]], dtype=np.float32)
    points = np.array([[0.1, 0.0, 2.0]], dtype=np.float32)  # 10cm off-axis at 2m depth
    uv = project_points(points, intrinsics)
    expected_u = 0.1 / 2.0 * 500.0 + 100  # = 125
    np.testing.assert_allclose(uv[0], [expected_u, 80.0], atol=1e-3)


def test_project_points_behind_camera_is_nan():
    intrinsics = np.array([[500.0, 0, 100], [0, 500.0, 80], [0, 0, 1]], dtype=np.float32)
    points = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)  # behind the camera
    uv = project_points(points, intrinsics)
    assert np.isnan(uv[0]).all()


def test_project_points_handles_batch_shape():
    intrinsics = np.array([[500.0, 0, 100], [0, 500.0, 80], [0, 0, 1]], dtype=np.float32)
    points = np.random.rand(4, 6, 3).astype(np.float32) + 1.0  # (T, N, 3), all in front of camera
    uv = project_points(points, intrinsics)
    assert uv.shape == (4, 6, 2)
    assert not np.isnan(uv).any()


def _video_fixture(f=20, h=48, w=64, cond_idx=10, t_obs=10, t_fc=14, s=2, n=4):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (f, h, w, 3), dtype=np.uint8)
    intrinsics = np.array([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]], dtype=np.float32)
    observed = rng.random((t_obs, n, 3)).astype(np.float32) + 1.0
    forecast = rng.random((s, t_fc, n, 3)).astype(np.float32) + 1.0
    return frames, cond_idx, observed, forecast, intrinsics




def test_demo_video_composite_layout_and_framecount(tmp_path):
    from foretrack.viz.demo_video import render_demo_video

    frames, cond_idx, observed, forecast, intrinsics = _video_fixture(f=30, cond_idx=10, t_obs=20, t_fc=24)
    out = tmp_path / "composite.mp4"
    render_demo_video(frames, cond_idx, observed, forecast, intrinsics, 5.0, str(out), playback_fps=15.0, hold_s=1.0)
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert (w, h) == (2 * frames.shape[2], 2 * frames.shape[1] + 30 + 44)
    assert n == 3 * (10 + 8 + 24 + 5)


