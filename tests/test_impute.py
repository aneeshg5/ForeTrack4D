from pathlib import Path

import cv2
import numpy as np

from foretrack.labeling.impute import (
    build_clip_depths,
    clip_frame_range,
    clip_output_path,
    find_nearby_depth,
    read_video_frames,
)


def test_clip_frame_range_matches_holoassist_release_indexing():
    # int(t * 30), per Ember-HoloAssist/holoassist-release's own frame-index scheme.
    assert clip_frame_range(0.0, 1.0) == (0, 30)
    assert clip_frame_range(1.5, 2.0) == (45, 60)
    assert clip_frame_range(0.033, 0.066) == (0, 1)


def test_clip_output_path():
    path = clip_output_path("/data/out", "train", "R001-video", 30, 60)
    assert path == Path("/data/out/train/R001-video_000030_000060.npz")


def _write_synthetic_video(path, num_frames=10, size=(64, 48)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), i * 10 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_read_video_frames_returns_requested_range(tmp_path):
    video_path = tmp_path / "video.mp4"
    _write_synthetic_video(video_path, num_frames=10)
    frames = read_video_frames(str(video_path), 2, 5)
    assert frames.shape[0] == 3
    assert frames.shape[1:] == (48, 64, 3)


def test_read_video_frames_empty_range_returns_empty_array(tmp_path):
    video_path = tmp_path / "video.mp4"
    _write_synthetic_video(video_path, num_frames=10)
    frames = read_video_frames(str(video_path), 5, 5)
    assert frames.shape[0] == 0


def test_build_clip_depths_shape_and_values(tmp_path):
    ahat_dir = tmp_path / "AhatDepth"
    ahat_dir.mkdir()
    depth_mm = np.zeros((64, 64), dtype=np.uint16)
    depth_mm[32, 32] = 2000
    cv2.imwrite(str(ahat_dir / "000000.png"), depth_mm)

    intrinsics = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1]], dtype=np.float32)
    identity_pose = np.eye(4, dtype=np.float32)
    t = 3
    ahat_frame_number = np.zeros(t, dtype=np.int64)  # every frame reuses the same AHAT image
    ahat_cam_poses = np.stack([identity_pose] * t)
    cam_poses = np.stack([identity_pose] * t)

    depths = build_clip_depths(
        tmp_path, 0, t, ahat_frame_number, ahat_cam_poses, intrinsics, intrinsics, cam_poses, 64, 64
    )
    assert depths.shape == (t, 64, 64)
    assert depths.dtype == np.float32
    np.testing.assert_allclose(depths[:, 32, 32], 2.0, atol=1e-2)


def test_build_clip_depths_missing_png_gives_zero_frame(tmp_path):
    (tmp_path / "AhatDepth").mkdir()
    intrinsics = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1]], dtype=np.float32)
    identity_pose = np.eye(4, dtype=np.float32)
    ahat_frame_number = np.array([999], dtype=np.int64)  # no such file on disk
    ahat_cam_poses = np.stack([identity_pose])
    cam_poses = np.stack([identity_pose])

    depths = build_clip_depths(
        tmp_path, 0, 1, ahat_frame_number, ahat_cam_poses, intrinsics, intrinsics, cam_poses, 64, 64
    )
    assert depths.shape == (1, 64, 64)
    assert (depths == 0).all()


def _write_depth_png(path, value=1000):
    depth = np.full((16, 16), value, dtype=np.uint16)
    cv2.imwrite(str(path), depth)


def test_find_nearby_depth_exact_frame_available(tmp_path):
    ahat_dir = tmp_path / "AhatDepth"
    ahat_dir.mkdir()
    _write_depth_png(ahat_dir / "000010.png")
    ahat_frame_number = np.array([10], dtype=np.int64)

    frame, depth = find_nearby_depth(tmp_path, ahat_frame_number, 0)
    assert frame == 0
    assert depth is not None


def test_find_nearby_depth_searches_outward_when_missing(tmp_path):
    ahat_dir = tmp_path / "AhatDepth"
    ahat_dir.mkdir()
    # frame_number[5] points at a PNG that doesn't exist; frame_number[7] does.
    _write_depth_png(ahat_dir / "000099.png")
    ahat_frame_number = np.array([0, 0, 0, 0, 0, 888, 0, 99, 0, 0], dtype=np.int64)

    frame, depth = find_nearby_depth(tmp_path, ahat_frame_number, 5, search_radius=15)
    assert frame == 7
    assert depth is not None


def test_find_nearby_depth_returns_none_when_nothing_in_radius(tmp_path):
    ahat_dir = tmp_path / "AhatDepth"
    ahat_dir.mkdir()
    ahat_frame_number = np.array([888] * 10, dtype=np.int64)

    frame, depth = find_nearby_depth(tmp_path, ahat_frame_number, 5, search_radius=3)
    assert frame is None
    assert depth is None
