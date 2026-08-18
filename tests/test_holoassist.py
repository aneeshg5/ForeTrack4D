import json

import numpy as np
import pytest

from foretrack.data.holoassist import (
    NUM_HAND_JOINTS,
    PALM_JOINT_INDEX,
    ahat_depth_filename,
    best_hand_contact_point,
    hand_contact_point,
    hand_landmark_pixels,
    lift_rgb_query_points,
    list_annotated_clips,
    parse_ahat_depth_timing,
    parse_hand_file,
    parse_intrinsics_file,
    parse_pose_file,
    project_to_pixels,
    read_split,
    reproject_depth_to_rgb,
    scale_intrinsics,
    unproject_depth,
)


def _write_pose_line(f, sync_time, orig_time, pose4x4):
    fields = [str(sync_time), str(orig_time)] + [repr(float(x)) for x in pose4x4.reshape(-1)]
    f.write("\t".join(fields) + "\n")


def test_parse_pose_file(tmp_path):
    path = tmp_path / "Pose_sync.txt"
    identity = np.eye(4)
    shifted = np.eye(4)
    shifted[:3, 3] = [1.0, 2.0, 3.0]
    with open(path, "w") as f:
        _write_pose_line(f, 0.0, 637917497145832487, identity)
        _write_pose_line(f, 0.033, 637917497145832487 + 330000, shifted)

    sync_time, orig_ticks, poses = parse_pose_file(str(path))
    np.testing.assert_allclose(sync_time, [0.0, 0.033])
    assert orig_ticks[0] == 637917497145832487
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses[0], identity)
    np.testing.assert_allclose(poses[1], shifted)


def test_parse_intrinsics_file(tmp_path):
    path = tmp_path / "Intrinsics.txt"
    k_fields = ["500.0", "0", "320.0", "0", "505.0", "240.0", "0", "0", "1"]
    middle = [str(0.0)] * 14
    trailing = ["640", "480"]
    with open(path, "w") as f:
        f.write("\t".join(k_fields + middle + trailing) + "\n")

    intrinsics, size = parse_intrinsics_file(str(path))
    expected_k = np.array([[500.0, 0, 320.0], [0, 505.0, 240.0], [0, 0, 1]])
    np.testing.assert_allclose(intrinsics, expected_k)
    np.testing.assert_allclose(size, [640, 480])


def test_scale_intrinsics_uniform_half_scale():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    scaled = scale_intrinsics(intrinsics, declared_size=(640, 480), actual_size=(320, 240))
    expected = np.array([[250.0, 0, 160.0], [0, 250.0, 120.0], [0, 0, 1]])
    np.testing.assert_allclose(scaled, expected)


def test_scale_intrinsics_nonuniform_axes():
    intrinsics = np.array([[681.388, 0, 445.448], [0, 682.386, 237.431], [0, 0, 1]])
    scaled = scale_intrinsics(intrinsics, declared_size=(896, 504), actual_size=(454, 256))
    np.testing.assert_allclose(scaled[0, 0], 681.388 * 454 / 896, atol=1e-3)
    np.testing.assert_allclose(scaled[1, 1], 682.386 * 256 / 504, atol=1e-3)
    np.testing.assert_allclose(scaled[0, 2], 445.448 * 454 / 896, atol=1e-3)
    np.testing.assert_allclose(scaled[1, 2], 237.431 * 256 / 504, atol=1e-3)
    assert scaled[2, 2] == 1.0
    assert scaled[0, 1] == 0.0


def test_parse_hand_file(tmp_path):
    path = tmp_path / "Left_sync.txt"
    joints = np.zeros((NUM_HAND_JOINTS, 4, 4))
    for j in range(NUM_HAND_JOINTS):
        joints[j] = np.eye(4)
        joints[j, :3, 3] = j
    valid = ["1"] * NUM_HAND_JOINTS
    tracked = ["1"] * NUM_HAND_JOINTS
    fields = ["0", "0", "1"] + [repr(float(x)) for x in joints.reshape(-1)] + valid + tracked
    with open(path, "w") as f:
        f.write("\t".join(fields) + "\n")

    leading, out_joints, out_valid, out_tracked = parse_hand_file(str(path))
    assert leading.shape == (1, 2)
    assert out_joints.shape == (1, NUM_HAND_JOINTS, 4, 4)
    assert out_valid.shape == (1, NUM_HAND_JOINTS)
    assert out_tracked.shape == (1, NUM_HAND_JOINTS)
    assert out_valid.all()
    assert out_tracked.all()
    np.testing.assert_allclose(out_joints[0, PALM_JOINT_INDEX, :3, 3], [0, 0, 0])
    np.testing.assert_allclose(out_joints[0, PALM_JOINT_INDEX + 1, :3, 3], [1, 1, 1])


def test_parse_hand_file_rejects_bad_column_count(tmp_path):
    path = tmp_path / "Left_sync.txt"
    with open(path, "w") as f:
        f.write("0\t1\t2\t3\n")
    try:
        parse_hand_file(str(path))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_project_to_pixels_point_directly_ahead():
    cam_pose = np.eye(4)
    point_world = np.array([[2.0, 0.0, 0.0]])
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    uv = project_to_pixels(point_world, cam_pose, intrinsics)
    np.testing.assert_allclose(uv, [[320.0, 240.0]], atol=1e-6)


def test_project_to_pixels_point_to_the_left():
    cam_pose = np.eye(4)
    point_world = np.array([[2.0, 1.0, 0.0]])
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    uv = project_to_pixels(point_world, cam_pose, intrinsics)
    assert uv[0, 0] < 320.0


def _write_label_root(label_root, sessions_by_split, events, label2idx=None):
    label_root.mkdir(parents=True, exist_ok=True)
    for subset, sessions in sessions_by_split.items():
        (label_root / f"{subset}_0724.txt").write_text("\n".join(sessions) + "\n")
    with open(label_root / "labels_20230724_2221_classes.json", "w") as f:
        json.dump(events, f)
    with open(label_root / "labels_20230724_2221_label2idx.json", "w") as f:
        json.dump(label2idx or {}, f)


def test_read_split(tmp_path):
    _write_label_root(tmp_path, {"train": ["R001-video", "R002-video"], "val": ["R003-video"]}, {})
    assert read_split(str(tmp_path), "train") == ["R001-video", "R002-video"]
    assert read_split(str(tmp_path), "val") == ["R003-video"]


def test_list_annotated_clips_filters_to_split_and_event_key(tmp_path):
    events = {
        "fine_grained_action": {
            "R001-video": [[0.0, 1.0, [1, 5]], [2.0, 2.5, [1, 6]]],
            "R999-not-in-split": [[0.0, 1.0, [1, 5]]],
        },
        "coarse_grained_action": {"R001-video": [[0.0, 5.0, [2, 1]]]},
    }
    _write_label_root(tmp_path, {"train": ["R001-video"]}, events)
    clips = list_annotated_clips(str(tmp_path), "train", event_key="fine_grained_action")
    assert clips == [("R001-video", 0.0, 1.0, 1, 5), ("R001-video", 2.0, 2.5, 1, 6)]


def test_list_annotated_clips_drops_degenerate_clips(tmp_path):
    events = {
        "fine_grained_action": {
            "R001-video": [
                [1.0, 0.5, [1, 5]],  # t_end < t_start
                [1.0, 1.01, [1, 6]],  # shorter than one frame at 30fps
                [1.0, 2.0, [1, 7]],  # valid
            ],
        }
    }
    _write_label_root(tmp_path, {"train": ["R001-video"]}, events)
    clips = list_annotated_clips(str(tmp_path), "train", event_key="fine_grained_action")
    assert clips == [("R001-video", 1.0, 2.0, 1, 7)]


def _make_joints():
    joints = np.zeros((NUM_HAND_JOINTS, 4, 4))
    for j in range(NUM_HAND_JOINTS):
        joints[j] = np.eye(4)
    joints[PALM_JOINT_INDEX, :3, 3] = [2.0, 0.0, 0.0]  # 2 units straight ahead (mathnet +x)
    return joints


def test_hand_contact_point_projects_tracked_joint():
    joints = _make_joints()
    valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked = np.ones(NUM_HAND_JOINTS, dtype=bool)
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    pt = hand_contact_point(joints, valid, tracked, cam_pose, intrinsics, 640, 480)
    assert pt == pytest.approx((320.0, 240.0), abs=1e-4)


def test_hand_contact_point_returns_none_when_untracked():
    joints = _make_joints()
    valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked[PALM_JOINT_INDEX] = False
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    assert hand_contact_point(joints, valid, tracked, cam_pose, intrinsics, 640, 480) is None


def test_hand_contact_point_returns_none_when_outside_frame():
    joints = _make_joints()
    valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked = np.ones(NUM_HAND_JOINTS, dtype=bool)
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    assert hand_contact_point(joints, valid, tracked, cam_pose, intrinsics, 10, 10) is None


def test_best_hand_contact_point_falls_back_to_tracked_hand():
    joints = _make_joints()
    all_valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    all_tracked = np.ones(NUM_HAND_JOINTS, dtype=bool)
    none_tracked = np.zeros(NUM_HAND_JOINTS, dtype=bool)
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])

    pt = best_hand_contact_point(
        joints, all_valid, none_tracked, joints, all_valid, all_tracked, cam_pose, intrinsics, 640, 480
    )
    assert pt == pytest.approx((320.0, 240.0), abs=1e-4)


def test_hand_landmark_pixels_returns_only_in_front_tracked_joints():
    joints = _make_joints()
    valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked = np.ones(NUM_HAND_JOINTS, dtype=bool)
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    uv = hand_landmark_pixels(joints, valid, tracked, cam_pose, intrinsics, 640, 480)
    assert uv.shape == (1, 2)
    np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-4)


def test_hand_landmark_pixels_excludes_untracked_joint():
    joints = _make_joints()
    valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked[PALM_JOINT_INDEX] = False
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    uv = hand_landmark_pixels(joints, valid, tracked, cam_pose, intrinsics, 640, 480)
    assert uv.shape == (0, 2)


def test_hand_landmark_pixels_excludes_out_of_bounds_joint():
    joints = _make_joints()
    valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    tracked = np.ones(NUM_HAND_JOINTS, dtype=bool)
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    uv = hand_landmark_pixels(joints, valid, tracked, cam_pose, intrinsics, 10, 10)
    assert uv.shape == (0, 2)


def test_best_hand_contact_point_none_when_neither_tracked():
    joints = _make_joints()
    all_valid = np.ones(NUM_HAND_JOINTS, dtype=bool)
    none_tracked = np.zeros(NUM_HAND_JOINTS, dtype=bool)
    cam_pose = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    pt = best_hand_contact_point(
        joints, all_valid, none_tracked, joints, all_valid, none_tracked, cam_pose, intrinsics, 640, 480
    )
    assert pt is None


def test_parse_ahat_depth_timing(tmp_path):
    path = tmp_path / "Timing_sync.txt"
    path.write_text(
        "0.0\t0\t637917497145832487\n"
        "0.03333333333333333\t43\t637917497156451749\n"
        "0.06666666666666667\t44\t637917497156671752\n"
    )
    sync_time, frame_number, orig_ticks = parse_ahat_depth_timing(str(path))
    np.testing.assert_allclose(sync_time, [0.0, 0.03333333333333333, 0.06666666666666667])
    np.testing.assert_array_equal(frame_number, [0, 43, 44])
    assert orig_ticks[1] == 637917497156451749


def test_ahat_depth_filename():
    assert ahat_depth_filename(33) == "000033.png"
    assert ahat_depth_filename(0) == "000000.png"


def test_unproject_depth_at_principal_point():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[240, 320] = 2000  # 2000mm = 2m, straight ahead
    points_world = unproject_depth(depth, intrinsics, cam_pose)
    np.testing.assert_allclose(points_world[240, 320], [2.0, 0.0, 0.0], atol=1e-3)
    assert np.isnan(points_world[0, 0, 0])  # untouched (zero-depth) pixels stay NaN


def test_unproject_depth_offset_camera():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [1.0, 0.0, 0.0]  # camera sits at mathnet (1,0,0)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[240, 320] = 3000  # 3m directly ahead of the camera
    points_world = unproject_depth(depth, intrinsics, cam_pose)
    np.testing.assert_allclose(points_world[240, 320], [4.0, 0.0, 0.0], atol=1e-3)


def test_lift_rgb_query_points_same_camera():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[240, 320] = 2000
    query_uv = np.array([[320.0, 240.0]])
    out = lift_rgb_query_points(query_uv, depth, intrinsics, cam_pose, intrinsics, cam_pose)
    np.testing.assert_allclose(out[0], [2.0, 0.0, 0.0], atol=1e-2)


def test_lift_rgb_query_points_no_nearby_depth_returns_nan():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[10, 10] = 2000  # far from the query pixel
    query_uv = np.array([[320.0, 240.0]])
    out = lift_rgb_query_points(query_uv, depth, intrinsics, cam_pose, intrinsics, cam_pose, max_pixel_dist=4.0)
    assert np.isnan(out[0]).all()


def test_reproject_depth_to_rgb_same_camera_round_trips():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[240, 320] = 2000
    depth[100, 100] = 1500
    out = reproject_depth_to_rgb(depth, intrinsics, cam_pose, intrinsics, cam_pose, 480, 640)
    assert out.shape == (480, 640)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[240, 320], 2.0, atol=1e-2)
    np.testing.assert_allclose(out[100, 100], 1.5, atol=1e-2)


def test_reproject_depth_to_rgb_empty_depth_returns_zeros():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    depth = np.zeros((480, 640), dtype=np.uint16)
    out = reproject_depth_to_rgb(depth, intrinsics, cam_pose, intrinsics, cam_pose, 480, 640)
    assert (out == 0).all()


def test_reproject_depth_to_rgb_uncovered_pixels_are_zero_not_nan():
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[240, 320] = 2000
    out = reproject_depth_to_rgb(depth, intrinsics, cam_pose, intrinsics, cam_pose, 480, 640)
    assert out[0, 0] == 0.0
    assert not np.isnan(out).any()


def test_reproject_depth_to_rgb_zbuffer_keeps_nearest():
    depth_intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    rgb_intrinsics = np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1]])
    cam_pose = np.eye(4)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[240, 320] = 3000  # farther
    depth[240, 321] = 1000  # nearer, adjacent pixel
    out = reproject_depth_to_rgb(depth, depth_intrinsics, cam_pose, rgb_intrinsics, cam_pose, 48, 64)
    np.testing.assert_allclose(out[24, 32], 1.0, atol=1e-2)
