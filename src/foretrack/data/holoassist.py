# File-format handling verified against taeinkwon/PyHoloAssist. See NOTICE.md.

import glob
import json
import os

import numpy as np

from .transforms import MATHNET_TO_CV

TICKS_PER_SECOND = 10_000_000  # .NET DateTime.Ticks: 100ns units

HAND_JOINT_NAMES = [
    "Palm", "Wrist",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyTip",
]
NUM_HAND_JOINTS = len(HAND_JOINT_NAMES)
PALM_JOINT_INDEX = HAND_JOINT_NAMES.index("Palm")

_HAND_SYNC_COLS = 2 + 1 + NUM_HAND_JOINTS * 16 + NUM_HAND_JOINTS + NUM_HAND_JOINTS
_HAND_JOINT_START = 3


def _parse_coordinate_system(fields: list) -> np.ndarray:
    return np.array(fields, dtype=np.float64).reshape(4, 4)


def parse_pose_file(path: str) -> tuple:
    sync_time, orig_ticks, poses = [], [], []
    with open(path) as f:
        for line in f:
            fields = line.split()
            sync_time.append(float(fields[0]))
            orig_ticks.append(float(fields[1]))
            poses.append(_parse_coordinate_system(fields[2:18]))
    return np.array(sync_time, dtype=np.float64), np.array(orig_ticks, dtype=np.float64), np.stack(poses)


def parse_intrinsics_file(path: str) -> tuple:
    with open(path) as f:
        fields = f.read().split()
    values = [float(x) for x in fields]
    intrinsics = np.array(values[:9], dtype=np.float64).reshape(3, 3)
    image_size = np.array(values[-2:], dtype=np.int64)
    return intrinsics, image_size


def scale_intrinsics(intrinsics: np.ndarray, declared_size: tuple, actual_size: tuple) -> np.ndarray:
    declared_w, declared_h = declared_size
    actual_w, actual_h = actual_size
    scale_x, scale_y = actual_w / declared_w, actual_h / declared_h
    scaled = intrinsics.copy()
    scaled[0, 0] *= scale_x  # fx
    scaled[0, 2] *= scale_x  # cx
    scaled[1, 1] *= scale_y  # fy
    scaled[1, 2] *= scale_y  # cy
    return scaled


def parse_hand_file(path: str) -> tuple:
    leading, joints, valid, tracked = [], [], [], []
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) != _HAND_SYNC_COLS:
                raise ValueError(f"{path}: unexpected column count {len(fields)} (expected {_HAND_SYNC_COLS})")
            leading.append([float(fields[0]), float(fields[1])])
            joint_end = _HAND_JOINT_START + NUM_HAND_JOINTS * 16
            joints.append(
                np.array(fields[_HAND_JOINT_START : joint_end], dtype=np.float64).reshape(NUM_HAND_JOINTS, 4, 4)
            )
            valid.append([f == "1" for f in fields[joint_end : joint_end + NUM_HAND_JOINTS]])
            tracked.append([f == "1" for f in fields[joint_end + NUM_HAND_JOINTS : joint_end + 2 * NUM_HAND_JOINTS]])
    return (
        np.array(leading, dtype=np.float64),
        np.stack(joints),
        np.array(valid, dtype=bool),
        np.array(tracked, dtype=bool),
    )


def world_to_camera_frame(points_world: np.ndarray, cam_pose_world: np.ndarray) -> np.ndarray:
    R, t = cam_pose_world[:3, :3].astype(np.float64), cam_pose_world[:3, 3].astype(np.float64)
    world_to_cam_R = MATHNET_TO_CV.astype(np.float64) @ R.T
    world_to_cam_t = -world_to_cam_R @ t
    return points_world @ world_to_cam_R.T + world_to_cam_t


def project_to_pixels(points_world: np.ndarray, cam_pose_world: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points_cam = world_to_camera_frame(points_world, cam_pose_world)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    z = points_cam[:, 2]
    return np.stack([points_cam[:, 0] / z * fx + cx, points_cam[:, 1] / z * fy + cy], axis=-1)


AHAT_DEPTH_SCALE_MM_TO_M = 1.0 / 1000.0


def parse_ahat_depth_timing(path: str) -> tuple:
    sync_time, frame_number, orig_ticks = [], [], []
    with open(path) as f:
        for line in f:
            fields = line.split()
            sync_time.append(float(fields[0]))
            frame_number.append(int(fields[1]))
            orig_ticks.append(float(fields[2]))
    return np.array(sync_time, dtype=np.float64), np.array(frame_number, dtype=np.int64), np.array(orig_ticks, dtype=np.float64)


def ahat_depth_filename(frame_number: int) -> str:
    return f"{frame_number:06d}.png"


def unproject_depth(depth_mm: np.ndarray, intrinsics: np.ndarray, cam_pose_world: np.ndarray) -> np.ndarray:
    h, w = depth_mm.shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth_mm.astype(np.float64) * AHAT_DEPTH_SCALE_MM_TO_M
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    points_cam_cv = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    R, t = cam_pose_world[:3, :3].astype(np.float64), cam_pose_world[:3, 3].astype(np.float64)
    cam_to_world_R = R @ MATHNET_TO_CV.T.astype(np.float64)
    points_world = (points_cam_cv @ cam_to_world_R.T + t).reshape(h, w, 3)
    points_world[depth_mm == 0] = np.nan
    return points_world


def lift_rgb_query_points(
    query_uv_rgb: np.ndarray,
    depth_mm: np.ndarray,
    depth_intrinsics: np.ndarray,
    depth_cam_pose: np.ndarray,
    rgb_intrinsics: np.ndarray,
    rgb_cam_pose: np.ndarray,
    max_pixel_dist: float = 4.0,
) -> np.ndarray:
    points_world = unproject_depth(depth_mm, depth_intrinsics, depth_cam_pose)
    valid = ~np.isnan(points_world[..., 0])
    pts_world = points_world[valid]
    if len(pts_world) == 0:
        return np.full((len(query_uv_rgb), 3), np.nan)

    pts_uv_rgb = project_to_pixels(pts_world, rgb_cam_pose, rgb_intrinsics)

    out = np.full((len(query_uv_rgb), 3), np.nan)
    for i in range(len(query_uv_rgb)):
        dists = np.linalg.norm(pts_uv_rgb - query_uv_rgb[i], axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= max_pixel_dist:
            out[i] = pts_world[j]
    return out


def reproject_depth_to_rgb(
    depth_mm: np.ndarray,
    depth_intrinsics: np.ndarray,
    depth_cam_pose: np.ndarray,
    rgb_intrinsics: np.ndarray,
    rgb_cam_pose: np.ndarray,
    rgb_height: int,
    rgb_width: int,
) -> np.ndarray:
    points_world = unproject_depth(depth_mm, depth_intrinsics, depth_cam_pose)
    valid = ~np.isnan(points_world[..., 0])
    pts_world = points_world[valid]
    depth_out = np.zeros((rgb_height, rgb_width), dtype=np.float32)
    if len(pts_world) == 0:
        return depth_out

    pts_cam_rgb = world_to_camera_frame(pts_world, rgb_cam_pose)
    fx, fy, cx, cy = rgb_intrinsics[0, 0], rgb_intrinsics[1, 1], rgb_intrinsics[0, 2], rgb_intrinsics[1, 2]
    z = pts_cam_rgb[:, 2]
    u = np.round(pts_cam_rgb[:, 0] / z * fx + cx).astype(np.int64)
    v = np.round(pts_cam_rgb[:, 1] / z * fy + cy).astype(np.int64)

    in_bounds = (z > 0) & (u >= 0) & (u < rgb_width) & (v >= 0) & (v < rgb_height)
    flat_idx = v[in_bounds] * rgb_width + u[in_bounds]
    depth_flat = np.full(rgb_height * rgb_width, np.inf, dtype=np.float32)
    np.minimum.at(depth_flat, flat_idx, z[in_bounds].astype(np.float32))
    depth_flat[np.isinf(depth_flat)] = 0.0
    return depth_flat.reshape(rgb_height, rgb_width)


def hand_contact_point(
    joints: np.ndarray,
    joints_valid: np.ndarray,
    joints_tracked: np.ndarray,
    cam_pose_world: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    joint_index: int = PALM_JOINT_INDEX,
) -> tuple:
    if not (joints_valid[joint_index] and joints_tracked[joint_index]):
        return None
    joint_world = joints[joint_index, :3, 3]
    point_cam = world_to_camera_frame(joint_world[None], cam_pose_world)[0]
    if point_cam[2] <= 0:
        return None
    uv = project_to_pixels(joint_world[None], cam_pose_world, intrinsics)[0]
    if not (0 <= uv[0] < image_width and 0 <= uv[1] < image_height):
        return None
    return float(uv[0]), float(uv[1])


def hand_landmark_pixels(
    joints: np.ndarray,
    joints_valid: np.ndarray,
    joints_tracked: np.ndarray,
    cam_pose_world: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    valid_tracked = joints_valid & joints_tracked
    if not valid_tracked.any():
        return np.zeros((0, 2), dtype=np.float32)
    joint_world = joints[valid_tracked, :3, 3]
    point_cam = world_to_camera_frame(joint_world, cam_pose_world)
    in_front = point_cam[:, 2] > 0
    joint_world = joint_world[in_front]
    if len(joint_world) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    uv = project_to_pixels(joint_world, cam_pose_world, intrinsics)
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < image_width) & (uv[:, 1] >= 0) & (uv[:, 1] < image_height)
    return uv[in_bounds].astype(np.float32)


def best_hand_contact_point(
    left_joints: np.ndarray, left_valid: np.ndarray, left_tracked: np.ndarray,
    right_joints: np.ndarray, right_valid: np.ndarray, right_tracked: np.ndarray,
    cam_pose_world: np.ndarray, intrinsics: np.ndarray, image_width: int, image_height: int,
    joint_index: int = PALM_JOINT_INDEX,
) -> tuple:
    left_pt = hand_contact_point(
        left_joints, left_valid, left_tracked, cam_pose_world, intrinsics, image_width, image_height, joint_index
    )
    right_pt = hand_contact_point(
        right_joints, right_valid, right_tracked, cam_pose_world, intrinsics, image_width, image_height, joint_index
    )
    if left_pt is not None and right_pt is not None:
        return left_pt if left_tracked.sum() >= right_tracked.sum() else right_pt
    return left_pt if left_pt is not None else right_pt


HOLOASSIST_FPS = 30  # Ember-HoloAssist/holoassist-release's own MultiRawDataset.fps


def read_split(label_root: str, subset: str) -> list:
    path = os.path.join(label_root, f"{subset}_0724.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _find_label_files(label_root: str) -> tuple:
    classes_files = sorted(glob.glob(os.path.join(label_root, "labels_*_classes.json")))
    if not classes_files:
        raise FileNotFoundError(f"no labels_*_classes.json found under {label_root}")
    classes_file = classes_files[-1]
    label2idx_file = classes_file.replace("_classes.json", "_label2idx.json")
    return classes_file, label2idx_file


def list_annotated_clips(label_root: str, subset: str, event_key: str = "fine_grained_action") -> list:
    classes_file, _ = _find_label_files(label_root)
    with open(classes_file) as f:
        events = json.load(f)[event_key]

    sessions = set(read_split(label_root, subset))
    clips = []
    for video_name, video_events in events.items():
        if video_name not in sessions:
            continue
        for t_start, t_end, label in video_events:
            if t_end < t_start or int((t_end - t_start) * HOLOASSIST_FPS) == 0:
                continue
            task_id, label_id = label
            clips.append((video_name, t_start, t_end, task_id, label_id))
    return clips
