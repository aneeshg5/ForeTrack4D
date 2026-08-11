import glob
import json
import os

import numpy as np

from .transforms import MATHNET_TO_CV

# All byte formats below are confirmed against real downloaded HoloAssist files (public,
# ungated Azure blob storage), not just derived from source code.
# "_sync" files (Pose_sync.txt, Hands/Left_sync.txt, Hands/Right_sync.txt, AhatDepth's own
# Timing_sync.txt/Pose_sync.txt) share one 30fps clock and carry sync-specific fields the
# generic microsoft/psi per-modality exporter doesn't have; Intrinsics.txt is NOT a "_sync"
# file and is a single whole-session line (a physical camera's calibration doesn't vary per
# frame), confirmed via taeinkwon/PyHoloAssist's own parser.

TICKS_PER_SECOND = 10_000_000  # .NET DateTime.Ticks: 100ns units

# HandJointIndex enum order (Microsoft.Psi.MixedReality.HandJointIndex), 26 joints. Each line
# of Hands/Left_sync.txt / Right_sync.txt stores one CoordinateSystem (4x4 pose, 16 doubles)
# per joint, in this exact order.
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

# Hands_sync.txt, per raw_loader_index.py's own offsets (`line[2:]` then `[1:-52]` before
# reshaping to (-1,4,4)): 2 sync-prefix/time fields + IsActive(1) + 26*16 joint values +
# 26 JointsValid + 26 JointsTracked. Confirmed self-consistent: the release loader's "bad
# hands" zero-fallback is shaped (max_samples, 469), which is exactly this total (471) minus
# the 2 fields it drops via `line[2:]` -- not independently guessed.
_HAND_SYNC_COLS = 2 + 1 + NUM_HAND_JOINTS * 16 + NUM_HAND_JOINTS + NUM_HAND_JOINTS
_HAND_JOINT_START = 3


def _parse_coordinate_system(fields: list) -> np.ndarray:
    return np.array(fields, dtype=np.float64).reshape(4, 4)


def parse_pose_file(path: str) -> tuple:
    """Pose_sync.txt / Head_sync.txt (Video, AhatDepth, or Head's own pose stream -- all the
    same format). Each line has 2 leading fields then 16 doubles, row-major 4x4 pose.
    Confirmed against a real downloaded file (session z095-july-11-22-gladom_disassemble,
    AhatDepth/Pose_sync.txt): field[0] is 0.0 on the first line and increases from there
    (a synced relative-seconds clock, starting at session start), field[1] is a huge integer
    (~6.38e17) matching the expected magnitude of .NET DateTime.Ticks for a 2023 capture date
    (the original, pre-sync absolute timestamp). Returns (sync_time_sec (T,), orig_ticks (T,),
    poses (T,4,4)) -- poses are local-to-world (verified against
    DepthImageCameraViewAsMeshVisualizationObject's `point.TransformBy(CameraPose)` usage in
    microsoft/psi, i.e. CameraPose maps camera-local points into world space, not the
    reverse)."""
    sync_time, orig_ticks, poses = [], [], []
    with open(path) as f:
        for line in f:
            fields = line.split()
            sync_time.append(float(fields[0]))
            orig_ticks.append(float(fields[1]))
            poses.append(_parse_coordinate_system(fields[2:18]))
    return np.array(sync_time, dtype=np.float64), np.array(orig_ticks, dtype=np.float64), np.stack(poses)


def parse_intrinsics_file(path: str) -> tuple:
    """Intrinsics.txt: a SINGLE line for the whole session (a physical camera's calibration
    doesn't change frame to frame), not a per-frame time series -- this corrects an earlier,
    wrong assumption derived from the generic per-modality exporter's multi-line format.
    Confirmed against two real downloaded files (session z095-july-11-22-gladom_disassemble,
    both Video/Intrinsics.txt and AhatDepth/Intrinsics.txt): 25 tab-separated fields, no
    leading time field. Matches taeinkwon/PyHoloAssist's own `read_intrinsics_txt` exactly:
    fields[:9] reshape directly to the 3x3 intrinsics matrix, fields[-2:] are (width, height).
    Video's real values (896x504) independently match forehand4d's own hardcoded
    `self.img_h, self.img_w = 504, 896` in holo_dataset.py -- further cross-validation.
    Returns (intrinsics (3,3), image_size (2,) as (width, height))."""
    with open(path) as f:
        fields = f.read().split()
    values = [float(x) for x in fields]
    intrinsics = np.array(values[:9], dtype=np.float64).reshape(3, 3)
    image_size = np.array(values[-2:], dtype=np.int64)
    return intrinsics, image_size


def scale_intrinsics(intrinsics: np.ndarray, declared_size: tuple, actual_size: tuple) -> np.ndarray:
    """Intrinsics.txt's declared (width, height) doesn't always match the actually-decoded
    image -- confirmed on a real session (R012-7July-Nespresso): Intrinsics.txt declares
    896x504, but Video_compress.mp4 (a separately re-encoded, downscaled copy) actually decodes
    to 454x256. The two axis ratios aren't identical (454/896=0.5067 vs 256/504=0.5079),
    consistent with H.264 macroblock-size rounding during compression, so this scales fx/cx and
    fy/cy independently rather than assuming one uniform factor. actual_size/declared_size are
    each (width, height)."""
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
    """Hands/Left_sync.txt or Right_sync.txt. Returns (leading_fields (T,2) raw/uninterpreted,
    joints (T,26,4,4), joints_valid (T,26) bool, joints_tracked (T,26) bool)."""
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
    """points_world: (N,3) in the mathnet (forward=x, left=y, up=z) world frame -- same frame
    joint/camera poses are expressed in. cam_pose_world: (4,4) local-to-world camera pose
    (mathnet basis, both the rotation block and the translation). Returns (N,3) points in that
    camera's own local frame, opencv convention (x right, y down, z forward) -- i.e. what our
    project's world convention means when this camera is the one at t=0.

    Derivation (column-vector form, p in mathnet world basis): local_mathnet = R^T (p - t);
    local_opencv = B @ local_mathnet, where B = MATHNET_TO_CV converts a single vector's own
    basis representation (not a change of reference frame -- p is already local to the camera
    at this point, only its axis labeling changes). Combined:
    local_opencv = (B @ R^T) @ p - (B @ R^T) @ t.
    """
    R, t = cam_pose_world[:3, :3].astype(np.float64), cam_pose_world[:3, 3].astype(np.float64)
    world_to_cam_R = MATHNET_TO_CV.astype(np.float64) @ R.T
    world_to_cam_t = -world_to_cam_R @ t
    return points_world @ world_to_cam_R.T + world_to_cam_t


def project_to_pixels(points_world: np.ndarray, cam_pose_world: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """points_world: (N,3) mathnet-basis world points. cam_pose_world: (4,4) mathnet-basis
    local-to-world camera pose. intrinsics: (3,3), opencv pinhole convention. Returns (N,2)
    pixel coords."""
    points_cam = world_to_camera_frame(points_world, cam_pose_world)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    z = points_cam[:, 2]
    return np.stack([points_cam[:, 0] / z * fx + cx, points_cam[:, 1] / z * fy + cy], axis=-1)


# AhatDepth PNGs are uint16 raw millimeters (0 = no return), confirmed empirically against a
# real downloaded sample (session R0027-12-GoPro, frame 33: 512x512, values 1-1055, mean ~586)
# -- not documented in any source code reviewed, but physically consistent with AHAT's publicly
# documented short-throw hand-tracking depth range (~0.2-1.0m): 1055mm sits almost exactly at
# that range's upper bound. No independent second source confirms the mm scale specifically;
# flagged as empirical, not textual, verification.
AHAT_DEPTH_SCALE_MM_TO_M = 1.0 / 1000.0


def parse_ahat_depth_timing(path: str) -> tuple:
    """AhatDepth/Timing_sync.txt: `sync_time_sec \\t native_depth_frame_number \\t
    orig_ticks` per line -- confirmed against a real downloaded file (session
    z095-july-11-22-gladom_disassemble): this file has the SAME line count (3554) and the SAME
    final sync_time (118.433s) as that session's own Video/Pose_sync.txt and
    AhatDepth/Pose_sync.txt, confirming it's on the identical shared 30fps clock (so index qf =
    int(t*30), clip_frame_range's scheme, indexes this file directly, unlike
    AhatDepth_synced.txt -- see ahat_depth_filename). AHAT's native capture rate is lower than
    30fps (this session's frame_number only advanced ~1185 times over 3554 rows), so the same
    frame_number repeats across consecutive rows -- nearest-neighbor depth reuse, not a bug.
    Returns (sync_time_sec (T,), frame_number (T,) int, orig_ticks (T,))."""
    sync_time, frame_number, orig_ticks = [], [], []
    with open(path) as f:
        for line in f:
            fields = line.split()
            sync_time.append(float(fields[0]))
            frame_number.append(int(fields[1]))
            orig_ticks.append(float(fields[2]))
    return np.array(sync_time, dtype=np.float64), np.array(frame_number, dtype=np.int64), np.array(orig_ticks, dtype=np.float64)


def ahat_depth_filename(frame_number: int) -> str:
    """AhatDepth/{frame_number:06d}.png -- confirmed against real downloaded filenames
    (000033.png, 000038.png, ...) matching Timing_sync.txt's own frame_number column."""
    return f"{frame_number:06d}.png"


def unproject_depth(depth_mm: np.ndarray, intrinsics: np.ndarray, cam_pose_world: np.ndarray) -> np.ndarray:
    """depth_mm: (H,W) raw AHAT millimeters (0 = invalid). intrinsics: (3,3), opencv pinhole.
    cam_pose_world: (4,4) mathnet-basis local-to-world (same convention as parse_pose_file).
    Returns (H,W,3) world-space points (mathnet basis), NaN where depth==0. Algebraic inverse
    of project_to_pixels: that function computes local_opencv = (B @ R^T) @ p_world -
    (B @ R^T) @ t; solving for p_world with B and R^T both orthogonal gives
    p_world = R @ B^T @ local_opencv + t."""
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
    """Lifts query points sampled in the RGB (Video) camera's pixel space to 3D world
    coordinates, using AHAT depth even though the two cameras have different resolution,
    intrinsics, and pose. AHAT (512x512) and Video (896x504) are physically separate sensors,
    so an RGB pixel has no direct AHAT-pixel correspondence -- unproject the full depth map to
    world points (mathnet basis), then reproject those into the RGB camera via the same
    project_to_pixels used everywhere else, and nearest-neighbor match against the query
    pixels. Returns (N,3) world points (mathnet basis, NaN rows where no depth point reprojects
    within max_pixel_dist of that query pixel)."""
    points_world = unproject_depth(depth_mm, depth_intrinsics, depth_cam_pose)
    valid = ~np.isnan(points_world[..., 0])
    pts_world = points_world[valid]
    if len(pts_world) == 0:
        return np.full((len(query_uv_rgb), 3), np.nan)

    pts_uv_rgb = project_to_pixels(pts_world, rgb_cam_pose, rgb_intrinsics)

    # brute-force nearest neighbor in RGB pixel space, chunked over query points to bound
    # peak memory (N query x M depth-points distance matrix, M can be up to ~512*512).
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
    """Dense per-frame version of lift_rgb_query_points: reprojects a full AHAT depth frame
    into the RGB camera's own pixel grid, giving a (rgb_height, rgb_width) metric depth map in
    the RGB camera's frame -- the format TAPIP3D's `depths` input expects, letting the full
    clip skip its internal MegaSaM depth-estimation fallback entirely (which has its own,
    separate bug in TAPIP3D's vendored code) in favor of real sensor depth.
    Z-buffer scatter (nearest depth wins per output pixel, `np.minimum.at` -- a forward splat,
    not interpolation) rather than a per-pixel nearest-neighbor search, since building a dense
    map this way needs to place ~262144 AHAT points, not just a few dozen query points.
    Returns 0 (not NaN) at pixels with no reprojected coverage, matching depth_mm's own
    invalid-pixel convention, since TAPIP3D's own depth filtering expects a plain float array,
    not NaNs."""
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
    """joints/joints_valid/joints_tracked: one frame's worth (26,4,4)/(26,)/(26,), from
    parse_hand_file. cam_pose_world/intrinsics: the Video camera's pose+intrinsics at the SAME
    frame (caller's responsibility to align -- hand and video streams are separately "_sync"'d
    and this module does not resolve cross-stream time alignment, see the leading-fields
    caveat in parse_pose_file/parse_hand_file's docstrings). Returns the palm joint (default)
    projected into pixel coords, or None if
    that joint wasn't tracked this frame OR its projection falls outside the RGB frame.

    The out-of-frame case is real and expected, not a fallback for a bug: HoloLens's hand
    tracking has a wider field of view than the RGB video camera (confirmed empirically against
    a real downloaded session -- ~30% of frames with an otherwise-tracked palm joint project
    outside the RGB frame's bounds, physically consistent with two different-FOV sensors, not
    a near-0% in-bounds rate a genuinely broken transform would produce). Without this check
    the caller would try to seed SAM2 with a pixel coordinate outside the actual image."""
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
    """All of one hand's tracked, in-frame joints (up to 26) projected to pixel coords -- meant
    as SAM2 negative prompts (see segment.Sam2ObjectSegmenter.object_mask), not just the single
    palm joint hand_contact_point returns. A single positive click at the palm joint alone
    reliably makes SAM2 select the hand/glove itself as the "largest coherent region touching
    this point" rather than the smaller grasped object (confirmed via mask-overlay
    spot-checks on HoloAssist data: most sampled clips had the mask covering the glove, not
    the manipulated object). Marking the hand's own extent as explicitly negative yields the
    largest coherent non-hand mask, which a single positive click doesn't. Returns (M,2),
    M <= 26 (0 if no joints qualify)."""
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
    """Picks whichever hand has its contact joint tracked AND in-frame this frame, preferring
    the hand with more total tracked joints when both qualify (the more reliably-tracked hand
    overall, a reasonable proxy for which hand is actually manipulating the object this frame --
    HoloAssist doesn't label which hand is "active"). Returns (u,v) or None if neither hand
    qualifies."""
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
    """{label_root}/{subset}_0724.txt: one session name per line -- Ember-HoloAssist/
    holoassist-release's own build_holoassist_data. subset in {"train", "val", "test"}."""
    path = os.path.join(label_root, f"{subset}_0724.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _find_label_files(label_root: str) -> tuple:
    """labels_<date>_<id>_classes.json / _label2idx.json -- multiple dated versions can exist
    (e.g. labels_20230724_2221_* per raw_loader_index.py's own default vs.
    labels_20240723_2221_* per the release repo's hand_forecast.yaml config); use the
    lexicographically latest (most recent date prefix) rather than hardcoding one."""
    classes_files = sorted(glob.glob(os.path.join(label_root, "labels_*_classes.json")))
    if not classes_files:
        raise FileNotFoundError(f"no labels_*_classes.json found under {label_root}")
    classes_file = classes_files[-1]
    label2idx_file = classes_file.replace("_classes.json", "_label2idx.json")
    return classes_file, label2idx_file


def list_annotated_clips(label_root: str, subset: str, event_key: str = "fine_grained_action") -> list:
    """Shot-selection. "fine_grained_action"/"coarse_grained_action"
    are HoloAssist's own hand-object-interaction event categories (as opposed to "Narration"/
    "Conversation", which aren't actions), so filtering to one of these event keys already
    gives single-shot, hand-object-visible clips with no extra filtering needed -- mirrors
    Ember-HoloAssist/holoassist-release's own MultiRawDataset._collect_clips, including its
    same drop conditions (t_end < t_start, or a clip shorter than one frame at 30fps).
    Returns a list of (video_name, t_start_sec, t_end_sec, task_id, label_id) tuples."""
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
