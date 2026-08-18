from pathlib import Path

import cv2
import numpy as np

from ..data.holoassist import (
    HOLOASSIST_FPS,
    ahat_depth_filename,
    best_hand_contact_point,
    hand_landmark_pixels,
    lift_rgb_query_points,
    parse_ahat_depth_timing,
    parse_hand_file,
    parse_intrinsics_file,
    parse_pose_file,
    reproject_depth_to_rgb,
    scale_intrinsics,
    world_to_camera_frame,
)
from .filter import accept_clip
from .run_tapip3d import run_with_query_points
from .segment import Sam2ObjectSegmenter, sample_query_points_in_mask, save_mask_overlay


def clip_frame_range(t_start: float, t_end: float) -> tuple:
    return int(t_start * HOLOASSIST_FPS), int(t_end * HOLOASSIST_FPS)


def clip_output_path(out_dir: str, split: str, video_name: str, query_frame: int, end_frame: int) -> Path:
    return Path(out_dir) / split / f"{video_name}_{query_frame:06d}_{end_frame:06d}.npz"


def read_video_frames(video_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)


def find_nearby_depth(
    session_dir: Path, ahat_frame_number: np.ndarray, query_frame: int, search_radius: int = 15
) -> tuple:
    for offset in range(search_radius + 1):
        for frame in ({query_frame + offset, query_frame - offset} if offset else {query_frame}):
            if not (0 <= frame < len(ahat_frame_number)):
                continue
            depth_path = session_dir / "AhatDepth" / ahat_depth_filename(int(ahat_frame_number[frame]))
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_mm is not None:
                return frame, depth_mm
    return None, None


def build_clip_depths(
    session_dir: Path,
    query_frame: int,
    end_frame: int,
    ahat_frame_number: np.ndarray,
    ahat_cam_poses: np.ndarray,
    ahat_intrinsics: np.ndarray,
    rgb_intrinsics: np.ndarray,
    cam_poses: np.ndarray,
    rgb_height: int,
    rgb_width: int,
) -> np.ndarray:
    depths = []
    for frame in range(query_frame, end_frame):
        depth_path = session_dir / "AhatDepth" / ahat_depth_filename(int(ahat_frame_number[frame]))
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            depths.append(np.zeros((rgb_height, rgb_width), dtype=np.float32))
            continue
        depths.append(
            reproject_depth_to_rgb(
                depth_mm, ahat_intrinsics, ahat_cam_poses[frame],
                rgb_intrinsics, cam_poses[frame], rgb_height, rgb_width,
            )
        )
    return np.stack(depths)


def process_clip(
    video_name: str, t_start: float, t_end: float, cfg: dict, segmenter: Sam2ObjectSegmenter, out_path: Path
) -> bool:
    session_dir = Path(cfg["holoassist_root"]) / video_name / "Export_py"
    _, _, cam_poses = parse_pose_file(session_dir / "Video" / "Pose_sync.txt")
    rgb_intrinsics_declared, rgb_size_declared = parse_intrinsics_file(session_dir / "Video" / "Intrinsics.txt")
    _, left_joints, left_valid, left_tracked = parse_hand_file(session_dir / "Hands" / "Left_sync.txt")
    _, right_joints, right_valid, right_tracked = parse_hand_file(session_dir / "Hands" / "Right_sync.txt")
    _, ahat_frame_number, _ = parse_ahat_depth_timing(session_dir / "AhatDepth" / "Timing_sync.txt")
    _, _, ahat_cam_poses = parse_pose_file(session_dir / "AhatDepth" / "Pose_sync.txt")
    ahat_intrinsics, _ = parse_intrinsics_file(session_dir / "AhatDepth" / "Intrinsics.txt")

    query_frame, end_frame = clip_frame_range(t_start, t_end)
    stream_lens = [len(cam_poses), len(left_joints), len(right_joints), len(ahat_frame_number), len(ahat_cam_poses)]
    if query_frame >= min(stream_lens):
        return False

    frames = read_video_frames(session_dir / "Video_compress.mp4", query_frame, end_frame)
    if len(frames) == 0:
        return False
    rgb_height, rgb_width = frames.shape[1:3]
    rgb_intrinsics = scale_intrinsics(
        rgb_intrinsics_declared, tuple(rgb_size_declared), (rgb_width, rgb_height)
    )

    work_dir = Path(cfg["work_dir"]) / f"{video_name}_{query_frame:06d}_{end_frame:06d}"

    contact = best_hand_contact_point(
        left_joints[query_frame], left_valid[query_frame], left_tracked[query_frame],
        right_joints[query_frame], right_valid[query_frame], right_tracked[query_frame],
        cam_poses[query_frame], rgb_intrinsics, rgb_width, rgb_height,
    )
    if contact is None:
        return False

    negative_points = np.concatenate([
        hand_landmark_pixels(
            left_joints[query_frame], left_valid[query_frame], left_tracked[query_frame],
            cam_poses[query_frame], rgb_intrinsics, rgb_width, rgb_height,
        ),
        hand_landmark_pixels(
            right_joints[query_frame], right_valid[query_frame], right_tracked[query_frame],
            cam_poses[query_frame], rgb_intrinsics, rgb_width, rgb_height,
        ),
    ], axis=0)
    mask = segmenter.object_mask(frames[0], contact, negative_points=negative_points)
    if not mask.any():
        return False

    depth_frame, depth_mm = find_nearby_depth(session_dir, ahat_frame_number, query_frame)
    if depth_mm is None:
        return False

    n = cfg["n"]
    candidate_uv = sample_query_points_in_mask(mask, n=n * 3)
    candidate_xyz_mathnet = lift_rgb_query_points(
        candidate_uv, depth_mm, ahat_intrinsics, ahat_cam_poses[depth_frame], rgb_intrinsics, cam_poses[query_frame]
    )
    valid = ~np.isnan(candidate_xyz_mathnet[:, 0])
    if valid.sum() < n:
        return False
    query_uv = candidate_uv[valid][:n]
    query_xyz_mathnet = candidate_xyz_mathnet[valid][:n]

    work_dir.mkdir(parents=True, exist_ok=True)
    save_mask_overlay(frames[0], mask, contact, work_dir / "mask_overlay.jpg", query_uv=query_uv)

    query_xyz_cam0 = world_to_camera_frame(query_xyz_mathnet, cam_poses[query_frame])
    query_point = np.concatenate(
        [np.zeros((n, 1), dtype=np.float32), query_xyz_cam0.astype(np.float32)], axis=-1
    )

    actual_end_frame = query_frame + len(frames)
    clip_depths = build_clip_depths(
        session_dir, query_frame, actual_end_frame, ahat_frame_number, ahat_cam_poses, ahat_intrinsics,
        rgb_intrinsics, cam_poses, rgb_height, rgb_width,
    )

    tapip3d_out = work_dir / "tapip3d_result.npz"
    run_with_query_points(
        video=frames,
        intrinsics=rgb_intrinsics,
        query_point=query_point,
        out_npz=str(tapip3d_out),
        tapip3d_python=cfg["tapip3d_python"],
        tapip3d_repo=cfg["tapip3d_repo"],
        checkpoint=cfg["tapip3d_checkpoint"],
        work_dir=str(work_dir),
        depths=clip_depths,
    )

    result = np.load(tapip3d_out)
    tracks, visibility = result["tracks"], result["visibility"]
    if not accept_clip(tracks, visibility, cfg.get("filter", {})):
        return False

    frame_path = work_dir / "query_frame.jpg"
    cv2.imwrite(str(frame_path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        tracks=tracks,
        visibility=visibility,
        intrinsics=rgb_intrinsics,
        image_paths=np.array([str(frame_path)] * len(tracks)),
        query_frame_idx=0,
        query_xyz_t0=tracks[0],
        object_id=-1,
        object_name=video_name,
        source="pseudo",
    )
    return True
