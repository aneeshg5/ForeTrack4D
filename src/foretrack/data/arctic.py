# Articulation math follows forehand4d's common/object_tensors.py. See NOTICE.md.

import glob
import json
from pathlib import Path

import numpy as np
import trimesh

from .dexycb import TrackFramesDataset
from .gt_tracks import render_depth_from_posed_mesh, render_visibility, sample_query_points

EGO_IMAGE_SCALE = 0.3

ARCTIC_OBJECTS = [
    "capsulemachine", "box", "ketchup", "laptop", "microwave", "mixer",
    "notebook", "espressomachine", "waffleiron", "scissors", "phone",
]

ARTICULATION_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / theta
    k = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)).astype(np.float32)


def load_object_template(meta_dir: Path, obj_name: str, n: int, seed: int = 0):
    obj_dir = meta_dir / "object_vtemplates" / obj_name
    top = trimesh.load(obj_dir / "top.obj", force="mesh")
    bottom = trimesh.load(obj_dir / "bottom.obj", force="mesh")

    n_top = max(1, round(n * top.area / (top.area + bottom.area)))
    n_bottom = n - n_top

    pts_top = sample_query_points(top, n_top, seed=seed)
    pts_bottom = sample_query_points(bottom, n_bottom, seed=seed + 1)

    points = np.concatenate([pts_top, pts_bottom], axis=0).astype(np.float32)
    is_top = np.concatenate([np.ones(n_top, dtype=bool), np.zeros(n_bottom, dtype=bool)])
    return points, is_top, top, bottom


def articulate_and_pose(points_obj: np.ndarray, is_top: np.ndarray, pose_7d: np.ndarray) -> np.ndarray:
    angle, global_aa, transl = pose_7d[0], pose_7d[1:4], pose_7d[4:7]

    points = points_obj.copy()
    if angle != 0:
        r_local = axis_angle_to_rotmat(ARTICULATION_AXIS * angle)
        points[is_top] = points[is_top] @ r_local.T

    r_global = axis_angle_to_rotmat(global_aa)
    return points @ r_global.T + transl


def world_to_camera(points_world_mm: np.ndarray, r_cam: np.ndarray, t_cam: np.ndarray) -> np.ndarray:
    points_world_m = points_world_mm / 1000.0
    return points_world_m @ r_cam.T + t_cam.reshape(1, 3)


def build_track_npz(seq_object_npy: str, out_path: str, arctic_root: str, n: int = 64, vis_tol: float = 0.005) -> None:
    seq_object_npy = Path(seq_object_npy)
    arctic_root = Path(arctic_root)
    meta_dir = arctic_root / "meta"

    seq_name = seq_object_npy.stem.replace(".object", "")
    subject = seq_object_npy.parent.name
    obj_name = seq_name.split("_")[0]
    if obj_name not in ARCTIC_OBJECTS:
        raise ValueError(f"unrecognized ARCTIC object '{obj_name}' parsed from '{seq_name}'")

    query_xyz_obj, is_top, top_mesh, bottom_mesh = load_object_template(meta_dir, obj_name, n)

    with open(meta_dir / "misc.json") as f:
        misc = json.load(f)
    ioi_offset = misc[subject]["ioi_offset"]

    pose_seq = np.load(seq_object_npy).astype(np.float32)  # (T, 7)
    cam_data = np.load(
        seq_object_npy.parent / f"{seq_name}.egocam.dist.npy", allow_pickle=True
    ).item()
    r_cam_all = np.asarray(cam_data["R_k_cam_np"], dtype=np.float32)  # (T, 3, 3)
    t_cam_all = np.asarray(cam_data["T_k_cam_np"], dtype=np.float32)  # (T, 3, 1)
    intrinsics = np.asarray(cam_data["intrinsics"], dtype=np.float32)  # (3, 3)

    num_frames = pose_seq.shape[0]
    r_cam_t0, t_cam_t0 = r_cam_all[0], t_cam_all[0]

    tracks = np.zeros((num_frames, n, 3), dtype=np.float32)
    visibility = np.zeros((num_frames, n), dtype=bool)

    for t in range(num_frames):
        points_world_mm = articulate_and_pose(query_xyz_obj, is_top, pose_seq[t])
        tracks[t] = world_to_camera(points_world_mm, r_cam_t0, t_cam_t0)

        top_verts_world = articulate_and_pose(
            top_mesh.vertices.astype(np.float32), np.ones(len(top_mesh.vertices), dtype=bool), pose_seq[t]
        )
        bottom_verts_world = articulate_and_pose(
            bottom_mesh.vertices.astype(np.float32), np.zeros(len(bottom_mesh.vertices), dtype=bool), pose_seq[t]
        )
        top_verts_cam = world_to_camera(top_verts_world, r_cam_t0, t_cam_t0)
        bottom_verts_cam = world_to_camera(bottom_verts_world, r_cam_t0, t_cam_t0)
        posed_mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=top_verts_cam, faces=top_mesh.faces, process=False),
            trimesh.Trimesh(vertices=bottom_verts_cam, faces=bottom_mesh.faces, process=False),
        ])
        depth = render_depth_from_posed_mesh(posed_mesh, intrinsics, width=2800, height=2000)
        visibility[t] = render_visibility(tracks[t], intrinsics, depth, tol=vis_tol)

    intrinsics_cropped = (intrinsics * EGO_IMAGE_SCALE).astype(np.float32)
    intrinsics_cropped[2, 2] = 1.0  # the homogeneous row's [0,0,1] must stay unscaled
    image_paths = [
        str(arctic_root / "cropped_images" / subject / seq_name / "0" / f"{t + ioi_offset:05d}.jpg")
        for t in range(num_frames)
    ]

    np.savez(
        out_path,
        tracks=tracks,
        visibility=visibility,
        intrinsics=intrinsics_cropped,
        image_paths=np.array(image_paths),
        query_frame_idx=0,
        query_xyz_t0=tracks[0],
        object_id=ARCTIC_OBJECTS.index(obj_name),
        object_name=obj_name,
        subject=subject,
        seq_name=seq_name,
    )


class ArcticTracks(TrackFramesDataset):

    def __init__(self, root: str, split: str, **kwargs):
        files = sorted(glob.glob(f"{root}/{split}/**/*.npz", recursive=True))
        if len(files) == 0:
            raise ValueError(f"no npz files found under {root}/{split}")
        super().__init__(files, **kwargs)
