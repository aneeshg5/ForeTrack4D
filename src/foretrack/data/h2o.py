import glob
from pathlib import Path

import numpy as np
import trimesh

from .dexycb import TrackFramesDataset
from .gt_tracks import render_depth_from_posed_mesh, render_visibility, sample_query_points

H2O_OBJECTS = {
    1: "book", 2: "espresso", 3: "lotion", 4: "spray",
    5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino",
}

H2O_MESH_FILENAMES = {"spray": "lotion_spray.obj"}


def load_intrinsics(cam4_dir: Path) -> tuple:
    fx, fy, cx, cy, width, height = (float(x) for x in (cam4_dir / "cam_intrinsics.txt").read_text().split())
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return intrinsics, int(width), int(height)


def load_frame_matrix(path: Path, has_class: bool) -> tuple:
    vals = path.read_text().split()
    cls = int(float(vals[0])) if has_class else None
    mat = np.array(vals[1:] if has_class else vals, dtype=np.float32).reshape(4, 4)
    return cls, mat


def build_track_npz(seq_dir: str, out_path: str, h2o_root: str, n: int = 64, vis_tol: float = 0.005) -> None:
    seq_dir = Path(seq_dir)
    h2o_root = Path(h2o_root)

    intrinsics, width, height = load_intrinsics(seq_dir)

    obj_pose_files = sorted((seq_dir / "obj_pose_rt").glob("*.txt"))
    num_frames = len(obj_pose_files)

    cls0, _ = load_frame_matrix(obj_pose_files[0], has_class=True)
    obj_name = H2O_OBJECTS[cls0]
    mesh_filename = H2O_MESH_FILENAMES.get(obj_name, f"{obj_name}.obj")
    mesh_path = h2o_root / "object" / "object" / obj_name / mesh_filename
    mesh = trimesh.load(mesh_path, force="mesh")

    query_xyz_obj = sample_query_points(mesh, n)

    tracks = np.zeros((num_frames, n, 3), dtype=np.float32)
    visibility = np.zeros((num_frames, n), dtype=bool)
    image_paths = []

    for t, obj_pose_file in enumerate(obj_pose_files):
        frame_id = obj_pose_file.stem
        cls, rt = load_frame_matrix(obj_pose_file, has_class=True)
        if cls != cls0:
            raise ValueError(f"{seq_dir}: object class changes mid-sequence ({cls0} -> {cls} at frame {frame_id})")
        _, cam_to_world = load_frame_matrix(seq_dir / "cam_pose" / f"{frame_id}.txt", has_class=False)

        r_obj, t_obj = rt[:3, :3], rt[:3, 3]
        r_cam, t_cam = cam_to_world[:3, :3], cam_to_world[:3, 3]
        r_comb = r_cam @ r_obj
        t_comb = r_cam @ t_obj + t_cam

        tracks[t] = query_xyz_obj @ r_comb.T + t_comb
        posed = trimesh.Trimesh(vertices=mesh.vertices @ r_comb.T + t_comb, faces=mesh.faces, process=False)
        depth = render_depth_from_posed_mesh(posed, intrinsics, width=width, height=height)
        visibility[t] = render_visibility(tracks[t], intrinsics, depth, tol=vis_tol)
        image_paths.append(str(seq_dir / "rgb" / f"{frame_id}.png"))

    np.savez(
        out_path,
        tracks=tracks,
        visibility=visibility,
        intrinsics=intrinsics,
        image_paths=np.array(image_paths),
        query_frame_idx=0,
        query_xyz_t0=tracks[0],
        object_id=cls0,
        object_name=obj_name,
    )


class H2OTracks(TrackFramesDataset):
    def __init__(self, root: str, split: str, **kwargs):
        files = sorted(glob.glob(f"{root}/{split}/**/*.npz", recursive=True))
        if len(files) == 0:
            raise ValueError(f"no npz files found under {root}/{split}")
        super().__init__(files, **kwargs)
