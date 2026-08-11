# GT track generation for ARCTIC's articulated objects -- our own schema (not adapted from
# forehand4d's dataloaders, per NOTICE.md's provenance convention), but the articulation math
# itself is grounded directly in forehand4d's common/object_tensors.py (ObjectTensors class).

import glob
import json
from pathlib import Path

import numpy as np
import trimesh

from .dexycb import TrackFramesDataset
from .gt_tracks import render_depth_from_posed_mesh, render_visibility, sample_query_points

# ARCTIC's own scripts_data/crop_images.py: for the egocentric view (view_idx == 0) specifically,
# "cropped_images" aren't actually cropped at all -- just the full frame uniformly resized by
# this fixed factor (no bbox, no offset). Verified directly: 2800x2000 * 0.3 == 840x600, exactly
# the resolution found in the real downloaded cropped_images/. Non-ego views DO use a real
# bbox-centered crop (a different code path we don't need, since this project only uses ego).
EGO_IMAGE_SCALE = 0.3

# ARCTIC's 11 articulated object categories (forehand4d's OBJECTS list, common/object_tensors.py)
ARCTIC_OBJECTS = [
    "capsulemachine", "box", "ketchup", "laptop", "microwave", "mixer",
    "notebook", "espressomachine", "waffleiron", "scissors", "phone",
]

# every ARCTIC object template is canonicalized so its articulation hinge is always the object's
# own local Z axis -- confirmed directly from forehand4d's construct_obj_tensors
# (obj_tensors["z_axis"] = [0, 0, -1], a single constant shared across all 11 objects, not
# something we chose or that varies per object). Negative Z, not positive; getting the sign
# wrong would silently flip every articulation direction.
ARTICULATION_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(3,) axis-angle -> (3,3) rotation matrix via Rodrigues' formula."""
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
    """FPS-sample n query points across an object's top + bottom parts TOGETHER (one combined
    point cloud), so each part's share of points reflects its share of surface area, rather than
    a fixed 50/50 split that would over-sample a small hinge relative to a large body. Returns
    (points, is_top, top_mesh, bottom_mesh): points in mm, the object's own canonical template
    frame (matching the raw released mesh scale -- converted to meters only once, alongside the
    object pose's own mm -> m conversion in build_track_npz, so the two never drift out of
    lockstep)."""
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
    """points_obj: (N,3) canonical template points, mm, object-local frame. is_top: (N,) bool.
    pose_7d: (7,) [angle, global_orient_aa(3), transl(3)] for one frame -- ARCTIC's own released
    format, verified directly against a real raw_seqs/*.object.npy file's shape and value ranges
    (angle near 0, orient/transl in plausible axis-angle/mm ranges) before trusting it, matching
    the project's established discipline of checking raw data rather than assuming a schema.

    Top-part points rotate around ARTICULATION_AXIS by `angle` radians FIRST; bottom-part points
    don't. Then ONE global rotation + translation applies to everything (top-after-articulation
    AND bottom together) -- this order matters, matches ObjectTensors.forward_7d_batch exactly.
    Output: mm, ARCTIC's own world (mocap-derived, camera-independent) frame -- NOT camera frame
    yet, see world_to_camera below."""
    angle, global_aa, transl = pose_7d[0], pose_7d[1:4], pose_7d[4:7]

    points = points_obj.copy()
    if angle != 0:
        r_local = axis_angle_to_rotmat(ARTICULATION_AXIS * angle)
        points[is_top] = points[is_top] @ r_local.T

    r_global = axis_angle_to_rotmat(global_aa)
    return points @ r_global.T + transl


def world_to_camera(points_world_mm: np.ndarray, r_cam: np.ndarray, t_cam: np.ndarray) -> np.ndarray:
    """points_world_mm: (N,3), ARCTIC's world frame, mm. r_cam: (3,3), t_cam: (3,) or (3,1) --
    one frame's world->camera extrinsics (`R_k_cam_np`/`T_k_cam_np` from *.egocam.dist.npy),
    which are in METERS (unlike the object pose's mm scale -- verified from real data: T_k_cam_np
    values are ~0.7-1.5, consistent with meter-scale camera-to-object distances, vs. the object
    pose's translation values in the hundreds/thousands, consistent with mm). Converts to meters
    before applying, matching this project's meters convention throughout."""
    points_world_m = points_world_mm / 1000.0
    return points_world_m @ r_cam.T + t_cam.reshape(1, 3)


def build_track_npz(seq_object_npy: str, out_path: str, arctic_root: str, n: int = 64, vis_tol: float = 0.005) -> None:
    """seq_object_npy: path to a raw_seqs/<subject>/<seq_name>.object.npy file, e.g.
    <ARCTIC_ROOT>/raw_seqs/s01/box_grab_01.object.npy. arctic_root:
    <...>/unpack/arctic_data/data.

    Unlike DexYCB (static camera, camera frame == world frame, decision 1's simple case),
    ARCTIC's egocam is head-mounted -- decision 1 still applies ("world coordinates defined as
    the camera frame at t=0"), but here that means every frame's points get mapped into the
    QUERY frame's (t=0's) camera extrinsics specifically, not each frame's own (moving) camera.
    ARCTIC's own world frame (mocap-derived) is camera-independent, so this is a single
    world_to_camera call per frame using t=0's R_cam/T_cam, not a two-step undo-then-reapply."""
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
    # per-subject offset between raw_seqs' 0-indexed pose array and cropped_images' 1-indexed
    # filenames (s01: ioi_offset=1, meaning 00001.jpg is frame 0); varies per subject.
    ioi_offset = misc[subject]["ioi_offset"]

    pose_seq = np.load(seq_object_npy).astype(np.float32)  # (T, 7)
    cam_data = np.load(
        seq_object_npy.parent / f"{seq_name}.egocam.dist.npy", allow_pickle=True
    ).item()
    r_cam_all = np.asarray(cam_data["R_k_cam_np"], dtype=np.float32)  # (T, 3, 3)
    t_cam_all = np.asarray(cam_data["T_k_cam_np"], dtype=np.float32)  # (T, 3, 1)
    intrinsics = np.asarray(cam_data["intrinsics"], dtype=np.float32)  # (3, 3)
    # NOTE: ARCTIC's egocam has real lens distortion (dist8, 8 coefficients) which this does not
    # yet correct for -- 3D tracks themselves are unaffected (distortion only matters for 2D
    # pixel projection), but query_uv / visibility rendering below uses raw pinhole intrinsics
    # as an approximation. Flagging explicitly rather than silently ignoring; revisit if
    # visibility/conditioning quality looks off in the visual sanity check.

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

    # image_paths / saved intrinsics must match the ACTUAL image resolution downstream code will
    # load (cropped_images/, which for the ego view is just a uniform EGO_IMAGE_SCALE resize --
    # see the module docstring). depth rendering above correctly used the full-res intrinsics;
    # this is a separate, later scaling for the npz's saved copy, matching TrackFramesDataset's
    # expectation (DexYCBTracks stores intrinsics matching image_paths' actual pixel resolution).
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
    """reads build_track_npz's output -- crop/augmentation/padding logic is shared with
    DexYCBTracks via TrackFramesDataset (data/dexycb.py), since it's dataset-agnostic once the
    npz schema matches (verified: build_track_npz above now saves image_paths and
    intrinsics scaled to match, exactly mirroring DexYCB's schema)."""

    def __init__(self, root: str, split: str, **kwargs):
        files = sorted(glob.glob(f"{root}/{split}/**/*.npz", recursive=True))
        if len(files) == 0:
            raise ValueError(f"no npz files found under {root}/{split}")
        super().__init__(files, **kwargs)
