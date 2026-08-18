from pathlib import Path

import numpy as np
import pyrender
import trimesh
import yaml

from .transforms import opencv_camera_pose_for_pyrender

YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


def sample_query_points(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    dense, _ = trimesh.sample.sample_surface(mesh, 20000, seed=seed)
    dense = np.asarray(dense, dtype=np.float32)

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(dense)))]
    dists = np.linalg.norm(dense - dense[selected[0]], axis=1)
    for _ in range(n - 1):
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)
        dists = np.minimum(dists, np.linalg.norm(dense - dense[next_idx], axis=1))
    return dense[selected]


def transform_to_camera_frame(points: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
    R, t = obj_pose[:, :3], obj_pose[:, 3]
    return points @ R.T + t


def render_depth_from_posed_mesh(
    posed_mesh: trimesh.Trimesh, intrinsics: np.ndarray, width: int, height: int
) -> np.ndarray:
    material = pyrender.MetallicRoughnessMaterial(doubleSided=True)
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(posed_mesh, material=material))
    cam = pyrender.IntrinsicsCamera(
        fx=intrinsics[0, 0], fy=intrinsics[1, 1],
        cx=intrinsics[0, 2], cy=intrinsics[1, 2],
        znear=0.01, zfar=5.0,
    )
    scene.add(cam, pose=opencv_camera_pose_for_pyrender())
    r = pyrender.OffscreenRenderer(width, height)
    try:
        _, depth = r.render(scene)
    finally:
        r.delete()
    return depth


def render_object_depth(
    mesh: trimesh.Trimesh, obj_pose: np.ndarray, intrinsics: np.ndarray, width: int, height: int
) -> np.ndarray:
    verts_cam = mesh.vertices @ obj_pose[:, :3].T + obj_pose[:, 3]
    posed = trimesh.Trimesh(vertices=verts_cam, faces=mesh.faces, process=False)
    return render_depth_from_posed_mesh(posed, intrinsics, width, height)


def render_visibility(
    points_cam: np.ndarray, intrinsics: np.ndarray, depth: np.ndarray, tol: float = 0.005
) -> np.ndarray:
    height, width = depth.shape
    z = points_cam[:, 2]
    u = np.round(points_cam[:, 0] / z * intrinsics[0, 0] + intrinsics[0, 2]).astype(int)
    v = np.round(points_cam[:, 1] / z * intrinsics[1, 1] + intrinsics[1, 2]).astype(int)

    in_bounds = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    visible = np.zeros(len(points_cam), dtype=bool)
    rendered_z = np.full(len(points_cam), np.nan, dtype=np.float32)
    rendered_z[in_bounds] = depth[v[in_bounds], u[in_bounds]]
    has_geom = in_bounds & (rendered_z > 0)
    visible[has_geom] = np.abs(z[has_geom] - rendered_z[has_geom]) < tol
    return visible


class _DexYCBLoader(yaml.SafeLoader):
    pass


_DexYCBLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)


def load_intrinsics(dexycb_root: Path, serial: str) -> np.ndarray:
    with open(dexycb_root / "calibration" / "intrinsics" / f"{serial}_640x480.yml") as f:
        c = yaml.load(f, Loader=_DexYCBLoader)["color"]
    return np.array(
        [[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], dtype=np.float32
    )


def build_track_npz(seq_dir: str, out_path: str, n: int = 64, vis_tol: float = 0.005) -> None:
    seq_dir = Path(seq_dir)
    serial = seq_dir.name
    subject_seq_dir = seq_dir.parent
    dexycb_root = subject_seq_dir.parent.parent

    with open(subject_seq_dir / "meta.yml") as f:
        meta = yaml.safe_load(f)

    ycb_id = meta["ycb_ids"][meta["ycb_grasp_ind"]]
    mesh_path = dexycb_root / "models" / YCB_CLASSES[ycb_id] / "textured_simple.obj"
    mesh = trimesh.load(mesh_path, force="mesh")

    query_xyz_obj = sample_query_points(mesh, n)
    intrinsics = load_intrinsics(dexycb_root, serial)

    num_frames = meta["num_frames"]
    tracks = np.zeros((num_frames, n, 3), dtype=np.float32)
    visibility = np.zeros((num_frames, n), dtype=bool)
    image_paths = []

    for t in range(num_frames):
        label = np.load(seq_dir / f"labels_{t:06d}.npz")
        obj_pose = label["pose_y"][meta["ycb_grasp_ind"]]  # (3, 4)

        tracks[t] = transform_to_camera_frame(query_xyz_obj, obj_pose)
        depth = render_object_depth(mesh, obj_pose, intrinsics, width=640, height=480)
        visibility[t] = render_visibility(tracks[t], intrinsics, depth, tol=vis_tol)
        image_paths.append(str(seq_dir / f"color_{t:06d}.jpg"))

    np.savez(
        out_path,
        tracks=tracks,
        visibility=visibility,
        intrinsics=intrinsics,
        image_paths=np.array(image_paths),
        query_frame_idx=0,
        query_xyz_t0=tracks[0],
        object_id=ycb_id,
        object_name=YCB_CLASSES[ycb_id],
    )
