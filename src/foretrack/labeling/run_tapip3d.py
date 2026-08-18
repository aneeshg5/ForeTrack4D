# Subprocess wrapper around zbw001/TAPIP3D's inference.py. See NOTICE.md.

import os
import subprocess
from pathlib import Path

import numpy as np


def _latest_result_npz(output_dir: Path) -> Path:
    candidates = sorted(Path(output_dir).glob("*/*.result.npz"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"no *.result.npz produced under {output_dir}")
    return candidates[-1]


def _invoke(
    input_npz: str, output_dir: str, tapip3d_python: str, tapip3d_repo: str, checkpoint: str,
    reuse_result: bool = False,
) -> Path:
    if reuse_result:
        try:
            existing = _latest_result_npz(output_dir)
            if existing is not None:
                return existing
        except (FileNotFoundError, IndexError, ValueError):
            pass
    env = os.environ.copy()
    tapip3d_bin_dir = os.path.dirname(os.path.abspath(tapip3d_python))
    env["PATH"] = f"{tapip3d_bin_dir}:{env.get('PATH', '')}"
    subprocess.run(
        [
            tapip3d_python, "inference.py",
            "--input_path", str(input_npz),
            "--output_dir", str(output_dir),
            "--checkpoint", str(checkpoint),
        ],
        check=True,
        cwd=tapip3d_repo,
        env=env,
    )
    return _latest_result_npz(output_dir)


def lift_query_points(query_uv: np.ndarray, depth_t0: np.ndarray, intrinsics_t0: np.ndarray) -> np.ndarray:
    u, v = query_uv[:, 0], query_uv[:, 1]
    z = depth_t0[np.round(v).astype(int), np.round(u).astype(int)]
    fx, fy, cx, cy = intrinsics_t0[0, 0], intrinsics_t0[1, 1], intrinsics_t0[0, 2], intrinsics_t0[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([np.zeros_like(z), x, y, z], axis=-1).astype(np.float32)


def _write_result_npz(result_path: Path, intrinsics_t0: np.ndarray, out_npz: str) -> None:
    result = np.load(result_path)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        tracks=result["coords"],
        visibility=result["visibs"],
        query_points=result["query_points"],
        intrinsics=intrinsics_t0,
    )


def run(
    video: np.ndarray,
    intrinsics: np.ndarray,
    query_uv: np.ndarray,
    out_npz: str,
    tapip3d_python: str,
    tapip3d_repo: str,
    checkpoint: str,
    work_dir: str,
) -> None:
    if os.path.exists(out_npz):
        return

    t = video.shape[0]
    intrinsics_seq = np.broadcast_to(intrinsics, (t, 3, 3)).astype(np.float32)

    depth_pass_dir = Path(work_dir) / "depth_pass"
    depth_pass_dir.mkdir(parents=True, exist_ok=True)
    depth_input_npz = depth_pass_dir / "input.npz"
    np.savez(depth_input_npz, video=video, intrinsics=intrinsics_seq)
    depth_result = np.load(_invoke(depth_input_npz, depth_pass_dir, tapip3d_python, tapip3d_repo, checkpoint, reuse_result=True))

    depth_h, depth_w = depth_result["depths"][0].shape
    orig_h, orig_w = video.shape[1:3]
    query_uv_at_depth_res = query_uv * np.array([depth_w / orig_w, depth_h / orig_h], dtype=np.float32)
    query_point = lift_query_points(query_uv_at_depth_res, depth_result["depths"][0], depth_result["intrinsics"][0])

    final_pass_dir = Path(work_dir) / "final_pass"
    final_pass_dir.mkdir(parents=True, exist_ok=True)
    final_input_npz = final_pass_dir / "input.npz"
    np.savez(
        final_input_npz,
        video=video,
        intrinsics=intrinsics_seq,
        depths=depth_result["depths"],
        query_point=query_point,
    )
    final_result_path = _invoke(final_input_npz, final_pass_dir, tapip3d_python, tapip3d_repo, checkpoint)
    _write_result_npz(final_result_path, intrinsics_seq[0], out_npz)


def run_with_query_points(
    video: np.ndarray,
    intrinsics: np.ndarray,
    query_point: np.ndarray,
    out_npz: str,
    tapip3d_python: str,
    tapip3d_repo: str,
    checkpoint: str,
    work_dir: str,
    depths: np.ndarray = None,
) -> None:
    if os.path.exists(out_npz):
        return

    t = video.shape[0]
    intrinsics_seq = np.broadcast_to(intrinsics, (t, 3, 3)).astype(np.float32)

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    input_npz = Path(work_dir) / "input.npz"
    save_kwargs = dict(video=video, intrinsics=intrinsics_seq, query_point=query_point.astype(np.float32))
    if depths is not None:
        save_kwargs["depths"] = depths.astype(np.float32)
    np.savez(input_npz, **save_kwargs)
    result_path = _invoke(input_npz, work_dir, tapip3d_python, tapip3d_repo, checkpoint)
    _write_result_npz(result_path, intrinsics_seq[0], out_npz)
