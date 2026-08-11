# subprocess wrapper: calls TAPIP3D's inference.py from the separate tapip3d conda env,
# exchanging data through npz files on disk (see README). CLI args and npz schema follow
# zbw001/TAPIP3D's inference.py.
#
# inference.py has no --output (exact path) flag: it writes
# {output_dir}/{timestamp}/{input_stem}.result.npz. It also has no world-mode flag and no way
# to pass pre-lifted 3D query points without already having depth: omitting `extrinsics` from
# the input npz gives per-frame identity extrinsics, and omitting `depths` makes it run
# MegaSaM internally (prepare_inputs). But its
# own query_point format is (N,4) = [query_time, x_world, y_world, z_world] -- already-lifted
# 3D points, not pixel coords -- so there's no way to hand it our own 2D query points on a
# single call. This wrapper therefore runs inference.py twice: once with no query_point (auto
# grid, MegaSaM-computed depth) purely to obtain per-frame depth, then lifts our own 2D query
# points using that depth and re-runs with those as the explicit query_point for the run whose
# tracks we keep.

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
    input_npz: str, output_dir: str, tapip3d_python: str, tapip3d_repo: str, checkpoint: str
) -> Path:
    # Several of TAPIP3D's own vendored MegaSaM subprocess stages spawn further NESTED
    # subprocesses as the bare string "python" (relying on PATH, not sys.executable) --
    # that nested lookup only succeeds if tapip3d_python's own directory is on PATH, true if
    # the caller activated the tapip3d venv first, but our real caller (impute_labels.py) runs
    # from an entirely different venv (venv_labeling), so PATH can't be relied on. Prepend it
    # explicitly rather than assume the calling process's environment happens to be right.
    #
    # `Path(tapip3d_python).resolve()` FOLLOWS SYMLINKS -- venv_tapip3d/bin/python is itself a
    # symlink to a shared base interpreter (e.g. a uv-managed python install), so `.resolve()`
    # silently computes that shared install's bin/ directory, NOT venv_tapip3d/bin, which is
    # where the venv's actually-installed packages (cv2, numpy, kornia, etc.) are wired up via
    # site-packages. A nested "python" resolved via the wrong (resolved-through-symlink) PATH
    # entry finds an interpreter with none of those packages -- this was the real, root cause
    # of a `ModuleNotFoundError` that persisted across three separate rounds of otherwise-correct
    # fixes (a PYTHONPATH type bug and a bare-"python"-vs-sys.executable bug elsewhere in this
    # same call chain, both real, neither sufficient) until finally instrumented and confirmed
    # directly. Use os.path.abspath, which normalizes without following
    # symlinks, so PATH points at the venv's own bin/ directory as intended.
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
    """query_uv: (N,2) pixel coords at t=0. depth_t0: (H,W) metric depth. intrinsics_t0: (3,3).
    Returns (N,4) = [query_time=0, x, y, z] in TAPIP3D's expected query_point format (world
    frame == camera(0) frame under the identity-extrinsics world-mode convention, so camera(0)
    coords are already world coords -- decision 1)."""
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
    """video: (T,H,W,3) uint8. intrinsics: (3,3), assumed constant over the clip (broadcast to
    (T,3,3)). query_uv: (N,2) pixel coords of query points at t=0. Writes out_npz with the same
    `coords`/`visibs`/`query_points` keys TAPIP3D's own result npz uses. Idempotent: skips both
    passes if out_npz already exists.

    Fallback path for sources with no real depth of their own: runs inference.py twice, once
    with no query_point (auto grid, MegaSaM-computed depth) purely to obtain per-frame depth,
    then lifts our own 2D query points using that depth and re-runs with those as the explicit
    query_point. When real sensor depth is available (e.g. HoloAssist's AhatDepth, via
    data/holoassist.py's lift_rgb_query_points), use run_with_query_points instead -- it skips
    this wasted first pass entirely."""
    if os.path.exists(out_npz):
        return

    t = video.shape[0]
    intrinsics_seq = np.broadcast_to(intrinsics, (t, 3, 3)).astype(np.float32)

    depth_pass_dir = Path(work_dir) / "depth_pass"
    depth_pass_dir.mkdir(parents=True, exist_ok=True)
    depth_input_npz = depth_pass_dir / "input.npz"
    np.savez(depth_input_npz, video=video, intrinsics=intrinsics_seq)
    depth_result = np.load(_invoke(depth_input_npz, depth_pass_dir, tapip3d_python, tapip3d_repo, checkpoint))

    # TAPIP3D's own prepare_inputs resizes video (and depth) to an internal working resolution
    # (confirmed directly: a 640x360 input clip came back with depths shaped (543, 724), not
    # (360, 640)) -- query_uv is in the ORIGINAL video's pixel space, so it must be rescaled to
    # the depth map's actual resolution before indexing into it, or lift_query_points silently
    # samples the wrong pixel entirely.
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
    """Single-pass variant of run(): use when 3D query points are already available from real
    sensor depth (e.g. HoloAssist's AhatDepth via data/holoassist.py's lift_rgb_query_points),
    skipping run()'s wasted first pass whose only purpose is lifting 2D query points with
    TAPIP3D's own MegaSaM depth. query_point: (N,4) = [query_time=0, x, y, z], already in the
    camera(0)/world frame.

    depths: optional (T,H,W) metric depth, e.g. from data/holoassist.py's
    reproject_depth_to_rgb run per-frame. When supplied, TAPIP3D's own prepare_inputs uses it
    directly and never invokes its internal MegaSaM call (sensor depth throughout, not
    monocular estimation). When omitted, TAPIP3D still
    computes depth/pose for the full clip's own tracking internally via MegaSaM (world mode via
    omitted extrinsics) for sources without native depth."""
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
