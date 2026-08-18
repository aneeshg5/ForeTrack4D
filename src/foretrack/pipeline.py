import subprocess
from pathlib import Path

import cv2
import numpy as np

from .data.dexycb import CROP_TARGET_SIZE, compute_object_crop
from .eval.metrics import ade_per_timestep
from .labeling.impute import read_video_frames
from .labeling.run_tapip3d import run as run_tapip3d
from .labeling.segment import (
    Sam2ObjectSegmenter,
    detect_hand_landmarks,
    detect_hands,
    sample_query_points_in_mask,
)
from .viz.demo_video import render_demo_video
from .viz.render_tracks import render_forecast_vs_reality

MAX_DURATION_S = 15.0
CUT_DIFF_THRESHOLD = 40.0


def estimate_intrinsics(h: int, w: int) -> np.ndarray:
    f = float(max(h, w))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)


def detect_likely_cuts(frames: np.ndarray) -> list:
    diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)).mean(axis=(1, 2, 3))
    return [int(i) + 1 for i, d in enumerate(diffs) if d > CUT_DIFF_THRESHOLD]


def validate_video(frames: np.ndarray, fps: float, hand_model_path: str, max_duration_s: float = MAX_DURATION_S) -> tuple:
    n_frames = len(frames)
    duration = n_frames / fps
    if duration > max_duration_s:
        return False, f"video is {duration:.1f}s, longer than the {max_duration_s:.0f}s cap -- trim it and try again"
    if n_frames < 2:
        return False, "video has no readable frames"

    cuts = detect_likely_cuts(frames)
    if cuts:
        return False, f"looks like a scene cut around frame {cuts[0]} (large frame-to-frame change) -- upload a single continuous shot"

    sample_idxs = np.linspace(0, n_frames - 1, min(5, n_frames)).astype(int)
    hand_hits = sum(1 for i in sample_idxs if len(detect_hands(frames[i], hand_model_path)) > 0)
    if hand_hits == 0:
        return False, "no hands detected in this video -- needs a hand manipulating an object"

    return True, ""


def run_demo_job(
    video_path: str,
    conditioning_frac: float,
    out_dir: str,
    *,
    hand_model_path: str,
    sam2_checkpoint: str,
    sam2_model_cfg: str,
    tapip3d_python: str,
    tapip3d_repo: str,
    tapip3d_checkpoint: str,
    forecast_python: str,
    forecast_script: str,
    forecast_config: str,
    forecast_checkpoint: str,
    k_samples: int = 5,
    sam2_device: str = "cuda",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames = read_video_frames(video_path, 0, n_frames)

    ok, reason = validate_video(frames, fps, hand_model_path)
    if not ok:
        return {"status": "rejected", "reason": reason}

    cond_idx = int(np.clip(conditioning_frac, 0.0, 1.0) * (n_frames - 1))
    cond_frame = frames[cond_idx]
    h, w = cond_frame.shape[:2]
    intrinsics = estimate_intrinsics(h, w)

    hand_points = detect_hands(cond_frame, hand_model_path)
    if not hand_points:
        return {"status": "rejected", "reason": f"no hand detected at the chosen conditioning frame ({cond_idx}) -- try a different point on the scrubber"}
    contact_point = hand_points[0]
    landmark_sets = detect_hand_landmarks(cond_frame, hand_model_path)
    neg_points = np.concatenate(landmark_sets, axis=0) if landmark_sets else None

    segmenter = Sam2ObjectSegmenter(sam2_checkpoint, sam2_model_cfg, device=sam2_device)
    mask = segmenter.object_mask(cond_frame, contact_point, neg_points)
    if not mask.any():
        return {"status": "rejected", "reason": "no manipulated object found near the detected hand -- make sure the object is clearly visible and being held"}

    if int(mask.sum()) < 300:
        return {"status": "rejected", "reason": "could not isolate the manipulated object (segmentation too small) -- try a frame where the object is clearly visible"}
    np.save(out_dir / "mask.npy", mask)
    query_uv_full = sample_query_points_in_mask(mask, n=64)

    x0, y0, x1, y1 = compute_object_crop(query_uv_full, h, w, image=cond_frame)
    crop = cond_frame[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
    crop_h, crop_w = crop.shape[:2]
    crop_resized = cv2.resize(crop, (CROP_TARGET_SIZE[1], CROP_TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    query_uv_crop = query_uv_full - np.array([x0, y0], dtype=np.float32)
    query_uv_crop = query_uv_crop * np.array([CROP_TARGET_SIZE[1] / crop_w, CROP_TARGET_SIZE[0] / crop_h], dtype=np.float32)

    observed_video = frames[cond_idx:]
    observed_out = out_dir / "observed.npz"
    run_tapip3d(
        observed_video, intrinsics, query_uv_full, str(observed_out),
        tapip3d_python=tapip3d_python, tapip3d_repo=tapip3d_repo, checkpoint=tapip3d_checkpoint,
        work_dir=str(out_dir / "tapip3d_work"),
    )
    observed = np.load(observed_out)
    observed_tracks = observed["tracks"]  # (T_obs, N, 3), starts at the conditioning frame

    query_xyz_t0 = observed_tracks[0]

    forecast_input = out_dir / "forecast_input.npz"
    np.savez(
        forecast_input, image=crop_resized, query_uv=query_uv_crop,
        query_xyz_t0=query_xyz_t0, orig_h=CROP_TARGET_SIZE[0], orig_w=CROP_TARGET_SIZE[1],
    )
    forecast_output = out_dir / "forecast_output.npz"
    subprocess.run(
        [
            forecast_python, forecast_script,
            "--config", forecast_config, "--diffusion_ckpt", forecast_checkpoint,
            "--input_npz", str(forecast_input), "--output_npz", str(forecast_output),
            "--samples", str(k_samples),
        ],
        check=True,
    )
    forecast = np.load(forecast_output)
    forecast_samples = forecast["samples"]  # (S, T, N, 3)

    t_compare = min(observed_tracks.shape[0], forecast_samples.shape[1])
    ade_curves = [
        ade_per_timestep(forecast_samples[s, :t_compare], observed_tracks[:t_compare]).tolist()
        for s in range(forecast_samples.shape[0])
    ]

    render_path = out_dir / "comparison.jpg"
    render_forecast_vs_reality(cond_frame, observed_tracks, forecast_samples, intrinsics, str(render_path))

    video_path = out_dir / "comparison.mp4"
    render_demo_video(
        frames, cond_idx, observed_tracks, forecast_samples, intrinsics, fps, str(video_path),
    )

    return {
        "status": "done",
        "conditioning_frame": cond_idx,
        "render_path": str(render_path),
        "video_path": str(video_path),
        "ade_curve_per_sample": ade_curves,
        "low_confidence": bool(forecast["low_confidence"]),
        "disagreement_cm": float(forecast["disagreement_cm"]),
    }
