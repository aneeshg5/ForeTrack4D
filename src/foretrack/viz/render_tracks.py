# Forecast-vs-reality overlay: projects 3D tracks to 2D via intrinsics and
# draws them on the conditioning frame, saturation fading with time (ForeHand4D's visual
# language) so early timesteps are bold and later, more uncertain ones fade out. K forecast
# samples get distinct colors so their spread (or collapse) is visible at a glance; the observed
# (TAPIP3D) track is drawn separately in a fixed color for contrast.

import cv2
import numpy as np

OBSERVED_COLOR = (255, 255, 255)  # white, BGR
FORECAST_COLORS = [  # BGR, distinct hues for up to 5 samples before cycling
    (60, 180, 255), (60, 255, 120), (255, 120, 60), (200, 60, 255), (60, 255, 255),
]


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """points: (..., 3) metric camera-frame coords -> (..., 2) pixel coords. Points behind the
    camera (z <= 0) are returned as NaN so callers can skip drawing them rather than plotting a
    nonsensical projection."""
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    z = points[..., 2]
    uv = np.stack([points[..., 0] / z * fx + cx, points[..., 1] / z * fy + cy], axis=-1)
    uv[z <= 0] = np.nan
    return uv


def _draw_faded_track(img: np.ndarray, uv: np.ndarray, color: tuple, mask: np.ndarray = None) -> None:
    """uv: (T, N, 2) pixel coords for one track (one observed sequence, or one forecast sample).
    Draws every point as a small dot, alpha-faded from 1.0 at t=0 to a fixed floor at the last
    timestep so the whole track is always visible, not exactly invisible by the end."""
    t_len = uv.shape[0]
    for t in range(t_len):
        if mask is not None and not mask[t]:
            continue
        alpha = 1.0 - 0.85 * (t / max(t_len - 1, 1))  # fades to 0.15, never fully invisible
        for x, y in uv[t]:
            if np.isnan(x) or np.isnan(y):
                continue
            overlay = img.copy()
            cv2.circle(overlay, (int(round(x)), int(round(y))), 3, color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def render_forecast_vs_reality(
    frame: np.ndarray,
    observed_tracks: np.ndarray,
    forecast_samples: np.ndarray,
    intrinsics: np.ndarray,
    out_path: str,
    observed_mask: np.ndarray = None,
) -> None:
    """frame: (H, W, 3) uint8 RGB, the conditioning frame. observed_tracks: (T_obs, N, 3) metric,
    TAPIP3D's tracked reality from the conditioning frame onward. forecast_samples: (S, T, N, 3)
    metric, K sampled forecasts from the model. intrinsics: (3, 3), the same camera model both
    track sets are expressed in. Writes a single annotated frame to out_path -- one static image,
    not a rendered video."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()

    observed_uv = project_points(observed_tracks, intrinsics)
    _draw_faded_track(img, observed_uv, OBSERVED_COLOR, observed_mask)

    for s in range(forecast_samples.shape[0]):
        color = FORECAST_COLORS[s % len(FORECAST_COLORS)]
        forecast_uv = project_points(forecast_samples[s], intrinsics)
        _draw_faded_track(img, forecast_uv, color)

    cv2.imwrite(str(out_path), img)


def _draw_trail(img: np.ndarray, uv: np.ndarray, k: int, color: tuple, trail_len: int, mask: np.ndarray = None) -> None:
    """Draws track positions at timesteps (k - trail_len, k], newest bold, older fading -- a
    comet trail showing recent motion, unlike _draw_faded_track's whole-horizon fade. uv: (T, N, 2)."""
    t_hi = min(k, uv.shape[0] - 1)
    for t in range(max(0, t_hi - trail_len + 1), t_hi + 1):
        if mask is not None and not mask[t]:
            continue
        age = t_hi - t
        alpha = 1.0 - 0.85 * (age / max(trail_len - 1, 1))
        radius = 4 if age == 0 else 2
        overlay = img.copy()
        drew = False
        for x, y in uv[t]:
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(overlay, (int(round(x)), int(round(y))), radius, color, -1)
            drew = True
        if drew:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def render_forecast_vs_reality_video(
    frames: np.ndarray,
    cond_idx: int,
    observed_tracks: np.ndarray,
    forecast_samples: np.ndarray,
    intrinsics: np.ndarray,
    fps: float,
    out_path: str,
    observed_mask: np.ndarray = None,
    trail_len: int = 12,
    hold_s: float = 1.5,
) -> str:
    """Side-by-side mp4: both panels play the real video up to the conditioning frame; then the
    left (REALITY) continues playing with TAPIP3D's observed tracks overlaid, while the right
    (FORECAST) freezes on the conditioning frame and animates the K sampled forecast trails over
    it. No pixel is ever synthesized -- the right panel is a real, frozen frame with predicted
    point trajectories drawn on top, which is why this stays on the right side of the "never
    call it generated video" rule: the freeze makes it visually unmistakable that only the
    tracks, not the imagery, are predicted.

    frames: (F, H, W, 3) uint8 RGB, the full uploaded video. observed_tracks: (T_obs, N, 3)
    metric, starting at cond_idx. forecast_samples: (S, T, N, 3). Timesteps map 1:1 to video
    frames (both track sets are per-frame of the clip suffix, same convention demo_pipeline
    uses for its ADE comparison). Returns the path actually written."""
    h, w = frames.shape[1:3]
    observed_uv = project_points(observed_tracks, intrinsics)
    forecast_uv = np.stack([project_points(forecast_samples[s], intrinsics) for s in range(forecast_samples.shape[0])])

    horizon = max(observed_tracks.shape[0], forecast_samples.shape[1])
    cond_frame_bgr = cv2.cvtColor(frames[cond_idx], cv2.COLOR_RGB2BGR)

    # avc1 (H.264) plays natively in browsers; opencv builds without it fall back to mp4v,
    # which most browsers refuse -- the worker should re-encode with ffmpeg in that case.
    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (2 * w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError(f"no usable mp4 codec (tried avc1, mp4v) for {out_path}")

    def composite(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        canvas = np.concatenate([left, right], axis=1)
        cv2.line(canvas, (w, 0), (w, h), (255, 255, 255), 1)
        return canvas

    for t in range(cond_idx):
        raw = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        left, right = raw.copy(), raw.copy()
        _label(left, "REALITY")
        _label(right, "FORECAST")
        writer.write(composite(left, right))

    last = None
    for k in range(horizon):
        left = cv2.cvtColor(frames[min(cond_idx + k, frames.shape[0] - 1)], cv2.COLOR_RGB2BGR).copy()
        _draw_trail(left, observed_uv, k, OBSERVED_COLOR, trail_len, observed_mask)
        _label(left, "REALITY")

        right = cond_frame_bgr.copy()
        for s in range(forecast_uv.shape[0]):
            _draw_trail(right, forecast_uv[s], k, FORECAST_COLORS[s % len(FORECAST_COLORS)], trail_len)
        _label(right, "FORECAST (frozen frame)")

        last = composite(left, right)
        writer.write(last)

    if last is not None:
        for _ in range(int(round(hold_s * fps))):
            writer.write(last)

    writer.release()
    return str(out_path)
