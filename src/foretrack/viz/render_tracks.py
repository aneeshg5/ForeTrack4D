import cv2
import numpy as np

OBSERVED_COLOR = (255, 255, 255)  # white, BGR
FORECAST_COLORS = [  # BGR, distinct hues for up to 5 samples before cycling
    (60, 180, 255), (60, 255, 120), (255, 120, 60), (200, 60, 255), (60, 255, 255),
]


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    z = points[..., 2]
    uv = np.stack([points[..., 0] / z * fx + cx, points[..., 1] / z * fy + cy], axis=-1)
    uv[z <= 0] = np.nan
    return uv


def _draw_faded_track(img: np.ndarray, uv: np.ndarray, color: tuple, mask: np.ndarray = None) -> None:
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
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()

    observed_uv = project_points(observed_tracks, intrinsics)
    _draw_faded_track(img, observed_uv, OBSERVED_COLOR, observed_mask)

    for s in range(forecast_samples.shape[0]):
        color = FORECAST_COLORS[s % len(FORECAST_COLORS)]
        forecast_uv = project_points(forecast_samples[s], intrinsics)
        _draw_faded_track(img, forecast_uv, color)

    cv2.imwrite(str(out_path), img)


def _draw_trail(img: np.ndarray, uv: np.ndarray, k: int, color: tuple, trail_len: int, mask: np.ndarray = None) -> None:
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
    h, w = frames.shape[1:3]
    observed_uv = project_points(observed_tracks, intrinsics)
    forecast_uv = np.stack([project_points(forecast_samples[s], intrinsics) for s in range(forecast_samples.shape[0])])

    horizon = max(observed_tracks.shape[0], forecast_samples.shape[1])
    cond_frame_bgr = cv2.cvtColor(frames[cond_idx], cv2.COLOR_RGB2BGR)

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
