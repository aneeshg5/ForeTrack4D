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
