# Composite forecast-vs-reality demo video. Layout: 2x2 panels (observed video, frozen-frame
# forecast, orbiting 3D world-space view, live ADE-vs-horizon chart) over a full-width timeline
# bar. Both track panels play the real video in sync to the conditioning frame; after it, the
# observed panel keeps playing real footage while the forecast panel freezes on the conditioning
# frame and animates predicted trajectories only -- no pixel is ever synthesized.

import cv2
import numpy as np

from .render_tracks import project_points

# Okabe-Ito colorblind-safe palette, BGR
SAMPLE_COLORS = [
    (0, 159, 230),    # orange
    (233, 180, 86),   # sky blue
    (115, 158, 0),    # bluish green
    (0, 94, 213),     # vermillion
    (167, 121, 204),  # reddish purple
]
OBSERVED_COLOR = (200, 200, 200)
STATIC_COLOR = (128, 128, 128)
BG = (24, 22, 20)
TIMELINE_H = 44
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, org, scale=0.45, color=(235, 235, 235), thick=1):
    # same-thickness offset shadow: opencv glyph advance grows with stroke thickness, so a
    # thicker outline pass would extend past the fill and leave ghost tails
    for dx, dy in ((1, 1), (2, 2)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _trail(img, uv, k, color, trail_len, thickness=2, radius=3):
    """comet trail: positions at timesteps (k - trail_len, k], newest bold. uv: (T, M, 2)."""
    t_hi = min(k, uv.shape[0] - 1)
    for t in range(max(0, t_hi - trail_len + 1), t_hi + 1):
        age = t_hi - t
        alpha = 1.0 - 0.8 * (age / max(trail_len - 1, 1))
        r = radius + 1 if age == 0 else radius - 1
        overlay = img.copy()
        drew = False
        for x, y in uv[t]:
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(overlay, (int(round(x)), int(round(y))), max(r, 1), color, -1 if age == 0 else thickness - 1)
            drew = True
        if drew:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def _select_display_points(query_xyz: np.ndarray, n_show: int) -> np.ndarray:
    """farthest-point-sample indices of the query set for display -- all points stay in the
    metrics, only the rendering is thinned."""
    n = query_xyz.shape[0]
    if n <= n_show:
        return np.arange(n)
    idx = [0]
    d = np.linalg.norm(query_xyz - query_xyz[0], axis=1)
    for _ in range(n_show - 1):
        idx.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(query_xyz - query_xyz[idx[-1]], axis=1))
    return np.array(sorted(idx))


class _Orbit3D:
    """world-space viewport: orthographic-ish pinhole orbiting the t=0 centroid."""

    def __init__(self, pts_for_bounds: np.ndarray, w: int, h: int):
        flat = pts_for_bounds.reshape(-1, 3)
        flat = flat[~np.isnan(flat).any(axis=1)]
        self.center = flat.mean(axis=0)
        self.radius = max(float(np.percentile(np.linalg.norm(flat - self.center, axis=1), 95)) * 1.6, 1e-3)
        self.w, self.h = w, h

    def project(self, pts: np.ndarray, azim: float) -> np.ndarray:
        """pts: (..., 3) world -> (..., 2) viewport pixels. Fixed elevation, azimuth in radians."""
        elev = 0.42
        ca, sa, ce, se = np.cos(azim), np.sin(azim), np.cos(elev), np.sin(elev)
        p = pts - self.center
        x = p[..., 0] * ca + p[..., 2] * sa
        z = -p[..., 0] * sa + p[..., 2] * ca
        y = p[..., 1] * ce - z * se
        scale = 0.42 * min(self.w, self.h) / self.radius
        u = x * scale + self.w / 2
        v = y * scale + self.h / 2
        return np.stack([u, v], axis=-1)

    def draw_ground(self, img: np.ndarray, azim: float) -> None:
        ext = self.radius
        n = 5
        ys = self.center[1] + 0.35 * self.radius
        for i in range(-n, n + 1):
            a = np.array([[self.center[0] + i * ext / n, ys, self.center[2] - ext]])
            b = np.array([[self.center[0] + i * ext / n, ys, self.center[2] + ext]])
            c = np.array([[self.center[0] - ext, ys, self.center[2] + i * ext / n]])
            d = np.array([[self.center[0] + ext, ys, self.center[2] + i * ext / n]])
            for p0, p1 in ((a, b), (c, d)):
                u0 = self.project(p0 - self.center + self.center, azim)[0]
                u1 = self.project(p1, azim)[0]
                cv2.line(img, (int(u0[0]), int(u0[1])), (int(u1[0]), int(u1[1])), (46, 44, 42), 1, cv2.LINE_AA)


def _chart_panel(w, h, curves, static_curve, k, fps, y_max):
    """live ADE-vs-horizon chart, drawn up to timestep k. curves: list of (T,) cm arrays."""
    img = np.full((h, w, 3), BG, np.uint8)
    ml, mr, mt, mb = 52, 14, 30, 34
    pw, ph = w - ml - mr, h - mt - mb
    t_max = max(len(static_curve), max(len(c) for c in curves))
    _put(img, "forecast error vs horizon (cm, all 64 points)", (ml, 19), 0.42)
    for frac in (0.0, 0.5, 1.0):
        y = int(mt + ph * (1 - frac))
        cv2.line(img, (ml, y), (ml + pw, y), (50, 48, 46), 1)
        _put(img, f"{frac * y_max:.0f}", (10, y + 4), 0.4, (150, 150, 150))
    x_of = lambda t: int(ml + pw * t / max(t_max - 1, 1))
    y_of = lambda v: int(mt + ph * (1 - min(v, y_max) / y_max))
    _put(img, "0s", (ml - 6, h - 12), 0.4, (150, 150, 150))
    _put(img, f"+{(t_max - 1) / fps:.1f}s", (ml + pw - 34, h - 12), 0.4, (150, 150, 150))

    def draw_curve(curve, color, dashed=False):
        upto = min(k + 1, len(curve))
        pts = [(x_of(t), y_of(curve[t])) for t in range(upto) if np.isfinite(curve[t])]
        for i in range(1, len(pts)):
            if dashed and i % 2 == 0:
                continue
            cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], 3, color, -1)

    draw_curve(static_curve, STATIC_COLOR, dashed=True)
    for i, c in enumerate(curves):
        draw_curve(c, SAMPLE_COLORS[i % len(SAMPLE_COLORS)])
    _put(img, "gray dashed = 'object never moves' baseline", (ml + 40, h - 12), 0.38, (150, 150, 150))
    return img


def _timeline(w, k_global, cond_idx, total, fps):
    img = np.full((TIMELINE_H, w, 3), BG, np.uint8)
    ml, mr = 14, 150
    bw = w - ml - mr
    y = TIMELINE_H // 2
    cv2.line(img, (ml, y), (ml + bw, y), (70, 68, 66), 3)
    x_now = int(ml + bw * min(k_global, total - 1) / max(total - 1, 1))
    x_cond = int(ml + bw * cond_idx / max(total - 1, 1))
    cv2.line(img, (ml, y), (min(x_now, ml + bw), y), (210, 210, 210), 3)
    for xh in range(x_cond, ml + bw, 8):  # hatched future-of-conditioning region
        cv2.line(img, (xh, y - 5), (xh + 4, y + 5), (95, 93, 90), 1)
    cv2.line(img, (x_cond, y - 9), (x_cond, y + 9), SAMPLE_COLORS[0], 2)
    _put(img, "forecast starts", (max(x_cond - 44, ml), y - 12), 0.38, SAMPLE_COLORS[0])
    t_rel = (k_global - cond_idx) / fps
    label = f"t = +{t_rel:.1f}s" if k_global >= cond_idx else f"t = {t_rel:.1f}s"
    _put(img, label, (ml + bw + 12, y + 5), 0.5)
    return img


def render_demo_video(
    frames: np.ndarray,
    cond_idx: int,
    observed_tracks: np.ndarray,
    forecast_samples: np.ndarray,
    intrinsics: np.ndarray,
    fps: float,
    out_path: str,
    n_show: int = 12,
    trail_len: int = 10,
    hold_s: float = 3.0,
    playback_fps: float = 15.0,
) -> str:
    """frames: (F, H, W, 3) uint8 RGB. observed_tracks: (T_obs, N, 3) metric, from cond_idx on.
    forecast_samples: (S, T, N, 3). Writes the composite mp4 and returns its path. Metrics use
    all N points; the panels display an FPS-thinned subset of n_show for legibility."""
    h, w = frames.shape[1:3]
    s_count, t_fc = forecast_samples.shape[0], forecast_samples.shape[1]
    t_obs = observed_tracks.shape[0]
    horizon = max(t_obs, t_fc)
    total = cond_idx + horizon

    show = _select_display_points(observed_tracks[0], n_show)
    observed_uv = project_points(observed_tracks[:, show], intrinsics)
    forecast_uv = [project_points(forecast_samples[s][:, show], intrinsics) for s in range(s_count)]

    # error curves on ALL points, vs observed reality; static = frozen t=0 positions
    t_cmp = min(t_obs, t_fc)
    err = lambda pred, gt: np.linalg.norm(pred - gt, axis=-1).mean(axis=1) * 100
    curves = [err(forecast_samples[s][:t_cmp], observed_tracks[:t_cmp]) for s in range(s_count)]
    static_curve = err(np.repeat(observed_tracks[:1], t_obs, axis=0), observed_tracks)
    y_max = max(1.0, float(np.nanmax([np.nanmax(c) for c in curves + [static_curve]])) * 1.15)

    orbit = _Orbit3D(np.concatenate([observed_tracks.reshape(-1, 3)] + [f.reshape(-1, 3) for f in forecast_samples]), w, h)
    cond_bgr = cv2.cvtColor(frames[cond_idx], cv2.COLOR_RGB2BGR)

    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), playback_fps, (2 * w, 2 * h + TIMELINE_H))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError(f"no usable mp4 codec for {out_path}")
    repeat = max(1, int(round(playback_fps / fps)))

    def emit(canvas, times=1):
        for _ in range(times * repeat):
            writer.write(canvas)

    def compose(left, right, k_global, flash=0.0):
        p3d = np.full((h, w, 3), BG, np.uint8)
        azim = 0.6 + 0.25 * (k_global / max(total - 1, 1)) * 2 * np.pi * 0.35
        orbit.draw_ground(p3d, azim)
        k = k_global - cond_idx
        if k >= 0:
            ouv = orbit.project(observed_tracks[: min(k + 1, t_obs), show], azim)
            _trail(p3d, ouv, k, OBSERVED_COLOR, trail_len + 4, radius=2)
            for s in range(s_count):
                fuv = orbit.project(forecast_samples[s][: min(k + 1, t_fc), show], azim)
                _trail(p3d, fuv, k, SAMPLE_COLORS[s % len(SAMPLE_COLORS)], trail_len + 4, radius=2)
        _put(p3d, "3D world space (camera orbits for depth)", (10, 22), 0.45)
        _put(p3d, "white = observed   colors = 5 sampled futures", (10, h - 12), 0.4, (170, 170, 170))

        chart = _chart_panel(w, h, curves, static_curve, max(k, 0), fps, y_max)
        top = np.concatenate([left, right], axis=1)
        bottom = np.concatenate([p3d, chart], axis=1)
        canvas = np.concatenate([top, bottom, _timeline(2 * w, k_global, cond_idx, total, fps)], axis=0)
        cv2.line(canvas, (w, 0), (w, 2 * h), (90, 88, 86), 1)
        cv2.line(canvas, (0, h), (2 * w, h), (90, 88, 86), 1)
        if flash > 0:
            cv2.rectangle(canvas, (2, 2), (2 * w - 3, 2 * h + TIMELINE_H - 3), SAMPLE_COLORS[0], int(2 + 4 * flash))
        return canvas

    # phase 1: shared playback up to the conditioning frame
    for t in range(cond_idx):
        raw = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        left, right = raw.copy(), raw.copy()
        _put(left, "OBSERVED (TAPIP3D)", (10, 22))
        _put(right, "FORECAST", (10, 22))
        emit(compose(left, right, t))

    # freeze event: hold ~1.2s with the information-boundary caption
    left0 = cond_bgr.copy()
    right0 = cond_bgr.copy()
    _put(left0, "OBSERVED (TAPIP3D)", (10, 22))
    _put(right0, "FORECAST", (10, 22))
    _put(right0, "model sees ONLY this frame -- no future frames", (10, h - 14), 0.46, SAMPLE_COLORS[0])
    n_freeze = max(1, int(round(1.2 * playback_fps / repeat)))
    for i in range(n_freeze):
        emit(compose(left0, right0, cond_idx, flash=1.0 - i / n_freeze))

    # phase 2: rollout
    last = None
    for k in range(horizon):
        li = min(cond_idx + k, frames.shape[0] - 1)
        left = cv2.cvtColor(frames[li], cv2.COLOR_RGB2BGR).copy()
        _trail(left, observed_uv, k, OBSERVED_COLOR, trail_len, radius=1)
        _put(left, "OBSERVED (TAPIP3D)", (10, 22))
        if k >= t_obs:
            _put(left, "clip ended", (10, h - 14), 0.42, (170, 170, 170))

        right = cond_bgr.copy()
        for s in range(s_count):
            _trail(right, forecast_uv[s], k, SAMPLE_COLORS[s % len(SAMPLE_COLORS)], trail_len)
        _put(right, "FORECAST", (10, 22))
        _put(right, f"frozen conditioning frame + {s_count} sampled futures ({observed_tracks.shape[1]} pts, {len(show)} shown)", (10, h - 14), 0.4, (190, 190, 190))

        last = compose(left, right, cond_idx + k)
        emit(last)

    # end hold with computed outcome numbers
    if last is not None:
        fde = [float(np.linalg.norm(forecast_samples[s][t_cmp - 1] - observed_tracks[t_cmp - 1], axis=-1).mean() * 100) for s in range(s_count)]
        summary = last.copy()
        box_h = 66
        cv2.rectangle(summary, (10, 2 * h - box_h - 8), (2 * w - 10, 2 * h - 6), (12, 11, 10), -1)
        _put(summary, f"final error vs observed at +{(t_cmp - 1) / fps:.1f}s -- best sample: {min(fde):.1f}cm, worst: {max(fde):.1f}cm", (20, 2 * h - box_h + 18), 0.5)
        _put(summary, f"'object never moves' baseline: {static_curve[t_cmp - 1]:.1f}cm", (20, 2 * h - box_h + 40), 0.5, STATIC_COLOR)
        _put(summary, "forecast made from ONE frame; tracks are 3D (see world-space view)", (20, 2 * h - box_h + 60), 0.44, (190, 190, 190))
        emit(summary, times=max(1, int(round(hold_s * fps))))

    writer.release()
    return str(out_path)
