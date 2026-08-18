import cv2
import numpy as np

from .render_tracks import project_points

SAMPLE_COLORS = [
    (0, 159, 230),    # orange
    (233, 180, 86),   # sky blue
    (115, 158, 0),    # bluish green
    (0, 94, 213),     # vermillion
    (167, 121, 204),  # reddish purple
]
OBSERVED_COLOR = (220, 220, 220)
QUERY_COLOR = (80, 220, 255)  # yellow, marks the points being forecast
STATIC_COLOR = (128, 128, 128)
BG = (24, 22, 20)
TIMELINE_H = 44
LEGEND_H = 30
HEADER_H = 34
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, org, scale=0.45, color=(235, 235, 235), thick=1):
    for dx, dy in ((1, 1), (2, 2)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _trail(img, uv, k, color, trail_len, radius=3):
    t_hi = min(k, uv.shape[0] - 1)
    for t in range(max(0, t_hi - trail_len + 1), t_hi + 1):
        age = t_hi - t
        alpha = 1.0 - 0.8 * (age / max(trail_len - 1, 1))
        r = radius + 1 if age == 0 else max(radius - 1, 1)
        overlay = img.copy()
        drew = False
        for x, y in uv[t]:
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(overlay, (int(round(x)), int(round(y))), r, color, -1)
            drew = True
        if drew:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def _mark_queries(img, uv0):
    for x, y in uv0:
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 6, QUERY_COLOR, 1, cv2.LINE_AA)


def _mark_mask(img, mask, alpha=0.3, outline=True):
    overlay = img.copy()
    overlay[mask] = (0.55 * np.array(QUERY_COLOR) + 0.45 * overlay[mask]).astype(np.uint8)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    if outline:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, QUERY_COLOR, 2, cv2.LINE_AA)


def _select_display_points(query_xyz: np.ndarray, n_show: int) -> np.ndarray:
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

    def __init__(self, pts_for_bounds: np.ndarray, w: int, h: int):
        flat = pts_for_bounds.reshape(-1, 3)
        flat = flat[~np.isnan(flat).any(axis=1)]
        self.center = flat.mean(axis=0)
        self.radius = max(float(np.percentile(np.linalg.norm(flat - self.center, axis=1), 95)) * 1.6, 1e-3)
        self.w, self.h = w, h

    def project(self, pts: np.ndarray, azim: float) -> np.ndarray:
        elev = 0.42
        ca, sa, ce, se = np.cos(azim), np.sin(azim), np.cos(elev), np.sin(elev)
        p = pts - self.center
        x = p[..., 0] * ca + p[..., 2] * sa
        z = -p[..., 0] * sa + p[..., 2] * ca
        y = p[..., 1] * ce - z * se
        scale = 0.42 * min(self.w, self.h) / self.radius
        return np.stack([x * scale + self.w / 2, y * scale + self.h / 2], axis=-1)

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
                u0 = self.project(p0, azim)[0]
                u1 = self.project(p1, azim)[0]
                cv2.line(img, (int(u0[0]), int(u0[1])), (int(u1[0]), int(u1[1])), (46, 44, 42), 1, cv2.LINE_AA)


def _chart_panel(w, h, curves, static_curve, k, fps, y_max):
    title = ("SCOREBOARD: how far off is the prediction? (cm, lower = better)" if len(curves) == 1
             else "SCOREBOARD: how far off is each predicted future? (cm, lower = better)")
    img = np.full((h, w, 3), BG, np.uint8)
    ml, mr, mt, mb = 52, 14, 46, 34
    pw, ph = w - ml - mr, h - mt - mb
    t_max = max(len(static_curve), max(len(c) for c in curves))
    _put(img, title, (ml, 20), 0.44)
    _put(img, "measured against where the object's points actually went", (ml, 38), 0.38, (160, 160, 160))
    for frac in (0.0, 0.5, 1.0):
        y = int(mt + ph * (1 - frac))
        cv2.line(img, (ml, y), (ml + pw, y), (50, 48, 46), 1)
        _put(img, f"{frac * y_max:.0f}", (10, y + 4), 0.4, (150, 150, 150))
    def x_of(t):
        return int(ml + pw * t / max(t_max - 1, 1))

    def y_of(v):
        return int(mt + ph * (1 - min(v, y_max) / y_max))

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
        return pts[-1] if pts else None

    end = draw_curve(static_curve, STATIC_COLOR, dashed=True)
    if end and k > 0.25 * t_max:
        x = int(np.clip(end[0] - 150, ml + 4, w - 235))
        _put(img, "baseline: 'object never moves'", (x, max(end[1] - 8, 52)), 0.38, STATIC_COLOR)
    for i, c in enumerate(curves):
        draw_curve(c, SAMPLE_COLORS[i % len(SAMPLE_COLORS)])
    return img


def _legend(w, s_count, sample_label=None):
    img = np.full((LEGEND_H, w, 3), (16, 15, 14), np.uint8)
    x = 14
    cv2.circle(img, (x, LEGEND_H // 2), 5, OBSERVED_COLOR, -1)
    _put(img, "what actually happened", (x + 12, LEGEND_H // 2 + 5), 0.42)
    x += 200
    cv2.circle(img, (x, LEGEND_H // 2), 6, QUERY_COLOR, 1)
    _put(img, "points being forecast", (x + 12, LEGEND_H // 2 + 5), 0.42)
    x += 195
    if s_count == 1:
        cv2.circle(img, (x, LEGEND_H // 2), 5, SAMPLE_COLORS[0], -1)
        _put(img, sample_label or "model prediction", (x + 12, LEGEND_H // 2 + 5), 0.42)
        return img
    _put(img, f"{s_count} predicted futures:", (x, LEGEND_H // 2 + 5), 0.42)
    x += 160
    for i in range(s_count):
        cv2.circle(img, (x, LEGEND_H // 2), 5, SAMPLE_COLORS[i % len(SAMPLE_COLORS)], -1)
        _put(img, f"#{i + 1}", (x + 9, LEGEND_H // 2 + 5), 0.42, (200, 200, 200))
        x += 52
    return img


def _timeline(w, k_global, cond_idx, total, fps):
    img = np.full((TIMELINE_H, w, 3), BG, np.uint8)
    ml, mr = 14, 210
    bw = w - ml - mr
    y = TIMELINE_H // 2
    cv2.line(img, (ml, y), (ml + bw, y), (70, 68, 66), 3)
    x_now = int(ml + bw * min(k_global, total - 1) / max(total - 1, 1))
    x_cond = int(ml + bw * cond_idx / max(total - 1, 1))
    cv2.line(img, (ml, y), (min(x_now, ml + bw), y), (210, 210, 210), 3)
    for xh in range(x_cond, ml + bw, 8):
        cv2.line(img, (xh, y - 5), (xh + 4, y + 5), (95, 93, 90), 1)
    cv2.line(img, (x_cond, y - 9), (x_cond, y + 9), SAMPLE_COLORS[0], 2)
    _put(img, "prediction made here", (max(x_cond - 60, ml), y - 12), 0.38, SAMPLE_COLORS[0])
    t_rel = (k_global - cond_idx) / fps
    label = f"+{t_rel:.1f}s after prediction" if k_global >= cond_idx else "before prediction"
    _put(img, label, (ml + bw + 12, y + 5), 0.46)
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
    show_3d: bool = True,
    sample_label: str = None,
    trail_len: int = 10,
    hold_s: float = 3.0,
    playback_fps: float = 15.0,
    max_horizon: int = None,
    object_mask: np.ndarray = None,
    caption: str = None,
    target_width: int = 720,
    static_camera: bool = True,
) -> str:
    if frames.shape[2] > target_width:
        r = target_width / frames.shape[2]
        frames = np.stack([cv2.resize(f, (int(round(frames.shape[2] * r)), int(round(frames.shape[1] * r)))) for f in frames])
        intrinsics = intrinsics.copy()
        intrinsics[:2] *= r

    h, w = frames.shape[1:3]
    s_count, t_fc = forecast_samples.shape[0], forecast_samples.shape[1]
    t_obs = observed_tracks.shape[0]
    horizon = max(t_obs, t_fc) if max_horizon is None else min(max(t_obs, t_fc), max_horizon)
    total = cond_idx + horizon

    show = _select_display_points(observed_tracks[0], n_show)
    observed_uv = project_points(observed_tracks[:, show], intrinsics)
    forecast_uv = [project_points(forecast_samples[s][:, show], intrinsics) for s in range(s_count)]
    query_uv0 = observed_uv[0]

    t_cmp = min(t_obs, t_fc, horizon)
    def err(pred, gt):
        return np.linalg.norm(pred - gt, axis=-1).mean(axis=1) * 100

    curves = [err(forecast_samples[s][:t_cmp], observed_tracks[:t_cmp]) for s in range(s_count)]
    static_curve = err(np.repeat(observed_tracks[:1], t_cmp, axis=0), observed_tracks[:t_cmp])
    y_max = max(1.0, float(np.nanmax([np.nanmax(c) for c in curves + [static_curve]])) * 1.15)

    orbit = _Orbit3D(np.concatenate([observed_tracks.reshape(-1, 3)] + [f.reshape(-1, 3) for f in forecast_samples]), w, h)
    cond_bgr = cv2.cvtColor(frames[cond_idx], cv2.COLOR_RGB2BGR)
    cond_dim = (cond_bgr * 0.55).astype(np.uint8)  # dimmed frozen frame reads as deliberate
    legend = _legend(2 * w, s_count, sample_label)

    header = None
    if caption:
        header = np.full((HEADER_H, 2 * w, 3), (16, 15, 14), np.uint8)
        _put(header, caption, (14, 22), 0.5)
    head_h = HEADER_H if header is not None else 0

    chart_h = h if show_3d else int(h * 0.62)
    canvas_h = head_h + h + LEGEND_H + chart_h + TIMELINE_H

    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), playback_fps, (2 * w, canvas_h))
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
        k = k_global - cond_idx
        if show_3d:
            p3d = np.full((h, w, 3), BG, np.uint8)
            azim = 0.6 + 0.25 * (k_global / max(total - 1, 1)) * 2 * np.pi * 0.35
            orbit.draw_ground(p3d, azim)
            if k >= 0:
                ouv = orbit.project(observed_tracks[: min(k + 1, t_obs), show], azim)
                _trail(p3d, ouv, k, OBSERVED_COLOR, trail_len + 4, radius=2)
                for s in range(s_count):
                    fuv = orbit.project(forecast_samples[s][: min(k + 1, t_fc), show], azim)
                    _trail(p3d, fuv, k, SAMPLE_COLORS[s % len(SAMPLE_COLORS)], trail_len + 4, radius=2)
            _put(p3d, "SAME TRACKS IN 3D -- the model predicts metric 3D motion,", (10, 22), 0.42)
            _put(p3d, "not screen positions (view slowly orbits to show depth)", (10, 40), 0.42)
            bottom = np.concatenate([p3d, _chart_panel(w, chart_h, curves, static_curve, max(k, 0), fps, y_max)], axis=1)
        else:
            bottom = _chart_panel(2 * w, chart_h, curves, static_curve, max(k, 0), fps, y_max)

        top = np.concatenate([left, right], axis=1)
        parts = ([header] if header is not None else []) + [top, legend, bottom, _timeline(2 * w, k_global, cond_idx, total, fps)]
        canvas = np.concatenate(parts, axis=0)
        cv2.line(canvas, (w, head_h), (w, head_h + h + LEGEND_H + (h if show_3d else 0)), (90, 88, 86), 1)
        if flash > 0:
            cv2.rectangle(canvas, (2, 2), (2 * w - 3, canvas.shape[0] - 3), SAMPLE_COLORS[0], int(2 + 4 * flash))
        return canvas

    for t in range(cond_idx):
        raw = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        left, right = raw.copy(), raw.copy()
        _put(left, "LIVE VIDEO", (10, 22))
        _put(right, "LIVE VIDEO (prediction starts soon)", (10, 22))
        emit(compose(left, right, t))

    left0 = cond_bgr.copy()
    _put(left0, "WHAT ACTUALLY HAPPENS NEXT", (10, 22))
    right0 = cond_dim.copy()
    if object_mask is not None:
        _mark_mask(left0, object_mask)
        _mark_mask(right0, object_mask, alpha=0.5)
    else:
        _mark_queries(left0, query_uv0)
        _mark_queries(right0, query_uv0)
    _put(right0, "PREDICTION -- frame frozen on purpose", (10, 22))
    _put(right0, "the model sees ONLY this frame, then predicts", (10, h - 30), 0.46, QUERY_COLOR)
    _put(right0, "where these points move next, in 3D", (10, h - 12), 0.46, QUERY_COLOR)
    n_freeze = max(1, int(round((2.6 if cond_idx == 0 else 1.6) * playback_fps / repeat)))
    for i in range(n_freeze):
        emit(compose(left0, right0, cond_idx, flash=1.0 - i / n_freeze))

    last = None
    for k in range(horizon):
        li = min(cond_idx + k, frames.shape[0] - 1)
        left = cv2.cvtColor(frames[li], cv2.COLOR_RGB2BGR).copy()
        if static_camera:
            _trail(left, observed_uv, k, OBSERVED_COLOR, trail_len, radius=1)
            _put(left, "WHAT ACTUALLY HAPPENS (real video + tracked points)", (10, 22))
        else:
            _put(left, "WHAT ACTUALLY HAPPENS (real video)", (10, 22))

        right = cond_dim.copy()
        if object_mask is not None:
            _mark_mask(right, object_mask, alpha=0.25, outline=k < 2 * trail_len)
        elif k < 2 * trail_len:
            _mark_queries(right, query_uv0)
        _trail(right, observed_uv, k, OBSERVED_COLOR, trail_len, radius=1)
        for s in range(s_count):
            _trail(right, forecast_uv[s], k, SAMPLE_COLORS[s % len(SAMPLE_COLORS)], trail_len)
        pred_word = "PREDICTED (color)" if s_count == 1 else "PREDICTED (colors)"
        _put(right, f"{pred_word} vs ACTUAL (white), frozen frame", (10, 22))
        _put(right, f"+{k / fps:.1f}s into the predicted future", (10, h - 12), 0.5, QUERY_COLOR)

        last = compose(left, right, cond_idx + k)
        emit(last)

    if last is not None:
        fde = [float(np.linalg.norm(forecast_samples[s][t_cmp - 1] - observed_tracks[t_cmp - 1], axis=-1).mean() * 100) for s in range(s_count)]
        best = int(np.argmin(fde))
        summary = last.copy()
        box_h = 70
        y0 = head_h + h - box_h - 8
        cv2.rectangle(summary, (10, y0), (2 * w - 10, y0 + box_h), (12, 11, 10), -1)
        if s_count == 1:
            headline = f"result at +{(t_cmp - 1) / fps:.1f}s: prediction off by {fde[0]:.1f}cm"
            second = f"'object never moves' baseline: off by {static_curve[t_cmp - 1]:.1f}cm"
        else:
            headline = f"result at +{(t_cmp - 1) / fps:.1f}s: closest prediction was future #{best + 1}, off by {fde[best]:.1f}cm"
            second = f"'object never moves' baseline: off by {static_curve[t_cmp - 1]:.1f}cm    worst future: {max(fde):.1f}cm"
        _put(summary, headline, (20, y0 + 22), 0.52, SAMPLE_COLORS[best % len(SAMPLE_COLORS)])
        _put(summary, second, (20, y0 + 44), 0.46, (200, 200, 200))
        tail = " (bottom-left view)" if show_3d else ""
        _put(summary, f"prediction used ONE frame; positions are metric 3D{tail}", (20, y0 + 64), 0.44, (170, 170, 170))
        emit(summary, times=max(1, int(round(hold_s * fps))))

    writer.release()
    return str(out_path)
