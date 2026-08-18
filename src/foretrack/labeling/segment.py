# Uses SAM 2 (facebookresearch/sam2) and mediapipe HandLandmarker. See NOTICE.md.

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MIDDLE_FINGER_MCP = 9


def detect_hands(frame: np.ndarray, model_path: str, min_confidence: float = 0.5) -> list:
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=min_confidence,
    )
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))

    h, w = frame.shape[:2]
    return [
        (landmarks[MIDDLE_FINGER_MCP].x * w, landmarks[MIDDLE_FINGER_MCP].y * h)
        for landmarks in result.hand_landmarks
    ]


def detect_hand_landmarks(frame: np.ndarray, model_path: str, min_confidence: float = 0.5) -> list:
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=min_confidence,
    )
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))

    h, w = frame.shape[:2]
    return [
        np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        for landmarks in result.hand_landmarks
    ]


class Sam2ObjectSegmenter:
    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cuda"):
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def object_mask(self, frame: np.ndarray, contact_point: tuple, negative_points: np.ndarray = None) -> np.ndarray:
        self.predictor.set_image(frame)
        point_coords = [contact_point]
        point_labels = [1]
        if negative_points is not None and len(negative_points) > 0:
            point_coords.extend(negative_points.tolist())
            point_labels.extend([0] * len(negative_points))
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int64)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=True
        )
        frame_area = masks[0].size
        best, best_area = None, -1
        for m in masks:
            m = m.astype(bool)
            area = int(m.sum())
            if area > 0.4 * frame_area:
                continue
            if negative_points is not None and len(negative_points) > 0:
                xs = np.clip(negative_points[:, 0].astype(int), 0, m.shape[1] - 1)
                ys = np.clip(negative_points[:, 1].astype(int), 0, m.shape[0] - 1)
                if m[ys, xs].mean() > 0.6:
                    continue
            if area > best_area:
                best, best_area = m, area
        if best is None:
            best = masks[int(np.argmax(scores))].astype(bool)
        return best


def save_mask_overlay(
    frame: np.ndarray, mask: np.ndarray, contact_point: tuple, out_path: str, query_uv: np.ndarray = None
) -> None:
    overlay = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * red[mask]).astype(np.uint8)
    cv2.circle(overlay, (int(contact_point[0]), int(contact_point[1])), 6, (255, 255, 0), -1)
    if query_uv is not None:
        for x, y in query_uv:
            cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 255), -1)
    cv2.imwrite(str(out_path), overlay)


def sample_query_points_in_mask(mask: np.ndarray, n: int = 64, seed: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("mask is empty, cannot sample query points")
    points = np.stack([xs, ys], axis=-1).astype(np.float32)

    rng = np.random.default_rng(seed)
    if len(points) <= n:
        idx = rng.choice(len(points), size=n, replace=True)
        return points[idx]
    selected = [int(rng.integers(len(points)))]
    dists = np.linalg.norm(points - points[selected[0]], axis=1)
    for _ in range(n - 1):
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)
        dists = np.minimum(dists, np.linalg.norm(points - points[next_idx], axis=1))
    return points[selected]
