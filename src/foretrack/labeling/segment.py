# SAM2 (third_party/sam2, cloned by scripts/setup_third_party.sh) and mediapipe run in-process
# in the foretrack env -- unlike TAPIP3D, neither needs the separate tapip3d conda env, so no
# subprocess boundary here (contrast with labeling/run_tapip3d.py).

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 21-point hand landmark model, MIDDLE_FINGER_MCP index -- mediapipe's legacy `solutions.hands`
# API is unavailable on this build (its pip wheel only ships the Tasks API as of 0.10.x), so
# this uses HandLandmarker directly. model_path: a downloaded hand_landmarker.task bundle, e.g.
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
MIDDLE_FINGER_MCP = 9


def detect_hands(frame: np.ndarray, model_path: str, min_confidence: float = 0.5) -> list:
    """returns one (x, y) pixel-coord contact-region estimate per detected hand."""
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
    """all 21 landmark pixel coords per detected hand, not just the single contact joint
    detect_hands returns -- meant as SAM2 negative prompts (Sam2ObjectSegmenter.object_mask),
    the same fix that solved HoloAssist's hand-vs-object mask confusion,
    for sources with no native hand-joint file format (mediapipe-only, e.g. EgoExo4D). Returns
    one (21,2) array per detected hand."""
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
        """largest coherent object mask near a hand's contact point.

        negative_points: (M,2) pixel coords SAM2 is told are NOT part of the object -- e.g.
        data/holoassist.hand_landmark_pixels' full set of hand joints. A single positive click
        at the contact point alone is not enough of a non-hand filter in practice: real
        mask-overlay spot-checking on HoloAssist data showed SAM2 frequently selecting the
        hand/glove itself as the "largest coherent region touching this point" instead of the
        smaller grasped object it's touching, since a gloved hand is
        larger and more visually coherent than many manipulated objects. Marking the hand's own
        extent as explicitly negative directly excludes it as a candidate region, rather than
        relying on click placement alone."""
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
        return masks[int(np.argmax(scores))].astype(bool)


def save_mask_overlay(
    frame: np.ndarray, mask: np.ndarray, contact_point: tuple, out_path: str, query_uv: np.ndarray = None
) -> None:
    """Spot-check overlay: red
    tint over the SAM2 mask, a cyan dot at the hand contact point SAM2 was seeded from, and
    (if given) yellow dots at the sampled query points -- lets a human tell apart three
    distinct failure modes at a glance: wrong object masked, contact point on/inside the hand
    rather than the grasped object, or a good mask with badly clustered query points."""
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
    """farthest-point-sample n pixel coords inside a 2D mask, mirroring gt_tracks.py's
    sample_query_points (same greedy FPS loop, over mask pixels instead of a mesh surface)."""
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
