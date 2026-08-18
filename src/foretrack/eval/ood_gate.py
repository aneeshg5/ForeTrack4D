import numpy as np

DISAGREEMENT_THRESHOLD_CM = 3.659


def conditioned_disagreement(pred_cond: np.ndarray, pred_uncond: np.ndarray, mask: np.ndarray = None) -> float:
    err = np.linalg.norm(pred_cond - pred_uncond, axis=-1)  # (T, N)
    if mask is not None:
        err = err[mask]
    return float(err.mean() * 100)


def gated_prediction(pred_cond: np.ndarray, pred_uncond: np.ndarray, pred_static: np.ndarray, mask: np.ndarray = None, threshold: float = DISAGREEMENT_THRESHOLD_CM) -> np.ndarray:
    if conditioned_disagreement(pred_cond, pred_uncond, mask) < threshold:
        return pred_static
    return pred_cond
