import numpy as np

_CV_TO_GL = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def opencv_to_opengl(points: np.ndarray) -> np.ndarray:
    return points @ _CV_TO_GL.T


def opengl_to_opencv(points: np.ndarray) -> np.ndarray:
    return points @ _CV_TO_GL.T


MATHNET_TO_CV = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float32
)


def mathnet_to_opencv(points: np.ndarray) -> np.ndarray:
    return points @ MATHNET_TO_CV.T


def opencv_to_mathnet(points: np.ndarray) -> np.ndarray:
    return points @ MATHNET_TO_CV


def opencv_camera_pose_for_pyrender() -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = _CV_TO_GL
    return pose


def normalize_translation(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def denormalize_translation(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def query_relative_stats(query_xyz_t0: np.ndarray):
    # centre and scale come from the t=0 query points alone, so an unseen scene needs no
    # dataset statistics it cannot have. Scheme A (per-dataset mean/std) leaves EgoExo4D at
    # 2.6-3.9x the training spread with 26% of coords past +/-3; this holds every dataset,
    # including one never trained on, inside the same support.
    mean = query_xyz_t0.mean(axis=-2, keepdims=True)
    # RMS distance to the centroid: a rotation-invariant scale. Averaging the per-axis standard
    # deviations instead is not invariant, and would turn a rigid rotation of the object into a
    # shear once normalized.
    centred = query_xyz_t0 - mean
    scale = np.sqrt((centred ** 2).sum(axis=-1, keepdims=True).mean(axis=-2, keepdims=True)) + 1e-6
    return mean, scale


def normalize_query_relative(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def denormalize_query_relative(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # metric motion is recoverable because the query points are an input, so their spread is
    # known at inference time
    return x * scale + mean
