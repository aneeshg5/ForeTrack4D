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
