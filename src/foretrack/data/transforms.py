import numpy as np

# opencv convention: x right, y down, z forward. all track data is stored this way.
# flip here, once, if a source (e.g. pyrender/opengl) uses y up, z backward.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def opencv_to_opengl(points: np.ndarray) -> np.ndarray:
    return points @ _CV_TO_GL.T


def opengl_to_opencv(points: np.ndarray) -> np.ndarray:
    return points @ _CV_TO_GL.T


# MathNet.Spatial.Euclidean basis used throughout Microsoft.Psi.MixedReality / HoloLens
# captures (HoloAssist), confirmed from Psi.Calibration.ICameraIntrinsics.GetCameraSpacePosition's
# own docstring ("Point in 3D space, assuming MathNet basis (Forward=X, Left=Y, Up=Z)"): x =
# forward, y = left, z = up. Mapping to our opencv world (x right, y down, z forward):
# opencv.x = -mathnet.y, opencv.y = -mathnet.z, opencv.z = mathnet.x. Exposed publicly (not
# just via the point-transform functions below) since data/holoassist.py's pose composition
# needs the raw rotation matrix, not just point conversion.
MATHNET_TO_CV = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float32
)


def mathnet_to_opencv(points: np.ndarray) -> np.ndarray:
    return points @ MATHNET_TO_CV.T


def opencv_to_mathnet(points: np.ndarray) -> np.ndarray:
    return points @ MATHNET_TO_CV


def opencv_camera_pose_for_pyrender() -> np.ndarray:
    """camera-to-world pose (pyrender/OpenGL convention) for a camera sitting
    at our world origin. our world is defined as the opencv camera frame
    (decision 1), so this is always the same constant flip, never per-frame."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = _CV_TO_GL
    return pose


def normalize_translation(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def denormalize_translation(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean
