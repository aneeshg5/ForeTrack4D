import numpy as np

from foretrack.data.transforms import (
    denormalize_translation,
    mathnet_to_opencv,
    normalize_translation,
    opencv_to_mathnet,
    opencv_to_opengl,
    opengl_to_opencv,
)


def test_normalize_round_trip():
    x = np.random.randn(10, 3).astype(np.float32)
    mean = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    std = np.array([1.5, 2.0, 0.5], dtype=np.float32)
    x_norm = normalize_translation(x, mean, std)
    x_back = denormalize_translation(x_norm, mean, std)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


def test_convention_round_trip():
    x = np.random.randn(10, 3).astype(np.float32)
    np.testing.assert_allclose(opengl_to_opencv(opencv_to_opengl(x)), x, atol=1e-5)


def test_mathnet_round_trip():
    x = np.random.randn(10, 3).astype(np.float32)
    np.testing.assert_allclose(opencv_to_mathnet(mathnet_to_opencv(x)), x, atol=1e-5)


def test_mathnet_forward_axis_maps_to_opencv_z():
    forward = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    out = mathnet_to_opencv(forward)
    np.testing.assert_allclose(out, [[0.0, 0.0, 1.0]], atol=1e-6)
