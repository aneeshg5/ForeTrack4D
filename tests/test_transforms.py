import numpy as np

from foretrack.data.transforms import (
    denormalize_query_relative,
    denormalize_translation,
    mathnet_to_opencv,
    normalize_query_relative,
    normalize_translation,
    opencv_to_mathnet,
    opencv_to_opengl,
    opengl_to_opencv,
    query_relative_stats,
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


def test_query_relative_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(64, 3)) * 0.2 + np.array([0.1, -0.3, 0.8])
    mean, scale = query_relative_stats(q)
    x = rng.normal(size=(128, 64, 3)) * 0.2 + np.array([0.1, -0.3, 0.8])
    out = denormalize_query_relative(normalize_query_relative(x, mean, scale), mean, scale)
    np.testing.assert_allclose(out, x, rtol=1e-5, atol=1e-6)


def test_query_relative_is_scale_invariant():
    # the whole point: two scenes differing only by a similarity transform must normalize to
    # the same coordinates, which is what puts unseen datasets inside the training support
    rng = np.random.default_rng(1)
    q = rng.normal(size=(64, 3))
    mean_a, scale_a = query_relative_stats(q)
    q2 = q * 7.0 + np.array([3.0, -2.0, 11.0])
    mean_b, scale_b = query_relative_stats(q2)
    np.testing.assert_allclose(
        normalize_query_relative(q, mean_a, scale_a),
        normalize_query_relative(q2, mean_b, scale_b),
        rtol=1e-4, atol=1e-5,
    )


def test_query_relative_produces_order_one_spread():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(64, 3)) * np.array([0.05, 0.4, 2.0]) + 1.0
    mean, scale = query_relative_stats(q)
    # scaling is isotropic, so an elongated object stays elongated rather than being forced to
    # unit spread on every axis -- the guarantee is order-1 magnitude, not exact unit variance
    assert 0.5 < float(normalize_query_relative(q, mean, scale).std()) < 2.0
    assert scale.shape[-1] == 1


def test_query_relative_preserves_rigid_motion():
    # per-axis scaling would turn a rotation into a shear; motion here is rigid, so the
    # normalization has to commute with rotation for the target to stay physically meaningful
    rng = np.random.default_rng(3)
    q = rng.normal(size=(64, 3)) * np.array([0.05, 0.4, 2.0])
    theta = 0.7
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    mean, scale = query_relative_stats(q)
    norm_then_rot = normalize_query_relative(q, mean, scale) @ rot.T
    qr = q @ rot.T
    mean_r, scale_r = query_relative_stats(qr)
    np.testing.assert_allclose(norm_then_rot, normalize_query_relative(qr, mean_r, scale_r), rtol=1e-5, atol=1e-6)
