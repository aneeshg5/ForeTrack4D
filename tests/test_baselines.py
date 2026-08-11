import numpy as np

from foretrack.eval.baselines import constant_velocity, static


def test_static_baseline_shape():
    query_xyz_t0 = np.random.randn(64, 3).astype(np.float32)
    out = static(query_xyz_t0, t=128)
    assert out.shape == (128, 64, 3)
    np.testing.assert_allclose(out[0], query_xyz_t0)
    np.testing.assert_allclose(out[-1], query_xyz_t0)


def test_constant_velocity_baseline():
    xyz_t0 = np.zeros((64, 3), dtype=np.float32)
    velocity = np.full((64, 3), 0.1, dtype=np.float32)
    xyz_t1 = xyz_t0 + velocity
    out = constant_velocity(xyz_t0, xyz_t1, t=10)
    assert out.shape == (10, 64, 3)
    np.testing.assert_allclose(out[0], xyz_t0)
    np.testing.assert_allclose(out[1], xyz_t1)
    np.testing.assert_allclose(out[9], xyz_t0 + 9 * velocity)
