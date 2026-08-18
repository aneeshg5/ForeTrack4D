import numpy as np


def static(query_xyz_t0: np.ndarray, t: int) -> np.ndarray:
    return np.repeat(query_xyz_t0[None], t, axis=0)


def constant_velocity(xyz_t0: np.ndarray, xyz_t1: np.ndarray, t: int) -> np.ndarray:
    velocity = xyz_t1 - xyz_t0
    steps = np.arange(t, dtype=xyz_t0.dtype).reshape(t, *([1] * xyz_t0.ndim))
    return xyz_t0[None] + steps * velocity[None]
