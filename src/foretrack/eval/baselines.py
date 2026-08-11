import numpy as np


def static(query_xyz_t0: np.ndarray, t: int) -> np.ndarray:
    return np.repeat(query_xyz_t0[None], t, axis=0)


def constant_velocity(xyz_t0: np.ndarray, xyz_t1: np.ndarray, t: int) -> np.ndarray:
    """extrapolate the velocity observed between the first two GT frames. oracle-ish: unlike
    the actual model, this peeks at real frame-1 motion rather than conditioning on a single
    image, so it isn't a fair comparison -- label it as such in any results table."""
    velocity = xyz_t1 - xyz_t0
    steps = np.arange(t, dtype=xyz_t0.dtype).reshape(t, *([1] * xyz_t0.ndim))
    return xyz_t0[None] + steps * velocity[None]
