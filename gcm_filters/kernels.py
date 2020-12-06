import functools


try:
    from cupy import get_array_module
except ImportError:
    import numpy as np

    def get_array_module(*args):
        return np


def try_to_use_cupy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        np = get_array_module(*args)
        return func(*args, **kwargs)
    return wrapper


@try_to_use_cupy
def simple_diffusion_kernel(phi):
    return (
        -4 * phi
        + np.roll(phi, -1, axis=-1)
        + np.roll(phi, 1, axis=-1)
        + np.roll(phi, -1, axis=-2)
        + np.roll(phi, 1, axis=-2)
    )
