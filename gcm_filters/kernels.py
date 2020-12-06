"""
Core smoothing routines that operate on 2D arrays.
"""

import functools


try:
    from cupy import get_array_module as _get_array_module
except ImportError:
    import numpy as np

    def _get_array_module(*args):
        return np


def _try_to_use_cupy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        np = _get_array_module(*args)
        return func(*args, **kwargs)

    return wrapper


@_try_to_use_cupy
def simple_diffusion_kernel(phi):
    """Classic diffusion stencil for regular grid.

    Parameters
    ----------
    phi : array_like

    Returns
    -------
    array_like
        The diffusive tendency for `phi`
    """
    return (
        -4 * phi
        + np.roll(phi, -1, axis=-1)
        + np.roll(phi, 1, axis=-1)
        + np.roll(phi, -1, axis=-2)
        + np.roll(phi, 1, axis=-2)
    )
