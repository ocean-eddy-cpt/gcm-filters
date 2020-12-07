import numpy as np
import pytest

from gcm_filters.kernels import simple_diffusion_kernel


@pytest.fixture
def numpy_array_2d():
    ny, nx = (128, 256)
    data = np.random.rand(ny, nx)
    return data


@pytest.mark.parametrize(
    "kernel_function",
    [
        simple_diffusion_kernel
    ]
)
def test_conservation(numpy_array_2d, kernel_function):
    res = kernel_function(numpy_array_2d)
    np.testing.assert_allclose(res.sum(), 0.0, atol=1e-12)
