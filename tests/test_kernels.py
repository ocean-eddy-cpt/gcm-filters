import numpy as np
import pytest

from gcm_filters.kernels import ALL_KERNELS, GridType


@pytest.fixture(scope="module", params=list(GridType))
def grid_type_field_and_extra_kwargs(request):
    grid_type = request.param

    ny, nx = (128, 256)
    data = np.random.rand(ny, nx)

    extra_kwargs = {}
    if grid_type == GridType.CARTESIAN_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        extra_kwargs["wet_mask"] = mask_data
    return grid_type, data, extra_kwargs


def test_conservation(grid_type_field_and_extra_kwargs):
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    res = laplacian(data)
    np.testing.assert_allclose(res.sum(), 0.0, atol=1e-12)
