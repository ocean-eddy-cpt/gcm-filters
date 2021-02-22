import numpy as np
import pytest

from gcm_filters.kernels import ALL_KERNELS, GridType, required_grid_vars


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
    if grid_type == GridType.IRREGULAR_CARTESIAN_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        extra_kwargs["wet_mask"] = mask_data
        grid_data = np.ones_like(data)
        extra_kwargs["dxw"] = grid_data
        extra_kwargs["dyw"] = grid_data
        extra_kwargs["dxs"] = grid_data
        extra_kwargs["dys"] = grid_data
        extra_kwargs["area"] = grid_data
        extra_kwargs["kappa_s"] = grid_data
        extra_kwargs["kappa_w"] = grid_data
    return grid_type, data, extra_kwargs


def test_conservation(grid_type_field_and_extra_kwargs):
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    res = laplacian(data)
    np.testing.assert_allclose(res.sum(), 0.0, atol=1e-12)


def test_required_grid_vars(grid_type_field_and_extra_kwargs):
    grid_type, _, extra_kwargs = grid_type_field_and_extra_kwargs
    grid_vars = required_grid_vars(grid_type)
    assert set(grid_vars) == set(extra_kwargs)
