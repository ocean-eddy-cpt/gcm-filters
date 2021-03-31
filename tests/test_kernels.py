import numpy as np
import pytest

from gcm_filters.kernels import ALL_KERNELS, GridType, required_grid_vars


@pytest.fixture(scope="module", params=list(GridType))
def grid_type_field_and_extra_kwargs(request):
    grid_type = request.param

    ny, nx = (128, 256)
    data = np.random.rand(ny, nx)
    mask_data = np.ones_like(data)
    mask_data[: (ny // 2), : (nx // 2)] = 0

    extra_kwargs = {}
    if grid_type == GridType.CARTESIAN_WITH_LAND:
        extra_kwargs["wet_mask"] = mask_data
    if grid_type == GridType.IRREGULAR_CARTESIAN_WITH_LAND:
        extra_kwargs["wet_mask"] = mask_data
        grid_data = np.ones_like(data)
        extra_kwargs["dxw"] = grid_data
        extra_kwargs["dyw"] = grid_data
        extra_kwargs["dxs"] = grid_data
        extra_kwargs["dys"] = grid_data
        extra_kwargs["area"] = grid_data
    if (grid_type == GridType.MOM5U) or (grid_type == GridType.MOM5T):
        dxu, dyu = np.meshgrid(np.random.rand(nx), np.random.rand(ny))
        dxt, dyt = dxu, dyu
        extra_kwargs["wet"] = mask_data
        extra_kwargs["dxu"] = dxu
        extra_kwargs["dyu"] = dyu
        extra_kwargs["dxt"] = dxt
        extra_kwargs["dyt"] = dyt
    if grid_type == GridType.MOM5U:
        extra_kwargs["area_u"] = dxu * dyu
    if grid_type == GridType.MOM5T:
        extra_kwargs["area_t"] = dxt * dyt
    return grid_type, data, extra_kwargs


def test_conservation(grid_type_field_and_extra_kwargs):
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    areas = 1.0
    for k, v in extra_kwargs.items():
        if k.startswith("area"):
            areas = v
            break
    res = laplacian(data) * areas
    np.testing.assert_allclose(res.sum(), 0.0, atol=1e-12)


def test_required_grid_vars(grid_type_field_and_extra_kwargs):
    grid_type, _, extra_kwargs = grid_type_field_and_extra_kwargs
    grid_vars = required_grid_vars(grid_type)
    assert set(grid_vars) == set(extra_kwargs)
