import enum

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
    if grid_type == GridType.POP_TRIPOLAR_GRID:
        mask_data = np.ones_like(data)
        mask_data[3 * (ny // 4) :, (nx // 4) : (nx // 2)] = 0
        mask_data[
            0, :
        ] = 0  #  we need Antarctic land to disable periodic boundary condition in y-direction
        extra_kwargs["wet_mask"] = mask_data
    return grid_type, data, extra_kwargs


# TODO: implement check (elsewhere) that makes sure that sourthernmost row of user-specified land mask is land


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


@pytest.fixture(
    scope="module",
    params=[
        member
        for name, member in GridType.__members__.items()
        if name == "POP_TRIPOLAR_GRID"
    ],
)
def grid_type_field_and_extra_kwargs_and_delta_function(request):
    grid_type = request.param
    ny, nx = (128, 256)
    data = np.zeros((ny, nx))
    delta_pos = nx // 5
    data[-1, delta_pos] = 1  # delta function at northern boundary
    extra_kwargs = {}
    mask_data = np.ones_like(data)
    extra_kwargs["wet_mask"] = mask_data

    return grid_type, data, extra_kwargs, nx, delta_pos


def test_tripolar_exchanges(grid_type_field_and_extra_kwargs_and_delta_function):
    """This test checks whether a delta function gets diffused isotropically across the northern boundary seam line of a tripolar grid. """

    (
        grid_type,
        data,
        extra_kwargs,
        nx,
        delta_pos,
    ) = grid_type_field_and_extra_kwargs_and_delta_function
    print(grid_type)
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    diffused = laplacian(data)

    np.testing.assert_allclose(
        diffused[-2, delta_pos], diffused[-1, nx - delta_pos - 1], atol=1e-12
    )
    # this test is only appropriate for tracer fields.
