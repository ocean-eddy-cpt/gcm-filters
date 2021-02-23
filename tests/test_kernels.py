import copy
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
    if grid_type == GridType.POP_SIMPLE_TRIPOLAR_T_GRID:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        mask_data[0, :] = 0  #  Antarctica
        extra_kwargs["wet_mask"] = mask_data
    if grid_type == GridType.POP_TRIPOLAR_T_GRID:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        mask_data[0, :] = 0  #  Antarctica
        extra_kwargs["wet_mask"] = mask_data
        grid_data = np.ones_like(data)
        extra_kwargs["dxe"] = grid_data
        extra_kwargs["dye"] = grid_data
        extra_kwargs["dxn"] = grid_data
        extra_kwargs["dyn"] = grid_data
        extra_kwargs["tarea"] = grid_data
    return grid_type, data, extra_kwargs


def test_conservation(grid_type_field_and_extra_kwargs):
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    diffused = laplacian(data)
    res = diffused.sum()  # integrate over full domain
    np.testing.assert_allclose(res, 0.0, atol=1e-12)


def test_required_grid_vars(grid_type_field_and_extra_kwargs):
    grid_type, _, extra_kwargs = grid_type_field_and_extra_kwargs
    grid_vars = required_grid_vars(grid_type)
    assert set(grid_vars) == set(extra_kwargs)


################## Tripolar grid tests ##############################################
tripolar_grids = [
    member for name, member in GridType.__members__.items() if name.startswith("POP")
]


def test_for_antarctica(grid_type_field_and_extra_kwargs):
    """This test checks that we get an error if southernmost row of wet_mask has entry not equal to zero. """
    grid_type, _, extra_kwargs = grid_type_field_and_extra_kwargs

    if grid_type in tripolar_grids:
        nx = np.shape(extra_kwargs["wet_mask"])[1]
        random_loc = np.random.randint(0, nx)
        bad_kwargs = copy.deepcopy(extra_kwargs)
        bad_kwargs["wet_mask"][0, random_loc] = 1

        LaplacianClass = ALL_KERNELS[grid_type]
        with pytest.raises(AssertionError, match=r"Wet mask requires .*"):
            laplacian = LaplacianClass(**bad_kwargs)


def test_tripolar_exchanges(grid_type_field_and_extra_kwargs):
    """This test checks that Laplacian exchanges across northern boundary seam line of tripolar grid are correct. """
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)

    delta_fct = np.zeros_like(data)
    nx = np.shape(delta_fct)[1]

    if grid_type in tripolar_grids:
        random_loc = np.random.randint(0, nx)
        delta_fct[-1, random_loc] = 1  # deploy mass at northern boundary
        diffused = laplacian(delta_fct)
        # check that delta function gets diffused isotropically across northern boundary
        # this would need to be replaced once we provide irregular grid data in fixture
        np.testing.assert_allclose(
            diffused[-2, random_loc], diffused[-1, nx - random_loc - 1], atol=1e-12
        )
