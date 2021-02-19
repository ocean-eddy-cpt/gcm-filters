import enum
import copy

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
    if grid_type == GridType.POP_SIMPLE_TRIPOLAR_GRID:
        mask_data = np.ones_like(data)
        mask_data[3 * (ny // 4) :, (nx // 4) : (nx // 2)] = 0
        mask_data[0, :] = 0  #  Antarctica
        extra_kwargs["wet_mask"] = mask_data
        extra_kwargs["position"] = "T"
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


################## Tripolar grid tests ##############################################
def fold_northern_boundary(ufield, nx, invert):
    folded = ufield[-1, :]  # grab northernmost row
    folded = folded[::-1]  # mirror it
    if invert:
        folded = -folded
    folded = np.roll(folded, -1)  # shift by 1 cell to the left
    ufield[-1, 0 : nx // 2] = folded[0 : nx // 2]
    ufield[-1, nx // 2 - 1] = 0  # pivot point (first Arctic singularity) is on land
    ufield[-2, nx // 2 - 1] = 0  # pivot point (first Arctic singularity) is on land
    ufield[-1, -1] = 0  # second Arctic singularity is on land too
    # ufield[-2,-1] = 0 # second Arctic singularity is on land too
    return ufield


@pytest.fixture(
    scope="module",
    params=[("POP_SIMPLE_TRIPOLAR_GRID", "T"), ("POP_SIMPLE_TRIPOLAR_GRID", "U")],
)
def tripolar_grid_field_and_extra_kwargs(request):
    grid_type = GridType.__members__[request.param[0]]
    position = request.param[1]

    ny, nx = (128, 256)
    data = np.random.rand(ny, nx)

    extra_kwargs = {}
    extra_kwargs["position"] = position

    mask_data = np.ones_like(data)
    # mask_data[3 * (ny // 4) :, (nx // 4) : (nx // 2)] = 0
    mask_data[: (ny // 2), : (nx // 2)] = 0
    mask_data[0, :] = 0  #  Antarctica

    if position == "U":
        mask_data = fold_northern_boundary(mask_data, nx, invert=False)
        data = fold_northern_boundary(data, nx, invert=True)

    extra_kwargs["wet_mask"] = mask_data

    return grid_type, data, extra_kwargs


def test_for_antarctica(tripolar_grid_field_and_extra_kwargs):
    """This test checks that we get an error if southernmost row of wet_mask has entry not equal to zero. """
    grid_type, _, extra_kwargs = tripolar_grid_field_and_extra_kwargs

    nx = np.shape(extra_kwargs["wet_mask"])[1]
    random_loc = np.random.randint(0, nx)
    bad_kwargs = copy.deepcopy(extra_kwargs)
    bad_kwargs["wet_mask"][0, random_loc] = 1

    LaplacianClass = ALL_KERNELS[grid_type]
    with pytest.raises(AssertionError, match=r"Wet mask requires .*"):
        laplacian = LaplacianClass(**bad_kwargs)


def test_northern_boundary_fold(tripolar_grid_field_and_extra_kwargs):
    """This test checks that wet get an error if uppermost row of wet_mask or data does not fold onto itself. """
    grid_type, data, extra_kwargs = tripolar_grid_field_and_extra_kwargs
    if extra_kwargs["position"] == "U":
        LaplacianClass = ALL_KERNELS[grid_type]
        laplacian = LaplacianClass(**extra_kwargs)

        nx = np.shape(extra_kwargs["wet_mask"])[1]
        bad_data = data.copy()
        bad_data[-1, : nx // 2] = 1
        bad_data[-1, nx // 2 :] = 0
        with pytest.raises(AssertionError, match=r"Uppermost row of input field .*"):
            diffused = laplacian(bad_data)

        bad_kwargs = copy.deepcopy(extra_kwargs)
        bad_kwargs["wet_mask"][-1, : nx // 2] = 1
        bad_kwargs["wet_mask"][-1, nx // 2 :] = 0
        with pytest.raises(AssertionError, match=r"Uppermost row of wet mask .*"):
            laplacian = LaplacianClass(**bad_kwargs)


def test_tripolar_exchanges(tripolar_grid_field_and_extra_kwargs):
    """This test checks that Laplacian exchanges across northern boundary seam line of tripolar grid are correct. """
    grid_type, data, extra_kwargs = tripolar_grid_field_and_extra_kwargs
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)

    delta_fct = np.zeros_like(data)
    nx = np.shape(delta_fct)[1]

    position = extra_kwargs["position"]
    if position == "T":
        random_loc = np.random.randint(0, nx)
        delta_fct[-1, random_loc] = 1  # deploy mass at northern boundary
        diffused = laplacian(delta_fct)
        # check that delta function gets diffused isotropically across northern boundary
        np.testing.assert_allclose(
            diffused[-2, random_loc], diffused[-1, nx - random_loc - 1], atol=1e-12
        )

    elif position == "U":
        random_loc = np.random.randint(0, nx // 2 - 1)
        delta_fct[
            -2, random_loc
        ] = 1  # deploy mass just below northern boundary on the left side
        diffused = laplacian(delta_fct)
        # check that delta function gets diffused isotropically across northern boundary
        np.testing.assert_allclose(
            diffused[-3, random_loc], -diffused[-1, nx - random_loc - 2], atol=1e-12
        )
        # check that northernmost row of diffused data folds onto itself
        diffused = laplacian(data)
        folded = diffused[-1, :]  # grab northernmost row
        folded = -folded[::-1]  # mirror and invert it
        folded = np.roll(folded, -1)  # shift by 1 cell to the left
        np.testing.assert_allclose(diffused[-1, :], folded, atol=1e-12)
