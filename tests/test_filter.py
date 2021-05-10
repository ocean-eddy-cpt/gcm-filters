import copy

import numpy as np
import pytest
import xarray as xr

from gcm_filters import Filter, FilterShape, GridType
from gcm_filters.filter import FilterSpec


def _check_equal_filter_spec(spec1, spec2):
    assert spec1.n_steps_total == spec2.n_steps_total
    np.testing.assert_allclose(spec1.s, spec2.s)
    assert (spec1.is_laplacian == spec2.is_laplacian).all()
    assert spec1.s_max == spec2.s_max
    np.testing.assert_allclose(spec1.p, spec2.p, rtol=1e-07, atol=1e-07)


# These values were just hard copied from my dev environment.
# All they do is check that the results match what I got when I ran the code.
# They do NOT assure that the filter spec is correct.
@pytest.mark.parametrize(
    "filter_args, expected_filter_spec",
    [
        (
            dict(
                filter_scale=10.0,
                dx_min=1.0,
                filter_shape=FilterShape.GAUSSIAN,
                transition_width=np.pi,
                ndim=2,
            ),
            FilterSpec(
                n_steps_total=10,
                s=[
                    8.0 + 0.0j,
                    3.42929331 + 0.0j,
                    7.71587822 + 0.0j,
                    2.41473596 + 0.0j,
                    7.18021542 + 0.0j,
                    1.60752541 + 0.0j,
                    6.42502377 + 0.0j,
                    0.81114415 - 0.55260985j,
                    5.50381534 + 0.0j,
                    4.48146765 + 0.0j,
                ],
                is_laplacian=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                ],
                s_max=8.0,
                p=[
                    0.09887381,
                    -0.19152534,
                    0.1748326,
                    -0.14975371,
                    0.12112337,
                    -0.09198484,
                    0.0662522,
                    -0.04479323,
                    0.02895827,
                    -0.0173953,
                    0.00995974,
                    -0.00454758,
                ],
            ),
        ),
        (
            dict(
                filter_scale=2.0,
                dx_min=1.0,
                filter_shape=FilterShape.TAPER,
                transition_width=np.pi,
                ndim=1,
            ),
            FilterSpec(
                n_steps_total=3,
                s=[
                    5.23887374 - 1.09644141j,
                    -0.76856043 - 1.32116962j,
                    3.00058907 - 2.95588288j,
                ],
                is_laplacian=[False, False, False],
                s_max=4.0,
                p=[
                    0.83380304,
                    -0.23622724,
                    -0.06554041,
                    0.01593978,
                    0.00481014,
                    -0.00495532,
                    0.00168445,
                ],
            ),
        ),
    ],
)
def test_filter_spec(filter_args, expected_filter_spec):
    """This test just verifies that the filter specification looks as expected."""
    filter = Filter(**filter_args)
    _check_equal_filter_spec(filter.filter_spec, expected_filter_spec)
    # TODO: check other properties of filter_spec?


@pytest.fixture(scope="module", params=list(GridType))
def grid_type_and_input_ds(request):
    grid_type = request.param

    ny, nx = (128, 256)
    data = np.random.rand(ny, nx)

    grid_vars = {}

    if grid_type == GridType.REGULAR_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        da_mask = xr.DataArray(mask_data, dims=["y", "x"])
        grid_vars = {"wet_mask": da_mask}
    if grid_type == GridType.IRREGULAR_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        da_mask = xr.DataArray(mask_data, dims=["y", "x"])
        grid_data = np.ones_like(data)
        da_grid = xr.DataArray(grid_data, dims=["y", "x"])
        grid_vars = {
            "wet_mask": da_mask,
            "dxw": da_grid,
            "dyw": da_grid,
            "dxs": da_grid,
            "dys": da_grid,
            "area": da_grid,
        }
    if grid_type == GridType.TRIPOLAR_REGULAR_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        mask_data[0, :] = 0  #  Antarctica
        da_mask = xr.DataArray(mask_data, dims=["y", "x"])
        grid_vars = {"wet_mask": da_mask}
    if grid_type == GridType.TRIPOLAR_POP_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        mask_data[0, :] = 0  #  Antarctica
        da_mask = xr.DataArray(mask_data, dims=["y", "x"])
        grid_data = np.ones_like(data)
        da_grid = xr.DataArray(grid_data, dims=["y", "x"])
        grid_vars = {
            "wet_mask": da_mask,
            "dxe": da_grid,
            "dye": da_grid,
            "dxn": da_grid,
            "dyn": da_grid,
            "tarea": da_grid,
        }
    da = xr.DataArray(data, dims=["y", "x"])

    return grid_type, da, grid_vars


@pytest.mark.parametrize(
    "filter_args",
    [dict(filter_scale=3.0, dx_min=1.0, n_steps=0, filter_shape=FilterShape.GAUSSIAN)],
)
def test_filter(grid_type_and_input_ds, filter_args):
    grid_type, da, grid_vars = grid_type_and_input_ds
    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)
    filter.plot_shape()
    filtered = filter.apply(da, dims=["y", "x"])

    # check conservation
    # this would need to be replaced by a proper area-weighted integral
    da_sum = da.sum()
    filtered_sum = filtered.sum()

    xr.testing.assert_allclose(da_sum, filtered_sum)

    # check variance reduction
    assert (filtered ** 2).sum() < (da ** 2).sum()

    # check that we get an error if we leave out any required grid_vars
    for gv in grid_vars:
        grid_vars_missing = {k: v for k, v in grid_vars.items() if k != gv}
        with pytest.raises(ValueError, match=r"Provided `grid_vars` .*"):
            filter = Filter(
                grid_type=grid_type, grid_vars=grid_vars_missing, **filter_args
            )

    bad_filter_args = copy.deepcopy(filter_args)
    # check that we get an error if ndim > 2 and n_steps = 0
    bad_filter_args["ndim"] = 3
    bad_filter_args["n_steps"] = 0
    with pytest.raises(ValueError, match=r"When ndim > 2, you .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    # check that we get a warning if n_steps < n_steps_default
    bad_filter_args["ndim"] = 2
    bad_filter_args["n_steps"] = 3
    with pytest.warns(UserWarning, match=r"Warning: You have set n_steps .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    # check that we get a warning if numerical instability possible
    bad_filter_args["n_steps"] = 0
    bad_filter_args["filter_scale"] = 1000
    with pytest.warns(UserWarning, match=r"Warning: Filter scale much larger .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
