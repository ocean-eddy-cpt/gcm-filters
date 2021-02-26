import numpy as np
import pytest
import xarray as xr

from gcm_filters import Filter, FilterShape, GridType
from gcm_filters.filter import FilterSpec


def _check_equal_filter_spec(spec1, spec2):
    assert spec1.n_lap_steps == spec2.n_lap_steps
    assert spec1.n_bih_steps == spec2.n_bih_steps
    np.testing.assert_allclose(spec1.s_l, spec2.s_l)
    np.testing.assert_allclose(spec1.s_b, spec2.s_b)


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
                n_steps=4,
            ),
            FilterSpec(
                n_lap_steps=4,
                s_l=[2.56046256, 8.47349198, 15.22333438, 19.7392088],
                n_bih_steps=0,
                s_b=[],
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
                n_lap_steps=1,
                s_l=[9.8696044],
                n_bih_steps=4,
                s_b=[
                    -0.74638043 - 1.24167777j,
                    3.06062496 - 3.94612205j,
                    7.80242999 - 3.18038659j,
                    9.81491354 - 0.44874939j,
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
    da = xr.DataArray(data, dims=["y", "x"])

    grid_vars = {}
    mask_data = np.ones_like(data)
    mask_data[: (ny // 2), : (nx // 2)] = 0

    def make_yx_da(np_arr):
        return xr.DataArray(np_arr, dims=["y", "x"])

    da_mask = make_yx_da(mask_data)

    if grid_type == GridType.CARTESIAN_WITH_LAND:
        grid_vars = {"wet_mask": da_mask}
    if grid_type == GridType.IRREGULAR_CARTESIAN_WITH_LAND:
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
    if (grid_type == GridType.MOM5U) or (grid_type == GridType.MOM5T):
        # Add one to random dxs due to dxmin = 1.
        dxu, dyu = np.meshgrid(1.0 + np.random.rand(nx), 1.0 + np.random.rand(ny))
        dxt, dyt = dxu, dyu
        dxu = make_yx_da(dxu)
        dyu = make_yx_da(dyu)
        dxt = make_yx_da(dxt)
        dyt = make_yx_da(dyt)
        area_u = dxu * dyu
        area_t = dxt * dyt
        grid_vars["wet"] = da_mask
        grid_vars["dxu"] = dxu
        grid_vars["dyu"] = dyu
        grid_vars["dxt"] = dxt
        grid_vars["dyt"] = dyt
    if grid_type == GridType.MOM5U:
        grid_vars["area_u"] = area_u
    if grid_type == GridType.MOM5T:
        grid_vars["area_t"] = area_t

    return grid_type, da, grid_vars


@pytest.mark.parametrize(
    "filter_args",
    [dict(filter_scale=1.0, dx_min=1.0, n_steps=10, filter_shape=FilterShape.TAPER)],
)
def test_filter(grid_type_and_input_ds, filter_args):
    grid_type, da, grid_vars = grid_type_and_input_ds
    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)
    filtered = filter.apply(da, dims=["y", "x"])

    # check conservation
    # this would need to be replaced by a proper area-weighted integral
    xr.testing.assert_allclose(da.sum(), filtered.sum())

    # check variance reduction
    assert (filtered ** 2).sum() < (da ** 2).sum()

    # check that we get an error if we leave out any required grid_vars
    for gv in grid_vars:
        grid_vars_missing = {k: v for k, v in grid_vars.items() if k != gv}
        with pytest.raises(ValueError, match=r"Provided `grid_vars` .*"):
            filter = Filter(
                grid_type=grid_type, grid_vars=grid_vars_missing, **filter_args
            )
