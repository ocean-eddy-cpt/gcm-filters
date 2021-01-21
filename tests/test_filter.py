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
            dict(filter_scale=10.0, dx_min=1.0, n_steps=4),
            FilterSpec(
                n_lap_steps=4,
                s_l=[2.36715983, 8.36124821, 15.17931706, 19.7392088],
                n_bih_steps=0,
                s_b=[],
            ),
        ),
        (
            dict(
                filter_scale=1.0, dx_min=1.0, n_steps=10, filter_shape=FilterShape.TAPER
            ),
            FilterSpec(
                n_lap_steps=6,
                s_l=[
                    9.87341331,
                    12.66526236,
                    15.23856752,
                    17.38217753,
                    18.91899844,
                    19.7392088,
                ],
                n_bih_steps=2,
                s_b=[-1.83974928 - 2.24294603j, 1.21758518 - 8.29775049j],
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

    if grid_type == GridType.CARTESIAN_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        da_mask = xr.DataArray(mask_data, dims=["y", "x"])
        grid_vars = {"wet_mask": da_mask}

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
