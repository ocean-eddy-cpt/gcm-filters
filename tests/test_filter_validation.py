import os

import numpy as np
import pytest
import xarray as xr
import zarr

from pytest_lazyfixture import lazy_fixture

from gcm_filters import Filter, FilterShape


# https://stackoverflow.com/questions/66970626/pytest-skip-test-condition-depending-on-environment
def requires_env(varname, value):
    env_value = os.environ.get(varname)
    return pytest.mark.skipif(
        not env_value == value,
        reason=f"Test skipped unless environment variable {varname}=={value}",
    )


all_grids_data_and_input_ds = [
    lazy_fixture("grid_type_and_input_ds"),
    lazy_fixture("vector_grid_type_and_input_ds"),
]


def _get_fname(grid_data_and_input_ds):
    grid_type, _, grid_vars = grid_data_and_input_ds
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, "test_data_filter", f"{grid_type.name}.zarr")


def _get_results(grid_data_and_input_ds, filter_args):
    grid_type, data, grid_vars = grid_data_and_input_ds

    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)

    if len(data) == 2:  # vector grid
        (filtered_u, filtered_v) = filter.apply_to_vector(*data, dims=["y", "x"])
        filtered = np.stack([filtered_u.data, filtered_v.data])
    else:
        filtered = filter.apply(data, dims=["y", "x"])
        filtered = filtered.data

    filtered = filtered.astype("f4")  # use single precision to save space
    return filtered


@pytest.mark.parametrize(
    "filter_args",
    [
        dict(
            filter_scale=8.0,
            dx_min=1.0,
            n_steps=0,
            filter_shape=FilterShape.GAUSSIAN,
        )
    ],
)
# this test will not be run by default
# to run it and overwrite the test data, invoke pytest with an environment variable as follows
# $ GCM_FILTERS_OVERWRITE_TEST_DATA=1 pytest tests/test_filter_validation.py
@requires_env("GCM_FILTERS_OVERWRITE_TEST_DATA", "1")
@pytest.mark.parametrize("grid_data_and_input_ds", all_grids_data_and_input_ds)
def test_save_results(grid_data_and_input_ds, filter_args):
    res = _get_results(grid_data_and_input_ds, filter_args)
    fname = _get_fname(grid_data_and_input_ds)
    z_arr = zarr.open(
        fname, mode="w", shape=res.shape, chunks=(-1, -1), dtype=res.dtype
    )
    z_arr[:] = res


@pytest.mark.parametrize(
    "filter_args",
    [
        dict(
            filter_scale=8.0,
            dx_min=1.0,
            n_steps=0,
            filter_shape=FilterShape.GAUSSIAN,
        )
    ],
)
@pytest.mark.parametrize("grid_data_and_input_ds", all_grids_data_and_input_ds)
def test_check_results(grid_data_and_input_ds, filter_args):
    res = _get_results(grid_data_and_input_ds, filter_args)
    fname = _get_fname(grid_data_and_input_ds)
    z_arr = zarr.open(
        fname, mode="r", shape=res.shape, chunks=(-1, -1), dtype=res.dtype
    )
    np.testing.assert_allclose(z_arr[:], res)
