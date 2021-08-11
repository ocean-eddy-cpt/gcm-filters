import os

import numpy as np
import pytest
import xarray as xr
import zarr

from pytest_lazyfixture import lazy_fixture

from gcm_filters.kernels import ALL_KERNELS


# https://stackoverflow.com/questions/66970626/pytest-skip-test-condition-depending-on-environment
def requires_env(varname, value):
    env_value = os.environ.get(varname)
    return pytest.mark.skipif(
        not env_value == value,
        reason=f"Test skipped unless environment variable {varname}=={value}",
    )


all_grids_data_and_extra_kwargs = [
    lazy_fixture("scalar_grid_type_data_and_extra_kwargs"),
    lazy_fixture("vector_grid_type_data_and_extra_kwargs"),
]


def _get_fname(grid_data_and_extra_kwargs):
    grid_type, data, extra_kwargs = grid_data_and_extra_kwargs
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, "test_data", f"{grid_type.name}.zarr")


def _get_results(grid_data_and_extra_kwargs):
    grid_type, data, extra_kwargs = grid_data_and_extra_kwargs

    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)

    if len(data) == 2:  # vector grid
        args = data
    else:
        args = (data,)

    res = laplacian(*args)

    if len(data) == 2:  # vector grid
        res = np.stack(res)

    res = res.astype("f4")  # use single precision to save space
    return res


# this test will not be run by default
# to run it and overwrite the test data, invoke pytest with an environment variable as follows
# $ GCM_FILTERS_OVERWRITE_TEST_DATA=1 pytest tests/test_kernels_validation.py
@requires_env("GCM_FILTERS_OVERWRITE_TEST_DATA", "1")
@pytest.mark.parametrize("grid_data_and_extra_kwargs", all_grids_data_and_extra_kwargs)
def test_save_results(grid_data_and_extra_kwargs):
    res = _get_results(grid_data_and_extra_kwargs)
    fname = _get_fname(grid_data_and_extra_kwargs)
    z_arr = zarr.open(
        fname, mode="w", shape=res.shape, chunks=(-1, -1), dtype=res.dtype
    )
    z_arr[:] = res


@pytest.mark.parametrize("grid_data_and_extra_kwargs", all_grids_data_and_extra_kwargs)
def test_check_results(grid_data_and_extra_kwargs):
    res = _get_results(grid_data_and_extra_kwargs)
    fname = _get_fname(grid_data_and_extra_kwargs)
    z_arr = zarr.open(
        fname, mode="r", shape=res.shape, chunks=(-1, -1), dtype=res.dtype
    )
    np.testing.assert_allclose(z_arr[:], res)
