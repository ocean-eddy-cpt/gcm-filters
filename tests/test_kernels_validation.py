import numpy as np
import pytest
import xarray as xr
import zarr

from pytest_lazyfixture import lazy_fixture

from gcm_filters.kernels import ALL_KERNELS


all_grids_data_and_extra_kwargs = [
    lazy_fixture("scalar_grid_type_data_and_extra_kwargs"),
    lazy_fixture("vector_grid_type_data_and_extra_kwargs"),
]


def _get_fname(grid_data_and_extra_kwargs):
    grid_type, data, extra_kwargs = grid_data_and_extra_kwargs
    return f"test_data/{grid_type.name}.zarr"


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
# to run it and overwrite the test data, invoke pytest as follows
# $ pytest -m overwrite_test_data tests/test_kernels_validation.py
@pytest.mark.overwrite_test_data
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
