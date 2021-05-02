import numpy as np
import pytest
import xarray as xr

from gcm_filters import Filter, FilterShape, GridType
from gcm_filters.filter import FilterSpec


def _check_equal_filter_spec(spec1, spec2):
    assert spec1.n_steps_total == spec2.n_steps_total
    np.testing.assert_allclose(spec1.s, spec2.s)
    assert (spec1.is_laplacian == spec2.is_laplacian).all()


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
                n_steps_total=4,
                s=[
                    19.7392088 + 0.0j,
                    2.56046256 + 0.0j,
                    15.22333438 + 0.0j,
                    8.47349198 + 0.0j,
                ],
                is_laplacian=[True, True, True, True],
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
                n_steps_total=5,
                s=[
                    9.8696044 + 0.0j,
                    -0.74638043 - 1.24167777j,
                    9.81491354 - 0.44874939j,
                    3.06062496 - 3.94612205j,
                    7.80242999 - 3.18038659j,
                ],
                is_laplacian=[True, False, False, False, False],
            ),
        ),
    ],
)
def test_filter_spec(filter_args, expected_filter_spec):
    """This test just verifies that the filter specification looks as expected."""
    filter = Filter(**filter_args)
    _check_equal_filter_spec(filter.filter_spec, expected_filter_spec)
    # TODO: check other properties of filter_spec?


vector_grids = [gt for gt in GridType if gt.name.startswith("VECTOR")]
scalar_grids = [gt for gt in GridType if gt not in vector_grids]


@pytest.fixture(scope="module", params=scalar_grids)
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
            "kappa_w": da_grid,
            "kappa_s": da_grid,
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


@pytest.fixture(scope="module", params=vector_grids)
def vector_grid_type_and_input_ds(request):
    grid_type = request.param
    ny, nx = (128, 256)

    grid_vars = {}
    if grid_type == GridType.VECTOR_C_GRID:
        # construct spherical coordinate system similar to MOM6 NeverWorld2 grid
        # define latitudes and longitudes
        lat_min = -70
        lat_max = 70
        lat_u = np.linspace(
            lat_min + 0.5 * (lat_max - lat_min) / ny,
            lat_max - 0.5 * (lat_max - lat_min) / ny,
            ny,
        )
        lat_v = np.linspace(lat_min + (lat_max - lat_min) / ny, lat_max, ny)
        lon_min = 0
        lon_max = 60
        lon_u = np.linspace(lon_min + (lon_max - lon_min) / nx, lon_max, nx)
        lon_v = np.linspace(
            lon_min + 0.5 * (lon_max - lon_min) / nx,
            lon_max - 0.5 * (lon_max - lon_min) / nx,
            nx,
        )
        (geolon_u, geolat_u) = np.meshgrid(lon_u, lat_u)
        (geolon_v, geolat_v) = np.meshgrid(lon_v, lat_v)
        # radius of a random planet smaller than Earth
        R = 6378000 * np.random.rand(1)
        # dx varies spatially
        dxCu = R * np.cos(geolat_u / 360 * 2 * np.pi)
        dxCv = R * np.cos(geolat_v / 360 * 2 * np.pi)
        dxBu = dxCv + np.roll(dxCv, -1, axis=1)
        dxT = dxCu + np.roll(dxCu, 1, axis=1)
        da_dxCu = xr.DataArray(dxCu, dims=["y", "x"])
        da_dxCv = xr.DataArray(dxCv, dims=["y", "x"])
        da_dxBu = xr.DataArray(dxBu, dims=["y", "x"])
        da_dxT = xr.DataArray(dxT, dims=["y", "x"])
        # dy is set constant, equal to dx at the equator
        dy = np.max(dxCu) * np.ones((ny, nx))
        da_dy = xr.DataArray(dy, dims=["y", "x"])
        # compute grid cell areas
        area_u = dxCu * dy
        area_v = dxCv * dy
        da_area_u = xr.DataArray(area_u, dims=["y", "x"])
        da_area_v = xr.DataArray(area_v, dims=["y", "x"])
        # set isotropic and anisotropic kappas
        kappa_data = np.ones((ny, nx))
        da_kappa = xr.DataArray(kappa_data, dims=["y", "x"])
        # put a big island in the middle
        mask_data = np.ones((ny, nx))
        mask_data[: (ny // 2), : (nx // 2)] = 0
        da_mask = xr.DataArray(mask_data, dims=["y", "x"])
        grid_vars = {
            "wet_mask_t": da_mask,
            "wet_mask_q": da_mask,
            "dxT": da_dxT,
            "dyT": da_dy,
            "dxCu": da_dxCu,
            "dyCu": da_dy,
            "dxCv": da_dxCv,
            "dyCv": da_dy,
            "dxBu": da_dxBu,
            "dyBu": da_dy,
            "area_u": da_area_u,
            "area_v": da_area_v,
            "kappa_iso": da_kappa,
            "kappa_aniso": da_kappa,
        }
    data_u = np.random.rand(ny, nx)
    data_v = np.random.rand(ny, nx)
    da_u = xr.DataArray(data_u, dims=["y", "x"])
    da_v = xr.DataArray(data_v, dims=["y", "x"])

    return grid_type, da_u, da_v, grid_vars, geolat_u


#################### Diffusion-based filter tests ########################################
@pytest.mark.parametrize(
    "filter_args",
    [dict(filter_scale=1.0, dx_min=1.0, n_steps=10, filter_shape=FilterShape.TAPER)],
)
def test_diffusion_filter(grid_type_and_input_ds, filter_args):
    """Test all diffusion-based filters: filters that use a scalar Laplacian."""
    grid_type, da, grid_vars = grid_type_and_input_ds

    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)
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


#################### Visosity-based filter tests ########################################
@pytest.mark.parametrize(
    "filter_args",
    [dict(filter_scale=1.0, dx_min=1.0, n_steps=10, filter_shape=FilterShape.TAPER)],
)
def test_viscosity_filter(vector_grid_type_and_input_ds, filter_args):
    """Test all viscosity-based filters: filters that use a vector Laplacian."""
    grid_type, da_u, da_v, grid_vars, geolat_u = vector_grid_type_and_input_ds

    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)
    filtered_u, filtered_v = filter.apply_to_vector(da_u, da_v, dims=["y", "x"])

    # check conservation under solid body rotation: u = cos(lat), v=0;
    # this test fails due to numerical instabilities
    data_u = np.cos(geolat_u / 360 * 2 * np.pi)
    data_v = np.zeros_like(data_u)
    da_u = xr.DataArray(data_u, dims=["y", "x"])
    da_v = xr.DataArray(data_v, dims=["y", "x"])
    filtered_u, filtered_v = filter.apply_to_vector(da_u, da_v, dims=["y", "x"])
    xr.testing.assert_allclose(filtered_u, da_u, atol=1e-12)
    xr.testing.assert_allclose(filtered_v, da_v, atol=1e-12)

    # check that we get an error if we leave out any required grid_vars
    for gv in grid_vars:
        grid_vars_missing = {k: v for k, v in grid_vars.items() if k != gv}
        with pytest.raises(ValueError, match=r"Provided `grid_vars` .*"):
            filter = Filter(
                grid_type=grid_type, grid_vars=grid_vars_missing, **filter_args
            )
