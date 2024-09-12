from typing import Tuple

import numpy as np
import pytest
import xarray as xr

from numpy.random import PCG64, Generator

from gcm_filters.kernels import ALL_KERNELS, GridType


_grid_kwargs = {
    GridType.REGULAR: [],
    GridType.REGULAR_AREA_WEIGHTED: ["area"],
    GridType.REGULAR_WITH_LAND: ["wet_mask"],
    GridType.REGULAR_WITH_LAND_AREA_WEIGHTED: ["wet_mask", "area"],
    GridType.IRREGULAR_WITH_LAND: [
        "wet_mask",
        "dxw",
        "dyw",
        "dxs",
        "dys",
        "area",
        "kappa_w",
        "kappa_s",
    ],
    GridType.MOM5U: ["wet_mask", "dxt", "dyt", "dxu", "dyu", "area_u"],
    GridType.MOM5T: ["wet_mask", "dxt", "dyt", "dxu", "dyu", "area_t"],
    GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED: ["wet_mask", "area"],
    GridType.TRIPOLAR_POP_WITH_LAND: ["wet_mask", "dxe", "dye", "dxn", "dyn", "tarea"],
}

_vector_grid_kwargs = {
    GridType.VECTOR_C_GRID: [
        "wet_mask_t",
        "wet_mask_q",
        "dxT",
        "dyT",
        "dxCu",
        "dyCu",
        "dxCv",
        "dyCv",
        "dxBu",
        "dyBu",
        "area_u",
        "area_v",
        "kappa_iso",
        "kappa_aniso",
    ],
    GridType.VECTOR_B_GRID: [
        "DXU",
        "DYU",
        "HUS",
        "HUW",
        "HTE",
        "HTN",
        "UAREA",
        "TAREA",
    ],
}

scalar_grids = [
    GridType.REGULAR,
    GridType.REGULAR_AREA_WEIGHTED,
    GridType.REGULAR_WITH_LAND,
    GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
    GridType.IRREGULAR_WITH_LAND,
    GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED,
    GridType.TRIPOLAR_POP_WITH_LAND,
]
irregular_grids = [GridType.IRREGULAR_WITH_LAND, GridType.TRIPOLAR_POP_WITH_LAND]
tripolar_grids = [
    GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED,
    GridType.TRIPOLAR_POP_WITH_LAND,
]
vector_grids = [GridType.VECTOR_C_GRID, GridType.VECTOR_B_GRID]


def _make_random_data(shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = Generator(PCG64(seed))
    return rng.random(shape)


def _make_mask_data(shape: Tuple[int, int]) -> np.ndarray:
    mask_data = np.ones(shape)
    ny, nx = shape
    mask_data[0, :] = 0  #  Antarctica; required for some kernels
    mask_data[: (ny // 2), : (nx // 2)] = 0
    return mask_data


def _make_irregular_grid_data(shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = Generator(PCG64(seed))
    # avoid large-amplitude variation, ensure positive values, mean of 1
    grid_data = 0.9 + 0.2 * rng.random(shape)
    assert np.all(grid_data > 0)
    return grid_data


def _make_irregular_tripole_grid_data(shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = Generator(PCG64(seed))
    # avoid large-amplitude variation, ensure positive values, mean of 1
    grid_data = 0.9 + 0.2 * rng.random(shape)
    assert np.all(grid_data > 0)
    # make northern edge grid data fold onto itself
    nx = shape[-1]
    half_northern_edge = grid_data[-1, : (nx // 2)]
    grid_data[-1, (nx // 2) :] = half_northern_edge[::-1]
    return grid_data


def _make_scalar_grid_data(grid_type):
    shape = (128, 256)

    data = _make_random_data(shape, 100)

    extra_kwargs = {}
    for seed, name in enumerate(_grid_kwargs[grid_type]):
        if name == "wet_mask":
            extra_kwargs[name] = _make_mask_data(shape)
        elif "kappa" in name:
            extra_kwargs[name] = np.ones(shape)
        else:
            extra_kwargs[name] = _make_irregular_grid_data(shape, seed)

    # northern edge grid data has to fold onto itself for tripole grids
    if grid_type == GridType.TRIPOLAR_POP_WITH_LAND:
        for name in _grid_kwargs[grid_type]:
            if name in ["dxn", "dyn"]:
                seed += 1
                extra_kwargs[name] = _make_irregular_tripole_grid_data(shape, seed)

    return grid_type, data, extra_kwargs


@pytest.fixture(scope="session", params=scalar_grids)
def scalar_grid_type_data_and_extra_kwargs(request):
    return _make_scalar_grid_data(request.param)


@pytest.fixture(scope="session", params=irregular_grids)
def irregular_scalar_grid_type_data_and_extra_kwargs(request):
    return _make_scalar_grid_data(request.param)


@pytest.fixture(scope="session", params=tripolar_grids)
# the following test data mirrors a regular grid because
# these are the assumptions of test_tripolar_exchanges
def tripolar_grid_type_data_and_extra_kwargs(request):
    grid_type = request.param
    shape = (128, 256)

    data = _make_random_data(shape, 30)

    extra_kwargs = {}
    for name in _grid_kwargs[grid_type]:
        if name == "wet_mask":
            extra_kwargs[name] = _make_mask_data(shape)
        else:
            extra_kwargs[name] = np.ones_like(data)

    return grid_type, data, extra_kwargs


@pytest.fixture(scope="session", params=scalar_grids)
# test data for filter.py: need xr.DataArray's
def grid_type_and_input_ds(request, scalar_grid_type_data_and_extra_kwargs):

    grid_type, data, extra_kwargs = scalar_grid_type_data_and_extra_kwargs

    da = xr.DataArray(data, dims=["y", "x"])

    grid_vars = {}
    for name in extra_kwargs:
        grid_vars[name] = xr.DataArray(extra_kwargs[name], dims=["y", "x"])

    return grid_type, da, grid_vars


@pytest.fixture(scope="session")
# compute latitudes and longitudes for a spherical C-grid geometry
def spherical_geometry():

    ny, nx = (128, 256)

    # construct spherical coordinate system similar to MOM6 NeverWorld2 grid
    # define latitudes and longitudes
    lat_min = -70
    lat_max = 70

    # compute latitude of u-points on C-grid
    latCu = np.linspace(
        lat_min + 0.5 * (lat_max - lat_min) / ny,
        lat_max - 0.5 * (lat_max - lat_min) / ny,
        ny,
    )
    # compute latitude of v-points on C-grid
    latCv = np.linspace(lat_min + (lat_max - lat_min) / ny, lat_max, ny)

    lon_min = 0
    lon_max = 60

    lonCu = np.linspace(lon_min + (lon_max - lon_min) / nx, lon_max, nx)
    lonCv = np.linspace(
        lon_min + 0.5 * (lon_max - lon_min) / nx,
        lon_max - 0.5 * (lon_max - lon_min) / nx,
        nx,
    )

    (geolonCu, geolatCu) = np.meshgrid(lonCu, latCu)
    (geolonCv, geolatCv) = np.meshgrid(lonCv, latCv)

    return geolonCu, geolatCu, geolonCv, geolatCv


@pytest.fixture(scope="session", params=vector_grids)
def vector_grid_type_data_and_extra_kwargs(request, spherical_geometry):
    grid_type = request.param
    geolonCu, geolatCu, geolonCv, geolatCv = spherical_geometry
    ny, nx = geolonCu.shape

    extra_kwargs = {}

    R = 6378000

    # dx varies spatially
    for name in _vector_grid_kwargs[grid_type]:
        if name in ["dxCu", "dxT", "HUS", "HTE"]:
            extra_kwargs[name] = R * np.cos(geolatCu / 360 * 2 * np.pi)
            # compute dy for later
            # dy is set constant, equal to dx at the equator
            dy = np.max(extra_kwargs[name]) * np.ones((ny, nx))
        if name in ["dxCv", "dxBu", "DXU", "HUW", "HTN"]:
            extra_kwargs[name] = R * np.cos(geolatCv / 360 * 2 * np.pi)

    # dy
    for name in _vector_grid_kwargs[grid_type]:
        if name in ["dyCu", "dyCv", "dyBu", "dyT", "DYU"]:
            extra_kwargs[name] = dy

    # compute grid cell areas
    for name in _vector_grid_kwargs[grid_type]:
        if name == "area_u":
            extra_kwargs[name] = extra_kwargs["dxCu"] * extra_kwargs["dyCu"]
        elif name == "area_v":
            extra_kwargs[name] = extra_kwargs["dxCv"] * extra_kwargs["dyCv"]
        elif name == "UAREA":
            extra_kwargs[name] = extra_kwargs["DXU"] * extra_kwargs["DYU"]
        elif name == "TAREA":
            # on a spherical grid, HTE is the same as x-spacing centered at tracer point
            # on a spherical grid, DYU is the same as y-spacing centered at tracer point
            extra_kwargs[name] = extra_kwargs["HTE"] * extra_kwargs["DYU"]

    # set isotropic and anisotropic kappas
    for name in _vector_grid_kwargs[grid_type]:
        if name in ["kappa_iso", "kappa_aniso"]:
            extra_kwargs[name] = np.ones((ny, nx))

    # put a big island in the middle
    mask_data = np.ones((ny, nx))
    mask_data[: (ny // 2), : (nx // 2)] = 0
    for name in _vector_grid_kwargs[grid_type]:
        if name in ["wet_mask_t", "wet_mask_q"]:
            extra_kwargs[name] = mask_data

    data_u = _make_random_data((ny, nx), 42)
    data_v = _make_random_data((ny, nx), 43)

    # use same return signature as other kernel fixtures
    return grid_type, (data_u, data_v), extra_kwargs


@pytest.fixture(scope="session", params=vector_grids)
def vector_grid_type_and_input_ds(request, vector_grid_type_data_and_extra_kwargs):
    # test data for filter.py: need xr.DataArray's
    grid_type, (data_u, data_v), extra_kwargs = vector_grid_type_data_and_extra_kwargs

    da_u = xr.DataArray(data_u, dims=["y", "x"])
    da_v = xr.DataArray(data_v, dims=["y", "x"])

    grid_vars = {}
    for name in extra_kwargs:
        grid_vars[name] = xr.DataArray(extra_kwargs[name], dims=["y", "x"])

    return grid_type, (da_u, da_v), grid_vars
