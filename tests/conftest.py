from typing import Tuple

import numpy as np
import pytest

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
vector_grids = [GridType.VECTOR_C_GRID]


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


@pytest.fixture(scope="session")
def spherical_geometry():
    ny, nx = (128, 256)

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

    return geolon_u, geolat_u, geolon_v, geolat_v


@pytest.fixture(scope="session", params=vector_grids)
def vector_grid_type_data_and_extra_kwargs(request, spherical_geometry):
    grid_type = request.param
    geolon_u, geolat_u, geolon_v, geolat_v = spherical_geometry
    ny, nx = geolon_u.shape

    extra_kwargs = {}

    # for now, we assume that the only implemented vector grid is VECTOR_C_GRID
    # we can relax this if we implement other vector grids
    assert grid_type == GridType.VECTOR_C_GRID

    R = 6378000
    # dx varies spatially
    extra_kwargs["dxCu"] = R * np.cos(geolat_u / 360 * 2 * np.pi)
    extra_kwargs["dxCv"] = R * np.cos(geolat_v / 360 * 2 * np.pi)
    extra_kwargs["dxBu"] = extra_kwargs["dxCv"] + np.roll(
        extra_kwargs["dxCv"], -1, axis=1
    )
    extra_kwargs["dxT"] = extra_kwargs["dxCu"] + np.roll(
        extra_kwargs["dxCu"], 1, axis=1
    )
    # dy is set constant, equal to dx at the equator
    dy = np.max(extra_kwargs["dxCu"]) * np.ones((ny, nx))
    extra_kwargs["dyCu"] = dy
    extra_kwargs["dyCv"] = dy
    extra_kwargs["dyBu"] = dy
    extra_kwargs["dyT"] = dy
    # compute grid cell areas
    extra_kwargs["area_u"] = extra_kwargs["dxCu"] * extra_kwargs["dyCu"]
    extra_kwargs["area_v"] = extra_kwargs["dxCv"] * extra_kwargs["dyCv"]
    # set isotropic and anisotropic kappas
    extra_kwargs["kappa_iso"] = np.ones((ny, nx))
    extra_kwargs["kappa_aniso"] = np.ones((ny, nx))
    # put a big island in the middle
    mask_data = np.ones((ny, nx))
    mask_data[: (ny // 2), : (nx // 2)] = 0
    extra_kwargs["wet_mask_t"] = mask_data
    extra_kwargs["wet_mask_q"] = mask_data

    data_u = _make_random_data((ny, nx), 42)
    data_v = _make_random_data((ny, nx), 43)

    # use same return signature as other kernel fixtures
    return grid_type, (data_u, data_v), extra_kwargs
