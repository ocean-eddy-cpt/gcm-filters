import copy

import numpy as np
import pytest

from gcm_filters.kernels import ALL_KERNELS, GridType, required_grid_vars


# define (for now: hard-code) which grids are associated with vector Laplacians
vector_grids = [gt for gt in GridType if gt.name in {"VECTOR_C_GRID"}]
# all remaining grids are for scalar Laplacians
scalar_grids = [gt for gt in GridType if gt not in vector_grids]

################## Scalar Laplacian tests ##############################################
@pytest.fixture(scope="module", params=scalar_grids)
def grid_type_field_and_extra_kwargs(request):
    grid_type = request.param
    ny, nx = (128, 256)
    data = np.random.rand(ny, nx)

    extra_kwargs = {}
    if grid_type == GridType.REGULAR_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        extra_kwargs["wet_mask"] = mask_data
    if grid_type == GridType.IRREGULAR_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        extra_kwargs["wet_mask"] = mask_data
        grid_data = np.ones_like(data)
        extra_kwargs["dxw"] = grid_data
        extra_kwargs["dyw"] = grid_data
        extra_kwargs["dxs"] = grid_data
        extra_kwargs["dys"] = grid_data
        extra_kwargs["area"] = grid_data
        extra_kwargs["kappa_w"] = grid_data
        extra_kwargs["kappa_s"] = grid_data
    if grid_type == GridType.TRIPOLAR_REGULAR_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        mask_data[0, :] = 0  #  Antarctica
        extra_kwargs["wet_mask"] = mask_data
    if grid_type == GridType.TRIPOLAR_POP_WITH_LAND:
        mask_data = np.ones_like(data)
        mask_data[: (ny // 2), : (nx // 2)] = 0
        mask_data[0, :] = 0  #  Antarctica
        extra_kwargs["wet_mask"] = mask_data
        grid_data = np.ones_like(data)
        extra_kwargs["dxe"] = grid_data
        extra_kwargs["dye"] = grid_data
        extra_kwargs["dxn"] = grid_data
        extra_kwargs["dyn"] = grid_data
        extra_kwargs["tarea"] = grid_data

    return grid_type, data, extra_kwargs


def test_conservation(grid_type_field_and_extra_kwargs):
    """This test checks that scalar Laplacians preserve the area integral."""
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs

    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    res = laplacian(data)
    np.testing.assert_allclose(res.sum(), 0.0, atol=1e-12)


def test_required_grid_vars(grid_type_field_and_extra_kwargs):
    grid_type, _, extra_kwargs = grid_type_field_and_extra_kwargs
    grid_vars = required_grid_vars(grid_type)
    assert set(grid_vars) == set(extra_kwargs)


################## Irregular grid tests for scalar Laplacians ##############################################
# Irregular grids are grids that allow spatially varying dx, dy

# The following definition of irregular_grids is hard coded; maybe a better definition
# would be: all grids that have len(required_grid_vars)>1 (more than just a wet_mask)
irregular_grids = [
    gt for gt in GridType if "IRREGULAR" in gt.name or "TRIPOLAR_POP" in gt.name
]


def test_flux_in_y_direction(grid_type_field_and_extra_kwargs):
    """This test checks that the Laplacian computes the correct fluxes in y-direction if the grid is irregular.
    The test will catch sign errors in the Laplacian rolling of array elements along the y-axis."""
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs

    if grid_type in irregular_grids:
        # deploy mass at random location away from Antarctica: delta_{j,i}
        delta = np.zeros_like(data)
        ny = np.shape(delta)[0]
        random_yloc = np.random.randint(5, ny - 2)
        nx = np.shape(delta)[1]
        random_xloc = np.random.randint(0, nx)
        delta[random_yloc, random_xloc] = 1

        test_kwargs = copy.deepcopy(extra_kwargs)
        ones = np.ones_like(data)
        # create highly irregular grid data for
        # - southern edge of cell {j-1,i};
        # - southern edge of cell {j+2,i};
        # this should leave laplacian(delta_{j,i}) unaffected
        if grid_type == GridType.IRREGULAR_WITH_LAND:
            test_kwargs["wet_mask"] = ones  # no land for simplicity
            test_kwargs["area"] = ones
            test_kwargs["dyw"] = ones
            test_kwargs["dys"] = ones
            test_kwargs["dxw"] = ones
            test_kwargs["dxs"] = ones.copy()
            test_kwargs["dxs"][random_yloc - 1, :] = 1000
            test_kwargs["dxs"][random_yloc + 2, :] = 2000
        if grid_type == GridType.TRIPOLAR_POP_WITH_LAND:
            test_kwargs["wet_mask"] = ones.copy()
            test_kwargs["wet_mask"][0, :] = 0  # Antarctica
            test_kwargs["tarea"] = ones
            test_kwargs["dye"] = ones
            test_kwargs["dyn"] = ones
            test_kwargs["dxe"] = ones
            test_kwargs["dxn"] = ones.copy()
            test_kwargs["dxn"][random_yloc - 2, :] = 1000
            test_kwargs["dxn"][random_yloc + 1, :] = 2000

        LaplacianClass = ALL_KERNELS[grid_type]
        laplacian = LaplacianClass(**test_kwargs)
        diffused = laplacian(delta)
        # check that delta function gets diffused isotropically in y-direction
        np.testing.assert_allclose(
            diffused[random_yloc - 1, random_xloc],
            diffused[random_yloc + 1, random_xloc],
            atol=1e-12,
        )


def test_flux_in_x_direction(grid_type_field_and_extra_kwargs):
    """This test checks that the Laplacian computes the correct fluxes in x-direction if the grid is irregular.
    The test will catch sign errors in the Laplacian rolling of array elements along the x-axis."""
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs

    if grid_type in irregular_grids:
        # deploy mass at random location away from Antarctica: delta_{j,i}
        delta = np.zeros_like(data)
        ny = np.shape(delta)[0]
        random_yloc = np.random.randint(5, ny)
        nx = np.shape(delta)[1]
        random_xloc = np.random.randint(2, nx - 2)
        delta[random_yloc, random_xloc] = 1

        test_kwargs = copy.deepcopy(extra_kwargs)
        ones = np.ones_like(data)
        # create highly irregular grid data for
        # - western edge of cell {j,i-1};
        # - western edge of cell {j,i+2}
        # this should leave laplacian(delta_{j,i}) unaffected
        if grid_type == GridType.IRREGULAR_WITH_LAND:
            test_kwargs["wet_mask"] = ones  # no land for simplicity
            test_kwargs["area"] = ones
            test_kwargs["dys"] = ones
            test_kwargs["dxw"] = ones
            test_kwargs["dxs"] = ones
            test_kwargs["dyw"] = ones.copy()
            test_kwargs["dyw"][:, random_xloc - 1] = 1000
            test_kwargs["dyw"][:, random_xloc + 2] = 2000
        if grid_type == GridType.TRIPOLAR_POP_WITH_LAND:
            test_kwargs["wet_mask"] = ones.copy()
            test_kwargs["wet_mask"][0, :] = 0  # Antarctica
            test_kwargs["tarea"] = ones
            test_kwargs["dyn"] = ones
            test_kwargs["dxe"] = ones
            test_kwargs["dxn"] = ones
            test_kwargs["dye"] = ones.copy()
            test_kwargs["dye"][:, random_xloc - 2] = 1000
            test_kwargs["dye"][:, random_xloc + 1] = 2000

        LaplacianClass = ALL_KERNELS[grid_type]
        laplacian = LaplacianClass(**test_kwargs)
        diffused = laplacian(delta)
        # check that delta function gets diffused isotropically in x-direction
        np.testing.assert_allclose(
            diffused[random_yloc, random_xloc - 1],
            diffused[random_yloc, random_xloc + 1],
            atol=1e-12,
        )


################## Tripolar grid tests for scalar Laplacians ##############################################
tripolar_grids = [gt for gt in GridType if gt.name.startswith("TRIPOLAR")]


def test_for_antarctica(grid_type_field_and_extra_kwargs):
    """This test checks that we get an error if southernmost row of wet_mask has entry not equal to zero."""
    grid_type, _, extra_kwargs = grid_type_field_and_extra_kwargs

    if grid_type in tripolar_grids:
        nx = np.shape(extra_kwargs["wet_mask"])[1]
        random_loc = np.random.randint(0, nx)
        bad_kwargs = copy.deepcopy(extra_kwargs)
        bad_kwargs["wet_mask"][0, random_loc] = 1

        LaplacianClass = ALL_KERNELS[grid_type]
        with pytest.raises(AssertionError, match=r"Wet mask requires .*"):
            laplacian = LaplacianClass(**bad_kwargs)


def test_tripolar_exchanges(grid_type_field_and_extra_kwargs):
    """This test checks that Laplacian exchanges across northern boundary seam line of tripolar grid are correct."""
    grid_type, data, extra_kwargs = grid_type_field_and_extra_kwargs

    if grid_type in tripolar_grids:
        LaplacianClass = ALL_KERNELS[grid_type]
        laplacian = LaplacianClass(**extra_kwargs)

        delta = np.zeros_like(data)
        nx = np.shape(delta)[1]
        # deploy mass at northern boundary, away from boundaries and pivot point in middle
        random_loc = np.random.randint(1, nx // 2 - 2)
        delta[-1, random_loc] = 1

        diffused = laplacian(delta)
        # check that delta function gets diffused isotropically across northern boundary
        # this would need to be replaced once we provide irregular grid data in fixture
        np.testing.assert_allclose(
            diffused[-2, random_loc], diffused[-1, nx - random_loc - 1], atol=1e-12
        )


#################### Vector Laplacian tests ########################################


@pytest.fixture(scope="module", params=vector_grids)
def vector_grid_type_field_and_extra_kwargs(request):
    grid_type = request.param
    ny, nx = (128, 256)

    extra_kwargs = {}
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

    data_u = np.random.rand(ny, nx)
    data_v = np.random.rand(ny, nx)

    return grid_type, data_u, data_v, extra_kwargs, geolat_u


def test_conservation_under_solid_body_rotation(
    vector_grid_type_field_and_extra_kwargs,
):
    """This test checks that vector Laplacians are invariant under solid body rotations:
    a corollary of conserving angular momentum."""

    grid_type, _, _, extra_kwargs, geolat_u = vector_grid_type_field_and_extra_kwargs

    # u = cos(lat), v=0 is solid body rotation
    data_u = np.cos(geolat_u / 360 * 2 * np.pi)
    data_v = np.zeros_like(data_u)

    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    res_u, res_v = laplacian(data_u, data_v)
    np.testing.assert_allclose(res_u, 0.0, atol=1e-12)
    np.testing.assert_allclose(res_v, 0.0, atol=1e-12)


def test_zero_area(vector_grid_type_field_and_extra_kwargs):
    """This test checks that if area_u, area_v contain zeros, the Laplacian will not blow up
    due to division by zero."""

    grid_type, data_u, data_v, extra_kwargs, _ = vector_grid_type_field_and_extra_kwargs
    test_kwargs = copy.deepcopy(extra_kwargs)
    # fill area_u, area_v with zeros over land; e.g., you will find that in MOM6 model output
    test_kwargs["area_u"] = np.where(
        extra_kwargs["wet_mask_t"] > 0, test_kwargs["area_u"], 0
    )
    test_kwargs["area_v"] = np.where(
        extra_kwargs["wet_mask_t"] > 0, test_kwargs["area_v"], 0
    )
    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**test_kwargs)
    res_u, res_v = laplacian(data_u, data_v)
    assert np.any(np.isinf(res_u)) == False
    assert np.any(np.isnan(res_u)) == False
    assert np.any(np.isinf(res_v)) == False
    assert np.any(np.isnan(res_v)) == False


def test_required_vector_grid_vars(vector_grid_type_field_and_extra_kwargs):
    grid_type, _, _, extra_kwargs, _ = vector_grid_type_field_and_extra_kwargs
    grid_vars = required_grid_vars(grid_type)
    assert set(grid_vars) == set(extra_kwargs)
