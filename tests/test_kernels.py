import copy

import numpy as np
import pytest

from gcm_filters.kernels import (
    ALL_KERNELS,
    AreaWeightedMixin,
    BaseScalarLaplacian,
    GridType,
    required_grid_vars,
)


def test_conservation(scalar_grid_type_data_and_extra_kwargs):
    """This test checks that scalar Laplacians preserve the area integral."""
    grid_type, data, extra_kwargs = scalar_grid_type_data_and_extra_kwargs

    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    area = 1  # default value for regular Cartesian grids
    # - Laplacians that belong to AreaWeithedMixin class
    #   act on (transformed) regular grid with dx = dy = 1;
    #   --> test with area = 1
    # - all other Laplacians  act on potentially irregular grid
    #   --> need area information
    if issubclass(LaplacianClass, AreaWeightedMixin):
        area = 1
    else:
        area = extra_kwargs.get("area", None)
        if area is None:
            area = extra_kwargs.get("tarea", 1)

    res = laplacian(data)

    np.testing.assert_allclose((area * res).sum(), 0.0, atol=1e-12)


def test_required_grid_vars(scalar_grid_type_data_and_extra_kwargs):
    grid_type, _, extra_kwargs = scalar_grid_type_data_and_extra_kwargs
    grid_vars = required_grid_vars(grid_type)
    assert set(grid_vars) == set(extra_kwargs)


################## Irregular grid tests for scalar Laplacians ##############################################
# Irregular grids are grids that allow spatially varying dx, dy


def test_for_large_kappas(scalar_grid_type_data_and_extra_kwargs):
    """This test checks that we get an error if either kappa_s or kappa_w are > 1."""
    grid_type, _, extra_kwargs = scalar_grid_type_data_and_extra_kwargs

    if grid_type == GridType.IRREGULAR_WITH_LAND:
        # pick a location outside of the mask
        random_yloc = 99
        random_xloc = 225

        bad_kwargs = copy.deepcopy(extra_kwargs)

        bad_kwargs["kappa_w"][random_yloc, random_xloc] = 2.0

        LaplacianClass = ALL_KERNELS[grid_type]
        with pytest.raises(ValueError, match=r"There are kappa_.*"):
            laplacian = LaplacianClass(**bad_kwargs)

        # restore good value in kappa_w and set bad value in kappa_s
        bad_kwargs["kappa_w"][random_yloc, random_xloc] = 1.0
        bad_kwargs["kappa_s"][random_yloc, random_xloc] = 2.0

        with pytest.raises(ValueError, match=r"There are kappa_.*"):
            laplacian = LaplacianClass(**bad_kwargs)


def test_for_kappas_not_equal_to_one(scalar_grid_type_data_and_extra_kwargs):
    """This test checks that we get an error if neither kappa_s or kappa_w are
    set to 1.0 somewhere in the domain"""

    grid_type, _, extra_kwargs = scalar_grid_type_data_and_extra_kwargs

    if grid_type == GridType.IRREGULAR_WITH_LAND:
        bad_kwargs = copy.deepcopy(extra_kwargs)
        bad_kwargs["kappa_w"][:, :] = 0.5
        bad_kwargs["kappa_s"][:, :] = 0.5

        LaplacianClass = ALL_KERNELS[grid_type]
        with pytest.raises(ValueError, match=r"At least one place*"):
            laplacian = LaplacianClass(**bad_kwargs)


@pytest.mark.parametrize("direction", ["X", "Y"])
def test_flux(irregular_scalar_grid_type_data_and_extra_kwargs, direction):
    """This test checks that the Laplacian computes the correct fluxes in x- and y-direction if the grid is irregular.
    The test will catch sign errors in the Laplacian rolling of array elements."""
    grid_type, data, extra_kwargs = irregular_scalar_grid_type_data_and_extra_kwargs

    # deploy mass at random location away from Antarctica
    delta = np.zeros_like(data)
    ny, nx = delta.shape
    # pick a location outside of the mask
    random_yloc = 99
    random_xloc = 225
    delta[random_yloc, random_xloc] = 1

    test_kwargs = extra_kwargs.copy()
    # start with spatially uniform area, dx, dy because we want *isotropic* diffusion
    for name in extra_kwargs:
        if not name == "wet_mask":
            test_kwargs[name] = np.ones_like(data)

    # now introduce some "outlier" data for dx, dy that Laplacian(delta) should not feel because
    # the outlier dx / dy are far enough from the delta function. Note that the outlier data is
    # still close enough that Laplacian(delta) will feel them if np.roll(dx, -1, axis) in kernels.py
    # is mistakenly coded as np.roll(dx, +1, axis) and vice versa
    replace_data = {
        GridType.IRREGULAR_WITH_LAND: {
            "Y": (
                "dxs",
                (random_yloc - 1, slice(None)),
                (random_yloc + 2, slice(None)),
            ),
            "X": (
                "dyw",
                (slice(None), random_xloc - 1),
                (slice(None), random_xloc + 2),
            ),
        },
        GridType.TRIPOLAR_POP_WITH_LAND: {
            "Y": (
                "dxn",
                (random_yloc - 2, slice(None)),
                (random_yloc + 1, slice(None)),
            ),
            "X": (
                "dye",
                (slice(None), random_xloc - 2),
                (slice(None), random_xloc + 1),
            ),
        },
    }

    var_to_modify, slice_left, slice_right = replace_data[grid_type][direction]
    new_data = np.ones_like(test_kwargs[var_to_modify])
    new_data[slice_left] = 1000
    new_data[slice_right] = 2000
    test_kwargs[var_to_modify] = new_data

    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**test_kwargs)
    diffused = laplacian(delta)

    # Check that delta function gets diffused isotropically in y-direction. Isotropic diffusion is
    # ensured unless Laplacian(delta) feels grid "outlier" data.
    np.testing.assert_allclose(
        diffused[random_yloc - 1, random_xloc],
        diffused[random_yloc + 1, random_xloc],
        atol=1e-12,
    )

    # check that delta function gets diffused isotropically in x-direction
    np.testing.assert_allclose(
        diffused[random_yloc, random_xloc - 1],
        diffused[random_yloc, random_xloc + 1],
        atol=1e-12,
    )


################## Tripolar grid tests for scalar Laplacians ##############################################


def test_for_antarctica(tripolar_grid_type_data_and_extra_kwargs):
    """This test checks that we get an error if southernmost row of wet_mask has entry not equal to zero."""
    grid_type, _, extra_kwargs = tripolar_grid_type_data_and_extra_kwargs

    nx = np.shape(extra_kwargs["wet_mask"])[1]
    random_loc = 10
    bad_kwargs = copy.deepcopy(extra_kwargs)
    bad_kwargs["wet_mask"][0, random_loc] = 1

    LaplacianClass = ALL_KERNELS[grid_type]
    with pytest.raises(AssertionError, match=r"Wet mask requires .*"):
        laplacian = LaplacianClass(**bad_kwargs)


def test_folding_of_northern_gridedge_data(tripolar_grid_type_data_and_extra_kwargs):
    """This test checks that we get an error if northern edge of tripole grid data does not fold onto itself."""
    grid_type, _, extra_kwargs = tripolar_grid_type_data_and_extra_kwargs

    if grid_type == GridType.TRIPOLAR_POP_WITH_LAND:
        LaplacianClass = ALL_KERNELS[grid_type]

        xloc = 3
        bad_kwargs = copy.deepcopy(extra_kwargs)
        # dxn has uppermost row equal to ones; introduce one outlier value
        # in left half of row that does not get mirrored onto right half of row
        bad_kwargs["dxn"][-1, xloc] = 10
        with pytest.raises(AssertionError, match=r"Northernmost row of dxn .*"):
            laplacian = LaplacianClass(**bad_kwargs)
        # restore dxn, and repeat test for dyn
        bad_kwargs["dxn"][-1, xloc] = 1
        bad_kwargs["dyn"][-1, xloc] = 10
        with pytest.raises(AssertionError, match=r"Northernmost row of dyn .*"):
            laplacian = LaplacianClass(**bad_kwargs)


def test_tripolar_exchanges(tripolar_grid_type_data_and_extra_kwargs):
    """This test checks that Laplacian exchanges across northern boundary seam line of tripolar grid are correct."""
    grid_type, data, extra_kwargs = tripolar_grid_type_data_and_extra_kwargs

    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)

    delta = np.zeros_like(data)
    nx = np.shape(delta)[1]
    # deploy mass at northern boundary, away from boundaries and pivot point in middle
    random_loc = 10
    delta[-1, random_loc] = 1

    regular_kwargs = copy.deepcopy(extra_kwargs)
    regular_kwargs["wet_mask"][0, random_loc] = 1

    diffused = laplacian(delta)
    # check that delta function gets diffused isotropically across northern boundary;
    # this assumes regular grid data in fixture
    np.testing.assert_allclose(
        diffused[-2, random_loc], diffused[-1, nx - random_loc - 1], atol=1e-12
    )


#################### Vector Laplacian tests ########################################


def test_conservation_under_solid_body_rotation(
    vector_grid_type_data_and_extra_kwargs, spherical_geometry
):
    """This test checks that vector Laplacians are invariant under solid body rotations:
    a corollary of conserving angular momentum."""

    grid_type, _, extra_kwargs = vector_grid_type_data_and_extra_kwargs

    _, geolat_u, _, _ = spherical_geometry
    # u = cos(lat), v=0 is solid body rotation
    data_u = np.cos(geolat_u / 360 * 2 * np.pi)
    data_v = np.zeros_like(data_u)

    LaplacianClass = ALL_KERNELS[grid_type]
    laplacian = LaplacianClass(**extra_kwargs)
    res_u, res_v = laplacian(data_u, data_v)
    np.testing.assert_allclose(res_u, 0.0, atol=1e-12)
    np.testing.assert_allclose(res_v, 0.0, atol=1e-12)


def test_zero_area(vector_grid_type_data_and_extra_kwargs):
    """This test checks that if area_u, area_v contain zeros, the Laplacian will not blow up
    due to division by zero."""

    grid_type, (data_u, data_v), extra_kwargs = vector_grid_type_data_and_extra_kwargs

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
    assert not np.any(np.isinf(res_u))
    assert not np.any(np.isnan(res_u))
    assert not np.any(np.isnan(res_v))


def test_required_vector_grid_vars(vector_grid_type_data_and_extra_kwargs):
    grid_type, _, extra_kwargs = vector_grid_type_data_and_extra_kwargs
    grid_vars = required_grid_vars(grid_type)
    assert set(grid_vars) == set(extra_kwargs)
