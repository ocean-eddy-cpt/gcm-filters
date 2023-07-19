import copy

from typing import Tuple

import numpy as np
import pytest
import xarray as xr

from gcm_filters import Filter, FilterShape, GridType
from gcm_filters.filter import FilterSpec


def _check_equal_filter_spec(spec1, spec2):
    assert spec1.n_steps == spec2.n_steps
    assert spec1.s_max == spec2.s_max
    np.testing.assert_allclose(spec1.p, spec2.p, rtol=1e-07, atol=1e-07)
    np.testing.assert_allclose(spec1.dx_min_sq, spec2.dx_min_sq)


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
                grid_vars={},
            ),
            FilterSpec(
                n_steps=11,
                s_max=8.0,
                p=[
                    0.09887381,
                    -0.19152534,
                    0.1748326,
                    -0.14975371,
                    0.12112337,
                    -0.09198484,
                    0.0662522,
                    -0.04479323,
                    0.02895827,
                    -0.0173953,
                    0.00995974,
                    -0.00454758,
                ],
                dx_min_sq=1.0,
            ),
        ),
        (
            dict(
                filter_scale=2.0,
                dx_min=1.0,
                filter_shape=FilterShape.TAPER,
                transition_width=np.pi,
                ndim=1,
                grid_vars={},
            ),
            FilterSpec(
                n_steps=6,
                s_max=4.0,
                p=[
                    0.83380304,
                    -0.23622724,
                    -0.06554041,
                    0.01593978,
                    0.00481014,
                    -0.00495532,
                    0.00168445,
                ],
                dx_min_sq=1.0,
            ),
        ),
    ],
)
def test_filter_spec(filter_args, expected_filter_spec):
    """This test just verifies that the filter specification looks as expected."""
    filter = Filter(**filter_args)
    _check_equal_filter_spec(filter.filter_spec, expected_filter_spec)
    # TODO: check other properties of filter_spec?


#################### Diffusion-based filter tests ########################################
area_weighted_regular_grids = [
    GridType.REGULAR_AREA_WEIGHTED,
    GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
    GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED,
]


@pytest.mark.parametrize(
    "filter_args",
    [
        dict(
            filter_scale=3.0,
            dx_min=1.0,
            n_steps=0,
            filter_shape=FilterShape.GAUSSIAN,
        )
    ],
)
def test_diffusion_filter(grid_type_and_input_ds, filter_args):
    """Test all diffusion-based filters: filters that use a scalar Laplacian."""
    grid_type, da, grid_vars = grid_type_and_input_ds

    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)
    filter.plot_shape()
    filtered = filter.apply(da, dims=["y", "x"])

    # check conservation
    area = 1
    for k, v in grid_vars.items():
        if "area" in k:
            area = v
            break
    da_sum = (da * area).sum()
    filtered_sum = (filtered * area).sum()
    xr.testing.assert_allclose(da_sum, filtered_sum)

    # check that we get an error if we pass scalar Laplacian to .apply_to vector,
    # where the latter method is for vector Laplacians only
    with pytest.raises(ValueError, match=r"Provided Laplacian *"):
        filtered_u, filtered_v = filter.apply_to_vector(da, da, dims=["y", "x"])

    # check variance reduction
    assert (filtered**2).sum() < (da**2).sum()

    # check that we get an error if we leave out any required grid_vars
    for gv in grid_vars:
        grid_vars_missing = {k: v for k, v in grid_vars.items() if k != gv}
        with pytest.raises(ValueError, match=r"Provided `grid_vars` .*"):
            filter = Filter(
                grid_type=grid_type, grid_vars=grid_vars_missing, **filter_args
            )

    bad_filter_args = copy.deepcopy(filter_args)
    # check that we get an error when transition_width <= 1
    bad_filter_args["transition_width"] = 1
    with pytest.raises(ValueError, match=r"Transition width .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    bad_filter_args["transition_width"] = np.pi
    # check that we get an error if ndim > 2 and n_steps = 0
    bad_filter_args["ndim"] = 3
    bad_filter_args["n_steps"] = 0
    with pytest.raises(ValueError, match=r"When ndim > 2, you .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    # check that we get a warning if n_steps < n_steps_default
    bad_filter_args["ndim"] = 2
    bad_filter_args["n_steps"] = 3
    with pytest.warns(UserWarning, match=r"You have set n_steps .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    # check that we get an error if we pass dx_min != 1 to a regular scalar Laplacian
    if grid_type in area_weighted_regular_grids:
        bad_filter_args["filter_scale"] = 3  # restore good value for filter scale
        bad_filter_args["dx_min"] = 3
        with pytest.raises(ValueError, match=r"Provided Laplacian .*"):
            filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)


def test_application_to_dataset():
    # Create a dataset with both spatial and temporal variables
    dataset = xr.Dataset(
        data_vars=dict(
            spatial=(("y", "x"), np.random.normal(size=(100, 100))),
            temporal=(("time",), np.random.normal(size=(10,))),
            spatiotemporal=(("time", "y", "x"), np.random.normal(size=(10, 100, 100))),
        ),
        coords=dict(
            time=np.linspace(0, 1, 10),
            x=np.linspace(0, 1e6, 100),
            y=np.linspace(0, 1e6, 100),
        ),
    )

    # Filter it using a Gaussian filter
    filter = Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=FilterShape.GAUSSIAN,
        grid_type=GridType.REGULAR,
    )
    filtered_dataset = filter.apply(dataset, ["y", "x"])

    # Temporal variables should be unaffected because the filter is only
    # applied over space
    xr.testing.assert_allclose(dataset.temporal, filtered_dataset.temporal)

    # Spatial variables should be changed
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(dataset.spatial, filtered_dataset.spatial)
        xr.testing.assert_allclose(
            dataset.spatiotemporal, filtered_dataset.spatiotemporal
        )

    # Spatially averaged spatiotemporal variables should be unchanged because
    # the filter shouldn't modify the temporal component
    xr.testing.assert_allclose(
        dataset.spatiotemporal.mean(dim=["y", "x"]),
        filtered_dataset.spatiotemporal.mean(dim=["y", "x"]),
    )

    # Warnings should be raised if no fields were filtered
    with pytest.warns(UserWarning, match=r".* nothing was filtered."):
        filter.apply(dataset, ["foo", "bar"])
    with pytest.warns(UserWarning, match=r".* nothing was filtered."):
        filter.apply(dataset, ["yy", "x"])


def test_nondimensional_invariance():
    # Create a dataset with spatial variables, as above
    dataset = xr.Dataset(
        data_vars=dict(
            spatial=(("y", "x"), np.random.normal(size=(100, 100))),
        ),
        coords=dict(
            x=np.linspace(0, 1e6, 100),
            y=np.linspace(0, 1e6, 100),
        ),
    )

    # Filter it using a nondimensional filter, dx_min = 1
    filter = Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=FilterShape.GAUSSIAN,
        grid_type=GridType.REGULAR,
    )
    filtered_dataset = filter.apply(dataset, ["y", "x"])

    # Filter it using a nondimensional filter, dx_min = 2
    filter = Filter(
        filter_scale=8,
        dx_min=2,
        filter_shape=FilterShape.GAUSSIAN,
        grid_type=GridType.REGULAR,
    )
    filtered_dataset_v2 = filter.apply(dataset, ["y", "x"])

    # Check if they are the same
    xr.testing.assert_allclose(filtered_dataset.spatial, filtered_dataset_v2.spatial)


#################### Visosity-based filter tests ########################################
@pytest.mark.parametrize(
    "filter_args",
    [dict(filter_scale=5.0, dx_min=1.0, n_steps=10, filter_shape=FilterShape.TAPER)],
)
def test_viscosity_filter(
    vector_grid_type_and_input_ds, filter_args, spherical_geometry
):
    """Test all viscosity-based filters: filters that use a vector Laplacian."""
    grid_type, _, grid_vars = vector_grid_type_and_input_ds

    _, geolat_u, _, _ = spherical_geometry

    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)

    # check conservation under solid body rotation: u = cos(lat), v=0;
    data_u = np.cos(geolat_u / 360 * 2 * np.pi)
    data_v = np.zeros_like(data_u)
    da_u = xr.DataArray(data_u, dims=["y", "x"])
    da_v = xr.DataArray(data_v, dims=["y", "x"])
    filtered_u, filtered_v = filter.apply_to_vector(da_u, da_v, dims=["y", "x"])
    xr.testing.assert_allclose(filtered_u, da_u, atol=1e-12)
    xr.testing.assert_allclose(filtered_v, da_v, atol=1e-12)

    # check that we get an error if we pass vector Laplacian to .apply, where
    # the latter method is for scalar Laplacians only
    with pytest.raises(ValueError, match=r"Provided Laplacian *"):
        filtered_u = filter.apply(da_u, dims=["y", "x"])

    # check that we get an error if we leave out any required grid_vars
    for gv in grid_vars:
        grid_vars_missing = {k: v for k, v in grid_vars.items() if k != gv}
        with pytest.raises(ValueError, match=r"Provided `grid_vars` .*"):
            filter = Filter(
                grid_type=grid_type, grid_vars=grid_vars_missing, **filter_args
            )
