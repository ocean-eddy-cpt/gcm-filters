import copy

from typing import Tuple

import numpy as np
import pytest
import xarray as xr

from gcm_filters import Filter, FilterShape, GridType
from gcm_filters.filter import FilterSpec


def _check_equal_filter_spec(spec1, spec2):
    assert spec1.n_steps_total == spec2.n_steps_total
    np.testing.assert_allclose(spec1.s, spec2.s)
    assert (spec1.is_laplacian == spec2.is_laplacian).all()
    assert spec1.s_max == spec2.s_max
    np.testing.assert_allclose(spec1.p, spec2.p, rtol=1e-07, atol=1e-07)
    assert spec1.n_iterations == spec2.n_iterations


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
                n_steps_total=10,
                s=[
                    8.0 + 0.0j,
                    3.42929331 + 0.0j,
                    7.71587822 + 0.0j,
                    2.41473596 + 0.0j,
                    7.18021542 + 0.0j,
                    1.60752541 + 0.0j,
                    6.42502377 + 0.0j,
                    0.81114415 - 0.55260985j,
                    5.50381534 + 0.0j,
                    4.48146765 + 0.0j,
                ],
                is_laplacian=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                ],
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
                n_iterations=1,
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
                n_steps_total=3,
                s=[
                    5.23887374 - 1.09644141j,
                    -0.76856043 - 1.32116962j,
                    3.00058907 - 2.95588288j,
                ],
                is_laplacian=[False, False, False],
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
                n_iterations=1,
            ),
        ),
    ],
)
def test_filter_spec(filter_args, expected_filter_spec):
    """This test just verifies that the filter specification looks as expected."""
    filter = Filter(**filter_args)
    _check_equal_filter_spec(filter.filter_spec, expected_filter_spec)
    # TODO: check other properties of filter_spec?


# define (for now: hard-code) which grids are associated with vector Laplacians
vector_grids = [gt for gt in GridType if gt.name in {"VECTOR_C_GRID"}]
# all remaining grids are for scalar Laplacians
scalar_grids = [gt for gt in GridType if gt not in vector_grids]
scalar_transformed_regular_grids = [
    gt
    for gt in GridType
    if gt.name
    in {
        "REGULAR_AREA_WEIGHTED",
        "REGULAR_WITH_LAND_AREA_WEIGHTED",
        "TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED",
    }
]

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


def _make_random_data(ny, nx):
    data = np.random.rand(ny, nx)
    da = xr.DataArray(data, dims=["y", "x"])
    return da


def _make_mask_data(ny, nx):
    mask_data = np.ones((ny, nx))
    mask_data[0, :] = 0  #  Antarctica; required for some kernels
    mask_data[: (ny // 2), : (nx // 2)] = 0
    da_mask = xr.DataArray(mask_data, dims=["y", "x"])
    return da_mask


def _make_kappa_data(ny, nx):
    kappa_data = np.ones((ny, nx))
    da_kappa = xr.DataArray(kappa_data, dims=["y", "x"])
    return da_kappa


def _make_irregular_grid_data(ny, nx):
    # avoid large-amplitude variation, ensure positive values, mean of 1
    grid_data = 0.9 + 0.2 * np.random.rand(ny, nx)
    assert np.all(grid_data > 0)
    da_grid = xr.DataArray(grid_data, dims=["y", "x"])
    return da_grid


def _make_irregular_tripole_grid_data(ny, nx):
    # avoid large-amplitude variation, ensure positive values, mean of 1
    grid_data = 0.9 + 0.2 * np.random.rand(ny, nx)
    assert np.all(grid_data > 0)
    # make northern edge grid data fold onto itself
    half_northern_edge = grid_data[-1, : (nx // 2)]
    grid_data[-1, (nx // 2) :] = half_northern_edge[::-1]
    da_grid = xr.DataArray(grid_data, dims=["y", "x"])
    return da_grid


@pytest.fixture(scope="module", params=scalar_grids)
def grid_type_and_input_ds(request):
    grid_type = request.param
    ny, nx = 128, 256

    da = _make_random_data(ny, nx)

    grid_vars = {}
    for name in _grid_kwargs[grid_type]:
        if name == "wet_mask":
            grid_vars[name] = _make_mask_data(ny, nx)
        elif "kappa" in name:
            grid_vars[name] = _make_kappa_data(ny, nx)
        else:
            grid_vars[name] = _make_irregular_grid_data(ny, nx)

    if grid_type == GridType.TRIPOLAR_POP_WITH_LAND:
        for name in _grid_kwargs[grid_type]:
            if name in ["dxn", "dyn"]:
                grid_vars[name] = _make_irregular_tripole_grid_data(ny, nx)

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
    [
        dict(
            filter_scale=3.0,
            dx_min=1.0,
            n_steps=0,
            filter_shape=FilterShape.GAUSSIAN,
            n_iterations=1,
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
    assert (filtered ** 2).sum() < (da ** 2).sum()

    # check that we get an error if we leave out any required grid_vars
    for gv in grid_vars:
        grid_vars_missing = {k: v for k, v in grid_vars.items() if k != gv}
        with pytest.raises(ValueError, match=r"Provided `grid_vars` .*"):
            filter = Filter(
                grid_type=grid_type, grid_vars=grid_vars_missing, **filter_args
            )

    bad_filter_args = copy.deepcopy(filter_args)
    # check that we get an error when n_iterations < 1
    bad_filter_args["n_iterations"] = 0
    with pytest.raises(ValueError, match=r"Number of intermediate .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    # check that we get an error when n_iterations > 1 with Taper
    bad_filter_args["n_iterations"] = 2
    bad_filter_args["filter_shape"] = FilterShape.TAPER
    with pytest.raises(ValueError, match=r"n_iterations must be .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    bad_filter_args["n_iterations"] = 1
    bad_filter_args["filter_shape"] = FilterShape.GAUSSIAN
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
    # check that we get a warning if numerical instability possible
    bad_filter_args["n_steps"] = 0
    bad_filter_args["filter_scale"] = 1000
    with pytest.warns(UserWarning, match=r"Filter scale much larger .*"):
        filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **bad_filter_args)
    # check that we get an error if we pass dx_min != 1 to a regular scalar Laplacian
    if grid_type in scalar_transformed_regular_grids:
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


@pytest.mark.parametrize(
    "filter_args",
    [
        dict(
            filter_scale=8.0,
            dx_min=1.0,
            n_steps=0,
            filter_shape=FilterShape.GAUSSIAN,
            n_iterations=1,
        )
    ],
)
@pytest.mark.parametrize(
    "n_iterations",
    [2, 3, 4],
)
def test_iterated_filter(grid_type_and_input_ds, filter_args, n_iterations):
    "Test that the iterated Gaussian filter gives a result close to the original"

    grid_type, da, grid_vars = grid_type_and_input_ds

    iterated_filter_args = filter_args.copy()
    iterated_filter_args["n_iterations"] = n_iterations

    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)
    iterated_filter = Filter(
        grid_type=grid_type, grid_vars=grid_vars, **iterated_filter_args
    )

    filtered = filter.apply(da, dims=["y", "x"])
    iteratively_filtered = iterated_filter.apply(da, dims=["y", "x"])

    area = 1
    for k, v in grid_vars.items():
        if "area" in k:
            area = v
            break

    # The following tests whether a relative error bound in L^2 holds.
    # See the "Factoring the Gaussian Filter" section of the docs for details.
    assert (((filtered - iteratively_filtered) ** 2) * area).sum() < (
        (0.01 * (1 + n_iterations)) ** 2
    ) * ((da ** 2) * area).sum()


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


@pytest.mark.parametrize(
    "filter_args",
    [dict(filter_scale=4.0, dx_min=1.0, n_steps=0, filter_shape=FilterShape.GAUSSIAN)],
)
@pytest.mark.parametrize(
    "n_iterations",
    [2, 3, 4],
)
def test_iterated_viscosity_filter(
    vector_grid_type_and_input_ds, filter_args, n_iterations
):
    """Test error in the iterated Gaussian filter for vectors"""
    grid_type, da_u, da_v, grid_vars, _ = vector_grid_type_and_input_ds

    filter = Filter(grid_type=grid_type, grid_vars=grid_vars, **filter_args)
    filtered_u, filtered_v = filter.apply_to_vector(da_u, da_v, dims=["y", "x"])

    iterated_filter_args = filter_args.copy()
    iterated_filter_args["n_iterations"] = n_iterations
    iterated_filter = Filter(
        grid_type=grid_type, grid_vars=grid_vars, **iterated_filter_args
    )
    iteratively_filtered_u, iteratively_filtered_v = iterated_filter.apply_to_vector(
        da_u, da_v, dims=["y", "x"]
    )

    area = 1
    for k, v in grid_vars.items():
        if "area" in k:
            area = v
            break

    # The following tests whether a relative error bound in L^2 holds.
    # See the "Factoring the Gaussian Filter" section of the docs for details.
    difference = (filtered_u - iteratively_filtered_u) ** 2 + (
        filtered_v - iteratively_filtered_v
    ) ** 2
    unfiltered = da_u ** 2 + da_v ** 2
    assert (difference * area).sum() < ((0.01 * (1 + n_iterations)) ** 2) * (
        unfiltered * area
    ).sum()
