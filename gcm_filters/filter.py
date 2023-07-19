"""Main Filter class."""
import enum
import warnings

from dataclasses import dataclass, field
from itertools import chain, zip_longest
from typing import Iterable, NamedTuple

import numpy as np
import xarray as xr

from scipy import interpolate

from .gpu_compat import get_array_module
from .kernels import (
    ALL_KERNELS,
    AreaWeightedMixin,
    BaseScalarLaplacian,
    BaseVectorLaplacian,
    GridType,
)


FilterShape = enum.Enum("FilterShape", ["GAUSSIAN", "TAPER"])


# These parameters are used to set the default n_steps
filter_params = {
    FilterShape.GAUSSIAN: {
        1: {"offset": 0.8, "factor": 0.0, "exponent": 1},
        2: {"offset": 1.1, "factor": 0.0, "exponent": 1},
    },
    FilterShape.TAPER: {
        1: {"offset": 2.2, "factor": 0.6, "exponent": 2.5},
        2: {"offset": 3.2, "factor": 0.7, "exponent": 2.7},
    },
}


class TargetSpec(NamedTuple):
    s_max: float
    filter_scale: float
    transition_width: float


# these functions return functions
def _gaussian_target(target_spec: TargetSpec):
    return lambda t: np.exp(
        -(target_spec.s_max * (t + 1) / 2) * (target_spec.filter_scale) ** 2 / 24
    )


def _taper_target(target_spec: TargetSpec):
    FK = interpolate.PchipInterpolator(
        np.array(
            [
                0,
                2 * np.pi / (target_spec.transition_width * target_spec.filter_scale),
                2 * np.pi / target_spec.filter_scale,
                8 * np.sqrt(target_spec.s_max),
            ]
        ),
        np.array([1, 1, 0, 0]),
    )
    return lambda t: FK(np.sqrt((t + 1) * (target_spec.s_max / 2)))


_target_function = {
    FilterShape.GAUSSIAN: _gaussian_target,
    FilterShape.TAPER: _taper_target,
}


class FilterSpec(NamedTuple):
    n_steps: int
    s_max: float
    p: Iterable[float]
    dx_min_sq: float


def _compute_filter_spec(
    filter_scale,
    dx_min,
    filter_shape,
    transition_width=np.pi,
    ndim=2,
    n_steps=0,
):

    # Set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi / 2) * (
        2 * np.eye(n_steps - 1)
        - np.diag(np.ones(n_steps - 3), 2)
        - np.diag(np.ones(n_steps - 3), -2)
    )
    M[0, 0] = 3 * np.pi / 2

    # The range of wavenumbers is 0<=|k|<=sqrt(ndim)*pi/dxMin.
    # However, our 2nd order laplacians only get to sqrt(ndim)*2/dxMin at most.
    # Caveat: Not sure what is a good max wavenumber for the C-grid vector Laplacian
    # Per the paper, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    s_max = ndim * (2 / dx_min) ** 2

    target_spec = TargetSpec(s_max, filter_scale, transition_width)
    F = _target_function[filter_shape](target_spec)

    # Compute inner products of Galerkin basis with target
    b = np.zeros(n_steps - 1)
    points, weights = np.polynomial.chebyshev.chebgauss(n_steps + 1)
    for i in range(n_steps - 1):
        tmp = np.zeros(n_steps + 1)
        tmp[i] = 1
        tmp[i + 2] = -1
        phi = np.polynomial.chebyshev.chebval(points, tmp)
        b[i] = np.sum(
            weights * phi * (F(points) - ((1 - points) / 2 + F(1) * (points + 1) / 2))
        )

    # Get polynomial coefficients in Galerkin basis
    c_hat = np.linalg.solve(M, b)
    # Convert back to Chebyshev basis coefficients
    p = np.zeros(n_steps + 1)
    p[0] = c_hat[0] + (1 + F(1)) / 2
    p[1] = c_hat[1] - (1 - F(1)) / 2
    for i in range(2, n_steps - 1):
        p[i] = c_hat[i] - c_hat[i - 2]
    p[n_steps - 1] = -c_hat[n_steps - 3]
    p[n_steps] = -c_hat[n_steps - 2]

    dx_min_sq = dx_min**2  # For nondimensional Laplacians

    return FilterSpec(n_steps, s_max, p, dx_min_sq)


def _create_filter_func(
    filter_spec: FilterSpec,
    Laplacian: BaseScalarLaplacian,
):
    """Returns a function whose first argument is the field to be filtered
    and whose subsequent arguments are the required grid variables
    """

    def shifted_laplacian(
        field,
        s_max,
        laplacian,
        dx_min_sq,
    ):
        # This function computes -(field + (2/s_max) * laplacian(field))
        output = laplacian(field)
        if laplacian.is_dimensional:
            output = -field - (2 / s_max) * output
        else:
            output = -field - (2 / (s_max * dx_min_sq)) * output

        return output

    def filter_func(field, *args):
        # these next steps are a kind of hack we have to turn keyword arugments into regular arguments
        # the reason for doing this is that Xarray's apply_ufunc machinery works a lot better
        # with regular arguments
        assert len(args) == len(Laplacian.required_grid_args())
        grid_vars = {k: v for k, v in zip(Laplacian.required_grid_args(), args)}
        laplacian = Laplacian(**grid_vars)
        np = get_array_module(field)
        field_bar = field.copy()  # Initalize the filtering process

        # prepare field for filtering (this multiplies by area for simple fixed factor
        # filters, and does nothing for all other filters)
        field_bar = laplacian.prepare(field_bar)

        T_minus_2 = field_bar.copy()
        T_minus_1 = shifted_laplacian(
            field_bar, filter_spec.s_max, laplacian, filter_spec.dx_min_sq
        )
        field_bar = filter_spec.p[0] * T_minus_2 + filter_spec.p[1] * T_minus_1
        for i in range(2, filter_spec.n_steps + 1):
            T_minus_0 = (
                2
                * shifted_laplacian(
                    T_minus_1, filter_spec.s_max, laplacian, filter_spec.dx_min_sq
                )
                - T_minus_2
            )
            field_bar += filter_spec.p[i] * T_minus_0
            T_minus_2 = T_minus_1.copy()
            T_minus_1 = T_minus_0.copy()

        # finalize filtering (this divides by area for simple fixed factor filters,
        # and does nothing for all other filters)
        field_bar = laplacian.finalize(field_bar)

        return field_bar

    return filter_func


def _create_filter_func_vec(
    filter_spec: FilterSpec,
    Laplacian: BaseVectorLaplacian,
):
    """Returns a function whose first two arguments are the vector components of the field to be filtered
    and whose subsequent arguments are the require grid variables
    """

    def shifted_laplacian_vec(
        ufield,
        vfield,
        s_max,
        laplacian,
        dx_min_sq,
    ):
        # This function computes -(field + (2/s_max) * laplacian(field))
        (u_output, v_output) = laplacian(ufield, vfield)
        if laplacian.is_dimensional:
            u_output = -ufield - (2 / s_max) * u_output
            v_output = -vfield - (2 / s_max) * v_output
        else:
            u_output = -ufield - (2 / (s_max * dx_min_sq)) * u_output
            v_output = -vfield - (2 / (s_max * dx_min_sq)) * v_output
        return (u_output, v_output)

    def filter_func_vec(ufield, vfield, *args):
        # these next steps are a kind of hack we have to turn keyword arugments into regular arguments
        # the reason for doing this is that Xarray's apply_ufunc machinery works a lot better
        # with regular arguments
        assert len(args) == len(Laplacian.required_grid_args())
        grid_vars = {k: v for k, v in zip(Laplacian.required_grid_args(), args)}
        laplacian = Laplacian(**grid_vars)
        np = get_array_module(ufield)
        ufield_bar = ufield.copy()  # Initalize the filtering process
        vfield_bar = vfield.copy()  # Initalize the filtering process

        # prepare field for filtering (this multiplies by area for simple fixed factor
        # filters, and does nothing for all other filters)
        (ufield_bar, vfield_bar) = laplacian.prepare(ufield_bar, vfield_bar)

        uT_minus_2 = ufield_bar.copy()
        vT_minus_2 = vfield_bar.copy()
        (uT_minus_1, vT_minus_1) = shifted_laplacian_vec(
            ufield_bar,
            vfield_bar,
            filter_spec.s_max,
            laplacian,
            filter_spec.dx_min_sq,
        )
        ufield_bar = filter_spec.p[0] * uT_minus_2 + filter_spec.p[1] * uT_minus_1
        vfield_bar = filter_spec.p[0] * vT_minus_2 + filter_spec.p[1] * vT_minus_1
        for i in range(2, filter_spec.n_steps + 1):
            (uT_minus_0, vT_minus_0) = shifted_laplacian_vec(
                uT_minus_1,
                vT_minus_1,
                filter_spec.s_max,
                laplacian,
                filter_spec.dx_min_sq,
            )
            uT_minus_0 = 2 * uT_minus_0 - uT_minus_2
            vT_minus_0 = 2 * vT_minus_0 - vT_minus_2
            ufield_bar += filter_spec.p[i] * uT_minus_0
            vfield_bar += filter_spec.p[i] * vT_minus_0
            uT_minus_2 = uT_minus_1.copy()
            uT_minus_1 = uT_minus_0.copy()
            vT_minus_2 = vT_minus_1.copy()
            vT_minus_1 = vT_minus_0.copy()

        # finalize filtering (this divides by area for simple fixed factor filters,
        # and does nothing for all other filters)
        (ufield_bar, vfield_bar) = laplacian.finalize(ufield_bar, vfield_bar)

        return (ufield_bar, vfield_bar)

    return filter_func_vec


@dataclass
class Filter:
    """A class for applying diffusion-based smoothing filters to gridded data.

    Parameters
    ----------
    filter_scale : float
        The filter scale, which has different meaning depending on filter shape
    dx_min : float
        The smallest grid spacing. Should have same units as ``filter_scale``
    n_steps : int, optional
        Number of total steps in the filter
        ``n_steps == 0`` means the number of steps is chosen automatically
    filter_shape : FilterShape
        - ``FilterShape.GAUSSIAN``: The target filter has shape :math:`e^{-(k filter_scale)^2/24}`
        - ``FilterShape.TAPER``: The target filter has target grid scale Lf. Smaller scales are zeroed out.
          Scales larger than ``pi * filter_scale / 2`` are left as-is. In between is a smooth transition.
    transition_width : float, optional
        Width of the transition region in the "Taper" filter.
        This is a nondimensional parameter. Theoretical minimum is 1; not recommended.
    ndim : int, optional
         Laplacian is applied on a grid of dimension ndim
    grid_type : GridType
        what sort of grid we are dealing with
    grid_vars : dict
        dictionary of extra parameters used to initialize the grid Laplacian

    Attributes
    ----------
    filter_spec: FilterSpec
    """

    filter_scale: float
    dx_min: float
    filter_shape: FilterShape = FilterShape.GAUSSIAN
    transition_width: float = np.pi
    ndim: int = 2
    n_steps: int = 0
    grid_type: GridType = GridType.REGULAR
    grid_vars: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):

        self.Laplacian = ALL_KERNELS[self.grid_type]

        # Determine whether this is simple fixed factor filter; in that case we need dx_min = 1
        if issubclass(self.Laplacian, AreaWeightedMixin):
            if self.dx_min != 1:
                raise ValueError(
                    f"Provided Laplacian is for simple fixed factor filtering, "
                    "where transformed field is filtered on a regular grid with dx = dy = 1. "
                    "dx_min must be set to 1."
                )

        # Check if transition_width is <=1
        if self.transition_width <= 1:
            raise ValueError(f"Transition width must be > 1.")

        # Get default number of steps
        filter_factor = self.filter_scale / self.dx_min
        if self.ndim > 2:
            if self.n_steps < 3:
                raise ValueError(f"When ndim > 2, you must set n_steps manually")
            else:
                n_steps_default = self.n_steps  # For ndim>2 we don't have a default
        else:
            n_steps_factor = filter_params[self.filter_shape][self.ndim][
                "offset"
            ] + filter_params[self.filter_shape][self.ndim]["factor"] * (
                (np.pi / self.transition_width)
                ** filter_params[self.filter_shape][self.ndim]["exponent"]
            )
            n_steps_default = np.ceil(n_steps_factor * filter_factor).astype(int)

        # Set n_steps if needed and issue n_step warning, if needed
        if self.n_steps < 3:
            self.n_steps = n_steps_default

        if self.n_steps < n_steps_default:
            warnings.warn(
                "You have set n_steps below the default. Results might not be accurate.",
                stacklevel=2,
            )

        self.filter_spec = _compute_filter_spec(
            self.filter_scale,
            self.dx_min,
            self.filter_shape,
            self.transition_width,
            self.ndim,
            self.n_steps,
        )

        # check that we have all the required grid aguments

        if not set(self.Laplacian.required_grid_args()) == set(self.grid_vars):
            raise ValueError(
                f"Provided `grid_vars` {list(self.grid_vars)} do not match expected "
                f"{list(self.Laplacian.required_grid_args())}"
            )
        self.grid_ds = xr.Dataset({name: da for name, da in self.grid_vars.items()})

    def plot_shape(self, ax=None):
        """Plot the shape of the target filter and approximation."""
        import matplotlib.pyplot as plt

        # Plot the target filter and the approximate filter
        s_max = self.filter_spec.s_max
        target_spec = TargetSpec(s_max, self.filter_scale, self.transition_width)
        F = _target_function[self.filter_shape](target_spec)
        x = np.linspace(-1, 1, 10001)
        k = np.sqrt(s_max * (x + 1) / 2)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(k, F(x), "g", label="target filter", linewidth=4)
        ax.plot(
            k,
            np.polynomial.chebyshev.chebval(x, self.filter_spec.p),
            "m",
            label="approximation",
            linewidth=4,
        )
        ax.axvline(
            2 * np.pi / self.filter_scale,
            color="k",
            label="filter cutoff wavenumber",
            linewidth=2,
        )
        ax.set_xlim(left=0)
        if self.filter_scale / self.dx_min > 10:
            ax.set_xlim(right=4 * np.pi / self.filter_scale)
        ax.set_ylim(bottom=-0.1)
        ax.set_ylim(top=1.1)
        ax.set_xlabel("Wavenumber k", fontsize=18)
        ax.grid(True)
        ax.legend()

    def apply(self, ds, dims):
        """Filter an `xarray.DataArray` or `xarray.Dataset`
        with a scalar Laplacian across the dimensions specified by `dims`.

        Parameters
        ----------
        ds : xarray.DataArray or xarray.Dataset
            The data to be filtered. If Dataset, filter will be applied to
            all data variables.
        dims : sequence of str
            The names of the dimensions over which to apply the filter.
            Usually this is two spatial dimensions, e.g. ``('lat', 'lon')``
            or ``('y', 'x')``.

            .. warning:: The dimension order matters! Since some filters deal
                with anisotropic grids, the latitude dimension must appear first
                in order to obtain the correct result.
        """
        if issubclass(self.Laplacian, BaseVectorLaplacian):
            raise ValueError(
                f"Provided Laplacian {self.Laplacian} is a vector Laplacian. "
                f"The ``.apply`` method is only suitable for scalar Laplacians."
            )

        if isinstance(ds, xr.Dataset):
            filtered = ds.copy(deep=True)
            any_filtered = False
            for key, var in filtered.variables.items():
                if all(dim in var.dims for dim in dims):
                    filtered[key] = self._apply_to_dataarray(var, dims=dims)
                    any_filtered = True
            if not any_filtered:
                warnings.warn(
                    f"No variables in the dataset had all of the given "
                    f"dimensions ({dims}), so nothing was filtered.",
                    stacklevel=2,
                )
            return filtered
        else:
            return self._apply_to_dataarray(ds, dims=dims)

    def _apply_to_dataarray(self, field, dims):
        """Filter an `xarray.DataArray` field with scalar Laplacian across the
        dimensions specified by dims."""
        filter_func = _create_filter_func(self.filter_spec, self.Laplacian)
        grid_args = [self.grid_ds[name] for name in self.Laplacian.required_grid_args()]
        assert len(dims) == 2
        n_args = 1 + len(grid_args)
        field_smooth = xr.apply_ufunc(
            filter_func,
            field,
            *grid_args,
            input_core_dims=n_args * [dims],
            output_core_dims=[dims],
            output_dtypes=[field.dtype],
            dask="parallelized",
        )

        return field_smooth

    def apply_to_vector(self, ufield, vfield, dims):
        """Filter a vector field with vector Laplacian across the dimensions specified by dims.

        Parameters
        ----------
        ufield : xarray.DataArray
            The zonal component of the data to be filtered.
        vfield : xarray.DataArray
            The meridional component of the data to be filtered.
        dims : sequence of str
            The names of the dimensions over which to apply the filter.
            Usually this is two spatial dimensions, e.g. ``('lat', 'lon')``
            or ``('y', 'x')``.

            .. warning:: The dimension order matters! Since some filters deal
                with anisotropic grids, the latitude dimension must appear first
                in order to obtain the correct result.
        """
        if not issubclass(self.Laplacian, BaseVectorLaplacian):
            raise ValueError(
                f"Provided Laplacian {self.Laplacian} is a scalar Laplacian. "
                f"The ``.apply_to_vector`` method is only suitable for vector Laplacians."
            )

        filter_func_vec = _create_filter_func_vec(self.filter_spec, self.Laplacian)
        grid_args = [self.grid_ds[name] for name in self.Laplacian.required_grid_args()]
        assert len(dims) == 2
        n_args = 2 + len(grid_args)
        (ufield_smooth, vfield_smooth) = xr.apply_ufunc(
            filter_func_vec,
            ufield,
            vfield,
            *grid_args,
            input_core_dims=n_args * [dims],
            output_core_dims=2 * [dims],
            output_dtypes=[ufield.dtype, vfield.dtype],
            dask="parallelized",
        )

        return (ufield_smooth, vfield_smooth)
