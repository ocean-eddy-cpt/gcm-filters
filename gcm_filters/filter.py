"""Main Filter class."""
import enum

from dataclasses import dataclass, field
from typing import Iterable, NamedTuple

import numpy as np
import xarray as xr

from scipy import interpolate

from .gpu_compat import get_array_module
from .kernels import ALL_KERNELS, BaseLaplacian, GridType


FilterShape = enum.Enum("FilterShape", ["GAUSSIAN", "TAPER"])


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
                2 * np.sqrt(target_spec.s_max),
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
    n_lap_steps: int
    s_l: Iterable[float]
    n_bih_steps: int
    s_b: Iterable[complex]


def _compute_filter_spec(
    filter_scale,
    dx_min,
    filter_shape,
    transition_width,
    ndim,
    n_steps=0,
    root_tolerance=1e-12,
):
    # First set number of steps if not supplied by user
    if n_steps == 0:
        if ndim > 2:
            raise ValueError(f"When ndim > 2, you must set n_steps manually")
        if filter_shape == FilterShape.GAUSSIAN:
            if ndim == 1:
                n_steps = np.ceil(1.3 * filter_scale / dx_min).astype(int)
            else:  # ndim==2
                n_steps = np.ceil(1.8 * filter_scale / dx_min).astype(int)
        else:  # Taper
            if ndim == 1:
                n_steps = np.ceil(4.5 * filter_scale / dx_min).astype(int)
            else:  # ndim==2
                n_steps = np.ceil(6.4 * filter_scale / dx_min).astype(int)

    # First set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi / 2) * (
        2 * np.eye(n_steps - 1)
        - np.diag(np.ones(n_steps - 3), 2)
        - np.diag(np.ones(n_steps - 3), -2)
    )
    M[0, 0] = 3 * np.pi / 2

    # The range of wavenumbers is 0<=|k|<=sqrt(ndim)*pi/dxMin.
    # Per the notes, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    s_max = ndim * (np.pi / dx_min) ** 2

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

    # Get roots of the polynomial
    r = np.polynomial.chebyshev.chebroots(p)

    # convert back to s in [0,sMax]
    s = s_max / 2 * (r + 1)
    # Separate out the real and complex roots
    n_lap_steps = np.size(s[np.where(np.abs(np.imag(r)) < root_tolerance)])
    s_l = np.real(s[np.where(np.abs(np.imag(r)) < root_tolerance)])
    n_bih_steps = (n_steps - n_lap_steps) // 2
    s_b_re, indices = np.unique(
        np.real(s[np.where(np.abs(np.imag(r)) > root_tolerance)]), return_index=True
    )
    s_b_im = np.imag(s[np.where(np.abs(np.imag(r)) > root_tolerance)])[indices]
    s_b = s_b_re + s_b_im * 1j

    return FilterSpec(n_lap_steps, s_l, n_bih_steps, s_b)


def _create_filter_func(
    filter_spec: FilterSpec,
    Laplacian: BaseLaplacian,
):
    """Returns a function whose first argument is the field to be filtered
    and whose subsequent arguments are the require grid variables
    """

    def filter_func(field, *args):
        # these next steps are a kind of hack we have to turn keyword arugments into regular arguments
        # the reason for doing this is that Xarray's apply_ufunc machinery works a lot better
        # with regular arguments
        assert len(args) == len(Laplacian.required_grid_args())
        grid_vars = {k: v for k, v in zip(Laplacian.required_grid_args(), args)}
        laplacian = Laplacian(**grid_vars)
        np = get_array_module(field)
        field_bar = field.copy()  # Initalize the filtering process
        for i in range(filter_spec.n_lap_steps):
            s_l = filter_spec.s_l[i]
            tendency = laplacian(field_bar)  # Compute Laplacian
            field_bar += (1 / s_l) * tendency  # Update filtered field
        for i in range(filter_spec.n_bih_steps):
            s_b = filter_spec.s_b[i]
            temp_l = laplacian(field_bar)  # Compute Laplacian
            temp_b = laplacian(temp_l)  # Compute Biharmonic (apply Laplacian twice)
            field_bar += (
                temp_l * 2 * np.real(s_b) / np.abs(s_b) ** 2
                + temp_b * 1 / np.abs(s_b) ** 2
            )
        return field_bar

    return filter_func


@dataclass
class Filter:
    """A class for applying diffusion-based smoothing filters to gridded data.

    ̦Parameters
    ----------
    filter_scale : float
        The filter scale, which has different meaning depending on filter shape
    dx_min : float
        The smallest grid spacing. Should have same units as ``filter_scale``
    n_steps : int, optional
        Number of total steps in the filter
        ``n_steps == 0`` means the number of steps is chosen automatically
    filter_shape : FilterShape
        - ``FilterShape.GAUSSIAN``: The target filter has kernel :math:`e^{-|x/Lf|^2}`
        - ``FilterShape.TAPER``: The target filter has target grid scale Lf. Smaller scales are zeroed out.
          Scales larger than ``pi * filter_scale / 2`` are left as-is. In between is a smooth transition.
    transition_width : float, optional
        Width of the transition region in the "Taper" filter.
    ndim : int, optional
         Laplacian is applied on a grid of dimension ndim
    grid_type : GridType
        what sort of grid we are dealing with
    grid_vars : dict
        dictionary of extra parameters used to initialize the grid laplacian

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
    grid_type: GridType = GridType.CARTESIAN
    grid_vars: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):

        if self.n_steps < 0:
            raise ValueError("Filter requires N>=0")

        self.filter_spec = _compute_filter_spec(
            self.filter_scale,
            self.dx_min,
            self.filter_shape,
            self.transition_width,
            self.ndim,
            self.n_steps,
        )

        # check that we have all the required grid aguments
        self.Laplacian = ALL_KERNELS[self.grid_type]
        if not set(self.Laplacian.required_grid_args()) == set(self.grid_vars):
            raise ValueError(
                f"Provided `grid_vars` {list(self.grid_vars)} do not match expected "
                f"{list(self.Laplacian.required_grid_args())}"
            )
        self.grid_ds = xr.Dataset({name: da for name, da in self.grid_vars.items()})

    def apply(self, field, dims):
        """Filter a field across the dimensions specified by dims."""

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
