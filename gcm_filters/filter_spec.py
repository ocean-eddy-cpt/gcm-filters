"""Main Filter class."""
from dataclasses import dataclass
import enum

import numpy as np
from scipy import interpolate, integrate
from typing import NamedTuple, Callable

from .gpu_compat import get_array_module

FilterShape = enum.Enum("FilterShape", "Gaussian Taper")


# these functions return functions
def _gaussian_target(s_max, filter_scale, transition_width):
    lambda t: np.exp(-(s_max * (t + 1) / 2) * (filter_scale / 2) ** 2)


def _taper_target(s_max, filter_scale, transition_width):
    return interpolate.PchipInterpolator(
        np.array(
            [
                -1,
                (2 / sMax) * (np.pi / (transition_width * filter_scale)) ** 2 - 1,
                (2 / s_max) * (np.pi / Lf) ** 2 - 1,
                2,
            ]
        ),
        np.array([1, 1, 0, 0]),
    )


_target_function = {
    FilterShape.Gaussian: _gaussian_target,
    FilterShape.Taper: _taper_target,
}


class FilterSpec(NamedTuple):
    n_lap_steps: int
    s_l: float
    n_bih_steps: int
    s_b: float


def _compute_filter_spec(
    filter_scale, dx_min, n_steps, filter_shape, transition_width, root_tolerance=1e-12
):
    # First set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi / 2) * (
        2 * np.eye(N - 1) - np.diag(np.ones(N - 3), 2) - np.diag(np.ones(N - 3), -2)
    )
    M[0, 0] = 3 * np.pi / 2

    # The range of wavenumbers is 0<=|k|<=sqrt(2)*pi/dxMin. Nyquist here is for a 2D grid.
    # Per the notes, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    s_max = 2 * (np.pi / dxMin) ** 2

    F = _target_function[filter_shape]

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
    filter_spec: FilterSpec, laplacian_kernel: Callable, **kernel_kwargs
):
    def laplacian(field):
        # wrap any auxiliary arguments using a closure
        return laplacian_kernel(field, **kernel_kwargs)

    def filter_func(field):
        np = get_array_module(field)
        field_bar = field.copy()  # Initalize the filtering process
        for i in range(filter_spec.n_lap_steps):
            tendency = laplacian(field_bar)  # Compute Laplacian
            field_bar += (1 / filter_spec.s_l[i]) * tendency  # Update filtered field
        for i in range(filter_spec.n_big_steps):
            temp_l = laplacian(filed_bar)  # Compute Laplacian
            temp_b = laplacian(temp_l)  # Compute Biharmonic (apply Laplacian twice)
            field_bar += (
                temp_l * 2 * np.real(sB[i]) / np.abs(sB[i]) ** 2
                + temp_b * 1 / np.abs(sB[i]) ** 2
            )
        return field_bar

    return filter_func


@dataclass
class Filter:
    """A class for applying diffusion-based smoothing filters to gridded data.

    ̦Parameters
    -----------
    filter_scale : float
        The filter scale, which has different meaning depending on filter shape
    dx_min : float
        The smallest grid spacing. Should have same units as ``filter_scale``
    n_steps : int, optional
        Number of total steps in the filter
    filter_shape : {"Gaussian", "Taper"}
        - Gaussian: The target filter has kernel $e^{-|x/Lf|^2}$
        - Taper: The target filter has target grid scale Lf. Smaller scales are zeroed out.
          Scales larger than $\pi Lf/2$ are left as-is. In between is a smooth transition.
    transition_width : float, optional
        Width of the transition region in the "Taper" filter.

    Attributes
    ----------
    filter_spec: FilterSpec
    """

    filter_scale: int
    dx_min: float
    n_steps: int = 40
    shape: FilterShape = FilterShape.Gaussian
    transition_width: float = np.pi

    def __post_init__(self):

        if self.n_steps <= 2:
            raise ValueError("Filter requires N>2")

        self.filter_spec = _compute_filter_spec(
            filter_scale, dx_min, n_steps, filter_shape, transition_width
        )

    def apply(field, landMask, dx, dy):
        """
        Filters a 2D field, applying an operator of type (*) above.
        Assumes dy=constant, dx varies in y direction
        Inputs:
        field: 2D array (y, x) to be filtered
        landMask: 2D array, same size as field: 0 if cell is not on land, 1 if it is on land.
        dx is a 1D array, same size as 1st dimension of field
        dy is constant
        NL is number of Laplacian steps, see output of filterSpec fct above
        sL is s_i for the Laplacian steps, see output of filterSpec fct above
        NB is the number of Biharmonic steps, see output of filterSpec fct above
        sB is s_i for the Biharmonic steps, see output of filterSpec fct above
        Output:
        Filtered field.
        """
        fieldBar = field.copy()  # Initalize the filtering process
        for i in range(self.NL):
            tempL = Laplacian2D_FV(fieldBar, landMask, dx, dy)  # Compute Laplacian
            fieldBar = fieldBar + (1 / self.sL[i]) * tempL  # Update filtered field
        for i in range(self.NB):
            tempL = Laplacian2D_FV(fieldBar, landMask, dx, dy)  # Compute Laplacian
            tempB = Laplacian2D_FV(
                tempL, landMask, dx, dy
            )  # Compute Biharmonic (apply Laplacian twice)
            fieldBar = (
                fieldBar
                + (2 * np.real(self.sB[i]) / np.abs(self.sB[i]) ** 2) * tempL
                + (1 / np.abs(self.sB[i]) ** 2) * tempB
            )
        return fieldBar
