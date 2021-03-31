"""
Core smoothing routines that operate on 2D arrays.
"""
import enum

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict

from .gpu_compat import ArrayType, get_array_module


# not married to the term "Cartesian"
GridType = enum.Enum(
    "GridType",
    [
        "CARTESIAN",
        "CARTESIAN_WITH_LAND",
        "IRREGULAR_CARTESIAN_WITH_LAND",
        "MOM5U",
        "MOM5T",
    ],
)

ALL_KERNELS = {}  # type: Dict[GridType, Any]


@dataclass
class BaseLaplacian(ABC):
    def __call__(self, field):
        pass  # pragma: no cover

    # change to property when we are using python 3.9
    # https://stackoverflow.com/questions/128573/using-property-on-classmethods
    @classmethod
    def required_grid_args(self):
        try:
            return list(self.__annotations__)
        except AttributeError:
            return []


@dataclass
class CartesianLaplacian(BaseLaplacian):
    """̵Laplacian for regularly spaced Cartesian grids."""

    def __call__(self, field: ArrayType):
        np = get_array_module(field)
        return (
            -4 * field
            + np.roll(field, -1, axis=-1)
            + np.roll(field, 1, axis=-1)
            + np.roll(field, -1, axis=-2)
            + np.roll(field, 1, axis=-2)
        )


ALL_KERNELS[GridType.CARTESIAN] = CartesianLaplacian


@dataclass
class MOM5LaplacianU(BaseLaplacian):
    """Laplacian for MOM5 gird (velocity points)"""

    dxt: ArrayType
    dyt: ArrayType
    dxu: ArrayType
    dyu: ArrayType
    area_u: ArrayType
    wet: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet)

        self.x_wet_mask = self.wet * np.roll(self.wet, -1, axis=-1)
        self.y_wet_mask = self.wet * np.roll(self.wet, -1, axis=-2)

    def __call__(self, field: ArrayType):
        """Uses code by Elizabeth"""
        np = get_array_module()
        field = np.nan_to_num(field)
        fx = 2 * (np.roll(field, shift=-1, axis=-2) - field)
        fx /= np.roll(self.dxt, -1, axis=-2) + np.roll(self.dxt, (-1, -1), axis=(0, 1))
        fy = 2 * (np.roll(field, shift=-1, axis=-1) - field)
        fy /= np.roll(self.dyt, -1, axis=-1) + np.roll(self.dyt, (-1, -1), axis=(0, 1))
        fx *= self.x_wet_mask
        fy *= self.y_wet_mask

        out1 = 0.5 * fx * (self.dyu + np.roll(self.dyu, -1, axis=-2))
        out1 -= (
            0.5 * np.roll(fx, 1, axis=-2) * (self.dyu + np.roll(self.dyu, 1, axis=-2))
        )
        out1 /= self.area_u

        out2 = 0.5 * fy * (self.dxu + np.roll(self.dxu, -1, axis=-1))
        out2 -= (
            0.5 * np.roll(fy, 1, axis=-1) * (self.dxu + np.roll(self.dxu, 1, axis=-1))
        )
        out2 /= self.area_u
        return out1 + out2


ALL_KERNELS[GridType.MOM5U] = MOM5LaplacianU


@dataclass
class MOM5LaplacianT(BaseLaplacian):
    """Laplacian for MOM5 grid (tracer points)."""

    dxt: ArrayType
    dyt: ArrayType
    dxu: ArrayType
    dyu: ArrayType
    area_t: ArrayType
    wet: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet)

        self.x_wet_mask = self.wet * np.roll(self.wet, -1, axis=-1)
        self.y_wet_mask = self.wet * np.roll(self.wet, -1, axis=-2)

    def __call__(self, field):
        np = get_array_module(field)
        field = np.nan_to_num(field)
        fx = 2 * (np.roll(field, -1, axis=-2) - field)
        fx /= self.dxu + np.roll(self.dxu, 1, axis=-1)
        fy = 2 * (np.roll(field, -1, axis=-1) - field)
        fy /= self.dyu + np.roll(self.dyu, 1, axis=-2)
        fx *= self.x_wet_mask
        fy *= self.y_wet_mask

        out1 = fx * 0.5 * (self.dyt + np.roll(self.dyt, -1, axis=-2))
        out1 -= (
            np.roll(fx, 1, axis=-2) * 0.5 * (self.dyt + np.roll(self.dyt, 1, axis=-2))
        )
        out1 /= self.area_t

        out2 = fy * 0.5 * (self.dxt + np.roll(self.dxt, -1, axis=-1))
        out2 -= (
            np.roll(fy, 1, axis=-1) * 0.5 * (self.dxt + np.roll(self.dxt, 1, axis=-1))
        )
        out2 /= self.area_t
        return out1 + out2


ALL_KERNELS[GridType.MOM5T] = MOM5LaplacianT


@dataclass
class CartesianLaplacianWithLandMask(BaseLaplacian):
    """̵Laplacian for regularly spaced Cartesian grids with land mask.

    Attributes
    ----------
    wet_mask: Mask array, 1 for ocean, 0 for land
    """

    wet_mask: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet_mask)

        self.wet_fac = (
            np.roll(self.wet_mask, -1, axis=-1)
            + np.roll(self.wet_mask, 1, axis=-1)
            + np.roll(self.wet_mask, -1, axis=-2)
            + np.roll(self.wet_mask, 1, axis=-2)
        )

    def __call__(self, field: ArrayType):
        np = get_array_module(field)

        out = np.nan_to_num(field)  # set all nans to zero
        out = self.wet_mask * out

        out = (
            -self.wet_fac * out
            + np.roll(out, -1, axis=-1)
            + np.roll(out, 1, axis=-1)
            + np.roll(out, -1, axis=-2)
            + np.roll(out, 1, axis=-2)
        )

        out = self.wet_mask * out
        return out


ALL_KERNELS[GridType.CARTESIAN_WITH_LAND] = CartesianLaplacianWithLandMask


@dataclass
class IrregularCartesianLaplacianWithLandMask(BaseLaplacian):
    """̵Laplacian for irregularly spaced Cartesian grids with land mask.

    Attributes
    ----------
    wet_mask: Mask array, 1 for ocean, 0 for land
    dxw: x-spacing centered at western cell edge
    dyw: y-spacing centered at western cell edge
    dxs: x-spacing centered at southern cell edge
    dys: y-spacing centered at southern cell edge
    area: cell area
    """

    wet_mask: ArrayType
    dxw: ArrayType
    dyw: ArrayType
    dxs: ArrayType
    dys: ArrayType
    area: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet_mask)

        self.w_wet_mask = self.wet_mask * np.roll(self.wet_mask, -1, axis=-1)
        self.s_wet_mask = self.wet_mask * np.roll(self.wet_mask, -1, axis=-2)

    def __call__(self, field: ArrayType):
        np = get_array_module(field)

        out = np.nan_to_num(field)

        wflux = (
            (out - np.roll(out, -1, axis=-1)) / self.dxw * self.dyw
        )  # flux across western cell edge
        sflux = (
            (out - np.roll(out, -1, axis=-2)) / self.dys * self.dxs
        )  # flux across southern cell edge

        wflux = wflux * self.w_wet_mask  # no-flux boundary condition
        sflux = sflux * self.s_wet_mask  # no-flux boundary condition

        out = np.roll(wflux, 1, axis=-1) - wflux + np.roll(sflux, 1, axis=-2) - sflux

        out = out / self.area
        return out


ALL_KERNELS[
    GridType.IRREGULAR_CARTESIAN_WITH_LAND
] = IrregularCartesianLaplacianWithLandMask


def required_grid_vars(grid_type: GridType):
    """Utility function for figuring out the required grid variables
    needed by each grid type.

    Parameters
    ----------
    grid_type : GridType
        The grid type

    Returns
    -------
    grid_vars : list
        A list of names of required grid variables.
    """

    laplacian = ALL_KERNELS[grid_type]
    return laplacian.required_grid_args()
