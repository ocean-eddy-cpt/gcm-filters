"""
Core smoothing routines that operate on 2D arrays.
"""
import enum

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict

from .gpu_compat import ArrayType, get_array_module


# not married to the term "Cartesian"
GridType = enum.Enum("GridType", ["CARTESIAN", "CARTESIAN_WITH_LAND",
                                  "MOM5"])

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
class MOM5Laplacian(BaseLaplacian):
    dxt: ArrayType
    dyt: ArrayType
    dxu: ArrayType
    dyu: ArrayType
    area_u: ArrayType

    def __call__(self, field: ArrayType):
        np = get_array_module()
        fx = (np.roll(field, shift=-1, axis=0) - field) \
                / np.roll(self.dxt, -1, 0)
        fy = (np.roll(field, shift=-1, axis=1) - field) \
             / np.roll(self.dyt, -1, 1)
        filtered_field1 = self.dyu * fx
        filtered_field1 -= np.roll(self.dyu, 1, 0) * np.roll(fx, 1, 0)
        filtered_field1 /= self.area_u
        filtered_field2 = self.dxu * fy
        filtered_field2 -= np.roll(self.dxu, 1, 1) * np.roll(fy, 1, 1)
        filtered_field2 /= self.area_u
        return filtered_field1 + filtered_field2

    def __old_call__(self, field: ArrayType):
        np = get_array_module(field)
        """Uses code by Elizabeth"""
        
        fx = np.empty(field.shape)
        fy = np.empty(field.shape)
        filtered_field = np.empty(field.shape)
        
        for i in range(1,field.shape[0]-1):
            for j in range(field.shape[1]):
                fx[i,j]=(field[i+1,j]-field[i,j])/(self.dxt[i+1,j])

        for i in range(field.shape[0]):
            for j in range(1,field.shape[1]-1):
                fy[i,j]=(field[i,j+1]-field[i,j])/(self.dyt[i,j+1])

        for i in range(1,field.shape[0]-1):
            for j in range(1,field.shape[1]-1):
                filtered_field[i,j]=((self.dyu[i,j]*fx[i,j]-self.dyu[i-1,
                                                                     j]*fx[
                    i-1,j])/self.area_u[i,j])+\
                                    ((self.dxu[i,j]*fy[i,j]-self.dxu[i,
                                                                     j-1]*fy[
                                        i,j-1])/self.area_u[i,j])
        return filtered_field


ALL_KERNELS[GridType.MOM5] = MOM5Laplacian




@dataclass
class CartesianLaplacianWithLandMask(BaseLaplacian):
    """̵Laplacian for regularly spaced Cartesian grids with land mask.

    Attributes
    ----------
    wet_mask: Mask array, 1 for ocean, 0 for land
    """

    wet_mask: ArrayType

    def __call__(self, field: ArrayType):
        np = get_array_module(field)

        out = field.copy()  # is this necessary?
        out = np.nan_to_num(field)  # is this necessary?
        out = self.wet_mask * out

        fac = (
            np.roll(self.wet_mask, -1, axis=-1)
            + np.roll(self.wet_mask, 1, axis=-1)
            + np.roll(self.wet_mask, -1, axis=-2)
            + np.roll(self.wet_mask, 1, axis=-2)
        )

        out = (
            -fac * out
            + np.roll(out, -1, axis=-1)
            + np.roll(out, 1, axis=-1)
            + np.roll(out, -1, axis=-2)
            + np.roll(out, 1, axis=-2)
        )

        out = self.wet_mask * out
        return out


ALL_KERNELS[GridType.CARTESIAN_WITH_LAND] = CartesianLaplacianWithLandMask


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
