"""
Core smoothing routines that operate on 2D arrays.
"""
from abc import ABC
from dataclasses import dataclass
import enum
from typing import Union

import numpy as np

from .gpu_compat import get_array_module

# this will work once we require numpy >= 1.20
# ArrayType = np.typing.ArrayLike
ArrayType = np.ndarray

# not married to the term "Cartesian"
GridType = enum.Enum("GridType", ["CARTESIAN", "CARTESIAN_WITH_LAND"])

ALL_KERNELS = {}

class AbstractLaplacian(ABC):

    def __call__(self, field):
        pass


class CartesianLaplcian(AbstractLaplacian):
    """̵Laplacian for regularly spaced Cartesian grids.
    """

    def __call__(self, field: ArrayType):
        np = get_array_module(field)
        return (
            -4 * field
            + np.roll(field, -1, axis=-1)
            + np.roll(field, 1, axis=-1)
            + np.roll(field, -1, axis=-2)
            + np.roll(field, 1, axis=-2)
        )


ALL_KERNELS[GridType.CARTESIAN] = CartesianLaplcian


@dataclass
class CartesianLaplcianWithLandMask(AbstractLaplacian):
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


ALL_KERNELS[GridType.CARTESIAN_WITH_LAND] = CartesianLaplcianWithLandMask
