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
        "REGULAR",
        "REGULAR_WITH_LAND",
        "IRREGULAR_WITH_LAND",
        "TRIPOLAR_REGULAR_WITH_LAND",
        "TRIPOLAR_POP_WITH_LAND",
    ],
)

ALL_KERNELS = {}  # type: Dict[GridType, Any]


def _prepare_tripolar_exchanges(field):
    """Auxiliary function that prepares T-field for northern boundary exchanges on tripolar grid"""
    np = get_array_module(field)

    folded = field[..., [-1], :]  # grab northernmost row
    folded = folded[..., ::-1]  # mirror it
    field_extended = np.concatenate((field, folded), axis=-2)  # append it
    return field_extended


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
class RegularLaplacian(BaseLaplacian):
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


ALL_KERNELS[GridType.REGULAR] = RegularLaplacian


@dataclass
class RegularLaplacianWithLandMask(BaseLaplacian):
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


ALL_KERNELS[GridType.REGULAR_WITH_LAND] = RegularLaplacianWithLandMask


@dataclass
class IrregularLaplacianWithLandMask(BaseLaplacian):
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

        self.w_wet_mask = self.wet_mask * np.roll(self.wet_mask, 1, axis=-1)
        self.s_wet_mask = self.wet_mask * np.roll(self.wet_mask, 1, axis=-2)

    def __call__(self, field: ArrayType):
        np = get_array_module(field)

        out = np.nan_to_num(field)

        wflux = (
            (out - np.roll(out, 1, axis=-1)) / self.dxw * self.dyw
        )  # flux across western cell edge
        sflux = (
            (out - np.roll(out, 1, axis=-2)) / self.dys * self.dxs
        )  # flux across southern cell edge

        wflux = wflux * self.w_wet_mask  # no-flux boundary condition
        sflux = sflux * self.s_wet_mask  # no-flux boundary condition

        out = np.roll(wflux, -1, axis=-1) - wflux + np.roll(sflux, -1, axis=-2) - sflux

        out = out / self.area
        return out


ALL_KERNELS[GridType.IRREGULAR_WITH_LAND] = IrregularLaplacianWithLandMask


@dataclass
class TripolarRegularLaplacianTpoint(BaseLaplacian):
    """̵Laplacian for fields defined at T-points on POP tripolar grid geometry with land mask, but assuming that dx = dy = 1

    Attributes
    ----------
    wet_mask: Mask array, 1 for ocean, 0 for land
    """

    wet_mask: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet_mask)

        # check that southernmost row of wet mask has only zeros
        if self.wet_mask[..., 0, :].any():
            raise AssertionError("Wet mask requires zeros in southernmost row")

        wet_mask_extended = _prepare_tripolar_exchanges(self.wet_mask)
        self.wet_fac = (
            np.roll(wet_mask_extended, -1, axis=-1)
            + np.roll(wet_mask_extended, 1, axis=-1)
            + np.roll(wet_mask_extended, -1, axis=-2)
            + np.roll(wet_mask_extended, 1, axis=-2)
        )  # todo: inherit this operation from CartesianLaplacianWithLandMask

    def __call__(self, field: ArrayType):
        np = get_array_module(field)

        data = np.nan_to_num(field)  # set all nans to zero
        data = self.wet_mask * data
        data = _prepare_tripolar_exchanges(data)

        out = (
            -self.wet_fac * data
            + np.roll(data, -1, axis=-1)
            + np.roll(data, 1, axis=-1)
            + np.roll(data, -1, axis=-2)
            + np.roll(data, 1, axis=-2)
        )  # todo: inherit this operation from CartesianLaplacianWithLandMask

        out = out[..., :-1, :]  # disregard appended row

        out = self.wet_mask * out
        return out


ALL_KERNELS[GridType.TRIPOLAR_REGULAR_WITH_LAND] = TripolarRegularLaplacianTpoint


@dataclass
class POPTripolarLaplacianTpoint(BaseLaplacian):
    """̵Laplacian for irregularly spaced Cartesian grids with land mask.

    Attributes
    ----------
    wet_mask: Mask array, 1 for ocean, 0 for land; can be obtained via xr.where(KMT>0, 1, 0)
    dxe: x-spacing centered at eastern T-cell edge, provided by model diagnostic HUS(nlat, nlon)
    dye: y-spacing centered at eastern  T-cell edge, provided by model diagnostic HTE(nlat, nlon)
    dxn: x-spacing centered at northern T-cell edge, provided by model diagnostic HTN(nlat, nlon)
    dyn: y-spacing centered at northern T-cell edge, provided by model diagnostic HUW(nlat, nlon)
    tarea: cell area, provided by model diagnostic TAREA(nlat, nlon)
    """

    wet_mask: ArrayType
    dxe: ArrayType
    dye: ArrayType
    dxn: ArrayType
    dyn: ArrayType
    tarea: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet_mask)

        # check that southernmost row of wet mask has only zeros
        if self.wet_mask[..., 0, :].any():
            raise AssertionError("Wet mask requires zeros in southernmost row")

        # prepare grid information for northern boundary exchanges
        self.dxe = _prepare_tripolar_exchanges(self.dxe)
        self.dye = _prepare_tripolar_exchanges(self.dye)
        self.dxn = _prepare_tripolar_exchanges(self.dxn)
        self.dyn = _prepare_tripolar_exchanges(self.dyn)
        self.wet_mask = _prepare_tripolar_exchanges(self.wet_mask)

        self.e_wet_mask = self.wet_mask * np.roll(self.wet_mask, -1, axis=-1)
        self.n_wet_mask = self.wet_mask * np.roll(self.wet_mask, -1, axis=-2)

    def __call__(self, field: ArrayType):
        np = get_array_module(field)
        data = np.nan_to_num(field)  # set all nans to zero

        # prepare data for northern boundary exchanges
        data = _prepare_tripolar_exchanges(data)

        eflux = (
            (np.roll(data, -1, axis=-1) - data) / self.dxe * self.dye
        )  # flux across eastern T-cell edge
        nflux = (
            (np.roll(data, -1, axis=-2) - data) / self.dyn * self.dxn
        )  # flux across northern T-cell edge

        eflux = eflux * self.e_wet_mask  # no-flux boundary condition
        nflux = nflux * self.n_wet_mask  # no-flux boundary condition

        out = eflux - np.roll(eflux, 1, axis=-1) + nflux - np.roll(nflux, 1, axis=-2)

        out = out[..., :-1, :]  # disregard appended row
        out = out / self.tarea
        return out


ALL_KERNELS[GridType.TRIPOLAR_POP_WITH_LAND] = POPTripolarLaplacianTpoint


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
