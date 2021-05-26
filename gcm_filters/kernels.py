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
        "VECTOR_C_GRID",
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
class BaseScalarLaplacian(ABC):
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
class BaseVectorLaplacian(ABC):
    def __call__(self, ufield, vfield):
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
class RegularLaplacian(BaseScalarLaplacian):
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
class RegularLaplacianWithLandMask(BaseScalarLaplacian):
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
class IrregularLaplacianWithLandMask(BaseScalarLaplacian):
    """̵Laplacian for irregularly spaced Cartesian grids with land mask.

    Attributes
    ----------
    wet_mask: Mask array, 1 for ocean, 0 for land
    dxw: x-spacing centered at western cell edge
    dyw: y-spacing centered at western cell edge
    dxs: x-spacing centered at southern cell edge
    dys: y-spacing centered at southern cell edge
    area: cell area
    kappa_w:  zonal diffusivity centered at western cell edge
    kappa_s:  zonal diffusivity centered at southern cell edge
    """

    wet_mask: ArrayType
    dxw: ArrayType
    dyw: ArrayType
    dxs: ArrayType
    dys: ArrayType
    area: ArrayType
    kappa_w: ArrayType
    kappa_s: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet_mask)

        # derive wet mask for western cell edge from wet_mask at T points via
        # w_wet_mask(j,i) = wet_mask(j,i) * wet_mask(j,i-1)
        # note: wet_mask(j,i-1) corresponds to np.roll(wet_mask, +1, axis=-1)
        self.w_wet_mask = (
            self.wet_mask * np.roll(self.wet_mask, 1, axis=-1) * self.kappa_w
        )

        # derive wet mask for southern cell edge from wet_mask at T points via
        # s_wet_mask(j,i) = wet_mask(j,i) * wet_mask(j-1,i)
        # note: wet_mask(j-1,i) corresponds to np.roll(wet_mask, +1, axis=-2)
        self.s_wet_mask = (
            self.wet_mask * np.roll(self.wet_mask, 1, axis=-2) * self.kappa_s
        )

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
class TripolarRegularLaplacianTpoint(BaseScalarLaplacian):
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
class POPTripolarLaplacianTpoint(BaseScalarLaplacian):
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

        # derive wet mask for eastern cell edge from wet_mask at T points via
        # e_wet_mask(j,i) = wet_mask(j,i) * wet_mask(j,i+1)
        # note: wet_mask(j,i+1) corresponds to np.roll(wet_mask, -1, axis=-1)
        self.e_wet_mask = self.wet_mask * np.roll(self.wet_mask, -1, axis=-1)

        # derive wet mask for northern cell edge from wet_mask at T points via
        # n_wet_mask(j,i) = wet_mask(j,i) * wet_mask(j+1,i)
        # note: wet_mask(j+1,i) corresponds to np.roll(wet_mask, -1, axis=-2)
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


@dataclass
class CgridVectorLaplacian(BaseVectorLaplacian):
    """̵Vector Laplacian on C-Grid.

    Attributes
    ----------
    wet_mask_t: Mask array for t points, 1 for ocean, 0 for land
    wet_mask_q: Mask array for q (vorticity) points, 1 for ocean, 0 for land
    dxT: x-spacing centered at t points
    dyT: y-spacing centered at t points
    dxCu: x-spacing centered at u points
    dyCu: y-spacing centered at u points
    dxCv: x-spacing centered at v points
    dyCv: y-spacing centered at v points
    dxBu: x-spacing centered at q points
    dyBu: y-spacing centered at q points
    area_u: U-cell area
    area_v: V-cell area
    kappa_iso: isotropic viscosity
    kappa_aniso: anisotropic viscosity aligned with x-direction
    """

    wet_mask_t: ArrayType
    wet_mask_q: ArrayType
    dxT: ArrayType
    dyT: ArrayType
    dxCu: ArrayType
    dyCu: ArrayType
    dxCv: ArrayType
    dyCv: ArrayType
    dxBu: ArrayType
    dyBu: ArrayType
    area_u: ArrayType
    area_v: ArrayType
    kappa_iso: ArrayType
    kappa_aniso: ArrayType

    def __post_init__(self):
        np = get_array_module(self.wet_mask_t)

        self.dx_dyT = self.dxT / self.dyT * self.wet_mask_t
        self.dy_dxT = self.dyT / self.dxT * self.wet_mask_t
        self.dx_dyBu = self.dxBu / self.dyBu * self.wet_mask_q
        self.dy_dxBu = self.dyBu / self.dxBu * self.wet_mask_q

        self.dx2h = self.dxT * self.dxT
        self.dy2h = self.dyT * self.dyT
        self.dx2q = self.dxBu * self.dxBu
        self.dy2q = self.dyBu * self.dyBu

        # compute reciprocal of areas while avoiding division by zero
        self.recip_area_u = np.where(self.area_u > 0, 1 / self.area_u, 0)
        self.recip_area_v = np.where(self.area_v > 0, 1 / self.area_v, 0)

    def __call__(self, ufield: ArrayType, vfield: ArrayType):
        np = get_array_module(ufield)

        ufield = np.nan_to_num(ufield)
        vfield = np.nan_to_num(vfield)

        dufield_dx = self.dy_dxT * (
            ufield / self.dyCu - np.roll(ufield / self.dyCu, 1, axis=-1)
        )
        dvfield_dy = self.dx_dyT * (
            vfield / self.dxCv - np.roll(vfield / self.dxCv, 1, axis=-2)
        )
        str_xx = dufield_dx - dvfield_dy  # horizontal tension
        # multiply by isotropic viscosity + anisotropic contribution in x-direction
        str_xx = -(self.kappa_iso + 0.5 * self.kappa_aniso) * str_xx

        dvfield_dx = self.dy_dxBu * (
            np.roll(vfield / self.dyCv, -1, axis=-1) - vfield / self.dyCv
        )
        dufield_dy = self.dx_dyBu * (
            np.roll(ufield / self.dxCu, -1, axis=-2) - ufield / self.dxCu
        )
        str_xy = dvfield_dx + dufield_dy  # horizontal shear strain
        str_xy = -self.kappa_iso * str_xy  # multiply by isotropic viscosity

        u_component = (
            1
            / self.dyCu
            * (self.dy2h * str_xx - np.roll(self.dy2h * str_xx, -1, axis=-1))
        )
        u_component += (
            1
            / self.dxCu
            * (np.roll(self.dx2q * str_xy, 1, axis=-2) - self.dx2q * str_xy)
        )
        u_component *= self.recip_area_u

        v_component = (
            1
            / self.dyCv
            * (np.roll(self.dy2q * str_xy, 1, axis=-1) - self.dy2q * str_xy)
        )
        v_component -= (
            1
            / self.dxCv
            * (self.dx2h * str_xx - np.roll(self.dx2h * str_xx, -1, axis=-2))
        )
        v_component *= self.recip_area_v

        return (u_component, v_component)


ALL_KERNELS[GridType.VECTOR_C_GRID] = CgridVectorLaplacian


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
