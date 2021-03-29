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
        "VECTOR_C_GRID",
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

        self.w_wet_mask = (self.wet_mask * np.roll(self.wet_mask, 1, axis=-1) * self.kappa_w)
        self.s_wet_mask = (self.wet_mask * np.roll(self.wet_mask, 1, axis=-2) * self.kappa_s)

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


ALL_KERNELS[
    GridType.IRREGULAR_CARTESIAN_WITH_LAND
] = IrregularCartesianLaplacianWithLandMask


@dataclass
class VectorLaplacian(BaseLaplacian):
    """̵Vector Laplacian on C-Grid.

    Attributes
    ----------
    wet_mask_t: Mask array for t points, 1 for ocean, 0 for land
    wet_mask_q: Mask array for q points, 1 for ocean, 0 for land
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
        str_xx = - (self.kappa_iso + 0.5 * self.kappa_aniso) * str_xx  # multiply by viscosity in x-direction

        dvfield_dx = self.dy_dxBu * (
            np.roll(vfield / self.dyCv, -1, axis=-1) - vfield / self.dyCv
        )
        dufield_dy = self.dx_dyBu * (
            np.roll(ufield / self.dxCu, -1, axis=-2) - ufield / self.dxCu
        )
        str_xy = dvfield_dx + dufield_dy  # horizontal shear strain
        str_xy = -self.kappa_iso * str_xy  # multiply by viscosity in y-direction

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
        u_component /= self.area_u

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
        v_component /= self.area_v

        return (u_component, v_component)


ALL_KERNELS[GridType.VECTOR_C_GRID] = VectorLaplacian


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
