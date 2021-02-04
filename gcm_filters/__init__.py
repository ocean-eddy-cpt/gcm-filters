"""
Diffusion-based smoothers for coarse-graining GCM data.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


from .filter import Filter, FilterShape
from .kernels import GridType, required_grid_vars


__all__ = ["Filter", "FilterShape", "GridType"]
