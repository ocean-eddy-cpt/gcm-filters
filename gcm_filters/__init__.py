"""
ioos_pkg_skeleton is not a reak package, just a set of best practices examples.
"""

from .gcm_filters import meaning_of_life, meaning_of_life_url


__all__ = [
    "meaning_of_life",
    "meaning_of_life_url",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
