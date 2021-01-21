"""GPU compatibility stuff."""
import numpy as np


try:
    from cupy import get_array_module
except ImportError:

    def get_array_module(*args):
        return np


# this will work once we require numpy >= 1.20
# ArrayType = np.typing.ArrayLike
ArrayType = np.ndarray
