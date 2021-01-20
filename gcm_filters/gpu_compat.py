"""GPU compatibility stuff."""


try:
    from cupy import get_array_module
except ImportError:
    import numpy as np

    def get_array_module(*args):
        return np
