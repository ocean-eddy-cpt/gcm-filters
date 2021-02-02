# Simple Gaussian filtering
# Script parameters
scale = 4

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter as raw_gaussian_filter
from read_data import read_data


grid, data = read_data()
grid = grid.compute()
data = data.isel(time=0)['usurf'].compute()


def _gaussian_kernel(filter_scale: float, truncate: int = 4):
    """Return the weights of a Gaussian kernel (only one side)"""
    xs = np.arange(int(filter_scale * truncate) + 1)
    weights = np.exp(-xs**2 / (2 * filter_scale**2))
    return weights / (2 * np.sum(weights) - weights[0])

def _gaussian_filter(array: np.ndarray, filter_scale: float,
                        truncate: int = 4, out: np.ndarray = None)\
        -> np.ndarray:
    """Applies a Gaussian kernel in 2d, by filtering along each dim
    successfully."""
    weights = _gaussian_kernel(filter_scale, truncate)
    out1 = np.zeros_like(array, dtype=np.float64)
    out2 = np.zeros_like(array, dtype=np.float64)
    for i, weight in enumerate(weights):
        if i == 0:
            out1 += weight * array
        else:
            out1[i: , ...] += weight * array[:-i, ...]
            out1[:-i, ...] += weight * array[i:, ...]
    for i, weight in enumerate(weights):
        if i == 0:
            out2 += weight * out1
        else:
            out2[..., i:] += weight * out1[..., :-i]
            out2[..., :-i] += weight * out1[..., i:]
    return out2


def gaussian_filter(data: xr.Dataset, grid: xr.Dataset, scale: \
    float)-> xr.Dataset:
    """
    Applies a Gaussian filter, taking into account the grid areas,
    to an xarray dataset.

    Parameters
    ----------
    data: data to be filtered
    grid: grid info with variables dxu and dyu
    scale: scale of the filters, in number of grid points
    """
    # TODO check the areas are at the right location
    areas_u = grid['dxu'] * grid['dyu']
    # The normalization term is such that applying this function to a constant
    # field returns the same constant field
    # TODO add a test based on this
    func = lambda x: _gaussian_filter(x, scale)
    normalization = xr.apply_ufunc(func, areas_u)
    filtered_data = xr.apply_ufunc(func, data * areas_u)
    return filtered_data / normalization

# a = np.arange(100).reshape((10,10))
# a = np.array(a, dtype=np.float64)
# a_bar = _gaussian_filter(a, 1)
# plt.imshow(a_bar)
# plt.colorbar()
# plt.show()
# a_bar2 = raw_gaussian_filter(a, 5)
# print(a_bar)
data_ = data.fillna(0.)
filtered_data = gaussian_filter(data_, grid, 5)
filtered_data = xr.where(np.isnan(data.values), np.nan,
                         filtered_data)
print(filtered_data)
filtered_data.plot(vmin=-1, vmax=1)
plt.show()