# Simple Gaussian filtering
# Script parameters

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter as raw_gaussian_filter
from read_data import read_data


def _gaussian_kernel(filter_scale: float, truncate: int):
    """Return the weights of a Gaussian kernel (only one side)"""
    xs = np.arange(int(filter_scale * truncate) + 1)
    weights = np.exp(-xs**2 / (2 * filter_scale**2))
    return weights / (2 * np.sum(weights) - weights[0])

def _gaussian_filter(array: np.ndarray, filter_scale: float,
                        truncate: int = 4, out: np.ndarray = None)\
        -> np.ndarray:
    """Applies a Gaussian kernel in 2d, by filtering along each dim
    by turn."""
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
    float, mode='own')-> xr.Dataset:
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
    if mode == 'own':
        func = lambda x: _gaussian_filter(x, scale)
    else:
        func = lambda x: raw_gaussian_filter(x, scale)
    normalization = xr.apply_ufunc(func, areas_u)
    filtered_data = xr.apply_ufunc(func, data * areas_u)
    return filtered_data / normalization

if __name__ == '__main__':
    grid, data = read_data()
    grid = grid.compute()
    data = data.isel(time=8)['usurf'].compute()
    data_ = data.fillna(0.)
    filtered_data = gaussian_filter(data_, grid, scale=4, mode='scipy')
    filtered_data = xr.where(np.isnan(data.values), np.nan,
                             filtered_data)
    filtered_data2 = gaussian_filter(data_, grid, scale=4)
    filtered_data2 = xr.where(np.isnan(data.values), np.nan,
                             filtered_data2)
    filtered_data2.sel(yu_ocean=slice(20, 60), xu_ocean=slice(-80, -30)).plot(
        vmin=-1, vmax=1, cmap='jet')
    plt.show()