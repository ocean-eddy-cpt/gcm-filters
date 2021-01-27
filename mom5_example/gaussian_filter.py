# Simple Gaussian filtering
# Script parameters
scale = 4

import matplotlib.pyplot as plt
import xarray as xr
from scipy.ndimage import gaussian_filter as raw_gaussian_filter
from read_data import read_data


grid, data = read_data()
grid = grid.compute()
data = data.isel(time=0)['usurf'].compute()

def gaussian_filter(data: xr.Dataset, grid: xr.Dataset, scale: \
    float)-> xr.Dataset:
    """
    Applies a Gaussian filter, taking into account the grid areas.

    Parameters
    ----------
    data: data to be filtered
    scale: scale of the filters, in number of grid points
    """
    # TODO check the areas are at the right location
    areas_u = grid['dxu'] * grid['dyu']
    # The normalization term is such that applying this function to a constant
    # field returns the same constant field
    # TODO add a test based on this
    func = lambda x: raw_gaussian_filter(x, scale)
    normalization = xr.apply_ufunc(func, areas_u)
    filtered_data = xr.apply_ufunc(func, data * areas_u)
    return filtered_data / normalization


filtered_data = gaussian_filter(data, grid, 5)
print(filtered_data)
filtered_data.plot()
plt.show()