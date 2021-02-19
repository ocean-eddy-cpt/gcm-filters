import numpy as np
import matplotlib.pyplot as plt
from gcm_filters import filter
from gcm_filters.kernels import GridType
from gaussian_filter import gaussian_filter
import xarray as xr

# Script parameters
filter_scale=4

from read_data import read_data

grid_data, data = read_data()
grid_data = grid_data.compute().reset_coords()

data = data['usurf'].isel(time=0).sel(xu_ocean=slice(-100, -0),
                                      yu_ocean=slice(-50, 50)).compute()
# How do we deal with continents?
grid_data = grid_data.sel(xu_ocean=slice(-100, 0), yu_ocean=slice(-50, 50))
grid_data = grid_data[['dxt', 'dyt']]

# PROBLEM 1: we need the grid vars to be on the same grid as the field to
# be filtered. Here I've interpolated, but I think this is not right,
# we can probably just redefine the coords of dxt and dyt to be those of the
# velocities, we'll just have to be careful.
grid_data = grid_data.interp(dict(xt_ocean=data.xu_ocean,
                                  yt_ocean=data.yu_ocean))

# Still not sure about the parameter dxmin
cartesian_filter = filter.Filter(filter_scale, dx_min=1, n_steps=10,
                                 filter_shape=filter.FilterShape.GAUSSIAN,
                                 grid_vars=grid_data[['dxt', 'dyt']],
                                 grid_type=GridType.MOM5)
print(cartesian_filter.filter_spec)

filtered_data = cartesian_filter.apply(data, dims=['yu_ocean', 'xu_ocean'])

# Plot
plt.figure()
filtered_data.plot(vmin=-0.1, vmax=0.1)
plt.figure()
data.plot(vmin=-0.1, vmax=0.1)
plt.show()

