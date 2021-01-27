# Script parameters
data_location = '/media/arthur/DATA/Data sets/CM2.6'
grid_filename = 'grid_dataforeli'
uv_filename = 'uv_dataforeli'

from os.path import join

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from gcm_filters import filter


# Load the data
grid_data = xr.open_zarr(join(data_location, grid_filename))
uv_data = xr.open_zarr(join(data_location, uv_filename))

# Try a cartesian filter
data = uv_data.isel(time=0)['usurf']
data = data.fillna(0.0)
print(data)
dx_min = np.min(grid_data['dxu'].values)
print('dx_min: {}'.format(dx_min))
cartesian_filter = filter.Filter(4 * 1e4, dx_min, grid_vars=grid_data)

filtered_data = cartesian_filter.apply(data, ['xu_ocean', 'yu_ocean'])
filtered_data.plot(vmin=-1, vmax=1)
plt.show()

