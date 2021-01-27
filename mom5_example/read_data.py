from os.path import join

import xarray as xr

# Script parameters
data_location = '/media/arthur/DATA/Data sets/CM2.6'
grid_filename = 'grid_dataforeli'
uv_filename = 'uv_dataforeli'

# Load the data
def read_data():
    grid_data = xr.open_zarr(join(data_location, grid_filename))
    uv_data = xr.open_zarr(join(data_location, uv_filename))
    return grid_data, uv_data