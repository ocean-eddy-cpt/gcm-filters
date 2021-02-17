from os.path import join

import xarray as xr

# Load the data
def read_data(data_location: str, uv_filename: str, grid_filename: str):
    grid_data = xr.open_zarr(join(data_location, grid_filename))
    uv_data = xr.open_zarr(join(data_location, uv_filename))
    return grid_data, uv_data