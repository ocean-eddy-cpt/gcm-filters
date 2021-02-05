import numpy as np
import matplotlib.pyplot as plt
from gcm_filters import filter
from gcm_filters.kernels import GridType
from gaussian_filter import gaussian_filter
import xarray as xr

from read_data import read_data

grid_data, uv_data = read_data()
data = uv_data['usurf'].isel(time=0).compute()
print(data)

# Try a cartesian filter
grid_data = grid_data.compute().reset_coords()
grid_data = grid_data.rename(dict(wet='wet_mask'))
grid_data = grid_data.interp(dict(xt_ocean=grid_data.xu_ocean,
                                  yt_ocean=grid_data.yu_ocean))

cartesian_filter = filter.Filter(20, 1, n_steps=30,
                                 filter_shape=filter.FilterShape.GAUSSIAN,
                                 grid_vars=dict(wet_mask=grid_data['wet_mask']),
                                 grid_type=GridType.MOM5)

filtered_data = cartesian_filter.apply(data, ['yu_ocean', 'xu_ocean'])

data_ = data.fillna(0.)
grid_data = grid_data.rename(dict(wet_mask='wet'))
filtered_data2 = gaussian_filter(data_, grid_data, 20, mode='own')
filtered_data2 = xr.where(np.isnan(data.values), np.nan,
                         filtered_data2)


(filtered_data - filtered_data2).plot(vmin=-1, vmax=1)
plt.show()

