import numpy as np
import matplotlib.pyplot as plt
from gcm_filters import filter

from read_data import read_data

grid_data, uv_data = read_data()

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

