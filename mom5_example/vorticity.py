import numpy as np
import xarray as xr

def compute_vorticity(field: np.ndarray, grid: np.ndarray):
    """Uses code by Elizabeth"""
    grid = grid / 1e4
    u = field['usurf'].values
    v = field['vsurf'].values
    dxt = grid['dxt'].values
    dyt = grid['dyt'].values
    vorticity = np.empty(u.shape)

    for i in range(1, vorticity.shape[0] - 1):
        for j in range(1, vorticity.shape[1] - 1):
            dvdx = 0.5 * ((v[i, j] - v[i - 1, j]) / (
                        0.5 * (dxt[i, j] + dxt[i, j + 1])) \
                          + (v[i, j - 1] - v[i - 1, j - 1]) / (0.5 * (
                                dxt[i, j] + dxt[i, j - 1])))
            dudy = 0.5 * ((u[i, j] - u[i, j - 1]) / (
                        0.5 * (dyt[i, j] + dyt[i + 1, j])) \
                          + (u[i - 1, j] - u[i - 1, j - 1]) / (0.5 * (
                                dyt[i, j] + dyt[i - 1, j])))
            vorticity[i, j] = dvdx - dudy
    xt_ocean = grid['dxt'].xt_ocean
    yt_ocean = grid['dxt'].yt_ocean
    vorticity = xr.DataArray(vorticity,
                             coords=dict(xt_ocean=xt_ocean,yt_ocean=yt_ocean),
                             dims=('yt_ocean','xt_ocean'))
    return vorticity