import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_grid(values, resolution):
    """
    Resize grid of values from its size to the shape resolution x resolution.
    Use linear interpolation for points in between current points.

    Parameters
    ----------
    values
        Two-dimensional grid of values
    resolution
        Resolution of resulting grid

    Returns
    -------
    The output grid with shape resolution x resolution
    """
    x = np.linspace(0, 1, values.shape[1])
    y = np.linspace(0, 1, values.shape[0])
    interpolator = RegularGridInterpolator((x, y), values)

    points = np.linspace(0, 1, resolution)
    xv, yv = np.meshgrid(points, points)
    return interpolator((yv, xv))
