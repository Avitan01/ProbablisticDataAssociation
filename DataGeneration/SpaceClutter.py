from scipy import stats
import numpy as np


class SpaceClutter:
    LEO_NUM = 4028
    MEO_NUM = 131
    GEO_NUM = 511
    EARTH_RADIUS = 6378

    def __init__(self, view_angle: float, std: float, clutter_size: int,
                 LEO_mean: float, GEO_mean: float, LEO_var: float = 1000,
                 GEO_var: float = 2000):
        self._view_angle = view_angle
        self._std = std
        self._clutter_size = clutter_size
        self._LEO_norm = stats.norm(loc=LEO_mean, scale=LEO_var)
        self._GEO_norm = stats.norm(loc=GEO_mean, scale=GEO_var)
        self._MEO_ray = stats.rayleigh(scale=np.mean((LEO_mean, GEO_mean)))

    def generate_clutter(self) -> set:
        radius_clutter = np.concatenate(
            (self._LEO_norm.rvs(self.LEO_NUM),
             self._MEO_ray.rvs(self.MEO_NUM),
             self._GEO_norm.rvs(self.GEO_NUM))
        )
        radius_clutter[radius_clutter < 0] = abs(radius_clutter[radius_clutter < 0])  # make sure there are no negative values
        radius_clutter += self.EARTH_RADIUS  # Shift by earth radius
        angles = np.random.uniform(0, np.deg2rad(self._view_angle), size=len(radius_clutter))
        x_clutter = radius_clutter * np.cos(angles)
        y_clutter = radius_clutter * np.sin(angles)
        clutter = {(x_clutter[idx], y_clutter[idx])
                   for idx in range(len(radius_clutter))}
        return clutter
