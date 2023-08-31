from scipy import stats
import numpy as np


class Clutter:
    """creates a clutter of measurements based on chosen distribution. choose the distribution
    then to center it around a desired point using the generate_clutter function"""

    def __init__(self,
                 dist_type: str,
                 std: float,
                 clutter_size: int) -> None:
        """Args:
            dist_type(str): Distribution of the clutter. the options are:
                            "normal" - for normal distribution.
                            "uniform" - for uniform distribution.
            std(float): Standard deviation of the distribution.
         """
        self.dist = dist_type
        self.std = std
        self.clutter_size = clutter_size
        mean_x = 0

        match self.dist.lower():
            case "normal":
                self._r_dist = stats.norm(loc=mean_x, scale=self.std)
            case "uniform":
                self._r_dist = stats.uniform(loc=mean_x, scale=self.std)
            case 'log normal':
                self._r_dist = stats.norm(loc=mean_x, scale=self.std)
            case 'rayleigh':
                self._r_dist = stats.rayleigh(scale=self.std)

    def generate_clutter(self, mean: tuple) -> set:
        """Returns a clutter with chosen distribution for object Clutter and a given mean
            Args:
                mean - the (x,y) coordinate that are then mean of the distribution """
        angles = np.random.uniform(0, 2 * np.pi, size=self.clutter_size)
        radius_clutter = self._r_dist.rvs(size=self.clutter_size)
        if self.dist == 'log normal':
            radius_clutter = np.exp(radius_clutter)
        x_clutter = radius_clutter * np.cos(angles)
        y_clutter = radius_clutter * np.sin(angles)
        clutter = {(x_clutter[idx] + mean[0], y_clutter[idx] + mean[1])
                   for idx in range(self.clutter_size)}
        return clutter
