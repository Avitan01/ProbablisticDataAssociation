from scipy import stats
import numpy as np


# class clutter - choose the dist (with switches) and have it with mean 0
# generation - create the data then move it to desired mean then returns it


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
                # Calculate parameters for the log-normal distribution based on the desired standard deviation
                # For a log-normal distribution, sigma = sqrt(ln(std^2 / (std^2 + mean^2)))
                # sigma = np.sqrt(np.abs(np.log(self.std ** 2 / (self.std ** 2 + 1))))

                # Calculate the corresponding mean for the log-normal distribution
                # mean = sqrt(std^2 / sqrt(std^2 + 1))
                # mean = np.sqrt(self.std ** 2 / np.sqrt(self.std ** 2 + 1))

                # Create a log-normal distribution with the calculated parameters
                # self._r_dist = stats.lognorm(s=sigma, scale=np.exp(mean))

                # Generate random numbers from the log-normal distribution
                # lognormal_samples = lognormal_dist.rvs(size=1000)  # You can change the sample size as needed
                self._r_dist = stats.norm(loc=mean_x, scale=self.std)
                # self._r_dist = stats.lognorm(s=np.exp(self.std))
                    # s=self.std, scale=np.exp(mean_x))
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
