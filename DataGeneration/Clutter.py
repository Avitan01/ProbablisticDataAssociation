import scipy.stats as dist

class Clutter:
    """create a clutter based on chosen distribiution. number of false alarmas is always 100
    The clutter class creates a seperate distribiution for x and y"""
    
    def __init__(self, origin: tuple,
                 dist_type: str, 
                 norm_std = None, 
                 uni_low_bound = None) -> None:
        """
        Args:
        dist_type - distribiution of the clutter. the options are:
            "normal" - for normal distribiution 
         """
        self._num_of_false_targets = 100 # constant for now

        match dist_type:
            case "normal":
                self._mean_x = origin[0]
                self._mean_y = origin[1]
                self._std = norm_std
                self._x_dist = dist.norm(loc = self._mean_x, scale = self._std)
                self._y_dist = dist.norm(loc = self._mean_y, scale = self._std)
                self.generate_clutter()

            case "uniform":
                self.low_bound = uni_low_bound
                self._mean_x = origin[0]
                self._mean_y = origin[1]
                self.high_bound_x = 2 * self._mean_x - self.low_bound
                self.high_bound_y = 2 * self._mean_y - self.low_bound
                self._x_dist = dist.uniform(loc = self.low_bound, scale = self.high_bound_x - self.low_bound)
                self._y_dist = dist.uniform(loc = self.low_bound, scale = self.high_bound_y - self.low_bound)
                self.generate_clutter()

            case "Poiss":
                self._mean_x = origin[0] #lambda parameter
                self._mean_y = origin[1]
                self._x_dist = dist.poisson(mu = self._mean_x)
                self._y_dist = dist.poisson(mu = self._mean_y)
                self.generate_clutter()
        

    def generate_clutter(self) -> set:
        "Returns a set of points distributed with the chosen distribiution"
        self._x_clutter = self._x_dist.rvs(size=self._num_of_false_targets)
        self._y_clutter = self._y_dist.rvs(size=self._num_of_false_targets)
        clutter = {(self._x_clutter[idx],self._y_clutter[idx]) 
                        for idx in range(self._num_of_false_targets)}
        return clutter