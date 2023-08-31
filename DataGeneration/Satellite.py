import numpy as np
import matplotlib.pyplot as plt


class Satellite:
    EARTH_RADIUS = 6378
    MU = 3.986 * 10 ** 5  # km^3 / s^2
    """Simulates a non-maneuvering target with near constant velocity and system noise as acceleration"""

    def __init__(self, initial_r: float,
                 initial_theta: float,
                 dt: float,
                 orbits: float,
                 system_variance: float) -> None:
        """
        Args:
            initial_r(float): initial radius of satellite [km]
            initial_theta(float): initial real animosity angle [km]
            dt(float): advancement in time [s]
            orbits(float): The number of orbits to take.
            system_variance(float): the variance of the noise that drives the system
        """
        self._radius_zero = self.EARTH_RADIUS + initial_r
        self._noise_var = system_variance
        self._r = np.random.normal(loc=self._radius_zero, scale=self._noise_var)
        self._theta = np.deg2rad(initial_theta)
        self._orbits = orbits
        self._omega = np.sqrt(self.MU / (initial_r ** 3))
        self._x = []
        self._y = []
        self._trajectory_x = []
        self._trajectory_y = []
        self._radius = []
        self._angle = []
        self._time_vector = []
        self._dt = dt
        self._time = int(np.ceil(self._orbits * 2 * np.pi / self._omega))
        self._time_vector = np.arange(0, self._time, self._dt)
        self.initiate()

    def initiate(self) -> None:
        """Run simulation through time and create the trajectory vectors"""
        for curr_time in self._time_vector:
            if curr_time > self._time:
                break
            if self._theta > (2 * np.pi):
                self._theta -= 2 * np.pi
            self._x = self._r * np.cos(self._theta)
            self._y = self._r * np.sin(self._theta)
            self._trajectory_x.append(self._x)
            self._trajectory_y.append(self._y)
            self._radius.append(self._r)
            self._angle.append(self._theta)
            self._theta += self._dt * self._omega
            self._r += np.random.normal(loc=0, scale=self._noise_var)

    def get_state(self, time: float) -> list | None:
        """Get the location of the target at a given time.
            Args:
                time(float): Required time to find.
            Return:
                list: A vector of the x, y trajectory location, velocities and the time in which it was given."""
        if time > self._time:
            print('Time is rather then flight time')
            return None
        idx = np.argmin(np.abs(self._time_vector - time))
        return [self._trajectory_x[idx],
                self._trajectory_y[idx],
                None,
                None,
                self._time_vector[idx]]

    def get_state_radial(self, time: float) -> list | None:
        """Get the location of the target at a given time.
            Args:
                time(float): Required time to find.
            Return:
                list: A vector of the x, y trajectory location, velocities and the time in which it was given."""
        if time > self._time:
            print('Time is rather then flight time')
            return None
        idx = np.argmin(np.abs(self._time_vector - time))
        return [self._radius[idx],
                self._angle[idx],
                None,
                None,
                self._time_vector[idx]]

    @property
    def x_trajectory(self) -> list:
        return self._trajectory_x

    @property
    def y_trajectory(self) -> list:
        return self._trajectory_y

    @property
    def time_vector(self) -> np.array:
        return self._time_vector

    @property
    def radius(self) -> np.array:
        return self._radius


if __name__ == "__main__":
    satellite = Satellite(initial_r=10000,
                          initial_theta=0,
                          dt=1,
                          orbits=1,
                          system_variance=100)

    plt.plot(satellite.x_trajectory, satellite.y_trajectory)
    plt.show()
