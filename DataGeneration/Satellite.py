import numpy as np
import matplotlib.pyplot as plt


class Satellite:
    EARTH_RADIUS = 6378

    """Simulates a non-maneuvering target with near constant velocity and system noise as acceleration"""

    def __init__(self, initial_r: float,
                 initial_theta: float,
                 dt: float,
                 simulation_duration: float,
                 orbit_time: float,
                 system_variance: float) -> None:
        """
        Args:
            initial_r(float): initial radius of satalie [km]
            initial_theta(float): initial real anomolity angle [km]
            dt(float): advancement in time [s]
            simulation_duration(float): overall time of simulation [s]
            orbit_time(float): the time it takes for the satalite to finish an orbit [m/s]
            system_variance(float): the variance of the noise that drives the system
        """
        self._radius_zero = self.EARTH_RADIUS + initial_r
        self._noise_var = system_variance
        self._r = np.random.normal(loc=self._radius_zero, scale=self._noise_var)
        self._theta = np.deg2rad(initial_theta)
        self._time = simulation_duration
        self._omega = (2 * np.pi) / orbit_time
        self._x = []
        self._y = []
        self._trajectory_x = []
        self._trajectory_y = []
        self._radius = []
        self._angle = []
        self._time_vector = []
        self._dt = dt
        self._time_vector = np.arange(0, self._time, self._dt)
        self.initiate()

    def initiate(self) -> None:
        """Run simulation through time and create the trajectory vectors"""
        for curr_time in self._time_vector:
            if curr_time > self._time:
                break
            if self._theta % (2 * np.pi) == 0:
                self._theta = 0
            self._x = self._r * np.cos(self._theta)
            self._y = self._r * np.sin(self._theta)
            self._trajectory_x.append(self._x)
            self._trajectory_y.append(self._y)
            self._radius.append(self._r)
            self._angle.append(self._theta)
            # noise = np.random.normal(scale=self._noise_var)
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
                          orbit_time=24 * 3600,
                          dt=1,
                          simulation_duration=25 * 3600,
                          system_variance=100)


    plt.plot(satellite.x_trajectory, satellite.y_trajectory)
    plt.show()
