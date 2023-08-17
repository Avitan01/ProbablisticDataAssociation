import numpy as np


class Target:
    """Simulates a non-maneuvering target with constant velocity"""
    def __init__(self, initial_x: float,
                 initial_y: float,
                 dt: float,
                 simulation_duration: float,
                 velocity_x: float,
                 velocity_y: float) -> None:
        """
        Args:
            initial_x(float): initial location on x axis [m]
            initial_y(float): initial location on y axis [m]
            dt(float): advancement in time
            simulation_duration(float): overall time of simulation
            velocity_x(float): velocity on the x axis [m/s]
            velocity_y(float): velocity on the y axis [m/s]
        """
        self._x = initial_x
        self._y = initial_y
        self._time = simulation_duration
        self._Vx = velocity_x
        self._Vy = velocity_y
        self._trajectory_x = []
        self._trajectory_y = []
        self._time_vector = []
        self._dt = dt
        self._time_vector = np.arange(0, self._time, self._dt)
        self.initiate()

    def initiate(self) -> None:
        """Run simulation through time and create the trajectory vectors"""
        for curr_time in self._time_vector:
            if curr_time > self._time:
                break
            self._trajectory_x.append(self._x)
            self._trajectory_y.append(self._y)
            self._x = self._x + self._Vx * self._dt
            self._y = self._y + self._Vy * self._dt

    def get_state(self, time: float) -> list | None:
        """Get the location of the target at a given time.
            Args:
                time(float): Required time to find.
            Return:
                list: A vector of the x, y trajectory location and the time in which it was given."""
        if time > self._time:
            print('Time is rather then flight time')
            return None
        idx = np.argmin(np.abs(self._time_vector - time))
        return [self._trajectory_x[idx],
                self._trajectory_y[idx],
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
