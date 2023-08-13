class Target:
    """Simulates a non-manuvaring target with constant velocity"""


    def __init__(self, inital_x: float,  
                 inital_y: float,  
                 steps: int,
                 simulation_duration: float,  
                 Velocity_x: float,  
                 Velocity_y: float) -> None:
        """
        Args:
            initial_x - initial location on x axis [m]
            initial_y - initial location on y axis [m]
            steps - number of steps in the simulation
            simulation_duration - overall time of simulation
            Velocity_x - velocity on the x axis [m/s]
            Velocity_y - velocity on the y axis [m/s]
        """
        self._x = inital_x
        self._y = inital_y
        self._steps = steps
        self._time = simulation_duration
        self._Vx = Velocity_x
        self._Vx = Velocity_y
        self._trajectory_x = []
        self._trajectory_y = []
        self._time_vector = []
        self._dt = self._time//self._steps

    def initiate(self) -> None:
        """initate simulation"""

        _current_t = 0

        for step in range(self._steps):
            self._trajectory_x.append(self._x)
            self._trajectory_y.append(self._y)
            self._time_vector.append(_current_t)
            self._x = self._x + self.Vx*self._dt
            self._y = self._y + self.Vy*self._dt
            _current_t = step*self._dt
    
    def pull_state(self,specific_time) -> list:
        index = specific_time//self._dt
        return [self._trajectory_x[index],
                self._trajectory_y[index],
                self._time_vector[index]]
        
    @property
    def entire_x(self) -> list:
        return self._trajectory_x
    
    @property
    def entire_y(self) -> list:
        return self._trajectory_y
    
    @property
    def entire_time(self) -> list:
        return self._time_vector
