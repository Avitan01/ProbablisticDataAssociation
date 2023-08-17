import numpy as np


class ProbabilisticDataAssociationFilter:
    """Implement the PDA filter under the assumption mentioned"""
    iX = 0
    iY = 1
    NUMVARS = iY + 1

    def __init__(self, initial_x: float, initial_y: float,
                 Pd: float) -> None:
        """Initialize filter class
            Args:
                initial_x(float): Initial x position of the tracked target in 2D space.
                initial_y(float): Initial y position of the tracked target in 2D space.
                Pd(float): The probability of detecting the true target."""
        self._x = np.zeros(self.NUMVARS)  # State vector
        self._x[self.iX] = initial_x
        self._x[self.iY] = initial_y
        self._P = 10 * np.eye(self.NUMVARS)
        self._z = []
        self._S = []
        self._V = []  # Innovation
        self._W = []  # Gain
        self._measurements = []  # List of sets, each index is the time and the sets are all the measurements at that time
        self._Pd = Pd  # Probability for detection
        self._Pg = 0  # Factor for probability
        self._F = 1*np.eye(self.NUMVARS)  # Transition matrix
        self._F[self.iX, self.iY] = 0.01
        self._F[self.iY, self.iX] = 0.01
        self._H = np.eye(self.NUMVARS)  # Observation matrix
        self._Q = np.zeros(self.NUMVARS)  # Process noise covariance
        self._gamma = 16  # Validation parameter
        self._log_state = []

    def predict(self):
        """Predict the future state, measurement and covariance based on a known model"""
        self._x = self._F.dot(self._x)
        self._z = self._H.dot(self._x)
        self._P = self._F.dot(self._P).dot(self._F.T) + self._Q

    def validate(self, measurements: set, R: np.array):
        """Validate the incoming measurements and define the innovation
            Args:
                measurements(set): A set of measurements containing tuples with the x, y measurements of each point.
                R(np.array): Noise covariance given to us."""
        self._S = self._H.dot(self._P).dot(self._H.T) + R
        validated_measurements = set()
        for measurement in measurements:
            validation_region = (measurement - self._z).T.dot(np.linalg.inv(self._S)).dot(measurement - self._z)
            if validation_region <= self._gamma:
                validated_measurements.add(measurement)
        """Create validation region based on the measurement"""
        # Todo: create measurements as list or set and create validation range
        pass

    def associate(self):
        pass

    def evaluate_association_probability(self):
        pass

    def update(self):
        """Update the state and covariance based on the measurements and innovation"""
        pass

    def run_filter(self, time):
        for _ in time:
            self._log_state.append(self._x)
            self.predict()

    @property
    def state(self) -> np.array:
        return self._x

    @property
    def state_log(self) -> list:
        return [(x, y)for x, y in self._log_state]

