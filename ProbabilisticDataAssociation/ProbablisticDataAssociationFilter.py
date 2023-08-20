import numpy as np
from scipy import stats
from scipy import special


class ProbabilisticDataAssociationFilter:
    """Implement the PDA filter under the assumption mentioned"""

    def __init__(self, number_of_state_variables: int, initial_state: tuple,
                 initial_covariance_magnitude: float, transition_matrix: np.array,
                 Pd: float, Pg: float, observation_matrix: np.array, number_of_measurement_variables: int,
                 process_noise_gain: float, measurement_noise_gain: float, validation_size: float,
               ) -> None:
        """Initialize filter class
            Args:
                initial_x(float): Initial x position of the tracked target in 2D space.
                initial_y(float): Initial y position of the tracked target in 2D space.
                initial_v_x(float): iii
                initial_v_y(float): lll
                dt(float): Time interval
                Pd(float): The probability of detecting the true target."""
        self.NUMVARS = number_of_state_variables
        self._x = np.zeros(self.NUMVARS)  # State vector
        for i, initial_val in enumerate(initial_state):
            self._x[i] = initial_val
        self._P = initial_covariance_magnitude * np.eye(self.NUMVARS)
        self._F = transition_matrix  # Transition matrix
        self._z = []
        self._S = []  # Innovation
        self._measurements = []  # List of sets, each index is the time and the sets are all the measurements at that time
        self._V = []
        self._W = []  # Gain
        self._Pd = Pd  # Probability for detection
        self._Pg = Pg  # Factor for probability
        self.NUMMEAS = number_of_measurement_variables
        self._H = observation_matrix # Observation matrix
        self._Q = process_noise_gain * np.eye(self.NUMVARS)  # Process noise covariance
        self._R = measurement_noise_gain * np.eye(self.NUMMEAS)
        self._gamma = validation_size  # Validation parameter
        self._lambda = 0  # Poisson dist of the number of targets in the clutter
        self._log_state = []

    def predict(self):
        """Predict the future state, measurement and covariance based on a known model"""
        self._x = self._F.dot(self._x)  # + np.array([1, 2, 1, 3])
        self._z = self._H.dot(self._x)
        self._P = self._F.dot(self._P).dot(self._F.T) + self._Q

    def validate(self, measurements: set) -> set:
        """Validate the incoming measurements and define the innovation
            Args:
                measurements(set): A set of measurements containing tuples with the x, y measurements of each point."""
        self._S = self._H.dot(self._P).dot(self._H.T) + self._R
        validated_measurements = set()
        for measurement in measurements:
            validation_region = (measurement - self._z).T.dot(np.linalg.inv(self._S)).dot(measurement - self._z)
            if validation_region <= self._gamma:
                validated_measurements.add(measurement)
        self._measurements.append(validated_measurements)
        return validated_measurements
        # Todo: create measurements as list or set and create validation range

    def associate(self, validated_measurement: set) -> tuple:
        likelihood = []
        beta = []
        nu = []
        c_hypersphere = np.pi ** (self.NUMMEAS / 2) / special.gamma((self.NUMMEAS / 2) + 1)
        self._V = c_hypersphere * self._gamma ** (self.NUMVARS / 2) * np.sqrt(np.linalg.det(self._S))
        self._lambda = len(validated_measurement) / self._V
        for valid_meas in validated_measurement:
            nu.append(valid_meas - self._z)
            pdf = stats.multivariate_normal.pdf(valid_meas, self._z, self._S)
            likelihood.append((pdf * self._Pd) / self._lambda)
        total_likelihood = sum(likelihood)
        for likely in likelihood:
            beta.append(
                likely / (1 - self._Pd * self._Pg + total_likelihood)
            )
        beta_zero = (1 - self._Pd * self._Pg) / (1 - self._Pd * self._Pg + total_likelihood)
        innovation_series = [beta_i * nu_i for beta_i, nu_i in zip(beta, nu)]  # vector nu_i * beta_i
        if innovation_series:
            sum_beta_double_nu = sum(np.array(innovation_series).T.dot((np.array(nu))))
            spread_of_cov = sum_beta_double_nu - (np.array(innovation_series).T.dot(np.array(innovation_series)))
        else:
            return np.zeros((2,)), beta_zero, np.zeros((2, 2))
        return sum(innovation_series), beta_zero, spread_of_cov

    def update(self, measurements: set):
        valid_measurement = self.validate(measurements)
        self._W = self._P.dot(self._H.T).dot(np.linalg.inv(self._S))
        combined_innovation, beta_zero, spread_cov = self.associate(valid_measurement)
        self._x = self._x + self._W.dot(combined_innovation)
        P_correct = self._P - self._W.dot(self._S).dot(self._W.T)
        spread_of_covariance = self._W.dot(spread_cov).dot(self._W.T)
        self._P = beta_zero*self._P + (1-beta_zero)*P_correct + spread_of_covariance
        return valid_measurement

    def run_filter(self, time):
        for _ in time:
            self._log_state.append(self._x)
            self.predict()

    @property
    def state(self) -> np.array:
        return self._x

    @property
    def state_log(self) -> list:
        return [(x, y) for x, y in self._log_state]
