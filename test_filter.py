import unittest
import numpy as np
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter
from DataGeneration.Clutter import Clutter
from DataGeneration.Target import Target

class TestFilter(unittest.TestCase):
    """Test the Kalman filter implemented"""

    def __init__(self, *args, **kwargs):
        super(TestFilter, self).__init__(*args, **kwargs)
        dt = 0.1
        self.target = Target(
            initial_x=0.0, initial_y=0.0, dt=dt, simulation_duration=10,
            initial_vx=1, initial_vy=2, system_variance=0.1
        )
        self.clutter = Clutter(dist_type='uniform', std=0.5, clutter_size=20)
        self.state_size = 4
        #                 x    y   vx   vy
        initial_state = (0.0, 0.0, 1.0, 2.0)
        initial_covariance_magnitude = 10
        transition_matrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        Pd = 0.3  # Probability for detection
        Pg = 0.995  # Factor for probability
        observation_size = 2
        observation_matrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]]
        )

        process_noise_gain = 0.01 ** 2
        measurement_noise_gain = 7 ** 2

        self.pdaf = ProbabilisticDataAssociationFilter(
            self.state_size, initial_state, initial_covariance_magnitude,
            transition_matrix, Pd, Pg, observation_matrix, observation_size,
            process_noise_gain, measurement_noise_gain
        )

    def test_state_vector_size(self):
        self.assertGreater(self.pdaf.state.size, 0)
        self.assertEqual(self.pdaf.state.size, self.state_size)

    def test_Pd_is_logical(self):
        self.assertGreater(self.pdaf._Pd, 0)
        self.assertLess(self.pdaf._Pd, 1)

    def test_F_in_correct_dimensions(self):
        self.assertEqual(self.pdaf._F.shape[0], self.pdaf.state.size)

    def test_prediction_increases_uncertainty(self):
        cov_before = np.linalg.det(self.pdaf.covariance)
        self.pdaf.predict()
        cov_after = np.linalg.det(self.pdaf.covariance)
        self.assertGreater(cov_after, cov_before)

    def test_update_decreases_uncertainty(self):
        cov_before = np.linalg.det(self.pdaf.covariance)
        [x_true, y_true, _, _, curr_time] = self.target.get_state(self.target.time_vector[0])
        self.pdaf.update(self.clutter.generate_clutter((x_true, y_true)))
        cov_after = np.linalg.det(self.pdaf.covariance)
        self.assertGreater(cov_after, cov_before)
