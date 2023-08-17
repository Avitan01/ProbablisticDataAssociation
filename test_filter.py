import unittest
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter


class TestFilter(unittest.TestCase):
    """Test the Kalman filter implemented"""

    def __init__(self, *args, **kwargs):
        super(TestFilter, self).__init__(*args, **kwargs)
        x0 = 2
        y0 = 3
        self.pdaf = ProbabilisticDataAssociationFilter(
            initial_x=x0, initial_y=y0, Pd=0.8)

    def test_state_vector_size(self):
        self.assertGreater(self.pdaf.state.size, 0)
        self.assertEqual(self.pdaf.state.size, 2)

    def test_Pd_is_logical(self):
        self.assertGreater(self.pdaf._Pd, 0)
        self.assertLess(self.pdaf._Pd, 1)

    def test_F_in_correct_dimensions(self):
        self.assertEqual(self.pdaf._F.shape[0], self.pdaf.state.size)
