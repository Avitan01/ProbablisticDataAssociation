import unittest


class TestTarget(unittest.TestCase):
    """Test Target"""
    def test_target_returning_x_y(self):
        x = 1
        y = 2
        self.assertGreater(y, x)
