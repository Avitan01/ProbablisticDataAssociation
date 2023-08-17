import unittest
from Tools.Plotter import Plotter


import unittest
from DataGeneration.Target import Target
from DataGeneration.Clutter import Clutter

clutter = Clutter(origin=(1,1), dist_type="normal", norm_std=2)
for _ in range(10):
    measurements = clutter.generate_clutter()
    measurements.add(ture_measuremet)
    
check = next(test)

#class TestTarget(unittest.TestCase):
#    """Test Target.py"""
#    def test_target(self):
#        my_target = Target(initial_x=0,
#                           initial_y=0,
#                           dt=0.1,
#                           simulation_duration=1000,
#                           velocity_x=1,
#                           velocity_y=1)
#        my_target.initiate()
#        x_path = my_target.entire_x
#        y_path = my_target.entire_y
#        time_vec = my_target.entire_time
#        assert len(x_path) > 0
#        assert len(y_path) > 0
#        assert len(time_vec) > #0

#        index = 5
#        dt = 1000/100
#        time = index*dt
#        my_state = my_target.pull_state(index=index)
#        x_test = 0 + 1*time
#        y_test = 0 + 1*time
#        true_state = [x_test,y_test,time]
#        self.assertEqual(my_state,true_state)

