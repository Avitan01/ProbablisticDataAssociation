import numpy as np

from Tools.Plotter import Plotter
import matplotlib.pyplot as plt

from DataGeneration.Target import Target
from DataGeneration.Clutter import Clutter
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter

plotter = Plotter()
target = Target(initial_x=0.0, initial_y=0.0, dt=0.1, simulation_duration=10,
                velocity_x=2, velocity_y=3)
clutter = Clutter(dist_type='Uniform', std=6)
pdaf = ProbabilisticDataAssociationFilter(
                initial_x=0.0, initial_y=0.0, initial_v_x=2, initial_v_y=3,
                dt=0.1, Pd=0.8)

plotter.set_axis(plot_title='Target Tracking')
saved_clutter = []
validated_measurements = []
for time in target.time_vector:
    [x_true, y_true, curr_time] = target.get_state(time)
    cluster = clutter.generate_clutter((x_true, y_true))
    saved_clutter.append(cluster)
    cluster.add((x_true, y_true))  # Add noise to true measurements
    pdaf.predict()
    validated_measurements.append(pdaf.validate(cluster))

x_vec, y_vec = target.x_trajectory, target.y_trajectory
plotter.plot_true_values((x_vec, y_vec))

for i, clutter_to_plot in enumerate(saved_clutter):
    if i != 1:
        plotter.plot_clutter(clutter_to_plot, **{'label': ''})
    else:
        plotter.plot_clutter(clutter_to_plot)
for i, measurements_to_plot in enumerate(validated_measurements):
    if measurements_to_plot and i != 1:
        plotter.plot_measurements(measurements_to_plot, **{'label': ''})
    elif measurements_to_plot:
        plotter.plot_measurements(measurements_to_plot)
# pdaf.run_filter(target.time_vector)
# plotter.plot_true_values((x_vec, y_vec))
# plotter.plot_measurements(pdaf.state_log)
# print(pdaf.state_log)
# data = {(1, 2), (2, 3), (4, 4)}
# plotter.plot_clutter(data)


plotter.add_grid()
plotter.add_labels()
plt.show()
